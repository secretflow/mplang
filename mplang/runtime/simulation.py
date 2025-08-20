# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import concurrent.futures
import faulthandler
import sys
import traceback
from collections.abc import Sequence
from typing import Any, cast

import spu.libspu as libspu

from mplang.backend.builtin import BuiltinHandler
from mplang.backend.spu import SpuHandler
from mplang.backend.sql_duckdb import DuckDBHandler
from mplang.backend.stablehlo import StablehloHandler
from mplang.core.interp import InterpContext, InterpVar
from mplang.core.mask import Mask
from mplang.core.mpir import Reader, Writer
from mplang.core.mpobject import MPObject
from mplang.core.mptype import MPType, TensorLike
from mplang.expr.ast import Expr
from mplang.expr.evaluator import Evaluator
from mplang.runtime.grpc_comm import LinkCommunicator
from mplang.runtime.mem_comm import ThreadCommunicator


class SimVar(InterpVar):
    """A variable that references a value in an interpreter.

    SimVar represents a value that has been computed and exists
    in the interpreter's variable store.
    """

    def __init__(self, ctx: Simulator, mptype: MPType, values: list[Any]):
        # Initialize the parent InterpVar with a generated name
        super().__init__(ctx, mptype)
        self._values = values

    @property
    def values(self) -> list[Any]:
        """The values of this variable across all ranks."""
        return self._values

    def __repr__(self) -> str:
        return f"SimVar({self.mptype})"


class Simulator(InterpContext):
    def __init__(
        self,
        psize: int,
        *,
        spu_protocol: libspu.ProtocolKind = libspu.ProtocolKind.SEMI2K,
        spu_field: libspu.FieldType = libspu.FieldType.FM64,
        spu_mask: Mask | None = None,
        trace_ranks: list[int] | None = None,
        **attrs: Any,
    ) -> None:
        """Initialize a simulator with the given process size and attributes."""
        if trace_ranks is None:
            trace_ranks = []
        all_attrs = {
            "trace_ranks": trace_ranks,
            "spu_protocol": int(spu_protocol),
            "spu_field": int(spu_field),
            "spu_mask": spu_mask or Mask((1 << psize) - 1),
            **attrs,
        }
        super().__init__(psize, all_attrs)

        # Setup communicators
        self._comms = [ThreadCommunicator(rank, psize) for rank in range(psize)]
        for comm in self._comms:
            comm.set_peers(self._comms)

        # Prepare link context and spu handlers.
        spu_mask_attr = self.attr("spu_mask")
        if spu_mask_attr is None:
            raise ValueError("spu_mask attribute is required")
        # Ensure spu_mask_attr is a Mask object
        if not isinstance(spu_mask_attr, Mask):
            spu_mask_attr = Mask(spu_mask_attr)
        spu_addrs = [f"P{spu_rank}" for spu_rank in spu_mask_attr]
        spu_comms = [
            LinkCommunicator(idx, spu_addrs, mem_link=True)
            for idx in range(spu_mask_attr.num_parties())
        ]
        spu_config = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.SEMI2K,
            field=libspu.FieldType.FM64,
        )
        # Create separate SpuHandler instances for each party to avoid sharing state
        spu_handlers = [
            SpuHandler(spu_mask_attr.num_parties(), spu_config) for _ in range(psize)
        ]
        for rank, handler in enumerate(spu_handlers):
            handler.set_link_context(
                spu_comms[spu_mask_attr.global_to_relative_rank(rank)]
                if rank in spu_mask_attr
                else None
            )

        # Setup evaluators
        self._evaluators = [
            Evaluator(
                rank,
                {},  # the global environment for this rank
                self._comms[rank],
                [
                    BuiltinHandler(),
                    StablehloHandler(),
                    spu_handlers[rank],
                    DuckDBHandler(),
                ],
            )
            for rank in range(self.psize())
        ]

    def _do_evaluate(self, expr: Expr, evaluator: Evaluator) -> Any:
        """
        Helper function to simulate real-world MPIR serialization/deserialization
        process instead of direct expr.accept execution.

        This exposes potential MPIR serialization bugs by forcing expressions
        to go through the full serialize->deserialize cycle.
        """
        writer = Writer()
        graph_proto = writer.dumps(expr)

        reader = Reader()
        deserialized_expr = reader.loads(graph_proto)

        if deserialized_expr is None:
            raise ValueError("Failed to deserialize expression")

        return deserialized_expr.accept(evaluator)

    # override
    def fetch(self, obj: MPObject) -> list[TensorLike]:
        if not isinstance(obj, SimVar):
            raise ValueError(f"Expected SimVar, got {type(obj)}")

        return list(obj.values)

    # override
    def evaluate(self, expr: Expr, bindings: dict[str, MPObject]) -> Sequence[MPObject]:
        # sanity check for bindings.
        for name, var in bindings.items():
            if var.ctx is not self:
                raise ValueError(f"Variable {name} not in this context, got {var.ctx}.")

        pts_env = [
            {name: cast(SimVar, var).values[rank] for name, var in bindings.items()}
            for rank in range(self.psize())
        ]

        pts_evaluators = [
            self._evaluators[rank].fork(pts_env[rank]) for rank in range(self.psize())
        ]

        # Collect evaluation results from all parties
        pts_results: list[Any] = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._do_evaluate, expr, evaluator)
                for evaluator in pts_evaluators
            ]

            # Collect results with proper exception handling
            for i, future in enumerate(futures):
                try:
                    result = future.result(100)  # 100 second timeout
                    pts_results.append(result)
                except concurrent.futures.TimeoutError:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    raise
                except Exception as e:
                    print(
                        f"Exception in party {i}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

        # Convert results to SimVar objects
        # pts_results is a list of party results, where each party result is a list of values
        # We need to transpose this to get (n_outputs, n_parties) structure
        assert len(pts_results) == self.psize()

        # Ensure all parties returned the same number of outputs (matrix validation)
        if pts_results and not all(
            len(row) == len(pts_results[0]) for row in pts_results
        ):
            raise ValueError("Inconsistent number of outputs across parties")

        # Transpose: (n_parties, n_outputs) -> (n_outputs, n_parties)
        output_values = list(zip(*pts_results, strict=False))

        # Get the output types from the expression
        output_types = expr.mptypes

        # Create SimVar objects for each output
        sim_vars = []
        for values, mptype in zip(output_values, output_types, strict=False):
            sim_var = SimVar(self, mptype, list(values))
            sim_vars.append(sim_var)

        return sim_vars
