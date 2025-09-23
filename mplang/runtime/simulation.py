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
import logging
import sys
import traceback
from collections.abc import Sequence
from typing import Any, cast

import spu.libspu as libspu

# Import flat backends for kernel registration side-effects
from mplang.backend import (
    builtin,  # noqa: F401
    crypto,  # noqa: F401
    phe,  # noqa: F401
    spu,  # noqa: F401  # ensure SPU kernels (spu.seed_env etc.) registered
    sql_duckdb,  # noqa: F401
    stablehlo,  # noqa: F401
    tee,  # noqa: F401
)
from mplang.backend.base import create_runtime  # explicit per-rank backend runtime
from mplang.core.cluster import ClusterSpec
from mplang.core.comm import CollectiveMixin, CommunicatorBase
from mplang.core.expr.ast import Expr
from mplang.core.expr.evaluator import IEvaluator, create_evaluator
from mplang.core.interp import InterpContext, InterpVar
from mplang.core.mask import Mask
from mplang.core.mpir import Reader, Writer
from mplang.core.mpobject import MPObject
from mplang.core.mptype import MPType, TensorLike
from mplang.core.pfunc import PFunction  # for spu.seed_env kernel seeding
from mplang.runtime.link_comm import LinkCommunicator
from mplang.utils.spu_utils import parse_field, parse_protocol


class ThreadCommunicator(CommunicatorBase, CollectiveMixin):
    """Thread-based communicator for in-memory communication between threads"""

    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.peers: list[ThreadCommunicator] = []
        logging.debug(
            f"ThreadCommunicator initialized with rank={self.rank}, world_size={self.world_size}"
        )

    def set_peers(self, peers: list[ThreadCommunicator]) -> None:
        assert self.world_size == len(peers)
        self.peers = peers

    def send(self, to: int, key: str, data: Any) -> None:
        assert 0 <= to < self.world_size
        # print(f"send {key}: {self.rank} -> {to_rank}")
        self.peers[to].onSent(self.rank, key, data)


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
        cluster_spec: ClusterSpec,
        *,
        trace_ranks: list[int] | None = None,
    ) -> None:
        """Initialize a simulator with the given cluster specification.

        Args:
            cluster_spec: The cluster specification defining the simulation environment.
            trace_ranks: List of ranks to trace execution for debugging.
        """
        super().__init__(cluster_spec)
        self._trace_ranks = trace_ranks or []

        spu_devices = cluster_spec.get_devices_by_kind("SPU")
        if not spu_devices:
            raise ValueError("No SPU device found in the cluster specification")
        if len(spu_devices) > 1:
            raise ValueError("Multiple SPU devices found in the cluster specification")
        spu_device = spu_devices[0]

        # compute spu_mask from spu_device members
        spu_mask = Mask.from_ranks([member.rank for member in spu_device.members])

        # Convert protocol and field from config using utility functions
        spu_protocol = parse_protocol(spu_device.config["protocol"])
        spu_field = parse_field(spu_device.config["field"])

        world_size = self.world_size()

        # Setup communicators
        self._comms = [
            ThreadCommunicator(rank, world_size) for rank in range(world_size)
        ]
        for comm in self._comms:
            comm.set_peers(self._comms)

        # Prepare link contexts for SPU parties (store for evaluator-time initialization)
        spu_addrs = [f"P{spu_rank}" for spu_rank in spu_mask]
        self._spu_link_ctxs: list[LinkCommunicator | None] = [None] * world_size
        link_ctx_list = [
            LinkCommunicator(idx, spu_addrs, mem_link=True)
            for idx in range(spu_mask.num_parties())
        ]
        for g_rank in range(world_size):
            if g_rank in spu_mask:
                rel = Mask(spu_mask).global_to_relative_rank(g_rank)
                self._spu_link_ctxs[g_rank] = link_ctx_list[rel]

        self._spu_runtime_cfg = libspu.RuntimeConfig(
            protocol=spu_protocol, field=spu_field
        )
        self._spu_world = spu_mask.num_parties()
        self._spu_mask = spu_mask

        # No per-backend handlers needed anymore (all flat kernels)
        self._handlers: list[list[Any]] = [[] for _ in range(self.world_size())]

        self._evaluators: list[IEvaluator] = []
        for rank in range(self.world_size()):
            runtime = create_runtime(rank, self.world_size())
            ev = create_evaluator(
                rank,
                {},  # the global environment for this rank
                self._comms[rank],
                runtime,
                None,
            )
            self._evaluators.append(ev)

    @classmethod
    def simple(
        cls,
        world_size: int,
        **kwargs: Any,
    ) -> Simulator:
        """Create a simple simulator with the given number of parties.

        This is a convenience method that creates a ClusterSpec.simple()
        configuration for quick testing and prototyping.

        Args:
            world_size: Number of simulated parties.
            **kwargs: Additional arguments passed to the Simulator constructor.

        Returns:
            A Simulator instance with a simple cluster configuration.
        """
        cluster_spec = ClusterSpec.simple(world_size)
        return cls(cluster_spec, **kwargs)

    def _do_evaluate(self, expr: Expr, evaluator_engine: IEvaluator) -> Any:
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

        return evaluator_engine.evaluate(deserialized_expr)

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
            for rank in range(self.world_size())
        ]

        # Build per-rank evaluators with the per-party environment
        pts_evaluators: list[IEvaluator] = []
        for rank in range(self.world_size()):
            runtime = create_runtime(rank, self.world_size())
            ev = create_evaluator(
                rank,
                pts_env[rank],
                self._comms[rank],
                runtime,
                None,
            )
            link_ctx = self._spu_link_ctxs[rank]
            seed_fn = PFunction(
                fn_type="spu.seed_env",
                ins_info=(),
                outs_info=(),
                config=self._spu_runtime_cfg,
                world=self._spu_world,
                link=link_ctx,
            )
            # Seed SPU backend environment explicitly via runtime (no evaluator fast-path)
            ev.runtime.run_kernel(seed_fn, [])  # type: ignore[arg-type]
            pts_evaluators.append(ev)

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
        assert len(pts_results) == self.world_size()

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
