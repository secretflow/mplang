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

import asyncio
import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import spu.libspu as libspu

from mplang.v1.core import (
    ClusterSpec,
    CollectiveMixin,
    CommunicatorBase,
    InterpContext,
    InterpVar,
    IrReader,
    IrWriter,
    Mask,
    MPObject,
    MPType,
    PFunction,  # for spu.seed_env kernel seeding
    TensorLike,
)
from mplang.v1.core.async_comm import AsyncThreadCommunicator
from mplang.v1.core.expr.ast import Expr
from mplang.v1.core.expr.async_evaluator import (
    AsyncIterativeEvaluator,
)
from mplang.v1.core.expr.evaluator import IEvaluator
from mplang.v1.kernels.context import RuntimeContext
from mplang.v1.runtime.link_comm import LinkCommunicator
from mplang.v1.utils.spu_utils import parse_field, parse_protocol


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
        """Converted values across all ranks for user inspection."""
        return [v.to_numpy() if hasattr(v, "to_numpy") else v for v in self._values]

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
                Per-node op binding overrides should now be provided via
                each node's `runtime_info.op_bindings` in the supplied
                `cluster_spec`.
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

        # Executor for CPU-bound tasks
        self._executor = ThreadPoolExecutor(max_workers=os.cpu_count())

        # Persistent per-rank RuntimeContext instances (reused across evaluates).
        # We no longer pre-create evaluators since each evaluate has different env bindings.
        # Build per-rank runtime contexts.
        self._runtimes: list[RuntimeContext] = []
        for rank in range(self.world_size()):
            node = self.cluster_spec.get_node_by_rank(rank)
            rt = RuntimeContext(
                rank=rank,
                world_size=self.world_size(),
                initial_bindings=node.runtime_info.op_bindings,
            )
            self._runtimes.append(rt)

    @classmethod
    def simple(
        cls,
        world_size: int,
        op_bindings: dict[str, str] | None = None,
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
        if op_bindings:
            # Apply the same op_bindings to every node's runtime_info for convenience
            for node in cluster_spec.nodes.values():
                node.runtime_info.op_bindings.update(op_bindings)
        return cls(cluster_spec, **kwargs)

    def _do_evaluate(self, expr: Expr, evaluator_engine: IEvaluator) -> Any:
        """
        Helper function to simulate real-world MPIR serialization/deserialization
        process instead of direct expr.accept execution.

        This exposes potential MPIR serialization bugs by forcing expressions
        to go through the full serialize->deserialize cycle.
        """
        writer = IrWriter()
        graph_proto = writer.dumps(expr)

        reader = IrReader()
        deserialized_expr = reader.loads(graph_proto)

        if deserialized_expr is None:
            raise ValueError("Failed to deserialize expression")

        return evaluator_engine.evaluate(deserialized_expr)

    # override
    def fetch(self, obj: MPObject) -> list[TensorLike]:
        if not isinstance(obj, SimVar):
            raise ValueError(f"Expected SimVar, got {type(obj)}")
        return [v.to_numpy() if hasattr(v, "to_numpy") else v for v in obj._values]

    def _ensure_spu_init(self, rank: int) -> None:
        """Ensure SPU environment is initialized for the given rank."""
        runtime = self._runtimes[rank]
        spu_meta = runtime.state.setdefault("_spu", {})
        if not spu_meta.get("inited", False):
            link_ctx = self._spu_link_ctxs[rank]
            seed_fn = PFunction(
                fn_type="spu.seed_env",
                ins_info=(),
                outs_info=(),
                config=self._spu_runtime_cfg,
                world=self._spu_world,
                link=link_ctx,
            )
            runtime.run_kernel(seed_fn, [])  # type: ignore[arg-type]
            spu_meta["inited"] = True

    # override
    def evaluate(self, expr: Expr, bindings: dict[str, MPObject]) -> Sequence[MPObject]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Case A: Inside an existing loop (e.g., Jupyter)
            try:
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(self._evaluate_async(expr, bindings))
            except ImportError as e:
                raise RuntimeError(
                    "Running in an active event loop (e.g. Jupyter). "
                    "Please install 'nest_asyncio' or use 'await simulator.evaluate_async(...)'."
                ) from e
        else:
            # Case B: Standard script
            return asyncio.run(self._evaluate_async(expr, bindings))

    async def _evaluate_async(
        self, expr: Expr, bindings: dict[str, MPObject]
    ) -> Sequence[MPObject]:
        """Async evaluation entry point."""
        # 1. Setup Async Communicators
        world_size = self.world_size()
        async_comms = [
            AsyncThreadCommunicator(rank, world_size) for rank in range(world_size)
        ]
        for comm in async_comms:
            comm.set_peers(async_comms)

        # 2. Prepare Environment
        # Validate that all variables belong to this simulator context
        for name, var in bindings.items():
            if not isinstance(var, SimVar):
                raise ValueError(
                    f"Expected SimVar for variable '{name}', got {type(var)}"
                )
            if var.ctx is not self:
                raise ValueError(f"Variable '{name}' not in this context")

        pts_env = [
            {name: cast(SimVar, var)._values[rank] for name, var in bindings.items()}
            for rank in range(world_size)
        ]

        # 3. Create Evaluators
        evaluators = []
        for rank in range(world_size):
            runtime = self._runtimes[rank]
            # Initialize SPU if needed (same logic as sync)
            self._ensure_spu_init(rank)

            ev = AsyncIterativeEvaluator(
                rank=rank,
                env=pts_env[rank],
                comm=async_comms[rank],
                runtime=runtime,
                executor=self._executor,
            )
            evaluators.append(ev)

        # 4. Run Evaluation concurrently
        # We need to run all evaluators.evaluate(expr) concurrently.
        tasks = [ev.evaluate(expr) for ev in evaluators]
        pts_results = await asyncio.gather(*tasks)

        # Ensure results are lists if expr has single output
        if expr.num_outputs == 1:
            # If each evaluator already returns a list (as async evaluators do), don't wrap again
            if pts_results and not isinstance(pts_results[0], list):
                pts_results = [[res] for res in pts_results]

        # 5. Process Results (Transpose and Wrap)
        assert len(pts_results) == world_size
        if pts_results and not all(
            len(row) == len(pts_results[0]) for row in pts_results
        ):
            raise ValueError("Inconsistent number of outputs across parties")

        output_values = list(zip(*pts_results, strict=False))
        output_types = expr.mptypes
        sim_vars = []
        for values, mptype in zip(output_values, output_types, strict=False):
            sim_var = SimVar(self, mptype, list(values))
            sim_vars.append(sim_var)

        return sim_vars
