# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simp Driver memory IPC (MemCluster, SimpMemDriver, make_simulator)."""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from mplang.v2.backends.simp_driver.state import SimpDriver
from mplang.v2.backends.simp_worker import WORKER_HANDLERS, SimpWorker
from mplang.v2.backends.simp_worker.mem import LocalMesh
from mplang.v2.runtime.interpreter import ExecutionTracer, Interpreter
from mplang.v2.runtime.object_store import ObjectStore

if TYPE_CHECKING:
    from concurrent.futures import Future

    from mplang.v2.edsl.graph import Graph
    from mplang.v2.edsl.spec import ClusterSpec


class MemCluster:
    """Orchestrator that creates and manages local worker Interpreters.

    This class handles worker lifecycle management. It does NOT attach to
    an Interpreter - instead, it creates a SimpMemDriver that can be attached.
    """

    def __init__(
        self,
        world_size: int,
        *,
        cluster_spec: ClusterSpec | None = None,
        enable_tracing: bool = False,
    ) -> None:
        """Create a local memory cluster.

        Args:
            world_size: Number of workers.
            cluster_spec: Optional cluster specification for metadata.
            enable_tracing: If True, enable execution tracing.
        """
        self._world_size = world_size
        self._cluster_spec = cluster_spec

        # Construct root_dir from cluster_id
        data_root = pathlib.Path(os.environ.get("MPLANG_DATA_ROOT", ".mpl"))
        cluster_id = cluster_spec.cluster_id if cluster_spec else f"local_{world_size}"
        cluster_root = data_root / cluster_id
        self.host_root = cluster_root / "__host__"

        # Create Local Mesh (communication mesh for workers)
        self._mesh = LocalMesh(world_size)

        # Create Execution Tracer
        self.tracer: ExecutionTracer = ExecutionTracer(
            enabled=enable_tracing, trace_dir=self.host_root / "trace"
        )
        self.tracer.start()

        # Create Workers
        self._workers: list[Interpreter] = []
        for rank in range(world_size):
            worker_root = cluster_root / f"node{rank}"
            store = ObjectStore(fs_root=str(worker_root / "store"))

            worker_state = SimpWorker(
                rank=rank,
                world_size=world_size,
                communicator=self._mesh.comms[rank],
                store=store,
            )

            w_handlers: dict[str, Callable[..., Any]] = {**WORKER_HANDLERS}  # type: ignore[dict-item]
            w_interp = Interpreter(
                name=f"Worker-{rank}",
                tracer=self.tracer,
                trace_pid=rank,
                store=store,
                root_dir=worker_root,
                handlers=w_handlers,
            )
            w_interp.set_dialect_state("simp", worker_state)

            w_interp.async_ops = {
                "bfv.add",
                "bfv.mul",
                "bfv.rotate",
                "bfv.batch_encode",
                "bfv.relinearize",
                "bfv.encrypt",
                "bfv.decrypt",
                "field.solve_okvs",
                "field.decode_okvs",
                "field.aes_expand",
                "field.mul",
                "simp.shuffle",
            }
            self._workers.append(w_interp)

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def workers(self) -> list[Interpreter]:
        return self._workers

    def create_state(self) -> SimpMemDriver:
        """Create a SimpMemDriver that can be attached to a Driver Interpreter."""
        return SimpMemDriver(
            world_size=self._world_size,
            workers=self._workers,
            mesh=self._mesh,
        )

    def shutdown(self, wait: bool = True) -> None:
        """Stop all workers and release resources."""
        self._mesh.shutdown(wait=wait)


class SimpMemDriver(SimpDriver):
    """Simp Driver for local memory IPC.

    Implements submit/fetch/collect interface for dispatching work to local workers.
    This class is created by MemCluster and attached to a Driver Interpreter.
    """

    dialect_name: str = "simp"

    def __init__(
        self,
        world_size: int,
        workers: list[Interpreter],
        mesh: Any,  # LocalMesh from simp_worker.mem
    ) -> None:
        self._world_size = world_size
        self._workers = workers
        self._mesh = mesh

    def shutdown(self) -> None:
        """Shutdown the local memory driver and its mesh."""
        self._mesh.shutdown()

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def workers(self) -> list[Interpreter]:
        """Worker interpreters (for backward compatibility)."""
        return self._workers

    def submit(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Future[Any]:
        """Submit execution to local worker thread."""
        return cast(
            "Future[Any]",
            self._mesh.executor.submit(
                self._run_worker, rank, graph, inputs, job_id=job_id
            ),
        )

    def collect(self, futures: list[Future[Any]]) -> list[Any]:
        """Wait for threads and collect results."""
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_EXCEPTION
        )
        for f in done:
            exc = f.exception()
            if exc:
                for nf in futures:
                    nf.cancel()
                self._mesh.shutdown(wait=False)
                raise exc
        return [f.result() for f in futures]

    def fetch(self, rank: int, uri: str) -> Future[Any]:
        """Fetch directly from worker store."""
        worker = self._workers[rank]
        worker_ctx = cast(SimpWorker, worker.get_dialect_state("simp"))
        return self._mesh.executor.submit(lambda: worker_ctx.store.get(uri))  # type: ignore[no-any-return]

    def _run_worker(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Execute on worker interpreter."""
        worker_interp = self._workers[rank]
        worker_ctx = cast(SimpWorker, worker_interp.get_dialect_state("simp"))

        # Resolve URI inputs (None means rank has no data)
        resolved_inputs = [
            worker_ctx.store.get(inp) if inp is not None else None for inp in inputs
        ]

        # Execute
        results = worker_interp.evaluate_graph(graph, resolved_inputs, job_id)

        # Store results (results is always a list)
        if not graph.outputs:
            return None
        return [
            worker_ctx.store.put(res) if res is not None else None for res in results
        ]


def make_simulator(
    world_size: int,
    *,
    cluster_spec: Any = None,
    enable_tracing: bool = False,
) -> Interpreter:
    """Create an Interpreter configured for local SIMP simulation.

    This factory creates a MemCluster with workers and returns an
    Interpreter with the simp dialect state attached.

    Args:
        world_size: Number of simulated parties.
        cluster_spec: Optional ClusterSpec for metadata.
        enable_tracing: If True, enable execution tracing.

    Returns:
        Configured Interpreter with simp state attached.

    Example:
        >>> interp = make_simulator(2)
        >>> with interp:
        ...     result = my_func()
    """
    from mplang.v2.backends.simp_driver.ops import DRIVER_HANDLERS

    if cluster_spec is None:
        from mplang.v2.libs.device import ClusterSpec

        cluster_spec = ClusterSpec.simple(world_size)

    cluster = MemCluster(
        world_size=world_size,
        cluster_spec=cluster_spec,
        enable_tracing=enable_tracing,
    )
    state = cluster.create_state()

    handlers: dict[str, Callable[..., Any]] = {**DRIVER_HANDLERS}  # type: ignore[dict-item]
    interp = Interpreter(
        name="HostInterpreter",
        root_dir=cluster.host_root,
        handlers=handlers,
        tracer=cluster.tracer,
        store=ObjectStore(fs_root=str(cluster.host_root)),
    )
    interp.set_dialect_state("simp", state)

    # Keep cluster alive (prevent GC)
    interp._simp_cluster = cluster  # type: ignore[attr-defined]
    interp._cluster_spec = cluster_spec  # type: ignore[attr-defined]

    return interp


# Backward compatibility alias
LocalCluster = MemCluster
