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

"""SIMP simulator module.

Provides SimpSimulator for local multi-threaded simulation.
This is useful for development and testing without network deployment.
"""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
import threading
from typing import Any

from mplang.v2.backends import simp_impl as _simp_impl  # noqa: F401
from mplang.v2.backends.simp_host import SimpHost
from mplang.v2.backends.simp_worker import WorkerInterpreter
from mplang.v2.edsl.graph import Graph
from mplang.v2.runtime.interpreter import ExecutionTracer


class ThreadCommunicator:
    """Thread-based communicator for in-memory communication.

    Args:
        rank: This communicator's rank.
        world_size: Total number of parties.
        use_serde: If True, serialize/deserialize data through serde on send,
            simulating real HTTP communication behavior.
    """

    def __init__(self, rank: int, world_size: int, *, use_serde: bool = False):
        self.rank = rank
        self.world_size = world_size
        self.use_serde = use_serde
        self.peers: list[ThreadCommunicator] = []
        self._mailbox: dict[str, Any] = {}
        self._cond = threading.Condition()
        self._sent_events: dict[str, threading.Event] = {}
        self._shutdown = False

    def set_peers(self, peers: list[ThreadCommunicator]) -> None:
        assert len(peers) == self.world_size
        self.peers = peers

    def shutdown(self) -> None:
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()

    def send(self, to: int, key: str, data: Any) -> None:
        assert 0 <= to < self.world_size
        # Optionally round-trip through serde to simulate HTTP communication
        if self.use_serde:
            from mplang.v2.edsl import serde

            data = serde.loads(serde.dumps(data))
        self.peers[to]._on_receive(self.rank, key, data)

    def recv(self, frm: int, key: str) -> Any:
        with self._cond:
            while key not in self._mailbox and not self._shutdown:
                self._cond.wait()
            if self._shutdown:
                raise RuntimeError("Communicator shut down")
            return self._mailbox.pop(key)

    def _on_receive(self, frm: int, key: str, data: Any) -> None:
        with self._cond:
            if key in self._mailbox:
                raise RuntimeError(
                    f"Mailbox overflow for key {key} at rank {self.rank}"
                )
            self._mailbox[key] = data
            self._cond.notify_all()


class Context:
    """Context for SIMP simulation."""

    def __init__(self, world_size: int, *, use_serde: bool = False):
        self.world_size = world_size
        self.use_serde = use_serde
        self.comms = [
            ThreadCommunicator(i, world_size, use_serde=use_serde)
            for i in range(world_size)
        ]
        for comm in self.comms:
            comm.set_peers(self.comms)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=world_size)

    def shutdown(self, wait: bool = True) -> None:
        for comm in self.comms:
            comm.shutdown()
        self.executor.shutdown(wait=wait)


# Global simulation context (can be set by user or initialized lazily)
_SIM_CONTEXT: Context | None = None


def get_or_create_context(world_size: int = 3, *, use_serde: bool = False) -> Context:
    global _SIM_CONTEXT
    if _SIM_CONTEXT is None:
        _SIM_CONTEXT = Context(world_size, use_serde=use_serde)
    elif _SIM_CONTEXT.world_size != world_size or _SIM_CONTEXT.use_serde != use_serde:
        # Recreate context if world_size or use_serde mismatch
        _SIM_CONTEXT.shutdown(wait=True)
        _SIM_CONTEXT = Context(world_size, use_serde=use_serde)
    return _SIM_CONTEXT


class SimpSimulator(SimpHost):
    """SIMP simulator running locally with threads.

    Args:
        world_size: Number of parties to simulate.
        use_serde: If True, serialize/deserialize data through serde on inter-party
            communication (send/recv). This simulates real HTTP behavior and validates
            that all transmitted types are properly registered with serde.
    """

    def __init__(
        self,
        world_size: int = 3,
        *,
        root_dir: pathlib.Path | None = None,
        use_serde: bool = False,
        enable_tracing: bool = False,
    ):
        """Initialize SimpSimulator.

        Args:
            world_size: Number of parties.
            root_dir: Host root directory. If None, generates default path.
            use_serde: Enable serialization for communication.
            enable_tracing: Enable execution tracing.
        """
        # Generate default root_dir if not provided
        if root_dir is None:
            data_root = pathlib.Path(os.environ.get("MPLANG_DATA_ROOT", ".mpl"))
            root_dir = data_root / f"__sim_{world_size}" / "__host__"

        super().__init__(world_size, root_dir=root_dir)
        self.use_serde = use_serde
        self.ctx = get_or_create_context(world_size, use_serde=use_serde)

        # Derive cluster_root from root_dir (root_dir is <cluster>/__host__)
        self.cluster_root = root_dir.parent

        # Tracer for host-side execution events
        self.tracer = ExecutionTracer(
            enabled=enable_tracing, trace_dir=self.root_dir / "trace"
        )
        self.tracer.start()

        # Workers with isolated sandboxes
        self.workers = [
            WorkerInterpreter(
                rank,
                world_size,
                self.ctx.comms[rank],
                tracer=self.tracer,
                root_dir=self.cluster_root / f"node{rank}",
            )
            for rank in range(world_size)
        ]

    def set_worker_executor(
        self, executor: concurrent.futures.Executor, async_ops: set[str]
    ) -> None:
        """Configure workers to use an executor for specific ops."""
        for worker in self.workers:
            worker.executor = executor
            worker.async_ops = async_ops

    def _submit(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        return self.ctx.executor.submit(
            self._run_party, rank, graph, inputs, job_id=job_id
        )

    def _collect(self, futures: list[Any]) -> list[Any]:
        # Wait for all to complete, or the first exception
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_EXCEPTION
        )

        # If any future raised an exception, re-raise it immediately
        for f in done:
            exc = f.exception()
            if exc:
                # Cancel pending futures (best effort)
                for nf in futures:
                    nf.cancel()
                # Shutdown context to unblock running threads
                self.ctx.shutdown(wait=False)
                raise exc

        # If no exceptions, all futures should be done (because FIRST_EXCEPTION
        # implies ALL_COMPLETED if no exception occurs)
        return [f.result() for f in futures]

    def _run_party(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        worker = self.workers[rank]
        if not isinstance(graph, Graph):
            raise TypeError(
                f"SimpSimulator only supports executing Graph tasks, got {type(graph)!r}"
            )
        return worker.execute_job(graph, inputs, job_id=job_id)

    def _fetch(self, rank: int, uri: str) -> Any:
        return self.ctx.executor.submit(lambda: self.workers[rank].store.get(uri))

    def shutdown(self, wait: bool = True) -> None:
        global _SIM_CONTEXT
        if self.ctx:
            self.ctx.shutdown(wait=wait)
        if _SIM_CONTEXT is self.ctx:
            _SIM_CONTEXT = None
