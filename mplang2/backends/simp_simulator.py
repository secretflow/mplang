"""SIMP simulator module."""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any

from mplang2.backends.simp_host import SimpHost
from mplang2.backends.simp_worker import WorkerInterpreter
from mplang2.edsl.graph import Graph


class ThreadCommunicator:
    """Thread-based communicator for in-memory communication."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.peers: list[ThreadCommunicator] = []
        self._mailbox: dict[str, Any] = {}
        self._cond = threading.Condition()
        self._sent_events: dict[str, threading.Event] = {}

    def set_peers(self, peers: list[ThreadCommunicator]) -> None:
        assert len(peers) == self.world_size
        self.peers = peers

    def send(self, to: int, key: str, data: Any) -> None:
        assert 0 <= to < self.world_size
        self.peers[to]._on_receive(self.rank, key, data)

    def recv(self, frm: int, key: str) -> Any:
        with self._cond:
            while key not in self._mailbox:
                self._cond.wait()
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

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for comm in self.comms:
            comm.set_peers(self.comms)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=world_size)

    def shutdown(self, wait: bool = True):
        self.executor.shutdown(wait=wait)


# Global simulation context (can be set by user or initialized lazily)
_SIM_CONTEXT: Context | None = None


def get_or_create_context(world_size: int = 3) -> Context:
    global _SIM_CONTEXT
    if _SIM_CONTEXT is None:
        _SIM_CONTEXT = Context(world_size)
    return _SIM_CONTEXT


class SimpSimulator(SimpHost):
    """SIMP simulator running locally with threads."""

    def __init__(self, world_size: int = 3):
        super().__init__(world_size)
        self.ctx = get_or_create_context(world_size)
        # Create persistent workers (Actors)
        self.workers = [
            WorkerInterpreter(rank, world_size, self.ctx.comms[rank])
            for rank in range(world_size)
        ]

    def _submit(self, rank: int, graph: Graph, inputs: dict[Any, Any]) -> Any:
        return self.ctx.executor.submit(self._run_party, rank, graph, inputs)

    def _collect(self, futures: list[Any]) -> list[Any]:
        return [f.result() for f in futures]

    def _run_party(self, rank: int, graph: Graph, inputs: dict[Any, Any]) -> Any:
        worker = self.workers[rank]
        return worker.evaluate_graph(graph, inputs)

    def shutdown(self, wait: bool = True):
        global _SIM_CONTEXT
        if self.ctx:
            self.ctx.shutdown(wait=wait)
        if _SIM_CONTEXT is self.ctx:
            _SIM_CONTEXT = None
