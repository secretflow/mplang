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

"""Simp Worker memory IPC runtime (LocalMesh, ThreadCommunicator)."""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any


class ThreadCommunicator:
    """Thread-based communicator for in-memory communication.

    Args:
        rank: This communicator's rank.
        world_size: Total number of parties.
        use_serde: If True, serialize/deserialize data through serde on send.
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


class LocalMesh:
    """Communication mesh for local SIMP simulation.

    Creates a set of ThreadCommunicators and a ThreadPoolExecutor for
    worker-side execution.
    """

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
