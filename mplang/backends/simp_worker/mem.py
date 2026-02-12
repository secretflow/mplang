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

from mplang.backends.simp_worker.base import SendRequest

# ---------------------------------------------------------------------------
# ThreadCommunicator
# ---------------------------------------------------------------------------


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
        # Mailbox keyed by (from_rank, tag): each key has exactly one message
        self._mailbox: dict[tuple[int, str], Any] = {}
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

    def send(
        self, to: int, key: str, data: Any, *, is_raw_bytes: bool = False
    ) -> SendRequest:
        """Send data to another rank (instant for in-memory).

        Returns a request handle for API consistency with HttpCommunicator.
        For ThreadCommunicator the handle is already completed since
        in-memory transfer is instant.

        Args:
            to: Target rank.
            key: Message key.
            data: Payload.
            is_raw_bytes: If True, treat data as raw bytes.

        Returns:
            SendRequest handle (already completed for in-memory).
        """
        assert 0 <= to < self.world_size
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        try:
            if self.use_serde:
                from mplang.edsl import serde

                data = serde.loads(serde.dumps(data))
            self.peers[to]._on_receive(self.rank, key, data)
            future.set_result(None)
        except Exception as e:
            future.set_exception(e)
        return SendRequest(future, to, key)

    # Alias for MPI-style naming
    isend = send

    def send_sync(
        self,
        to: int,
        key: str,
        data: Any,
        *,
        is_raw_bytes: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Send data to another rank synchronously.

        For ThreadCommunicator, this is identical to send() since in-memory
        transfer is instant. The timeout parameter is accepted for interface
        compatibility but ignored.

        Args:
            to: Target rank.
            key: Message key.
            data: Payload.
            is_raw_bytes: If True, treat data as raw bytes.
            timeout: Timeout in seconds (ignored, for interface compatibility).
        """
        _ = timeout  # Unused, in-memory transfer is instant
        req = self.send(to, key, data, is_raw_bytes=is_raw_bytes)
        req.wait()  # Immediate for in-memory, but ensures any exception is raised

    def recv(self, frm: int, key: str, *, timeout: float | None = None) -> Any:
        """Receive data from another rank.

        Args:
            frm: Source rank.
            key: Message key.
            timeout: Timeout in seconds. Currently ignored for ThreadCommunicator
                (waits indefinitely), but accepted for interface compatibility.

        Returns:
            The received data.
        """
        # Note: timeout is accepted for interface compatibility but not implemented
        # ThreadCommunicator is used for in-process simulation where timeouts are less critical
        _ = timeout  # Unused, for interface compatibility
        mailbox_key = (frm, key)
        with self._cond:
            while mailbox_key not in self._mailbox and not self._shutdown:
                self._cond.wait()
            if self._shutdown:
                raise RuntimeError("Communicator shut down")
            return self._mailbox.pop(mailbox_key)

    def _on_receive(self, frm: int, key: str, data: Any) -> None:
        mailbox_key = (frm, key)
        with self._cond:
            if mailbox_key in self._mailbox:
                raise RuntimeError(
                    f"Mailbox overflow: key {mailbox_key} already exists"
                )
            self._mailbox[mailbox_key] = data
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
