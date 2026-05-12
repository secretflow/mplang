# Copyright 2026 Ant Group Co., Ltd.
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

"""Communication context with automatic key generation.

Analogous to ``libspu.link.Context`` — wraps a shared Communicator with
per-peer counters for collision-free messaging.  Supports ``spawn()`` for
creating child contexts with independent counter namespaces.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mplang.backends.simp_worker.base import CommunicatorProtocol


class CommContext:
    """Communication context with automatic key generation.

    Wraps a shared :class:`CommunicatorProtocol` and generates deterministic,
    collision-free message keys via per-peer counters.

    Key format: ``"{context_id}:{seq}"``

    Send and recv to the same peer share **one** counter to maintain SPMD
    symmetry: for any peer pair (A, B), A's N-th interaction with B matches
    B's N-th interaction with A.
    """

    __slots__ = ("_comm", "_id", "_rank", "_seq", "_spawn_counter")

    def __init__(
        self,
        comm: CommunicatorProtocol,
        context_id: str,
        my_rank: int,
    ) -> None:
        self._comm = comm
        self._id = context_id
        self._rank = my_rank
        self._seq: dict[int, int] = defaultdict(int)
        self._spawn_counter: int = 0

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._comm.world_size

    def spawn(self) -> CommContext:
        """Create a child context with independent counter namespace.

        Child ``context_id = f"{parent_id}.{spawn_seq}"``.  ``spawn_seq``
        increments in the calling thread's order, so if all ranks call
        ``spawn()`` in the same program-order position, children get matching
        IDs across ranks.
        """
        child_id = f"{self._id}.{self._spawn_counter}"
        self._spawn_counter += 1
        return CommContext(self._comm, child_id, self._rank)

    def send(self, to: int, data: Any, *, is_raw_bytes: bool = False) -> SendRequest:
        seq = self._seq[to]
        self._seq[to] = seq + 1
        key = f"{self._id}:{seq}"
        return self._comm.send(to, key, data, is_raw_bytes=is_raw_bytes)

    def send_sync(
        self,
        to: int,
        data: Any,
        *,
        is_raw_bytes: bool = False,
        timeout: float | None = None,
    ) -> None:
        seq = self._seq[to]
        self._seq[to] = seq + 1
        key = f"{self._id}:{seq}"
        self._comm.send_sync(to, key, data, is_raw_bytes=is_raw_bytes, timeout=timeout)

    def recv(self, frm: int, *, timeout: float | None = None) -> Any:
        seq = self._seq[frm]
        self._seq[frm] = seq + 1
        key = f"{self._id}:{seq}"
        return self._comm.recv(frm, key, timeout=timeout)
