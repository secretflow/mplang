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

import logging
import threading
from typing import Any

from mplang.core.comm import CollectiveMixin, ICommunicator


class CommunicatorBase(ICommunicator):
    """Base implementation providing message box functionality for local communication"""

    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self._world_size = world_size
        self._msgboxes: dict = {}
        self._cond = threading.Condition()
        self._counter = 0

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    # override
    def new_id(self) -> str:
        res = self._counter
        self._counter += 1
        return str(res)

    def recv(self, frm: int, key: str) -> Any:
        """Wait until the key is set, returns the value"""
        # print(f"recv {key}: {sender_rank} -> {self.rank}")
        mkey = (frm, key)
        with self._cond:
            var = self._msgboxes.get(mkey, None)
            while var is None:
                self._cond.wait()
                var = self._msgboxes.get(mkey, None)
            return var

    def onSent(self, frm: int, key: str, data: Any) -> None:
        """Called when a key is sent to self"""
        with self._cond:
            assert key not in self._msgboxes, f"{key} exist {self._msgboxes.keys()}"
            mkey = (frm, key)
            self._msgboxes[mkey] = data
            self._cond.notify_all()


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
