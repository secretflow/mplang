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
from typing import Any

from mplang.core.comm import CollectiveMixin, CommunicatorBase


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
