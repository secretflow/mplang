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

import spu.libspu as libspu


class LinkCommunicator:
    """Minimal wrapper for libspu link context.

    This class serves only to create and hold a libspu link context for SPU runtime.
    It does NOT provide general-purpose communication methods - those are handled by
    the underlying libspu.link.Context which SPU runtime uses directly.

    All serialization is handled internally by libspu - no pickle needed here.
    """

    def __init__(self, rank: int, addrs: list[str], *, mem_link: bool = False):
        """Initialize link communicator for SPU.

        Args:
            rank: Rank of this party
            addrs: List of addresses for all parties
            mem_link: If True, use in-memory link (for testing); otherwise use BRPC
        """
        self._rank = rank
        self._world_size = len(addrs)

        desc = libspu.link.Desc()  # type: ignore
        desc.recv_timeout_ms = 100 * 1000  # 100 seconds
        desc.http_max_payload_size = 32 * 1024 * 1024  # Default set link payload to 32M
        for rank, addr in enumerate(addrs):
            desc.add_party(f"P{rank}", addr)

        if mem_link:
            self.lctx = libspu.link.create_mem(desc, self._rank)
        else:
            self.lctx = libspu.link.create_brpc(desc, self._rank)

        logging.info(
            f"LinkCommunicator initialized: rank={self._rank}, world_size={self._world_size}, "
            f"addrs={addrs}, mem_link={mem_link}"
        )

    @property
    def rank(self) -> int:
        """Get rank from underlying link context."""
        return self.lctx.rank  # type: ignore[no-any-return]

    @property
    def world_size(self) -> int:
        """Get world size from underlying link context."""
        return self.lctx.world_size  # type: ignore[no-any-return]

    def get_lctx(self) -> libspu.link.Context:
        """Get the underlying libspu link context.

        This is the primary interface - SPU runtime uses this context directly.
        All communication and serialization is handled by libspu internally.

        Returns:
            The underlying libspu.link.Context instance.
        """
        return self.lctx
