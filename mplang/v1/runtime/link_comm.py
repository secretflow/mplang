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
from typing import TYPE_CHECKING

import spu.libspu as libspu

if TYPE_CHECKING:
    from mplang.v1.core.comm import CommunicatorBase
    from mplang.v1.core.mask import Mask


class LinkCommunicator:
    """Minimal wrapper for libspu link context.

    Supports three modes:
    1. BRPC: Production mode with separate BRPC ports (legacy)
    2. Mem: In-memory links for testing (legacy)
    3. Channels: Reuse MPLang communicator via IChannel bridge (NEW)

    The mode is selected based on constructor arguments:
    - If `comm` is provided: Channels mode (NEW)
    - Elif `mem_link` is True: Mem mode
    - Else: BRPC mode
    """

    def __init__(
        self,
        rank: int,
        addrs: list[str] | None = None,
        *,
        mem_link: bool = False,
        comm: CommunicatorBase | None = None,
        spu_mask: Mask | None = None,
    ):
        """Initialize link communicator for SPU.

        Args:
            rank: Global rank of this party
            addrs: List of addresses for all SPU parties (required for BRPC/Mem mode)
            mem_link: If True, use in-memory link (Mem mode)
            comm: MPLang communicator to reuse (Channels mode, NEW)
            spu_mask: SPU parties mask (required for Channels mode)

        Raises:
            ValueError: If arguments are invalid for the selected mode
        """
        self._rank = rank

        # Mode 1: Channels (NEW) - Reuse MPLang communicator
        if comm is not None:
            if spu_mask is None:
                raise ValueError("spu_mask required when using comm")
            if rank not in spu_mask:
                raise ValueError(f"rank {rank} not in spu_mask {spu_mask}")

            # Lazy import to avoid circular dependency
            from mplang.v1.runtime.channel import BaseChannel

            # Create channels to ALL SPU parties (including self)
            # libspu expects world_size channels, with self channel being None
            channels = []
            rel_rank = spu_mask.global_to_relative_rank(rank)
            
            for idx, peer_rank in enumerate(spu_mask):
                if peer_rank == rank:
                    # For self, use None (won't be accessed by SPU)
                    channel = None
                else:
                    channel = BaseChannel(comm, rank, peer_rank)
                channels.append(channel)

            # Create link context with custom channels
            desc = libspu.link.Desc()  # type: ignore
            desc.recv_timeout_ms = 100 * 1000  # 100 seconds
            
            # Add party info to desc (required for world_size inference)
            for idx, peer_rank in enumerate(spu_mask):
                desc.add_party(f"P{idx}", f"dummy_{peer_rank}")

            self.lctx = libspu.link.create_with_channels(desc, rel_rank, channels)
            self._world_size = spu_mask.num_parties()

            logging.info(
                f"LinkCommunicator initialized with BaseChannel: "
                f"rank={rank}, rel_rank={rel_rank}, spu_mask={spu_mask}, "
                f"world_size={self._world_size}"
            )

        # Mode 2 & 3: BRPC or Mem (legacy)
        else:
            if addrs is None:
                raise ValueError("addrs required for BRPC/Mem mode")
            self._world_size = len(addrs)

            desc = libspu.link.Desc()  # type: ignore
            desc.recv_timeout_ms = 100 * 1000  # 100 seconds
            desc.http_max_payload_size = 32 * 1024 * 1024  # Default set link payload to 32M
            for rank_idx, addr in enumerate(addrs):
                desc.add_party(f"P{rank_idx}", addr)

            if mem_link:
                self.lctx = libspu.link.create_mem(desc, self._rank)
                logging.info(
                    f"LinkCommunicator initialized with Mem: "
                    f"rank={self._rank}, world_size={self._world_size}, addrs={addrs}"
                )
            else:
                self.lctx = libspu.link.create_brpc(desc, self._rank)
                logging.info(
                    f"LinkCommunicator initialized with BRPC: "
                    f"rank={self._rank}, world_size={self._world_size}, addrs={addrs}"
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
