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

"""SPU IChannel implementation that bridges to MPLang CommunicatorBase.

This module provides BaseChannel, which allows SPU to reuse MPLang's
existing communication layer (ThreadCommunicator/HttpCommunicator) instead
of creating separate BRPC connections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import spu.libspu as libspu

if TYPE_CHECKING:
    from mplang.v1.core.comm import CommunicatorBase


class BaseChannel(libspu.link.IChannel):
    """Bridge MPLang CommunicatorBase to SPU IChannel interface.

    This adapter allows SPU to use MPLang's existing communication layer
    (ThreadCommunicator or HttpCommunicator) instead of creating separate
    BRPC connections.

    Each MPLangChannel represents a channel to ONE peer rank.

    Communication Protocol:
        - SPU calls send(tag, bytes_data) -> MPLang comm.send(peer, key, bytes_data)
        - SPU calls recv(tag) -> bytes_data <- MPLang comm.recv(peer, key)

    Tag Namespace:
        All tags are prefixed with "spu:" to avoid collision with other
        MPLang traffic on the same communicator.
    """

    def __init__(
        self,
        comm: CommunicatorBase,
        local_rank: int,
        peer_rank: int,
        tag_prefix: str = "spu",
    ):
        """Initialize channel to a specific peer.

        Args:
            comm: MPLang communicator instance (Thread/Http)
            local_rank: Global rank of this party (for logging/debugging)
            peer_rank: Global rank of the peer party
            tag_prefix: Prefix for all tags to avoid collision (default: "spu")
        """
        super().__init__()
        self._comm = comm
        self._local_rank = local_rank
        self._peer_rank = peer_rank
        self._tag_prefix = tag_prefix

        logging.debug(
            f"BaseChannel initialized: local_rank={local_rank}, "
            f"peer_rank={peer_rank}, tag_prefix={tag_prefix}"
        )

    def _make_key(self, tag: str) -> str:
        """Create unique key for MPLang comm.

        Prefixes the tag to avoid collision with non-SPU traffic.

        Args:
            tag: SPU-provided tag (e.g., "send_0", "recv_0")

        Returns:
            Prefixed key (e.g., "spu:send_0")
        """
        return f"{self._tag_prefix}:{tag}"

    def Send(self, tag: str, data: bytes) -> None:
        """Send bytes to peer (synchronous in SPU semantics).

        Args:
            tag: Message tag for matching send/recv pairs
            data: Raw bytes to send
        """
        key = self._make_key(tag)
        logging.debug(
            f"BaseChannel.Send: {self._local_rank} -> {self._peer_rank}, "
            f"tag={tag}, key={key}, size={len(data)}"
        )

        # Send raw bytes directly
        # Note: CommunicatorBase.send expects Any type, bytes is acceptable
        self._comm.send(self._peer_rank, key, data)

    def Recv(self, tag: str) -> bytes:
        """Receive bytes from peer (blocking).

        Args:
            tag: Message tag for matching send/recv pairs

        Returns:
            Raw bytes received
        """
        key = self._make_key(tag)
        logging.debug(
            f"BaseChannel.Recv: {self._local_rank} <- {self._peer_rank}, "
            f"tag={tag}, key={key}"
        )

        # Receive data (should be bytes)
        data = self._comm.recv(self._peer_rank, key)

        # Validate data type
        if not isinstance(data, bytes):
            raise TypeError(
                f"Expected bytes from communicator, got {type(data).__name__}. "
                f"Communicator must support raw bytes transmission for SPU channels."
            )

        logging.debug(
            f"BaseChannel.Recv complete: {self._local_rank} <- {self._peer_rank}, "
            f"tag={tag}, size={len(data)}"
        )
        return data

    def SendAsync(self, tag: str, data: bytes) -> None:
        """Async send (MPLang's send is already async at network layer).

        For HttpCommunicator, the underlying httpx.put() is non-blocking
        at the HTTP client level. For ThreadCommunicator, send is instant
        (memory transfer).

        Args:
            tag: Message tag
            data: Raw bytes to send
        """
        # Reuse synchronous send - it's already async underneath
        self.Send(tag, data)

    def SendAsyncThrottled(self, tag: str, data: bytes) -> None:
        """Throttled async send.

        Currently maps to regular SendAsync. Future optimization could
        implement rate limiting if needed.

        Args:
            tag: Message tag
            data: Raw bytes to send
        """
        self.SendAsync(tag, data)

    def TestSend(self, timeout: int) -> None:
        """Test if this channel can send a dummy msg to peer.

        Uses fixed 0 seq_id as dummy msg's id to make this function reentrant.
        ConnectToMesh will retry on this multiple times.

        Args:
            timeout: Timeout in milliseconds
        """
        # Send a handshake message to test connectivity
        # Use fixed tag "__test__" to make this reentrant (idempotent)
        test_data = b"\x00"  # Minimal 1-byte message with seq_id=0
        self.Send("__test__", test_data)

    def TestRecv(self) -> None:
        """Wait for dummy msg from peer.

        Timeout is controlled by recv_timeout_ms in link descriptor.
        """
        # Receive the handshake message from peer
        # This blocks until message arrives (timeout from desc.recv_timeout_ms)
        test_data = self.Recv("__test__")
        # Validate it's the expected handshake message
        if test_data != b"\x00":
            logging.warning(
                f"TestRecv: unexpected handshake data from {self._peer_rank}, "
                f"expected b'\\x00', got {test_data!r}"
            )

    def WaitLinkTaskFinish(self) -> None:
        """Wait for all pending async tasks.

        For MPLang communicators:
        - ThreadCommunicator: No-op (instant memory transfer)
        - HttpCommunicator: No explicit wait needed (httpx handles it)

        This is a no-op in current implementation.
        """
        pass

    def Abort(self) -> None:
        """Abort communication (cleanup resources).

        This could be extended to notify the communicator to drop pending
        messages for this channel, but currently is a no-op.
        """
        logging.warning(
            f"BaseChannel.Abort called: {self._local_rank} <-> {self._peer_rank}"
        )
        # Future: Could call comm.abort_session() if implemented

    def SetThrottleWindowSize(self, size: int) -> None:
        """Set throttle window size.

        Not applicable to MPLang communicators. No-op.

        Args:
            size: Window size (ignored)
        """
        pass

    def SetChunkParallelSendSize(self, size: int) -> None:
        """Set chunk parallel send size.

        Not applicable to MPLang communicators. No-op.

        Args:
            size: Chunk size (ignored)
        """
        pass
