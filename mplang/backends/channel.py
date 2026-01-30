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

"""SPU IChannel implementation for MPLang v2.

Bridges v2's simp_worker communicators (ThreadCommunicator/HttpCommunicator)
to libspu's IChannel interface, enabling SPU to reuse existing communication
infrastructure instead of creating separate BRPC connections.
"""

from __future__ import annotations

from typing import Protocol

import spu.libspu as libspu

from mplang.logging_config import get_logger

logger = get_logger(__name__)


class CommunicatorProtocol(Protocol):
    """Protocol for v2 communicators (duck typing).

    Both ThreadCommunicator and HttpCommunicator implement this interface.
    """

    def send(
        self, to: int, key: str, data: bytes, *, is_raw_bytes: bool = False
    ) -> None: ...
    def recv(self, frm: int, key: str) -> bytes: ...


class BaseChannel(libspu.link.IChannel):
    """Bridge v2 communicator to SPU IChannel interface.

    Supports both ThreadCommunicator and HttpCommunicator via duck typing.
    Each BaseChannel represents a channel to ONE peer rank.

    Communication Protocol:
        - SPU calls send(tag, bytes) -> comm.send(peer, "spu:tag", bytes)
        - SPU calls recv(tag) -> bytes <- comm.recv(peer, "spu:tag")

    Tag Namespace:
        All tags are prefixed with "spu:" to avoid collision with other
        traffic on the same communicator.
    """

    def __init__(
        self,
        comm: CommunicatorProtocol,
        local_rank: int,
        peer_rank: int,
        tag_prefix: str = "spu",
    ):
        """Initialize channel to a specific peer.

        Args:
            comm: v2 communicator (any object implementing send/recv)
            local_rank: Global rank of this party
            peer_rank: Global rank of the peer party
            tag_prefix: Prefix for all tags (default: "spu")
        """
        super().__init__()
        self._comm = comm
        self._local_rank = local_rank
        self._peer_rank = peer_rank
        self._tag_prefix = tag_prefix

        logger.debug(
            "BaseChannel initialized: local_rank=%d, peer_rank=%d, tag_prefix=%s",
            local_rank,
            peer_rank,
            tag_prefix,
        )

    def _make_key(self, tag: str) -> str:
        """Create unique key for communicator.

        Args:
            tag: SPU-provided tag (e.g., "send_0")

        Returns:
            Prefixed key (e.g., "spu:send_0")
        """
        return f"{self._tag_prefix}:{tag}"

    def Send(self, tag: str, data: bytes) -> None:
        """Send bytes to peer.

        Args:
            tag: Message tag for matching send/recv pairs
            data: Raw bytes to send
        """
        key = self._make_key(tag)
        logger.debug(
            "BaseChannel.Send: %d -> %d, tag=%s, key=%s, size=%d",
            self._local_rank,
            self._peer_rank,
            tag,
            key,
            len(data),
        )

        # Send raw bytes directly.
        self._comm.send(self._peer_rank, key, data, is_raw_bytes=True)

    def Recv(self, tag: str) -> bytes:
        """Receive bytes from peer (blocking).

        Args:
            tag: Message tag for matching send/recv pairs

        Returns:
            Raw bytes received

        Raises:
            TypeError: If received data is not bytes
        """
        key = self._make_key(tag)
        logger.debug(
            "BaseChannel.Recv: %d <- %d, tag=%s, key=%s",
            self._local_rank,
            self._peer_rank,
            tag,
            key,
        )

        # Receive data (should be bytes)
        data = self._comm.recv(self._peer_rank, key)

        # Validate data type
        if not isinstance(data, bytes):
            raise TypeError(
                f"Expected bytes from communicator, got {type(data).__name__}. "
                f"Communicator must support raw bytes transmission for SPU channels."
            )

        logger.debug(
            "BaseChannel.Recv complete: %d <- %d, tag=%s, size=%d",
            self._local_rank,
            self._peer_rank,
            tag,
            len(data),
        )
        return data

    def SendAsync(self, tag: str, data: bytes) -> None:
        """Async send.

        For HttpCommunicator, underlying HTTP client is non-blocking.
        For ThreadCommunicator, send is instant (memory transfer).

        Args:
            tag: Message tag
            data: Raw bytes to send
        """
        self.Send(tag, data)

    def SendAsyncThrottled(self, tag: str, data: bytes) -> None:
        """Throttled async send.

        Currently maps to regular SendAsync.

        Args:
            tag: Message tag
            data: Raw bytes to send
        """
        self.SendAsync(tag, data)

    def TestSend(self, timeout: int) -> None:
        """Test if channel can send a dummy message to peer.

        Uses fixed tag "__test__" for idempotency.

        Args:
            timeout: Timeout in milliseconds (informational)
        """
        test_data = b"\x00"  # Minimal 1-byte handshake
        self.Send("__test__", test_data)

    def TestRecv(self) -> None:
        """Wait for dummy message from peer.

        Timeout controlled by recv_timeout_ms in link descriptor.

        Raises:
            Warning if unexpected handshake data received
        """
        test_data = self.Recv("__test__")
        if test_data != b"\x00":
            logger.warning(
                "TestRecv: unexpected handshake from %d, expected b'\\x00', got %r",
                self._peer_rank,
                test_data,
            )

    def WaitLinkTaskFinish(self) -> None:
        """Wait for all pending async tasks.

        No-op for v2 communicators (handled automatically).
        """

    def Abort(self) -> None:
        """Abort communication (cleanup).

        Currently a no-op. Could be extended for resource cleanup.
        """
        logger.warning(
            "BaseChannel.Abort: %d <-> %d", self._local_rank, self._peer_rank
        )

    def SetThrottleWindowSize(self, size: int) -> None:
        """Set throttle window size (no-op).

        Args:
            size: Window size (ignored)
        """

    def SetChunkParallelSendSize(self, size: int) -> None:
        """Set chunk parallel send size (no-op).

        Args:
            size: Chunk size (ignored)
        """
