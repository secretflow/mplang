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

import logging
from typing import Protocol

import spu.libspu as libspu


class CommunicatorProtocol(Protocol):
    """Protocol for v2 communicators (duck typing).

    Both ThreadCommunicator and HttpCommunicator implement this interface.

    Required Methods:
        send(): Send data to peer rank
        recv(): Receive data from peer rank

    Optional Methods (detected via hasattr() at runtime):
        get_context_id(): If implemented, returns context identifier (e.g., session_id)
                         for multi-session isolation. If not implemented or returns None,
                         no context isolation is applied (backward compatible).

    Note:
        get_context_id() is NOT defined in this Protocol to avoid forcing
        all implementations to provide it. BaseChannel checks for its existence
        at runtime using hasattr() and gracefully degrades if not available.
    """

    def send(self, to: int, key: str, data: bytes) -> None: ...
    def recv(self, frm: int, key: str) -> bytes: ...


class BaseChannel(libspu.link.IChannel):
    """Bridge v2 communicator to SPU IChannel interface.

    Supports both ThreadCommunicator and HttpCommunicator via duck typing.
    Each BaseChannel represents a channel to ONE peer rank.

    Communication Protocol:
        - SPU calls send(tag, bytes) -> comm.send(peer, key, bytes)
        - SPU calls recv(tag) -> bytes <- comm.recv(peer, key)

        Where key format is determined by context_id and tag_prefix:
        - Without context: "{tag_prefix}:{tag}" (e.g., "spu:send_0")
        - With context: "{context_id}:{tag_prefix}:{tag}" (e.g., "session_abc:spu:send_0")

    Tag Namespace:
        - tag_prefix: Identifies protocol/application type (spu, tee, custom, etc.)
        - context_id: Isolates concurrent sessions/contexts (from communicator)
        - tag: SPU-provided message tag (send_0, recv_1, __test__, etc.)

    Multi-Session Isolation:
        Upper-layer applications can enable session isolation by providing
        communicators that implement get_context_id() method:

        Example 1 - Single session (default, no isolation):
            comm = ThreadCommunicator(rank=0, world_size=2)
            channel = BaseChannel(comm, 0, 1, tag_prefix="spu")
            # Keys: "spu:send_0", "spu:__test__"
            # (comm.get_context_id() not implemented or returns None)

        Example 2 - Multi-session with context-aware communicator:
            # Option A: Wrapper pattern
            base_comm = ThreadCommunicator(rank=0, world_size=2)
            session_comm = SessionAwareCommunicator(base_comm, session_id="abc")
            channel = BaseChannel(session_comm, 0, 1, tag_prefix="spu")
            # Keys: "abc:spu:send_0", "abc:spu:__test__"

            # Option B: Extended communicator
            comm = ThreadCommunicator(rank=0, world_size=2, context_id="abc")
            channel = BaseChannel(comm, 0, 1, tag_prefix="spu")
            # Keys: "abc:spu:send_0", "abc:spu:__test__"

        Example 3 - Multiple protocols with session isolation:
            session_comm = SessionAwareCommunicator(base_comm, session_id="abc")
            spu_channel = BaseChannel(session_comm, 0, 1, tag_prefix="spu")
            tee_channel = BaseChannel(session_comm, 0, 1, tag_prefix="tee")
            # SPU keys: "abc:spu:send_0"
            # TEE keys: "abc:tee:send_0"
            # Both isolated by session + differentiated by protocol

    Design Rationale:
        - BaseChannel remains agnostic to session semantics
        - Communicator owns context/session identity
        - Automatic isolation without manual key construction
        - Backward compatible (get_context_id() is optional)
        - Preserves tag_prefix semantic meaning (protocol type)
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
            comm: v2 communicator (any object implementing send/recv).
                  If comm implements get_context_id(), the returned context ID
                  (e.g., session_id) will be automatically prepended to all keys
                  for multi-session isolation.
            local_rank: Global rank of this party
            peer_rank: Global rank of the peer party
            tag_prefix: Prefix identifying protocol/application type (default: "spu").
                        Should NOT encode session_id here; use communicator's
                        get_context_id() instead for proper session isolation.
                        Examples: "spu", "tee", "custom_protocol"
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
        """Create unique key for communicator.

        Key format:
            - Without context: "{tag_prefix}:{tag}"
            - With context: "{context_id}:{tag_prefix}:{tag}"

        Context ID is obtained from communicator via get_context_id() method.
        If communicator doesn't implement this method or returns None, no
        context prefix is added (backward compatible).

        Args:
            tag: SPU-provided tag (e.g., "send_0", "__test__")

        Returns:
            Prefixed key with optional context isolation

        Examples:
            # Communicator without get_context_id():
            "spu:send_0", "spu:__test__"

            # Communicator with get_context_id() returning "session_abc":
            "session_abc:spu:send_0", "session_abc:spu:__test__"
        """
        # Try to get context ID from communicator (optional method)
        context_id = None
        if hasattr(self._comm, "get_context_id"):
            try:
                context_id = self._comm.get_context_id()
            except Exception as e:
                # Log but don't fail - graceful degradation
                logger.debug(
                    "Failed to get context_id from communicator: %s. "
                    "Proceeding without context isolation.",
                    e,
                )

        # Build key with optional context prefix
        if context_id:
            return f"{context_id}:{self._tag_prefix}:{tag}"
        return f"{self._tag_prefix}:{tag}"

    def Send(self, tag: str, data: bytes) -> None:
        """Send bytes to peer.

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
        # v2 communicators accept Any, bytes is valid
        self._comm.send(self._peer_rank, key, data)

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

        Uses fixed tag "__test__" combined with tag_prefix and optional context_id.
        If communicator implements get_context_id(), the handshake will be
        automatically isolated per context (e.g., "session_abc:spu:__test__").

        Args:
            timeout: Timeout in milliseconds (informational)
        """
        test_data = b"\x00"  # Minimal 1-byte handshake
        self.Send("__test__", test_data)

    def TestRecv(self) -> None:
        """Wait for dummy message from peer.

        Timeout controlled by recv_timeout_ms in link descriptor.
        If communicator implements get_context_id(), the handshake will be
        automatically received from context-isolated mailbox (e.g.,
        "session_abc:spu:__test__"), preventing crosstalk with other sessions.

        Raises:
            Warning if unexpected handshake data received
        """
        test_data = self.Recv("__test__")
        if test_data != b"\x00":
            logging.warning(
                f"TestRecv: unexpected handshake from {self._peer_rank}, "
                f"expected b'\\x00', got {test_data!r}"
            )

    def WaitLinkTaskFinish(self) -> None:
        """Wait for all pending async tasks.

        No-op for v2 communicators (handled automatically).
        """

    def Abort(self) -> None:
        """Abort communication (cleanup).

        Currently a no-op. Could be extended for resource cleanup.
        """
        logging.warning(f"BaseChannel.Abort: {self._local_rank} <-> {self._peer_rank}")

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


# Type hint for context-aware communicators (optional runtime feature)
# This is NOT a Protocol because get_context_id() is optional and checked via hasattr().
# Example implementation:
#
#   class SessionAwareCommunicator:
#       def __init__(self, base_comm: CommunicatorProtocol, context_id: str):
#           self._base_comm = base_comm
#           self._context_id = context_id
#
#       def send(self, to: int, key: str, data: bytes) -> None:
#           self._base_comm.send(to, key, data)
#
#       def recv(self, frm: int, key: str) -> bytes:
#           return self._base_comm.recv(frm, key)
#
#       def get_context_id(self) -> str | None:
#           """Enable multi-session isolation in BaseChannel."""
#           return self._context_id
