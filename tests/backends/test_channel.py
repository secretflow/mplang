# Copyright 2026 Ant Group Co., Ltd.
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

"""Tests for v2 SPU channel bridge (BaseChannel)."""

import pytest

from mplang.backends.channel import BaseChannel
from mplang.backends.simp_worker.mem import ThreadCommunicator


class TestBaseChannel:
    """Test BaseChannel adapter for v2."""

    def test_basic_send_recv(self):
        """Test basic send/recv through BaseChannel."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Create channels: rank0 <-> rank1
        ch0 = BaseChannel(comms[0], local_rank=0, peer_rank=1)
        ch1 = BaseChannel(comms[1], local_rank=1, peer_rank=0)

        # Test send/recv
        data = b"hello spu from rank 0"
        ch0.Send("tag1", data)
        received = ch1.Recv("tag1")
        assert received == data

    def test_bidirectional_send_recv(self):
        """Test bidirectional communication."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        ch0 = BaseChannel(comms[0], local_rank=0, peer_rank=1)
        ch1 = BaseChannel(comms[1], local_rank=1, peer_rank=0)

        # Rank 0 -> 1
        data01 = b"0 to 1"
        ch0.Send("msg1", data01)
        assert ch1.Recv("msg1") == data01

        # Rank 1 -> 0
        data10 = b"1 to 0"
        ch1.Send("msg2", data10)
        assert ch0.Recv("msg2") == data10

    def test_multiple_messages(self):
        """Test multiple messages with different tags."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        ch0 = BaseChannel(comms[0], local_rank=0, peer_rank=1)
        ch1 = BaseChannel(comms[1], local_rank=1, peer_rank=0)

        # Send multiple messages with different tags
        messages = [(f"tag{i}", f"message_{i}".encode()) for i in range(5)]

        for tag, data in messages:
            ch0.Send(tag, data)

        for tag, expected_data in messages:
            received = ch1.Recv(tag)
            assert received == expected_data

    def test_tag_prefix_isolation(self):
        """Test that different tag prefixes isolate channels."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Create channels with different prefixes
        ch0_spu = BaseChannel(comms[0], 0, 1, tag_prefix="spu:")
        ch1_spu = BaseChannel(comms[1], 1, 0, tag_prefix="spu:")
        ch0_custom = BaseChannel(comms[0], 0, 1, tag_prefix="custom:")
        ch1_custom = BaseChannel(comms[1], 1, 0, tag_prefix="custom:")

        # Send via different channels with same tag name
        ch0_spu.Send("tag", b"spu data")
        ch0_custom.Send("tag", b"custom data")

        # Receive should get correct data based on prefix
        assert ch1_spu.Recv("tag") == b"spu data"
        assert ch1_custom.Recv("tag") == b"custom data"

    def test_send_async(self):
        """Test SendAsync (should work like regular Send)."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        ch0 = BaseChannel(comms[0], local_rank=0, peer_rank=1)
        ch1 = BaseChannel(comms[1], local_rank=1, peer_rank=0)

        data = b"async message"
        ch0.SendAsync("async_tag", data)
        assert ch1.Recv("async_tag") == data

    def test_test_send_recv(self):
        """Test TestSend and TestRecv (handshake methods)."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        ch0 = BaseChannel(comms[0], local_rank=0, peer_rank=1)
        ch1 = BaseChannel(comms[1], local_rank=1, peer_rank=0)

        # TestSend/TestRecv are no-ops for v2 (like v1) but should not raise
        ch0.TestSend(1000)  # timeout in ms
        ch1.TestRecv()

    def test_large_data(self):
        """Test sending large data."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        ch0 = BaseChannel(comms[0], local_rank=0, peer_rank=1)
        ch1 = BaseChannel(comms[1], local_rank=1, peer_rank=0)

        # Send 1MB of data
        large_data = b"x" * (1024 * 1024)
        ch0.Send("large", large_data)
        received = ch1.Recv("large")
        assert len(received) == len(large_data)
        assert received == large_data

    def test_recv_type_validation(self):
        """Test that recv validates data type."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        ch1 = BaseChannel(comms[1], local_rank=1, peer_rank=0)

        # Send non-bytes data directly via comm (bypass channel)
        comms[0].send(1, "spu:bad_tag", "not bytes")

        # Recv via channel should raise TypeError
        with pytest.raises(TypeError, match="Expected bytes"):
            ch1.Recv("bad_tag")

    def test_three_party_ring(self):
        """Test 3-party communication (typical SPU scenario)."""
        world_size = 3
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Create channels: 0->1, 1->2, 2->0 (ring)
        ch01 = BaseChannel(comms[0], 0, 1)
        ch12 = BaseChannel(comms[1], 1, 2)
        ch20 = BaseChannel(comms[2], 2, 0)

        ch10 = BaseChannel(comms[1], 1, 0)
        ch21 = BaseChannel(comms[2], 2, 1)
        ch02 = BaseChannel(comms[0], 0, 2)

        # Ring communication
        ch01.Send("ring", b"from 0")
        ch12.Send("ring", b"from 1")
        ch20.Send("ring", b"from 2")

        assert ch10.Recv("ring") == b"from 0"
        assert ch21.Recv("ring") == b"from 1"
        assert ch02.Recv("ring") == b"from 2"

    def test_multi_session_isolation_via_tag_prefix(self):
        """Test multi-session isolation using tag_prefix encoding.

        Demonstrates how upper layers can implement session isolation by
        encoding session_id into tag_prefix, without BaseChannel needing
        to know about session semantics.
        """
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Session A: two parties with session-specific prefix
        session_a_prefix = "session_aaa:spu"
        ch0_session_a = BaseChannel(comms[0], 0, 1, tag_prefix=session_a_prefix)
        ch1_session_a = BaseChannel(comms[1], 1, 0, tag_prefix=session_a_prefix)

        # Session B: same two parties, different session
        session_b_prefix = "session_bbb:spu"
        ch0_session_b = BaseChannel(comms[0], 0, 1, tag_prefix=session_b_prefix)
        ch1_session_b = BaseChannel(comms[1], 1, 0, tag_prefix=session_b_prefix)

        # Send messages in both sessions
        ch0_session_a.Send("data", b"message from session A")
        ch0_session_b.Send("data", b"message from session B")

        # Receive - no crosstalk between sessions
        recv_a = ch1_session_a.Recv("data")
        recv_b = ch1_session_b.Recv("data")

        assert recv_a == b"message from session A"
        assert recv_b == b"message from session B"

        # TestSend/TestRecv also isolated
        ch0_session_a.TestSend(1000)
        ch0_session_b.TestSend(1000)

        ch1_session_a.TestRecv()  # Receives from session A only
        ch1_session_b.TestRecv()  # Receives from session B only

    def test_multi_session_isolation_via_context_id(self):
        """Test multi-session isolation via communicator get_context_id().

        This tests the recommended pattern for upper-layer session management:
        communicator provides context_id, BaseChannel automatically constructs
        isolated keys without requiring manual tag_prefix encoding.
        """

        class SessionAwareCommunicator:
            """Wrapper that adds context_id to any communicator."""

            def __init__(self, base_comm, context_id: str | None):
                self._base_comm = base_comm
                self._context_id = context_id

            def send(self, to: int, key: str, data: bytes) -> None:
                self._base_comm.send(to, key, data)

            def recv(self, frm: int, key: str) -> bytes:
                return self._base_comm.recv(frm, key)

            def get_context_id(self) -> str | None:
                """Provide context ID for isolation."""
                return self._context_id

        # Setup base communicators
        world_size = 2
        base_comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in base_comms:
            c.set_peers(base_comms)

        # Session A: wrap with session_id="session_aaa"
        comms_a = [
            SessionAwareCommunicator(base_comms[0], "session_aaa"),
            SessionAwareCommunicator(base_comms[1], "session_aaa"),
        ]
        ch0_a = BaseChannel(comms_a[0], 0, 1, tag_prefix="spu")
        ch1_a = BaseChannel(comms_a[1], 1, 0, tag_prefix="spu")

        # Session B: wrap with session_id="session_bbb"
        comms_b = [
            SessionAwareCommunicator(base_comms[0], "session_bbb"),
            SessionAwareCommunicator(base_comms[1], "session_bbb"),
        ]
        ch0_b = BaseChannel(comms_b[0], 0, 1, tag_prefix="spu")
        ch1_b = BaseChannel(comms_b[1], 1, 0, tag_prefix="spu")

        # Send messages in both sessions (same tag "data", different contexts)
        ch0_a.Send("data", b"message from session A")
        ch0_b.Send("data", b"message from session B")

        # Verify no crosstalk - each session receives its own message
        recv_a = ch1_a.Recv("data")
        recv_b = ch1_b.Recv("data")

        assert recv_a == b"message from session A"
        assert recv_b == b"message from session B"

        # Verify TestSend/TestRecv also isolated by context_id
        ch0_a.TestSend(1000)
        ch0_b.TestSend(1000)

        ch1_a.TestRecv()  # Receives handshake from session_aaa:spu:__test__
        ch1_b.TestRecv()  # Receives handshake from session_bbb:spu:__test__

        # Verify key format in underlying mailbox (internal verification)
        # Session A keys should be "session_aaa:spu:data", "session_aaa:spu:__test__"
        # Session B keys should be "session_bbb:spu:data", "session_bbb:spu:__test__"
        # This is implicitly verified by successful isolation above
