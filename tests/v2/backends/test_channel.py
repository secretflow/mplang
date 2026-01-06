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

"""Tests for v2 SPU channel bridge (BaseChannel)."""

import pytest

from mplang.v2.backends.channel import BaseChannel
from mplang.v2.backends.simp_worker.mem import ThreadCommunicator


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
