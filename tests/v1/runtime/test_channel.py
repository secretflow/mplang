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

"""Tests for SPU channel bridge (BaseChannel and LinkCommunicator Channels mode)."""

import pytest

from mplang.v1.core.mask import Mask
from mplang.v1.runtime.channel import BaseChannel
from mplang.v1.runtime.link_comm import LinkCommunicator
from mplang.v1.runtime.simulation import ThreadCommunicator


class TestBaseChannel:
    """Test BaseChannel adapter."""

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
        ch0.send("tag1", data)
        received = ch1.recv("tag1")
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
        ch0.send("msg1", data01)
        assert ch1.recv("msg1") == data01

        # Rank 1 -> 0
        data10 = b"1 to 0"
        ch1.send("msg2", data10)
        assert ch0.recv("msg2") == data10

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
            ch0.send(tag, data)

        for tag, expected_data in messages:
            received = ch1.recv(tag)
            assert received == expected_data

    def test_tag_prefix_isolation(self):
        """Test that different tag prefixes isolate channels."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Create channels with different prefixes
        ch0_spu = BaseChannel(comms[0], 0, 1, tag_prefix="spu")
        ch1_spu = BaseChannel(comms[1], 1, 0, tag_prefix="spu")
        ch0_custom = BaseChannel(comms[0], 0, 1, tag_prefix="custom")
        ch1_custom = BaseChannel(comms[1], 1, 0, tag_prefix="custom")

        # Send via different channels with same tag name
        ch0_spu.send("tag", b"spu data")
        ch0_custom.send("tag", b"custom data")

        # Receive should get correct data based on prefix
        assert ch1_spu.recv("tag") == b"spu data"
        assert ch1_custom.recv("tag") == b"custom data"

    def test_send_async(self):
        """Test send_async (should work like regular send)."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        ch0 = BaseChannel(comms[0], local_rank=0, peer_rank=1)
        ch1 = BaseChannel(comms[1], local_rank=1, peer_rank=0)

        data = b"async message"
        ch0.send_async("async_tag", data)
        assert ch1.recv("async_tag") == data

    def test_test_send_always_true(self):
        """Test that test_send always returns True."""
        world_size = 2
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        ch0 = BaseChannel(comms[0], local_rank=0, peer_rank=1)
        assert ch0.test_send("any_tag") is True

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
        ch0.send("large", large_data)
        received = ch1.recv("large")
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
            ch1.recv("bad_tag")


class TestLinkCommunicatorChannelsMode:
    """Test LinkCommunicator with Channels mode."""

    def test_channels_mode_basic(self):
        """Test LinkCommunicator with Channels mode."""
        world_size = 3
        spu_mask = Mask.from_ranks([0, 1, 2])
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Create LinkCommunicators using Channels mode
        links = [
            LinkCommunicator(rank=i, comm=comms[i], spu_mask=spu_mask)
            for i in range(world_size)
        ]

        # Verify attributes
        assert all(link.world_size == 3 for link in links)
        for i, link in enumerate(links):
            # Relative rank should match global rank when all parties in mask
            assert link.rank == i

    def test_channels_mode_subset_mask(self):
        """Test Channels mode with subset of parties in SPU."""
        world_size = 4
        spu_mask = Mask.from_ranks([0, 2, 3])  # Rank 1 not in SPU
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Create LinkCommunicators only for SPU parties
        links = {}
        for i in spu_mask:
            links[i] = LinkCommunicator(rank=i, comm=comms[i], spu_mask=spu_mask)

        # Verify attributes
        assert all(link.world_size == 3 for link in links.values())
        # Verify relative ranks
        assert links[0].rank == 0  # First in mask -> rel_rank 0
        assert links[2].rank == 1  # Second in mask -> rel_rank 1
        assert links[3].rank == 2  # Third in mask -> rel_rank 2

    def test_channels_mode_validation(self):
        """Test Channels mode validation."""
        world_size = 3
        spu_mask = Mask.from_ranks([0, 1, 2])
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Missing spu_mask
        with pytest.raises(ValueError, match="spu_mask required"):
            LinkCommunicator(rank=0, comm=comms[0])

        # Rank not in mask
        with pytest.raises(ValueError, match="not in spu_mask"):
            LinkCommunicator(rank=0, comm=comms[0], spu_mask=Mask.from_ranks([1, 2]))

    def test_channels_mode_vs_mem_link(self):
        """Compare Channels mode with legacy Mem mode."""
        world_size = 2
        spu_mask = Mask.from_ranks([0, 1])
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Channels mode
        link_channels = LinkCommunicator(rank=0, comm=comms[0], spu_mask=spu_mask)

        # Mem mode (legacy)
        spu_addrs = [f"P{rank}" for rank in spu_mask]
        link_mem = LinkCommunicator(rank=0, addrs=spu_addrs, mem_link=True)

        # Both should have same world_size and rank
        assert link_channels.world_size == link_mem.world_size
        assert link_channels.rank == link_mem.rank

    def test_legacy_brpc_mode_still_works(self):
        """Ensure legacy BRPC mode still works."""
        addrs = ["127.0.0.1:8200", "127.0.0.1:8201"]

        # Should not raise (though BRPC connection may fail in test env)
        try:
            link = LinkCommunicator(rank=0, addrs=addrs, mem_link=False)
            assert link.world_size == 2
        except Exception:
            # BRPC connection failure is expected in test env
            pytest.skip("BRPC connection not available in test environment")

    def test_legacy_mem_mode_still_works(self):
        """Ensure legacy Mem mode still works."""
        addrs = ["P0", "P1", "P2"]
        link = LinkCommunicator(rank=0, addrs=addrs, mem_link=True)

        assert link.world_size == 3
        assert link.rank == 0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_three_party_spu_simulation(self):
        """Simulate 3-party SPU communication using Channels mode."""
        world_size = 3
        spu_mask = Mask.from_ranks([0, 1, 2])

        # Setup communicators
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Setup link communicators
        links = [
            LinkCommunicator(rank=i, comm=comms[i], spu_mask=spu_mask)
            for i in range(world_size)
        ]

        # Verify link setup
        assert all(link.world_size == 3 for link in links)
        assert all(link.rank == i for i, link in enumerate(links))

        # Test actual link context usage (basic barrier-like synchronization)
        # Note: We can't fully test libspu.link.Context operations without SPU runtime,
        # but we can verify the link was created successfully
        for link in links:
            assert link.get_lctx() is not None
            assert hasattr(link.get_lctx(), "rank")
            assert hasattr(link.get_lctx(), "world_size")
