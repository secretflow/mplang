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
        ch0_spu = BaseChannel(comms[0], 0, 1, tag_prefix="spu")
        ch1_spu = BaseChannel(comms[1], 1, 0, tag_prefix="spu")
        ch0_custom = BaseChannel(comms[0], 0, 1, tag_prefix="custom")
        ch1_custom = BaseChannel(comms[1], 1, 0, tag_prefix="custom")

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

        # These are no-ops for MPLang but should not raise
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


class TestLinkCommunicatorChannelsMode:
    """Test LinkCommunicator with Channels mode."""

    def test_channels_mode_basic(self):
        """Test LinkCommunicator with Channels mode."""
        import threading

        world_size = 3
        spu_mask = Mask.from_ranks([0, 1, 2])
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Create LinkCommunicators in parallel to avoid deadlock
        # (create_with_channels does handshake via TestSend/TestRecv)
        links = [None] * world_size
        exceptions = [None] * world_size

        def create_link(rank):
            try:
                links[rank] = LinkCommunicator(
                    rank=rank, comm=comms[rank], spu_mask=spu_mask
                )
            except Exception as e:
                exceptions[rank] = e

        threads = [
            threading.Thread(target=create_link, args=(i,)) for i in range(world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for exceptions
        for _, exc in enumerate(exceptions):
            if exc is not None:
                raise exc

        # Verify attributes
        assert all(link.world_size == 3 for link in links)
        for i, link in enumerate(links):
            # Relative rank should match global rank when all parties in mask
            assert link.rank == i

    def test_channels_mode_subset_mask(self):
        """Test Channels mode with subset of parties in SPU."""
        import threading

        world_size = 4
        spu_mask = Mask.from_ranks([0, 2, 3])  # Rank 1 not in SPU
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Create LinkCommunicators only for SPU parties (in parallel)
        links = {}
        exceptions = {}

        def create_link(rank):
            try:
                links[rank] = LinkCommunicator(
                    rank=rank, comm=comms[rank], spu_mask=spu_mask
                )
            except Exception as e:
                exceptions[rank] = e

        threads = [threading.Thread(target=create_link, args=(i,)) for i in spu_mask]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for exceptions
        for _, exc in exceptions.items():
            if exc is not None:
                raise exc

        # Verify attributes
        assert all(link.world_size == 3 for link in links.values())
        # Verify relative ranks
        assert links[0].rank == 0  # First in mask -> rel_rank 0
        assert links[2].rank == 1  # Second in mask -> rel_rank 1
        assert links[3].rank == 2  # Third in mask -> rel_rank 2

    def test_channels_mode_validation(self):
        """Test Channels mode validation."""
        world_size = 3
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Missing spu_mask
        with pytest.raises(ValueError, match="spu_mask required"):
            LinkCommunicator(rank=0, comm=comms[0])

        # Rank not in mask
        with pytest.raises(ValueError, match="not in spu_mask"):
            LinkCommunicator(rank=0, comm=comms[0], spu_mask=Mask.from_ranks([1, 2]))


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_three_party_spu_simulation(self):
        """Simulate 3-party SPU communication using Channels mode."""
        import threading

        world_size = 3
        spu_mask = Mask.from_ranks([0, 1, 2])

        # Setup communicators
        comms = [ThreadCommunicator(i, world_size) for i in range(world_size)]
        for c in comms:
            c.set_peers(comms)

        # Setup link communicators (in parallel to avoid deadlock)
        links = [None] * world_size
        exceptions = [None] * world_size

        def create_link(rank):
            try:
                links[rank] = LinkCommunicator(
                    rank=rank, comm=comms[rank], spu_mask=spu_mask
                )
            except Exception as e:
                exceptions[rank] = e

        threads = [
            threading.Thread(target=create_link, args=(i,)) for i in range(world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for exceptions
        for _, exc in enumerate(exceptions):
            if exc is not None:
                raise exc

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
