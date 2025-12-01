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

import numpy as np
import pytest

import mplang
import mplang.v1.simp.mpi as mpi
import mplang.v1.simp.random as mpr
from mplang.v1.core.primitive import pshfl
from mplang.v1.simp.api import run_jax_at


def eval_and_fetch(sim, fn, *args, **kwargs):
    """Helper function to evaluate a function and fetch results."""
    result = mplang.evaluate(sim, fn, *args, **kwargs)
    return mplang.fetch(sim, result)


class TestMPICommunication:
    """Test MPI communication primitives like gather, scatter, broadcast, etc."""

    def test_gather_m_basic(self):
        """Test basic gather_m primitive with 3 parties"""
        num_parties = 3
        sim = mplang.Simulator.simple(num_parties)

        data = mplang.evaluate(sim, mpr.prandint, 0, 100)
        gathered = mplang.evaluate(sim, mpi.gather_m, (1 << 3) - 1, 0, data)
        data, gathered = mplang.fetch(sim, (data, gathered))

        assert len(gathered) == 3  # Gathered from 3 parties
        assert gathered[0] == [data[0], None, None]
        assert gathered[1] == [data[1], None, None]
        assert gathered[2] == [data[2], None, None]

    def test_gather_m_different_root(self):
        """Test gather_m with different root party"""
        num_parties = 4
        sim = mplang.Simulator.simple(num_parties)

        # Test with root = 2
        data = mplang.evaluate(sim, mpr.prandint, 0, 100)
        gathered = mplang.evaluate(sim, mpi.gather_m, (1 << 4) - 1, 2, data)
        data, gathered = mplang.fetch(sim, (data, gathered))

        assert len(gathered) == 4
        # gathered is a list of 4 elements (one for each party's data being gathered)
        # Each element is a tuple showing where that data resides
        # Only root party 2 should have the actual gathered data
        for i in range(4):
            assert gathered[i][0] is None  # Party 0 doesn't have gathered data
            assert gathered[i][1] is None  # Party 1 doesn't have gathered data
            assert gathered[i][2] is not None  # Party 2 (root) has gathered data
            assert gathered[i][3] is None  # Party 3 doesn't have gathered data

    def test_gather_m_subset_parties(self):
        """Test gather_m from subset of parties"""
        num_parties = 4
        sim = mplang.Simulator.simple(num_parties)

        # Create data at parties 0, 1, 3
        data = mplang.evaluate(sim, mpr.prandint, 0, 100)
        # Gather from parties 0, 1, 3 (mask = 1011 = 11)
        src_mask = (1 << 0) | (1 << 1) | (1 << 3)
        gathered = mplang.evaluate(sim, mpi.gather_m, src_mask, 0, data)
        data, gathered = mplang.fetch(sim, (data, gathered))

        assert len(gathered) == 3  # Only 3 parties in the mask

    def test_gather_m_two_parties(self):
        """Test gather_m with minimum number of parties"""
        num_parties = 2
        sim = mplang.Simulator.simple(num_parties)

        data = mplang.evaluate(sim, mpr.prandint, 0, 50)
        gathered = mplang.evaluate(sim, mpi.gather_m, (1 << 2) - 1, 1, data)
        data, gathered = mplang.fetch(sim, (data, gathered))

        assert len(gathered) == 2
        # gathered is a list of 2 elements (one for each party's data being gathered)
        # Only root party 1 should have the actual gathered data
        for i in range(2):
            assert gathered[i][0] is None  # Party 0 doesn't have gathered data
            assert gathered[i][1] is not None  # Party 1 (root) has gathered data

    def test_scatter_m_basic(self):
        """Test basic scatter_m primitive"""
        num_parties = 3
        sim = mplang.Simulator.simple(num_parties)

        # Create data to scatter at party 0
        data = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: [10, 20, 30]))
        scattered = mplang.evaluate(sim, mpi.scatter_m, (1 << 3) - 1, 0, data)
        data, scattered = mplang.fetch(sim, (data, scattered))

        assert len(scattered) == 3  # Scattered to 3 parties
        assert scattered == [10, 20, 30]

    def test_scatter_m_different_root(self):
        """Test scatter_m with different root party"""
        num_parties = 4
        sim = mplang.Simulator.simple(num_parties)

        # Create data to scatter at party 2
        data = mplang.evaluate(sim, lambda: run_jax_at(2, lambda: [100, 200, 300, 400]))
        scattered = mplang.evaluate(sim, mpi.scatter_m, (1 << 4) - 1, 2, data)
        data, scattered = mplang.fetch(sim, (data, scattered))

        assert len(scattered) == 4
        assert scattered == [100, 200, 300, 400]

    def test_scatter_m_subset_parties(self):
        """Test scatter_m to subset of parties"""
        num_parties = 4
        sim = mplang.Simulator.simple(num_parties)

        # Create data to scatter to parties 0, 2, 3
        data = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: [5, 15, 25]))
        # Scatter to parties 0, 2, 3 (mask = 1101 = 13)
        to_mask = (1 << 0) | (1 << 2) | (1 << 3)
        scattered = mplang.evaluate(sim, mpi.scatter_m, to_mask, 0, data)
        data, scattered = mplang.fetch(sim, (data, scattered))

        # Only parties 0, 2, 3 should receive data
        assert scattered == [5, None, 15, 25]

    def test_scatter_m_large_data(self):
        """Test scatter_m with larger data values"""
        num_parties = 2
        sim = mplang.Simulator.simple(num_parties)

        # Create larger data values
        data = mplang.evaluate(sim, lambda: run_jax_at(1, lambda: [9999, 8888]))
        scattered = mplang.evaluate(sim, mpi.scatter_m, (1 << 2) - 1, 1, data)
        data, scattered = mplang.fetch(sim, (data, scattered))

        assert scattered == [9999, 8888]

    def test_bcast_m_basic(self):
        """Test basic bcast_m primitive"""
        num_parties = 3
        sim = mplang.Simulator.simple(num_parties)

        # Create data at party 0
        data = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: 42))
        # Broadcast from party 0 to all parties
        broadcasted = mplang.evaluate(sim, mpi.bcast_m, (1 << 3) - 1, 0, data)
        data, broadcasted = mplang.fetch(sim, (data, broadcasted))

        # All parties should receive the broadcasted value
        assert broadcasted == [42, 42, 42]

    def test_bcast_m_different_root(self):
        """Test bcast_m with different root party"""
        num_parties = 4
        sim = mplang.Simulator.simple(num_parties)

        # Create data at party 3
        data = mplang.evaluate(sim, lambda: run_jax_at(3, lambda: 123))
        # Broadcast from party 3 to all parties
        broadcasted = mplang.evaluate(sim, mpi.bcast_m, (1 << 4) - 1, 3, data)
        data, broadcasted = mplang.fetch(sim, (data, broadcasted))

        # All parties should receive the broadcasted value
        assert broadcasted == [123, 123, 123, 123]

    def test_bcast_m_subset_parties(self):
        """Test bcast_m to subset of parties"""
        num_parties = 5
        sim = mplang.Simulator.simple(num_parties)

        # Create data at party 1
        data = mplang.evaluate(sim, lambda: run_jax_at(1, lambda: 777))
        # Broadcast from party 1 to parties 0, 1, 3, 4 (mask = 11011 = 27)
        to_mask = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)
        broadcasted = mplang.evaluate(sim, mpi.bcast_m, to_mask, 1, data)
        data, broadcasted = mplang.fetch(sim, (data, broadcasted))

        # Only selected parties should receive the data
        assert broadcasted == [777, 777, None, 777, 777]

    def test_bcast_m_negative_value(self):
        """Test bcast_m with negative values"""
        num_parties = 2
        sim = mplang.Simulator.simple(num_parties)

        # Create negative data
        data = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: -456))
        broadcasted = mplang.evaluate(sim, mpi.bcast_m, (1 << 2) - 1, 0, data)
        data, broadcasted = mplang.fetch(sim, (data, broadcasted))

        assert len(broadcasted) == 2
        assert broadcasted == [-456, -456]

    def test_bcast_m_zero_value(self):
        """Test bcast_m with zero value"""
        num_parties = 3
        sim = mplang.Simulator.simple(num_parties)

        # Create zero data
        data = mplang.evaluate(sim, lambda: run_jax_at(2, lambda: 0))
        broadcasted = mplang.evaluate(sim, mpi.bcast_m, (1 << 3) - 1, 2, data)
        data, broadcasted = mplang.fetch(sim, (data, broadcasted))

        assert len(broadcasted) == 3
        assert broadcasted == [0, 0, 0]

    def test_p2p_basic(self):
        """Test basic p2p primitive"""
        num_parties = 3
        sim = mplang.Simulator.simple(num_parties)

        # Create data at party 0
        data = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: 99))
        # Send from party 0 to party 2
        sent = mplang.evaluate(sim, mpi.p2p, 0, 2, data)
        data, sent = mplang.fetch(sim, (data, sent))

        # Only party 2 should receive the data
        assert sent == [None, None, 99]

    def test_p2p_different_pairs(self):
        """Test p2p between different party pairs"""
        num_parties = 4
        sim = mplang.Simulator.simple(num_parties)

        # Test party 3 -> party 1
        data = mplang.evaluate(sim, lambda: run_jax_at(3, lambda: 555))
        sent = mplang.evaluate(sim, mpi.p2p, 3, 1, data)
        data, sent = mplang.fetch(sim, (data, sent))

        assert sent == [None, 555, None, None]

    def test_p2p_multiple_transfers(self):
        """Test multiple p2p transfers in sequence"""
        num_parties = 3
        sim = mplang.Simulator.simple(num_parties)

        # First transfer: party 0 -> party 1
        data1 = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: 100))
        sent1 = mplang.evaluate(sim, mpi.p2p, 0, 1, data1)

        # Second transfer: party 1 -> party 2
        data2 = mplang.evaluate(sim, lambda: run_jax_at(1, lambda: 200))
        sent2 = mplang.evaluate(sim, mpi.p2p, 1, 2, data2)

        data1, sent1, data2, sent2 = mplang.fetch(sim, (data1, sent1, data2, sent2))

        assert sent1 == [None, 100, None]
        assert sent2 == [None, None, 200]

    def test_p2p_large_value(self):
        """Test p2p with large values"""
        num_parties = 3
        sim = mplang.Simulator.simple(num_parties)

        large_value = 999999999
        data = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: large_value))
        result = mplang.evaluate(sim, mpi.p2p, 0, 2, data)

        data, result = mplang.fetch(sim, (data, result))

        assert result == [None, None, large_value]

    def test_p2p_same_party(self):
        """Test p2p primitive when source and destination are the same"""
        num_parties = 2
        sim = mplang.Simulator.simple(num_parties)

        # Create data at party 0
        data = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: 77))
        # Send from party 0 to party 0 (same party)
        sent = mplang.evaluate(sim, mpi.p2p, 0, 0, data)
        data, sent = mplang.fetch(sim, (data, sent))

        # Party 0 should still have the data (only party 0 has the data)
        assert sent == [77, None]

    def test_p2p_same_party_different_parties(self):
        """Test p2p same party with different party configurations"""
        num_parties = 4
        sim = mplang.Simulator.simple(num_parties)

        # Test party 2 -> party 2
        data = mplang.evaluate(sim, lambda: run_jax_at(2, lambda: 333))
        sent = mplang.evaluate(sim, mpi.p2p, 2, 2, data)
        data, sent = mplang.fetch(sim, (data, sent))

        assert sent == [None, None, 333, None]

    # TODO: add test for allgather_m when it's implemented
    def test_allgather_m_placeholder(self):
        """Placeholder test for allgather_m - will be enabled when implemented"""
        num_parties = 3
        sim = mplang.Simulator.simple(num_parties)

        data = mplang.evaluate(sim, mpr.prandint, 0, 50)

        # This should raise NotImplementedError for now
        with pytest.raises(NotImplementedError, match="Allgather not implemented"):
            mplang.evaluate(sim, mpi.allgather_m, (1 << 3) - 1, data)

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling for MPI primitives"""
        num_parties = 3
        mplang.Simulator.simple(num_parties)

        # Test with single party
        single_party_sim = mplang.Simulator.simple(1)
        original_data = mplang.evaluate(
            single_party_sim, lambda: run_jax_at(0, lambda: 42)
        )

        # Broadcast to single party
        broadcasted = mplang.evaluate(
            single_party_sim, mpi.bcast_m, 1, 0, original_data
        )
        _original_data_result, broadcasted = mplang.fetch(
            single_party_sim, (original_data, broadcasted)
        )
        assert broadcasted == [42]

        # P2P to same party in single party setup - use original MPObject
        sent = mplang.evaluate(single_party_sim, mpi.p2p, 0, 0, original_data)
        sent_result = mplang.fetch(single_party_sim, sent)
        assert sent_result == [42]

    def test_stress_with_many_parties(self):
        """Test MPI operations with more parties"""
        num_parties = 8
        sim = mplang.Simulator.simple(num_parties)

        # Test broadcast to all 8 parties
        data = mplang.evaluate(sim, lambda: run_jax_at(4, lambda: 888))
        broadcasted = mplang.evaluate(sim, mpi.bcast_m, (1 << 8) - 1, 4, data)
        data, broadcasted = mplang.fetch(sim, (data, broadcasted))

        expected = [888] * 8
        assert broadcasted == expected

        # Test p2p chain communication
        data = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: 111))
        # 0 -> 7
        sent = mplang.evaluate(sim, mpi.p2p, 0, 7, data)
        data, sent = mplang.fetch(sim, (data, sent))

        expected_sent = [None] * 7 + [111]
        assert sent == expected_sent

    def test_data_types_variety(self):
        """Test MPI operations with different data types and values"""
        num_parties = 3
        sim = mplang.Simulator.simple(num_parties)

        # Test with float-like values (represented as integers)
        data = mplang.evaluate(sim, lambda: run_jax_at(1, lambda: 1234))
        broadcasted = mplang.evaluate(sim, mpi.bcast_m, (1 << 3) - 1, 1, data)
        data, broadcasted = mplang.fetch(sim, (data, broadcasted))
        assert broadcasted == [1234, 1234, 1234]

        # Test with very large numbers
        large_data = mplang.evaluate(sim, lambda: run_jax_at(0, lambda: 2**20))
        sent = mplang.evaluate(sim, mpi.p2p, 0, 2, large_data)
        large_data, sent = mplang.fetch(sim, (large_data, sent))
        assert sent == [None, None, 2**20]


class TestPShfl:
    """Test pshfl related functions"""

    def test_pshfl_basic(self):
        """Test basic pshfl_s functionality"""
        num_parties = 10
        sim = mplang.Simulator.simple(num_parties)

        src = mplang.evaluate(sim, mpr.prandint, 0, 100)
        key = mplang.evaluate(sim, mpr.ukey, 42)
        index = mplang.evaluate(sim, mpr.pperm, key)

        # shuffle data with range, nothing changed.
        shuffled = mplang.evaluate(sim, pshfl, src, index)

        data, index, shuffled = mplang.fetch(sim, (src, index, shuffled))
        data, index, shuffled = np.stack(data), np.stack(index), np.stack(shuffled)
        np.testing.assert_array_equal(data[index], shuffled)

    # TODO(jint): add shfl complicated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
