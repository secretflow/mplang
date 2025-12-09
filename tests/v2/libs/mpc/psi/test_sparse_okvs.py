# Copyright 2025 Ant Group Co., Ltd.
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

"""Tests for Sparse OKVS (Unbalanced PSI)."""

import numpy as np
import pytest

from mplang.v2.libs.mpc.psi.sparse_okvs import (
    CUCKOO_EXPANSION,
    NUM_HASHES,
    compute_hash_positions,
    sparse_decode_numpy,
    sparse_encode_numpy,
)


class TestSparseOKVSNumpyBasic:
    """Test basic Sparse OKVS operations."""

    def test_hash_positions_deterministic(self):
        """Verify hash positions are deterministic."""
        key = np.uint64(12345)
        table_size = 1000

        pos1 = compute_hash_positions(key, table_size)
        pos2 = compute_hash_positions(key, table_size)

        np.testing.assert_array_equal(pos1, pos2)
        assert len(pos1) == NUM_HASHES

    def test_hash_positions_distinct(self):
        """Verify different keys produce different positions."""
        table_size = 10000

        pos1 = compute_hash_positions(np.uint64(100), table_size)
        pos2 = compute_hash_positions(np.uint64(200), table_size)

        # At least one position should differ
        assert not np.array_equal(pos1, pos2)

    def test_hash_positions_in_range(self):
        """Verify positions are within table bounds."""
        table_size = 1000

        for key in range(100):
            pos = compute_hash_positions(np.uint64(key), table_size)
            assert np.all(pos < table_size)
            assert np.all(pos >= 0)

    def test_sparse_encode_shapes(self):
        """Verify sparse encode output shapes."""
        n = 100
        keys = np.arange(n, dtype=np.uint64)
        values = np.random.randint(0, 2**63, size=(n, 2), dtype=np.uint64)
        server_table_size = 10000

        positions, coefficients, client_values = sparse_encode_numpy(
            keys, values, server_table_size
        )

        assert positions.shape == (n * NUM_HASHES,)
        assert coefficients.shape == (n * NUM_HASHES,)
        assert client_values.shape == (n, 2)

    def test_sparse_encode_positions_in_range(self):
        """Verify encoded positions are within table bounds."""
        n = 100
        keys = np.arange(n, dtype=np.uint64)
        values = np.random.randint(0, 2**63, size=(n, 2), dtype=np.uint64)
        server_table_size = 10000

        positions, _, _ = sparse_encode_numpy(keys, values, server_table_size)

        assert np.all(positions < server_table_size)
        assert np.all(positions >= 0)


class TestSparseOKVSRoundtrip:
    """Test Sparse OKVS encode/decode roundtrip."""

    def test_decode_xor_roundtrip(self):
        """Verify decode returns XOR of table entries at positions."""
        n = 10
        table_size = 100

        # Create a simple table
        table = np.random.randint(0, 2**63, size=(table_size, 2), dtype=np.uint64)

        # Create positions that we know
        keys = np.arange(n, dtype=np.uint64)
        values = np.zeros((n, 2), dtype=np.uint64)  # dummy values

        positions, _, _ = sparse_encode_numpy(keys, values, table_size)

        # Decode
        decoded = sparse_decode_numpy(positions, table)

        # Verify manually
        positions_2d = positions.reshape(n, NUM_HASHES)
        expected = np.zeros((n, 2), dtype=np.uint64)
        for i in range(n):
            for j in range(NUM_HASHES):
                expected[i] ^= table[positions_2d[i, j]]

        np.testing.assert_array_equal(decoded, expected)


class TestSparseOKVSCommunication:
    """Test communication size properties."""

    def test_communication_sublinear(self):
        """Verify communication is O(n), not O(N)."""
        client_n = 1000
        server_n = 1000000  # 1000x larger

        # Client set determines communication
        keys = np.arange(client_n, dtype=np.uint64)
        values = np.random.randint(0, 2**63, size=(client_n, 2), dtype=np.uint64)
        server_table_size = int(server_n * CUCKOO_EXPANSION)

        positions, _coefficients, _ = sparse_encode_numpy(
            keys, values, server_table_size
        )

        # Communication = positions + server response
        # positions: O(n * 3) * 8 bytes
        # response: O(n) * 16 bytes
        position_bytes = positions.nbytes
        response_bytes = client_n * 16  # (n, 2) uint64

        total_comm = position_bytes + response_bytes

        # Compare to dense OKVS communication: O(N) * 16 bytes
        dense_comm = server_n * 16

        # Sparse should be much smaller
        assert total_comm < dense_comm / 100  # At least 100x smaller
        print(
            f"Sparse: {total_comm / 1024:.2f} KB, Dense: {dense_comm / 1024 / 1024:.2f} MB"
        )
        print(f"Improvement: {dense_comm / total_comm:.1f}x")


class TestSparseOKVSIntegration:
    """Integration tests with EDSL."""

    def test_psi_unbalanced_basic(self):
        """Test unbalanced PSI end-to-end."""
        import mplang.v2 as mp
        from mplang.v2.dialects import tensor
        from mplang.v2.libs.mpc.psi.unbalanced import psi_unbalanced

        sim = mp.Simulator.simple(2)

        server = 0
        client = 1
        server_n = 1000
        client_n = 100

        # Create test data with known intersection
        server_items = np.arange(server_n, dtype=np.uint64)
        client_items = np.arange(50, 150, dtype=np.uint64)  # 50 intersect

        def job():
            s_items = tensor.constant(server_items)
            c_items = tensor.constant(client_items)

            mask = psi_unbalanced(server, client, server_n, client_n, s_items, c_items)
            return mask

        traced = mp.compile(sim, job)
        mask_obj = mp.evaluate(sim, traced)

        mask_val = mp.fetch(sim, mask_obj)[client]

        # Items 50-99 should be in intersection (first 50 client items)
        # Items 100-149 should be in intersection (last 50 client items overlap with 100-149)
        # Actually: client=[50..149], server=[0..999]
        # Intersection: [50..149] all in server
        expected_intersect = client_n  # All 100 client items are in server
        actual_intersect = np.sum(mask_val)

        print(f"Expected intersection: {expected_intersect}")
        print(f"Actual intersection: {actual_intersect}")

        assert actual_intersect == expected_intersect


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
