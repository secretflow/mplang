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

"""Tests for Sparse OKVS (Unbalanced PSI) using EDSL."""

import numpy as np
import pytest

import mplang.v2.dialects.tensor as tensor
from mplang.v2 import SimpSimulator, evaluate, fetch
from mplang.v2.libs.mpc.psi.sparse_okvs import (
    NUM_HASHES,
    compute_indices,
    decode_values,
)


class TestSparseOKVSEDSL:
    """Test Sparse OKVS operations using EDSL execution."""

    @pytest.fixture
    def sim(self):
        return SimpSimulator()

    def _fetch_one(self, sim, obj):
        """Helper to fetch result from the first party."""
        vals = fetch(sim, obj)
        if isinstance(vals, list):
            return vals[0]
        return vals

    def test_compute_indices_deterministic(self, sim):
        """Verify hash positions are deterministic via EDSL."""
        n = 10
        table_size = 1000
        keys = np.arange(n, dtype=np.uint64)
        seed = np.array([123, 456], dtype=np.uint64)

        def _prog(k, s):
            k_t = tensor.constant(k)
            s_t = tensor.constant(s)
            return compute_indices(k_t, table_size, s_t)

        # Run twice
        pos1_obj = evaluate(sim, _prog, keys, seed)
        pos2_obj = evaluate(sim, _prog, keys, seed)

        pos1 = self._fetch_one(sim, pos1_obj)
        pos2 = self._fetch_one(sim, pos2_obj)

        np.testing.assert_array_equal(pos1, pos2)
        assert pos1.shape == (n * NUM_HASHES,)

    def test_compute_indices_range(self, sim):
        """Verify positions are within table bounds."""
        n = 100
        table_size = 500
        keys = np.random.randint(0, 100000, size=(n,), dtype=np.uint64)
        seed = np.array([0, 0], dtype=np.uint64)

        def _prog(k, s):
            k_t = tensor.constant(k)
            s_t = tensor.constant(s)
            return compute_indices(k_t, table_size, s_t)

        positions_obj = evaluate(sim, _prog, keys, seed)
        positions = self._fetch_one(sim, positions_obj)

        assert np.all(positions < table_size)
        assert np.all(positions >= 0)

    def test_lookup_correctness(self, sim):
        """Verify sparse lookup returns correct XOR sum of table entries."""
        n = 5
        table_size = 20

        # Inputs
        keys = np.arange(n, dtype=np.uint64)
        seed = np.array([42, 99], dtype=np.uint64)

        # Server Table (M, 2)
        table_np = np.random.randint(0, 2**63, size=(table_size, 2), dtype=np.uint64)

        def _prog(k, s, tbl):
            k_t = tensor.constant(k)
            s_t = tensor.constant(s)
            tbl_t = tensor.constant(tbl)

            # 1. Client computes positions
            pos = compute_indices(k_t, table_size, s_t)

            # 2. Server looks up values
            res = decode_values(pos, tbl_t)

            return pos, res

        positions_obj, result_obj = evaluate(sim, _prog, keys, seed, table_np)

        positions = self._fetch_one(sim, positions_obj)
        result = self._fetch_one(sim, result_obj)

        # Verify manually in Python
        # We trust the EDSL execution, but we verify the logic:
        # result[i] should be XOR(table[pos[i][0]], table[pos[i][1]], table[pos[i][2]])

        positions_2d = positions.reshape(n, NUM_HASHES)
        expected = np.zeros((n, 2), dtype=np.uint64)

        for i in range(n):
            for j in range(NUM_HASHES):
                idx = positions_2d[i, j]
                expected[i] ^= table_np[idx]

        np.testing.assert_array_equal(result, expected)

    def test_different_seeds_different_positions(self, sim):
        """Verify changing seed changes positions."""
        n = 10
        table_size = 1000
        keys = np.arange(n, dtype=np.uint64)
        seed1 = np.array([1, 1], dtype=np.uint64)
        seed2 = np.array([2, 2], dtype=np.uint64)

        def _prog(k, s1, s2):
            k_t = tensor.constant(k)
            s1_t = tensor.constant(s1)
            s2_t = tensor.constant(s2)

            p1 = compute_indices(k_t, table_size, s1_t)
            p2 = compute_indices(k_t, table_size, s2_t)
            return p1, p2

        pos1_obj, pos2_obj = evaluate(sim, _prog, keys, seed1, seed2)

        pos1 = self._fetch_one(sim, pos1_obj)
        pos2 = self._fetch_one(sim, pos2_obj)

        assert not np.array_equal(pos1, pos2)

    def test_query_e2e(self, sim):
        """Verify the full query protocol between two parties."""
        from mplang.v2.libs.mpc.psi.sparse_okvs import query

        client_rank = 0
        server_rank = 1
        n = 5
        table_size = 20

        # Inputs
        keys = np.arange(n, dtype=np.uint64)
        seed = np.array([42, 99], dtype=np.uint64)
        # Server Table (M, 2)
        table_np = np.random.randint(0, 2**63, size=(table_size, 2), dtype=np.uint64)

        def _prog(k, s, tbl):
            k_t = tensor.constant(k)
            s_t = tensor.constant(s)
            tbl_t = tensor.constant(tbl)

            return query(client_rank, server_rank, k_t, tbl_t, s_t, table_size)

        # Run simulation
        result_obj = evaluate(sim, _prog, keys, seed, table_np)

        # Fetch results. Since query returns a value on the server (P1),
        # we expect the result to be available at index 1 (assuming P0, P1 order).
        results = fetch(sim, result_obj)

        # In SimpSimulator with 2 parties (default), results is [val_p0, val_p1]
        # The result is produced on P1.
        server_result = results[1]

        # Verify manually
        # 1. Re-run compute_indices logic locally (or via a separate sim call) to get positions
        # We can use a separate evaluate to get the expected positions for verification
        def _get_pos(k, s):
            return compute_indices(tensor.constant(k), table_size, tensor.constant(s))

        pos_obj = evaluate(sim, _get_pos, keys, seed)
        positions = self._fetch_one(sim, pos_obj)

        positions_2d = positions.reshape(n, NUM_HASHES)
        expected = np.zeros((n, 2), dtype=np.uint64)

        for i in range(n):
            for j in range(NUM_HASHES):
                idx = positions_2d[i, j]
                expected[i] ^= table_np[idx]

        np.testing.assert_array_equal(server_result, expected)
