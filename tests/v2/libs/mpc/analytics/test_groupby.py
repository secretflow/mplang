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

"""Tests for Oblivious Group-by Sum library."""

# Register implementations
import numpy as np

import mplang.v2 as mp
import mplang.v2.backends.tensor_impl  # noqa: F401
from mplang.v2.dialects import simp, tensor
from mplang.v2.libs.mpc.analytics import groupby


def _unwrap(val):
    """Unwrap TensorValue if needed."""
    if hasattr(val, "unwrap"):
        return val.unwrap()
    return val


class TestGroupbyBFV:
    def setup_method(self):
        self.interp = simp.make_simulator(world_size=2)

    def test_small_k(self):
        # N=10, K=3
        # Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Bins: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        # Sums:
        # Bin 0: 1 + 4 + 7 + 10 = 22
        # Bin 1: 2 + 5 + 8 = 15
        # Bin 2: 3 + 6 + 9 = 18

        data_np = np.arange(1, 11, dtype=np.int64)
        bins_np = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)
        K = 3

        with self.interp:
            # P0 holds data
            d = simp.pcall_static((0,), lambda: tensor.constant(data_np))
            # P1 holds bins
            b = simp.pcall_static((1,), lambda: tensor.constant(bins_np))

            # Run protocol
            res = groupby.oblivious_groupby_sum_bfv(d, b, K, sender=0, receiver=1)

            # Result is on Receiver (P1)
            res_vals = mp.fetch(res)
            p1_res = _unwrap(res_vals[1])

            expected = np.array([22, 15, 18], dtype=np.int64)
            np.testing.assert_array_equal(p1_res, expected)

    def test_chunking(self):
        # N=5000, degree=4096.
        # This forces multiple chunks (2 chunks).
        N = 5000
        K = 5
        degree = 4096

        np.random.seed(42)
        data_np = np.ones(N, dtype=np.int64)
        bins_np = np.random.randint(0, K, size=N, dtype=np.int64)

        # Calculate expected
        expected = np.zeros(K, dtype=np.int64)
        for i in range(N):
            expected[bins_np[i]] += data_np[i]

        with self.interp:
            d = simp.pcall_static((0,), lambda: tensor.constant(data_np))
            b = simp.pcall_static((1,), lambda: tensor.constant(bins_np))

            res = groupby.oblivious_groupby_sum_bfv(
                d, b, K, sender=0, receiver=1, poly_modulus_degree=degree
            )

            res_vals = mp.fetch(res)
            p1_res = _unwrap(res_vals[1])
            np.testing.assert_array_equal(p1_res, expected)

    def test_exact_chunk_multiple(self):
        # N=8192, degree=8192. Exactly 2 chunks (B=4096).
        N = 8192
        K = 2
        degree = 8192

        data_np = np.arange(N, dtype=np.int64)
        bins_np = (np.arange(N) % K).astype(np.int64)

        expected = np.zeros(K, dtype=np.int64)
        for i in range(N):
            expected[bins_np[i]] += data_np[i]

        with self.interp:
            d = simp.pcall_static((0,), lambda: tensor.constant(data_np))
            b = simp.pcall_static((1,), lambda: tensor.constant(bins_np))

            res = groupby.oblivious_groupby_sum_bfv(
                d,
                b,
                K,
                sender=0,
                receiver=1,
                poly_modulus_degree=degree,
                plain_modulus=536903681,
            )

            res_vals = mp.fetch(res)
            p1_res = _unwrap(res_vals[1])
            np.testing.assert_array_equal(p1_res, expected)

    def test_empty_bins(self):
        # K=5, but only bin 0 and 4 have data.
        N = 20
        K = 5
        degree = 4096

        data_np = np.ones(N, dtype=np.int64)
        bins_np = np.zeros(N, dtype=np.int64)
        bins_np[10:] = 4

        expected = np.array([10, 0, 0, 0, 10], dtype=np.int64)

        with self.interp:
            d = simp.pcall_static((0,), lambda: tensor.constant(data_np))
            b = simp.pcall_static((1,), lambda: tensor.constant(bins_np))

            res = groupby.oblivious_groupby_sum_bfv(
                d, b, K, sender=0, receiver=1, poly_modulus_degree=degree
            )

            res_vals = mp.fetch(res)
            p1_res = _unwrap(res_vals[1])
            np.testing.assert_array_equal(p1_res, expected)


class TestGroupbyShuffle:
    def setup_method(self):
        self.interp = simp.make_simulator(world_size=3)

    def test_small_k(self):
        # N=10, K=3
        # Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Bins: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        # Sums:
        # Bin 0: 1 + 4 + 7 + 10 = 22
        # Bin 1: 2 + 5 + 8 = 15
        # Bin 2: 3 + 6 + 9 = 18

        data_np = np.arange(1, 11, dtype=np.int64)
        bins_np = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)
        K = 3

        with self.interp:
            # P0 holds data
            d = simp.pcall_static((0,), lambda: tensor.constant(data_np))
            # P1 holds bins
            b = simp.pcall_static((1,), lambda: tensor.constant(bins_np))

            # Run protocol
            res = groupby.oblivious_groupby_sum_shuffle(
                d, b, K, sender=0, receiver=1, helper=2
            )

            # Result is on Receiver (P1)
            res_vals = mp.fetch(res)
            p1_res = _unwrap(res_vals[1])

            expected = np.array([22, 15, 18], dtype=np.int64)
            np.testing.assert_array_equal(p1_res, expected)

    def test_random(self):
        N = 100
        K = 5

        np.random.seed(42)
        data_np = np.random.randint(0, 100, size=N, dtype=np.int64)
        bins_np = np.random.randint(0, K, size=N, dtype=np.int64)

        expected = np.zeros(K, dtype=np.int64)
        for i in range(N):
            expected[bins_np[i]] += data_np[i]

        with self.interp:
            d = simp.pcall_static((0,), lambda: tensor.constant(data_np))
            b = simp.pcall_static((1,), lambda: tensor.constant(bins_np))

            res = groupby.oblivious_groupby_sum_shuffle(
                d, b, K, sender=0, receiver=1, helper=2
            )

            res_vals = mp.fetch(res)
            p1_res = res_vals[1]
            np.testing.assert_array_equal(_unwrap(p1_res), expected)
