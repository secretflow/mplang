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

"""Tests for Sparse OKVS (RR22 Style)."""

import numpy as np
import pytest

import mplang.v2 as mp
from mplang.v2.dialects import simp, tensor
from mplang.v2.libs.mpc.psi.okvs_gct import SparseOKVS, get_okvs_expansion


class TestSparseOKVSEDSL:
    """Test Sparse OKVS operations using EDSL execution."""

    @pytest.fixture(autouse=True)
    def setup_sim(self):
        sim = simp.make_simulator(world_size=3)
        mp.set_root_context(sim)
        yield

    def _fetch_one(self, obj):
        """Helper to fetch result from the first party."""
        vals = mp.fetch(obj)
        if isinstance(vals, list):
            return vals[0]
        return vals

    def test_encode_decode_correctness(self):
        """Verify OKVS Encode/Decode roundtrip."""
        n = 100
        expansion = get_okvs_expansion(n)
        m = int(n * expansion)
        # Align to 128
        if m % 128 != 0:
            m = ((m // 128) + 1) * 128

        keys = np.arange(n, dtype=np.uint64)
        # Random items to store (values)
        values = np.random.randint(0, 2**63, size=(n, 2), dtype=np.uint64)
        seed = np.array([0xDEADBEEF, 0xCAFEBABE], dtype=np.uint64)

        def _prog(k, v, s):
            k_t = tensor.constant(k)
            v_t = tensor.constant(v)
            s_t = tensor.constant(s)

            okvs = SparseOKVS(m)
            storage = okvs.encode(k_t, v_t, s_t)
            recovered = okvs.decode(k_t, storage, s_t)

            return recovered

        recovered_obj = mp.evaluate(_prog, keys, values, seed)
        recovered = self._fetch_one(recovered_obj)

        np.testing.assert_array_equal(recovered, values)

    def test_seed_diversity(self):
        """Verify different seeds produce different storage but correct values."""
        n = 50
        m = int(n * 1.5)
        keys = np.arange(n, dtype=np.uint64)
        values = np.random.randint(0, 2**63, size=(n, 2), dtype=np.uint64)

        seed1 = np.array([1, 1], dtype=np.uint64)
        seed2 = np.array([2, 2], dtype=np.uint64)

        def _prog(k, v, s1, s2):
            k_t = tensor.constant(k)
            v_t = tensor.constant(v)
            s1_t = tensor.constant(s1)
            s2_t = tensor.constant(s2)

            okvs = SparseOKVS(m)

            # Encode with different seeds
            store1 = okvs.encode(k_t, v_t, s1_t)
            store2 = okvs.encode(k_t, v_t, s2_t)

            # Decode to verify correctness
            rec1 = okvs.decode(k_t, store1, s1_t)
            rec2 = okvs.decode(k_t, store2, s2_t)

            return store1, store2, rec1, rec2

        res = mp.evaluate(_prog, keys, values, seed1, seed2)
        store1, store2, rec1, rec2 = (self._fetch_one(r) for r in res)

        # Storages should be different
        assert not np.array_equal(store1, store2)

        # Recovered values should match original
        np.testing.assert_array_equal(rec1, values)
        np.testing.assert_array_equal(rec2, values)

    def test_expansion_factor(self):
        """Verify expansion factor logic."""
        assert get_okvs_expansion(10) >= 1.2
        assert get_okvs_expansion(1000000) <= 1.4
