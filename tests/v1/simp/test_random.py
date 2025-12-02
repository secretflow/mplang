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
import mplang.v1.simp.random as mpr


def eval_and_fetch(sim, fn, *args, **kwargs):
    """Helper function to evaluate a function and fetch results."""
    result = mplang.evaluate(sim, fn, *args, **kwargs)
    return mplang.fetch(sim, result)


def test_ukey():
    num_parties = 3
    sim = mplang.Simulator.simple(num_parties)

    # Test ukey with a specific seed
    results = eval_and_fetch(sim, mpr.ukey, 42)

    # All parties should generate the same key since it's uniform
    assert len(results) == num_parties
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])

    # Key should be of the expected shape (2,) for JAX default PRNG
    assert results[0].shape == (2,)
    assert results[0].dtype == np.uint32

    # Test with different seeds should produce different keys
    seed2 = 123
    results2 = eval_and_fetch(sim, mpr.ukey, seed2)
    assert not np.array_equal(results[0], results2[0])


def test_urandint():
    num_parties = 4
    sim = mplang.Simulator.simple(num_parties)

    # test with constant key
    key = np.random.randint(0, 2**32, (2,), dtype=np.uint32)
    results = eval_and_fetch(sim, mpr.urandint, key, 0, 1000, (3, 2))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    assert results[0].shape == (3, 2)
    assert np.all(results[0] >= 0) and np.all(results[0] < 1000)

    key = mplang.evaluate(sim, mpr.ukey, 42)
    # test with shape
    results = eval_and_fetch(sim, mpr.urandint, key, 0, 1000, (3, 2))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    assert results[0].shape == (3, 2)
    assert np.all(results[0] >= 0) and np.all(results[0] < 1000)

    # test without shape (scalar)
    results = eval_and_fetch(sim, mpr.urandint, key, -500, 500, ())
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    assert results[0].shape == ()
    assert np.all(results[0] >= -500) and np.all(results[0] < 500)

    # test with small range
    results = eval_and_fetch(sim, mpr.urandint, key, 10, 11, (2,))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    assert results[0].shape == (2,)
    assert np.all(results[0] == 10)  # since range is [10, 11), should always be 10


def test_prandint():
    num_parties = 3
    sim = mplang.Simulator.simple(num_parties)

    # test with shape
    results = eval_and_fetch(sim, mpr.prandint, 0, 100, (2, 3))
    assert len(results) == num_parties
    for res in results:
        assert res.shape == (2, 3)
        assert np.all(res >= 0) and np.all(res < 100)

    # assert results from parties are different
    for i in range(1, len(results)):
        assert not np.array_equal(results[0], results[i])

    # test without shape (scalar)
    results = eval_and_fetch(sim, mpr.prandint, -50, 50, ())
    for res in results:
        assert res.shape == ()
        assert np.all(res >= -50) and np.all(res < 50)

    # test fixed range
    results = eval_and_fetch(sim, mpr.prandint, 10, 11, (2,))
    for res in results:
        assert res.shape == (2,)
        assert np.all(res == 10)  # since range is [10, 11), should always be 10


def test_pperm():
    num_parties = 10
    sim = mplang.Simulator.simple(num_parties)

    # Generate a key with mask covering all parties
    key = mplang.evaluate(sim, mpr.ukey, 42)

    # Test pperm with the key
    results = eval_and_fetch(sim, mpr.pperm, key)
    assert len(results) == num_parties

    # All results together should form a permutation of [0, 1, 2]
    result = np.stack(results)
    np.testing.assert_array_equal(np.sort(result), np.arange(num_parties))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
