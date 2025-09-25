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
import mplang.simp as simp
import mplang.simp.random as mpr


def eval_and_fetch(sim, fn, *args, **kwargs):
    """Helper function to evaluate a function and fetch results."""
    result = mplang.evaluate(sim, fn, *args, **kwargs)
    return mplang.fetch(sim, result)


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
        shuffled = mplang.evaluate(sim, simp.pshfl, src, index)

        data, index, shuffled = mplang.fetch(sim, (src, index, shuffled))
        data, index, shuffled = np.stack(data), np.stack(index), np.stack(shuffled)
        np.testing.assert_array_equal(data[index], shuffled)

    # TODO(jint): add shfl complicated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
