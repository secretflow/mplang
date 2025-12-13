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

import numpy as np

import mplang.v2 as mp
from mplang.v2.dialects import field, simp, tensor


def test_field_mul_integration():
    """Verify field.mul invokes C++ kernel correctly."""
    sim = simp.make_simulator(1)
    mp.set_root_context(sim)

    def protocol():
        # Create Inputs (uint64 pairs representing GF128 elements)
        # A = [1, 2] (low, high)
        # B = [3, 4]
        a = tensor.constant(np.array([1, 2], dtype=np.uint64))
        b = tensor.constant(np.array([3, 4], dtype=np.uint64))

        # Use field.mul
        res = field.mul(a, b)
        return res

    traced = mp.compile(protocol)
    result = mp.evaluate(traced)
    val = mp.fetch(result)

    print("Result:", val)

    # Expected: [1083, 2]
    expected = np.array([1083, 2], dtype=np.uint64)
    assert np.array_equal(val, expected), f"Expected {expected}, got {val}"


if __name__ == "__main__":
    test_field_mul_integration()
