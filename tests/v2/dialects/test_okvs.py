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


def test_okvs_edsl() -> None:
    n = 100
    m = int(n * 1.6)

    # Generate data
    keys_np = np.arange(n, dtype=np.uint64)
    values_np = np.zeros((n, 2), dtype=np.uint64)
    for i in range(n):
        values_np[i, 0] = i
        values_np[i, 1] = i * 10

    sim = simp.make_simulator(1)
    mp.set_root_context(sim)

    def protocol():
        # Create inputs as tensor constants (field.solve_okvs expects TensorType, not MPType)
        keys = tensor.constant(keys_np)
        values = tensor.constant(values_np)
        seed = tensor.constant(np.array([0xDEADBEEF, 0xCAFEBABE], dtype=np.uint64))

        # Solve OKVS
        storage = field.solve_okvs(keys, values, m, seed)
        # Decode OKVS
        decoded = field.decode_okvs(keys, storage, seed)
        return decoded

    traced = mp.compile(protocol)
    result = mp.evaluate(traced)
    decoded_res = mp.fetch(result)

    assert np.array_equal(decoded_res, values_np), "Decoded values do not match!"
    print("OKVS EDSL Test Passed!")


if __name__ == "__main__":
    test_okvs_edsl()
