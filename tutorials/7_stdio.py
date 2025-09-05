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

import jax.numpy as jnp
import numpy as np

import mplang
import mplang.simp as simp
import mplang.smpc as smpc
from mplang.frontend import builtin


@mplang.function
def save_data():
    # Party 0 creates and saves data
    x = simp.constant(np.array([[1, 2], [3, 4]], dtype=np.float32))
    y = simp.constant(np.array([[5, 6], [7, 8]], dtype=np.float32))

    x = simp.runAt(0, builtin.write)(x, "tmp/x.npy")
    y = simp.runAt(1, builtin.write)(y, "tmp/y.npy")

    return x, y


@mplang.function
def load_data():
    tensor_info = mplang.core.TensorType(shape=(2, 2), dtype=jnp.float32)

    x = simp.runAt(0, builtin.read)("tmp/x.npy", tensor_info)
    y = simp.runAt(1, builtin.read)("tmp/y.npy", tensor_info)

    x_ = smpc.sealFrom(x, 0)
    y_ = smpc.sealFrom(y, 1)
    z_ = smpc.srun(lambda a, b: a + b)(x_, y_)
    z = smpc.reveal(z_)

    return z


def run_stdio_example():
    print("\n--- Session 1: Creating and saving data ---")
    print("Creating matrices:")
    print("x = [[1, 2], [3, 4]]")
    print("y = [[5, 6], [7, 8]]")

    sim2 = mplang.Simulator(2)
    _ = mplang.evaluate(sim2, save_data)
    print("Data saved to files successfully")

    print("\n--- Session 2: Loading data and computing x + y ---")
    sim3 = mplang.Simulator(3)
    z = mplang.evaluate(sim3, load_data)
    result = mplang.fetch(sim3, z)
    print(result)


if __name__ == "__main__":
    run_stdio_example()
