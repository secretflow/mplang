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

import mplang.v1 as mp

sim3 = mp.Simulator.simple(3)

# make two variables on the simulator, one is a random integer, the other is a prank.
x = mp.evaluate(sim3, mp.prank)
y = mp.evaluate(sim3, lambda: mp.prandint(0, 100))
print(mp.fetch(sim3, (x, y)))


def pass_and_capture(x):
    # pass x as a parameter, and capture y from the outer scope
    return mp.run_jax(jnp.multiply, x, y)


z = mp.evaluate(sim3, pass_and_capture, x)
print(mp.fetch(sim3, z))

# jit it, still works.
jitted = mp.function(pass_and_capture)

z1 = mp.evaluate(sim3, jitted, x)
print(mp.fetch(sim3, z1))
