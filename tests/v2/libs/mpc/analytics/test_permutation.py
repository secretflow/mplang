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

"""Tests for Permutation Network library."""

import jax.numpy as jnp

import mplang.v2 as mp
import mplang.v2.backends.tensor_impl  # noqa: F401
from mplang.v2.dialects import simp, tensor
from mplang.v2.libs.mpc.analytics import permutation


def test_secure_switch_straight():
    # World size 2: Party 0 (Sender), Party 1 (Receiver)
    interp = simp.make_simulator(world_size=2)

    # Sender: x0=10, x1=20
    # Receiver: c=0 (Straight) -> y0=10, y1=20

    def protocol(x0, x1, c):
        return permutation.secure_switch(x0, x1, c, sender=0, receiver=1)

    with interp:
        x0_obj = simp.constant((0,), 10)
        x1_obj = simp.constant((0,), 20)
        c_obj = simp.constant((1,), 0)

        y0, y1 = protocol(x0_obj, x1_obj, c_obj)

        y0_val = mp.fetch(y0)
        y1_val = mp.fetch(y1)
        assert y0_val[1] == 10
        assert y1_val[1] == 20


def test_secure_switch_swap():
    # World size 2: Party 0 (Sender), Party 1 (Receiver)
    interp = simp.make_simulator(world_size=2)

    # Sender: x0=10, x1=20

    # Receiver: c=1 (Swap) -> y0=20, y1=10

    def protocol(x0, x1, c):
        return permutation.secure_switch(x0, x1, c, sender=0, receiver=1)

    with interp:
        x0_obj = simp.constant((0,), 10)
        x1_obj = simp.constant((0,), 20)
        c_obj = simp.constant((1,), 1)

        y0, y1 = protocol(x0_obj, x1_obj, c_obj)

        y0_val = mp.fetch(y0)
        y1_val = mp.fetch(y1)
        assert y0_val[1] == 20
        assert y1_val[1] == 10


def test_apply_permutation_n2():
    # World size 2
    interp = simp.make_simulator(world_size=2)

    # Sender: data=[10, 20]
    # Receiver: perm=[1, 0] (Swap) -> [20, 10]

    def protocol(d0, d1, p0, p1):
        data = [d0, d1]
        # Pack permutation into a tensor/list
        # In this test, we pass individual elements to construct the list
        # But apply_permutation expects a list of Objects.
        # And permutation is expected to be an Object (Tensor) or list of Objects?
        # The implementation expects `permutation` to be an Object (Tensor) in `get_control_bit`.

        # Let's construct the permutation tensor on Receiver

        def make_perm(a, b):
            return tensor.run_jax(lambda x, y: jnp.array([x, y]), a, b)

        perm = simp.pcall_static((1,), make_perm, p0, p1)

        return permutation.apply_permutation(data, perm, sender=0, receiver=1)

    with interp:
        d0_obj = simp.constant((0,), 10)
        d1_obj = simp.constant((0,), 20)
        p0_obj = simp.constant((1,), 1)
        p1_obj = simp.constant((1,), 0)

        res = protocol(d0_obj, d1_obj, p0_obj, p1_obj)

        # res is a list of Objects on Receiver
        res0_val = mp.fetch(res[0])
        res1_val = mp.fetch(res[1])
        assert res0_val[1] == 20
        assert res1_val[1] == 10
