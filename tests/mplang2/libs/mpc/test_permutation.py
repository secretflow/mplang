"""Tests for Permutation Network library."""

import jax.numpy as jnp

import mplang2.backends.crypto_impl
import mplang2.backends.tensor_impl  # noqa: F401
from mplang2.backends.simp_simulator import SimpSimulator
from mplang2.dialects import simp, tensor
from mplang2.libs.mpc import permutation


def test_secure_switch_straight():
    # World size 2: Party 0 (Sender), Party 1 (Receiver)
    interp = SimpSimulator(world_size=2)

    # Sender: x0=10, x1=20
    # Receiver: c=0 (Straight) -> y0=10, y1=20

    def protocol(x0, x1, c):
        return permutation.secure_switch(x0, x1, c, sender=0, receiver=1)

    with interp:
        x0_obj = simp.constant((0,), 10)
        x1_obj = simp.constant((0,), 20)
        c_obj = simp.constant((1,), 0)

        y0, y1 = protocol(x0_obj, x1_obj, c_obj)

    assert y0.runtime_obj.values[1] == 10
    assert y1.runtime_obj.values[1] == 20


def test_secure_switch_swap():
    # World size 2: Party 0 (Sender), Party 1 (Receiver)
    interp = SimpSimulator(world_size=2)

    # Sender: x0=10, x1=20
    # Receiver: c=1 (Swap) -> y0=20, y1=10

    def protocol(x0, x1, c):
        return permutation.secure_switch(x0, x1, c, sender=0, receiver=1)

    with interp:
        x0_obj = simp.constant((0,), 10)
        x1_obj = simp.constant((0,), 20)
        c_obj = simp.constant((1,), 1)

        y0, y1 = protocol(x0_obj, x1_obj, c_obj)

    assert y0.runtime_obj.values[1] == 20
    assert y1.runtime_obj.values[1] == 10


def test_apply_permutation_n2():
    # World size 2
    interp = SimpSimulator(world_size=2)

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

    assert res[0].runtime_obj.values[1] == 20
    assert res[1].runtime_obj.values[1] == 10
