"""Tests for mplang2.device module."""

import jax.numpy as jnp
import pytest

from mplang2.backends.simp_simulator import SimpSimulator
from mplang2.edsl import Interpreter
from mplang2.edsl.context import pop_context, push_context
from mplang2.libs.device import (
    ClusterSpec,
    device,
    get_dev_attr,
    put,
    set_global_cluster,
)


@pytest.fixture(autouse=True)
def setup_context():
    # Ensure we start with a clean state
    sim = SimpSimulator(world_size=3)
    push_context(sim)
    yield
    pop_context()


# Mocking simp and spu for testing purposes
# In a real test, we would use the actual dialects, but they might require
# a full runtime setup. For unit testing device logic, we can mock them
# or rely on the fact that they return Objects.

# However, since we are testing the device logic which calls simp/spu,
# and those calls are expected to work, we should probably use the real ones
# if possible, or mock them if they are too heavy.
# Given the current state of the repo, let's try to use the real ones
# but with a mock interpreter context if needed.

# Actually, let's just use the real Interpreter context.
# We need to set up a global cluster first.


@pytest.fixture(autouse=True)
def setup_cluster():
    cluster = ClusterSpec.simple(
        world_size=3, enable_ppu_device=True, enable_spu_device=True
    )
    set_global_cluster(cluster)

    # Also push a default interpreter context
    ctx = Interpreter()
    push_context(ctx)
    yield
    pop_context()


def test_put_ppu():
    x = jnp.array([1, 2, 3])
    x_p0 = put("P0", x)

    assert get_dev_attr(x_p0) == "P0"
    # In a real execution, x_p0 would hold the value on P0.
    # Since we are using the Interpreter, it should execute immediately.
    # However, simp.pcall_static implementation details matter here.
    # Assuming it works, we just check the device attribute.


def test_put_spu():
    print("Running test_put_spu")
    x = jnp.array([1, 2, 3])
    x_sp0 = put("SP0", x)

    assert get_dev_attr(x_sp0) == "SP0"


def test_device_explicit():
    print("Running test_device_explicit")

    @device("P0")
    def add(a, b):
        return a + b

    x = put("P0", jnp.array(1))
    y = put("P0", jnp.array(2))

    z = add(x, y)
    assert get_dev_attr(z) == "P0"


def test_device_auto_ppu():
    print("Running test_device_auto_ppu")

    @device
    def add(a, b):
        return a + b

    x = put("P0", jnp.array(1))
    y = put("P0", jnp.array(2))

    z = add(x, y)
    assert get_dev_attr(z) == "P0"


def test_device_auto_spu():
    print("Running test_device_auto_spu")

    @device
    def add(a, b):
        return a + b

    x = put("SP0", jnp.array(1))
    y = put("SP0", jnp.array(2))

    z = add(x, y)
    assert get_dev_attr(z) == "SP0"


def test_device_auto_transfer_ppu_to_spu():
    print("Running test_device_auto_transfer_ppu_to_spu")

    @device
    def add(a, b):
        return a + b

    x = put("P0", jnp.array(1))
    y = put("SP0", jnp.array(2))

    # Should infer SPU because one arg is on SPU
    z = add(x, y)
    assert get_dev_attr(z) == "SP0"


def test_explicit_transfer():
    print("Running test_explicit_transfer")
    x = put("P0", jnp.array(1))
    x_p1 = put("P1", x)
    assert get_dev_attr(x_p1) == "P1"

    x_sp0 = put("SP0", x)
    assert get_dev_attr(x_sp0) == "SP0"

    x_back = put("P0", x_sp0)
    assert get_dev_attr(x_back) == "P0"
