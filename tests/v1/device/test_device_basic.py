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

"""Tests for device API: explicit placement, auto-inference, and device operations."""

import pytest

import mplang.v1 as mp


def test_explicit_device_placement():
    """Test explicit device specification for computation."""
    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        x = mp.device("P0")(lambda: 41)()
        y = mp.device("P0")(lambda a: a + 1)(x)
        return y

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    assert int(fetched[0]) == 42


def test_device_computation():
    """Test basic computation on a device."""
    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        a = mp.device("P0")(lambda: 2)()
        b = mp.device("P0")(lambda x: x * 3)(a)
        return b

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    assert int(fetched[0]) == 6


def test_auto_device_inference_same_device():
    """Test auto device inference when all args are on same device."""
    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        x = mp.device("P0")(lambda: 10)()
        y = mp.device("P0")(lambda: 5)()
        # Auto inference: should use P0 since both x and y are on P0
        z = mp.device(lambda a, b: a + b)(x, y)
        return z

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    assert int(fetched[0]) == 15


def test_auto_device_inference_no_args_error():
    """Test that auto device without MPObject args raises error."""
    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        # No MPObject arguments - should fail
        return mp.device(lambda: 42)()

    with pytest.raises(ValueError, match=r"Cannot infer device.*no MPObject"):
        mp.evaluate(sim, compute)


def test_auto_device_inline():
    """Test inline lambda with auto device inference."""
    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        x = mp.device("P0")(lambda: 20)()
        # Inline lambda - should infer P0 from x
        y = mp.device(lambda a: a * 2)(x)
        return y

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    assert int(fetched[0]) == 40


def test_auto_device_decorator_style():
    """Test decorator style with auto device inference."""
    sim = mp.Simulator.simple(2)

    @mp.device
    def multiply(a, b):
        return a * b

    @mp.function
    def compute():
        x = mp.device("P0")(lambda: 3)()
        y = mp.device("P0")(lambda: 7)()
        # multiply should infer P0 from x and y
        z = multiply(x, y)
        return z

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    assert int(fetched[0]) == 21


def test_device_transfer_ppu_to_spu():
    """Test automatic data transfer from PPU to SPU."""
    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        x = mp.device("P0")(lambda: 10)()
        y = mp.device("P1")(lambda: 20)()
        # Transfer to SP0 for secure computation
        z = mp.device("SP0")(lambda a, b: a + b)(x, y)
        # Reveal result to P0
        result = mp.put("P0", z)
        return result

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    assert int(fetched[0]) == 30


def test_device_transfer_spu_to_ppu():
    """Test revealing SPU result back to PPU."""
    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        x = mp.device("P0")(lambda: 15)()
        y = mp.device("P1")(lambda: 25)()
        # Compute on SP0
        z = mp.device("SP0")(lambda a, b: a < b)(x, y)
        # Reveal to P0
        result = mp.put("P0", z)
        return result

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    assert bool(fetched[0]) is True


def test_multiple_device_operations():
    """Test chaining multiple device operations."""
    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        # Create data on different devices
        x = mp.device("P0")(lambda: 100)()
        y = mp.device("P1")(lambda: 50)()

        # Process on P0
        x_processed = mp.device("P0")(lambda a: a * 2)(x)

        # Compare on SP0
        comparison = mp.device("SP0")(lambda a, b: a > b)(x_processed, y)

        # Reveal result to P0
        result = mp.put("P0", comparison)
        return result

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    assert bool(fetched[0]) is True  # 200 > 50
