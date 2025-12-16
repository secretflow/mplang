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

"""Tests for PyTree input/output handling in compile/evaluate/fetch."""

import numpy as np
import pytest

import mplang.v2 as mp
from mplang.v2.dialects import simp, tensor


@pytest.fixture
def simulator():
    """Create a simple simulator for testing."""
    sim = simp.make_simulator(world_size=1)
    mp.set_root_context(sim, force=True)
    yield sim
    mp.set_root_context(None, force=True)


class TestPyTreeOutputs:
    """Test that functions can return various PyTree structures."""

    def test_single_value_output(self, simulator):
        """Single value output."""

        def fn():
            return tensor.constant(np.array([1.0, 2.0]))

        traced = mp.compile(fn)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        np.testing.assert_allclose(fetched, [1.0, 2.0])

    def test_tuple_output(self, simulator):
        """Tuple of values as output."""

        def fn():
            a = tensor.constant(np.array([1.0]))
            b = tensor.constant(np.array([2.0]))
            return a, b

        traced = mp.compile(fn)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        assert isinstance(fetched, tuple)
        assert len(fetched) == 2
        np.testing.assert_allclose(fetched[0], [1.0])
        np.testing.assert_allclose(fetched[1], [2.0])

    def test_list_output(self, simulator):
        """List of values as output."""

        def fn():
            a = tensor.constant(np.array([1.0]))
            b = tensor.constant(np.array([2.0]))
            c = tensor.constant(np.array([3.0]))
            return [a, b, c]

        traced = mp.compile(fn)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        assert isinstance(fetched, list)
        assert len(fetched) == 3
        np.testing.assert_allclose(fetched[0], [1.0])
        np.testing.assert_allclose(fetched[1], [2.0])
        np.testing.assert_allclose(fetched[2], [3.0])

    def test_dict_output(self, simulator):
        """Dict of values as output."""

        def fn():
            return {
                "x": tensor.constant(np.array([1.0])),
                "y": tensor.constant(np.array([2.0])),
            }

        traced = mp.compile(fn)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        assert isinstance(fetched, dict)
        assert set(fetched.keys()) == {"x", "y"}
        np.testing.assert_allclose(fetched["x"], [1.0])
        np.testing.assert_allclose(fetched["y"], [2.0])

    def test_nested_pytree_output(self, simulator):
        """Nested PyTree (dict containing tuple) as output."""

        def fn():
            a = tensor.constant(np.array([1.0]))
            b = tensor.constant(np.array([2.0]))
            c = tensor.constant(np.array([3.0]))
            return {
                "pair": (a, b),
                "single": c,
            }

        traced = mp.compile(fn)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        assert isinstance(fetched, dict)
        assert isinstance(fetched["pair"], tuple)
        np.testing.assert_allclose(fetched["pair"][0], [1.0])
        np.testing.assert_allclose(fetched["pair"][1], [2.0])
        np.testing.assert_allclose(fetched["single"], [3.0])

    def test_mixed_traced_and_constant_output(self, simulator):
        """Output containing both traced values and Python constants."""

        def fn():
            a = tensor.constant(np.array([1.0]))
            return a, 42, "hello"

        traced = mp.compile(fn)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        assert isinstance(fetched, tuple)
        assert len(fetched) == 3
        np.testing.assert_allclose(fetched[0], [1.0])
        assert fetched[1] == 42
        assert fetched[2] == "hello"


class TestPyTreeInputs:
    """Test that functions can accept various PyTree input structures."""

    def test_multiple_positional_args(self, simulator):
        """Multiple positional arguments."""

        def add_fn(a, b):
            return tensor.run_jax(lambda x, y: x + y, a, b)

        # Use tensor.constant to create traced values
        def create_inputs():
            x = tensor.constant(np.array([1.0, 2.0]))
            y = tensor.constant(np.array([3.0, 4.0]))
            return x, y

        # Trace a function that creates inputs and computes
        def workflow():
            x = tensor.constant(np.array([1.0, 2.0]))
            y = tensor.constant(np.array([3.0, 4.0]))
            return tensor.run_jax(lambda a, b: a + b, x, y)

        traced = mp.compile(workflow)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        np.testing.assert_allclose(fetched, [4.0, 6.0])

    def test_nested_dict_inputs(self, simulator):
        """Nested dict structure as function input."""

        def workflow():
            data = {
                "x": tensor.constant(np.array([1.0])),
                "y": tensor.constant(np.array([2.0])),
            }
            return tensor.run_jax(lambda a, b: a * b, data["x"], data["y"])

        traced = mp.compile(workflow)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        np.testing.assert_allclose(fetched, [2.0])


class TestPyTreeRoundTrip:
    """Test complete round-trip with complex PyTree structures."""

    def test_complex_pytree_roundtrip(self, simulator):
        """Complex nested structure round-trip."""

        def workflow():
            v0 = tensor.constant(np.array([1.0, 2.0]))
            v1 = tensor.constant(np.array([3.0, 4.0]))
            result = tensor.run_jax(lambda a, b: a + b, v0, v1)
            return {
                "sum": result,
                "original": (v0, v1),
            }

        traced = mp.compile(workflow)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        assert isinstance(fetched, dict)
        np.testing.assert_allclose(fetched["sum"], [4.0, 6.0])
        assert isinstance(fetched["original"], tuple)
        np.testing.assert_allclose(fetched["original"][0], [1.0, 2.0])
        np.testing.assert_allclose(fetched["original"][1], [3.0, 4.0])

    def test_list_of_dicts_output(self, simulator):
        """List containing dict structures."""

        def workflow():
            a = tensor.constant(np.array([1.0]))
            b = tensor.constant(np.array([2.0]))
            return [
                {"value": a, "name": "first"},
                {"value": b, "name": "second"},
            ]

        traced = mp.compile(workflow)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        assert isinstance(fetched, list)
        assert len(fetched) == 2
        np.testing.assert_allclose(fetched[0]["value"], [1.0])
        assert fetched[0]["name"] == "first"
        np.testing.assert_allclose(fetched[1]["value"], [2.0])
        assert fetched[1]["name"] == "second"
