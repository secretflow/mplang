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

"""Tests for fetch semantics: identity during tracing, real fetch during evaluation.

These tests protect the Design A semantics choice for mp.fetch:
- During tracing (compile): fetch returns identity (input unchanged)
- During execution (evaluate): fetch actually retrieves data from workers

See mp.fetch docstring for the A vs B tradeoff discussion.
"""

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


class TestFetchIdentityDuringTracing:
    """Test that fetch returns identity during tracing (Design A)."""

    def test_fetch_returns_same_object_during_compile(self, simulator):
        """fetch(x) during compile should return the same Object x."""

        def workflow():
            x = tensor.constant(np.array([1.0, 2.0]))
            y = mp.fetch(x)  # During tracing, should return x unchanged
            return y

        traced = mp.compile(workflow)

        # Should trace successfully - fetch didn't break the trace
        assert traced is not None
        assert len(traced.graph.outputs) == 1

    def test_fetch_in_trace_is_identity(self, simulator):
        """Verify fetch acts as identity: fetch(x) == x during tracing."""

        def workflow():
            x = tensor.constant(np.array([1.0, 2.0]))
            y = mp.fetch(x)
            # y should be the same Object as x during tracing
            # so we can continue to use y in further operations
            z = tensor.run_jax(lambda a: a * 2, y)
            return z

        traced = mp.compile(workflow)

        # Should compile successfully - y was still a valid Object
        assert traced is not None
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)
        np.testing.assert_allclose(fetched, [2.0, 4.0])

    def test_multiple_fetch_only_return_determines_output(self, simulator):
        """Multiple fetch calls don't affect graph outputs - only return does."""

        def workflow():
            a = tensor.constant(np.array([1.0]))
            b = tensor.constant(np.array([2.0]))
            _x = mp.fetch(a)  # fetch a but don't return it
            y = mp.fetch(b)  # fetch b
            return y  # only b is returned

        traced = mp.compile(workflow)

        # Only one output (b), not two (a and b)
        assert len(traced.graph.outputs) == 1

        result = mp.evaluate(traced)
        fetched = mp.fetch(result)
        np.testing.assert_allclose(fetched, [2.0])


class TestFetchDuringExecution:
    """Test that fetch actually retrieves data during execution."""

    def test_fetch_returns_python_value_during_evaluate(self, simulator):
        """fetch during evaluate should return actual Python values."""

        def workflow():
            return tensor.constant(np.array([1.0, 2.0, 3.0]))

        traced = mp.compile(workflow)
        result = mp.evaluate(traced)

        # result is InterpObject, fetch should convert to numpy array
        fetched = mp.fetch(result)
        assert isinstance(fetched, np.ndarray)
        np.testing.assert_allclose(fetched, [1.0, 2.0, 3.0])

    def test_fetch_nested_pytree(self, simulator):
        """fetch should work on nested PyTree structures."""

        def workflow():
            a = tensor.constant(np.array([1.0]))
            b = tensor.constant(np.array([2.0]))
            return {"x": a, "y": (b, b)}

        traced = mp.compile(workflow)
        result = mp.evaluate(traced)
        fetched = mp.fetch(result)

        assert isinstance(fetched, dict)
        np.testing.assert_allclose(fetched["x"], [1.0])
        assert isinstance(fetched["y"], tuple)
        np.testing.assert_allclose(fetched["y"][0], [2.0])
        np.testing.assert_allclose(fetched["y"][1], [2.0])
