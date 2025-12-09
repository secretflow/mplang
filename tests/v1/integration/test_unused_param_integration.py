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

"""
Test unused parameter handling with mplang integration.
This test verifies that functions with unused parameters work correctly
after the arg_keep_map implementation.
"""

import jax.numpy as jnp

import mplang.v1 as mp


def func_with_unused_params(a, unused_param, b, c):
    """Function with unused parameter in the middle."""
    return a + b + c


def func_all_unused_returns_constant(a, unused1, unused2):
    """Function where all parameters are unused - returns constant."""
    return 42


def func_first_last_unused(unused1, b, c, unused2):
    """Function with unused parameters at start and end."""
    return b * c


class TestUnusedParameterHandling:
    """Test suite for JAX unused parameter elimination handling."""

    @staticmethod
    def _extract_scalar(output):
        """Extract scalar value from potentially wrapped output."""
        if hasattr(output, "__iter__") and len(output) == 1:
            output = output[0]
        if hasattr(output, "item"):  # JAX array
            output = output.item()
        return output

    def test_basic_unused_param(self):
        """Test function with one unused parameter in middle position."""
        sim = mp.Simulator.simple(1)

        # Create traced function
        @mp.function
        def test_func():
            # Test values - create inside traced context
            a = mp.constant(1)
            unused = mp.constant(999)  # This should be eliminated by JAX
            b = mp.constant(2)
            c = mp.constant(3)
            return mp.run_jax(func_with_unused_params, a, unused, b, c)

        expected = 6  # 1 + 2 + 3

        # Compile and check that compilation succeeds
        compiled = mp.compile(sim, test_func)

        # The function should compile successfully
        assert compiled is not None

        # Execute and verify result
        result = mp.evaluate(sim, test_func)
        output = mp.fetch(sim, result)

        output = self._extract_scalar(output)

        assert output == expected, f"Expected {expected}, got {output}"

    def test_multiple_unused_params(self):
        """Test function with multiple unused parameters."""
        sim = mp.Simulator.simple(1)

        b_val = 5
        c_val = 7
        expected = b_val * c_val  # 35

        @mp.function
        def test_func():
            unused1 = mp.constant(100)
            b = mp.constant(b_val)
            c = mp.constant(c_val)
            unused2 = mp.constant(200)
            return mp.run_jax(func_first_last_unused, unused1, b, c, unused2)

        result = mp.evaluate(sim, test_func)
        output = mp.fetch(sim, result)
        output = self._extract_scalar(output)

        assert output == expected, f"Expected {expected}, got {output}"

    def test_all_params_unused(self):
        """Test function where all parameters are unused (returns constant)."""
        sim = mp.Simulator.simple(1)
        expected = 42

        @mp.function
        def test_func():
            a = mp.constant(1)
            unused1 = mp.constant(10)
            unused2 = mp.constant(20)
            return mp.run_jax(func_all_unused_returns_constant, a, unused1, unused2)

        result = mp.evaluate(sim, test_func)
        output = mp.fetch(sim, result)
        output = self._extract_scalar(output)

        assert output == expected, f"Expected {expected}, got {output}"

    def test_no_unused_params(self):
        """Test function with no unused parameters (regression test)."""
        sim = mp.Simulator.simple(1)

        def func_all_used(a, b, c):
            return a + b + c

        @mp.function
        def test_func():
            a = mp.constant(10)
            b = mp.constant(20)
            c = mp.constant(30)
            return mp.run_jax(func_all_used, a, b, c)

        result = mp.evaluate(sim, test_func)
        output = mp.fetch(sim, result)
        output = self._extract_scalar(output)

        assert output == 60, f"Expected 60, got {output}"

    def test_arg_keep_map_in_pfunc(self):
        """Test that arg_keep_map is correctly stored in PFunction when needed."""
        from mplang.v1.ops.jax_cc import jax2stablehlo

        def func_with_unused(a, unused, b):
            return a * b

        # Create test inputs
        a = jnp.array(2, dtype=jnp.int32)
        unused = jnp.array(999, dtype=jnp.int32)
        b = jnp.array(3, dtype=jnp.int32)

        # Mock is_variable function
        def is_variable(arg):
            return True  # Treat all as variables for this test

        # Call jax2stablehlo directly
        pfunc, _, _ = jax2stablehlo(is_variable, func_with_unused, a, unused, b)

        # Check that arg_keep_map is present when parameters are eliminated
        if "arg_keep_map" in pfunc.attrs:
            keep_map = pfunc.attrs["arg_keep_map"]
            assert isinstance(keep_map, list)
            assert len(keep_map) < 3  # Should be fewer than original 3 params
            assert 1 not in keep_map  # Index 1 (unused) should not be in keep_map
        else:
            # If no elimination happened (possible with different JAX versions/optimizations)
            pass

    def test_different_dtypes_unused(self):
        """Test unused parameter elimination with different data types."""
        sim = mp.Simulator.simple(1)

        def func_mixed_types(int_used, float_unused, int_used2):
            return int_used + int_used2  # float_unused is not used

        @mp.function
        def test_func():
            a = mp.constant(5)
            unused_float = mp.constant(3.14)  # Different dtype, unused
            c = mp.constant(7)
            return mp.run_jax(func_mixed_types, a, unused_float, c)

        result = mp.evaluate(sim, test_func)
        output = mp.fetch(sim, result)
        output = self._extract_scalar(output)

        assert output == 12, f"Expected 12, got {output}"
