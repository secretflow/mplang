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

from __future__ import annotations

import base64

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mplang.core import PFunction
from mplang.kernels.context import RuntimeContext
from mplang.kernels.jax_xla import _jax_exec
from mplang.kernels.value import TensorValue
from mplang.ops.jax_cc import jax_export

# Enable 64-bit precision for testing
jax.config.update("jax_enable_x64", True)


class TestJaxXla:
    """Test suite for JAX XLA kernel functionality."""

    def setup_method(self):
        """Initialize backend context for each test."""
        self.runtime = RuntimeContext(rank=0, world_size=1)

    def _create_test_pfunction(self, fn, *args, **kwargs) -> PFunction:
        """Helper to create a PFunction from a JAX function."""
        is_variable = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _, _ = jax_export(is_variable, fn, *args, **kwargs)
        return pfunc

    def _tensor_value(self, arr: np.ndarray | jnp.ndarray) -> TensorValue:
        """Helper to create TensorValue from array."""
        return TensorValue(np.array(arr))

    def test_simple_function_execution(self):
        """Test execution of a simple JAX function."""

        def add_fn(x, y):
            return x + y

        # Create test data
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])

        # Create PFunction
        pfunc = self._create_test_pfunction(add_fn, x, y)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x), self._tensor_value(y))

        # Verify result
        expected = np.array([5.0, 7.0, 9.0])
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_scalar_function_execution(self):
        """Test execution with scalar inputs and outputs."""

        def multiply_fn(x, y):
            return x * y

        # Create test data
        x = jnp.array(3.0)
        y = jnp.array(4.0)

        # Create PFunction
        pfunc = self._create_test_pfunction(multiply_fn, x, y)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x), self._tensor_value(y))

        # Verify result
        expected = np.array(12.0)
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_matrix_operations(self):
        """Test execution with matrix operations."""

        def matrix_mul_fn(x, y):
            return jnp.dot(x, y)

        # Create test matrices
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        # Create PFunction
        pfunc = self._create_test_pfunction(matrix_mul_fn, x, y)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x), self._tensor_value(y))

        # Verify result
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_multiple_outputs(self):
        """Test execution with multiple outputs."""

        def multi_output_fn(x, y):
            return x + y, x - y, x * y

        # Create test data
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        # Create PFunction
        pfunc = self._create_test_pfunction(multi_output_fn, x, y)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x), self._tensor_value(y))

        # Verify results
        assert isinstance(result, tuple)
        assert len(result) == 3

        # Check each output
        expected_sum = np.array([4.0, 6.0])
        expected_diff = np.array([-2.0, -2.0])
        expected_prod = np.array([3.0, 8.0])

        np.testing.assert_array_equal(result[0].to_numpy(), expected_sum)
        np.testing.assert_array_equal(result[1].to_numpy(), expected_diff)
        np.testing.assert_array_equal(result[2].to_numpy(), expected_prod)

    def test_complex_operations(self):
        """Test execution with complex JAX operations."""

        def complex_fn(x):
            return jnp.sin(x) * jnp.cos(x) + jnp.sum(x**2)

        # Create test data
        x = jnp.array([0.5, 1.0, 1.5])

        # Create PFunction
        pfunc = self._create_test_pfunction(complex_fn, x)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x))

        # Verify result
        expected = np.sin(x) * np.cos(x) + np.sum(x**2)
        assert isinstance(result, TensorValue)
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_different_dtypes(self):
        """Test execution with different data types."""

        def mixed_dtype_fn(x, y):
            return x.astype(jnp.float32) + y.astype(jnp.int32)

        # Create test data with different dtypes
        x = jnp.array([1.5, 2.5], dtype=jnp.float64)
        y = jnp.array([10, 20], dtype=jnp.int64)

        # Create PFunction
        pfunc = self._create_test_pfunction(mixed_dtype_fn, x, y)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x), self._tensor_value(y))

        # Verify result
        expected = np.array([11.5, 22.5], dtype=np.float32)
        assert isinstance(result, TensorValue)
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_high_dimensional_arrays(self):
        """Test execution with high-dimensional arrays."""

        def reduce_fn(x):
            return jnp.sum(x, axis=(1, 2))

        # Create 4D tensor
        x = jnp.ones((2, 3, 4, 5))

        # Create PFunction
        pfunc = self._create_test_pfunction(reduce_fn, x)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x))

        # Verify result shape and values
        assert isinstance(result, TensorValue)
        assert result.to_numpy().shape == (2, 5)
        expected = np.full((2, 5), 12.0)  # 3*4 = 12 for each sum
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_error_handling_wrong_fn_type(self):
        """Test error handling for wrong function type."""
        # Create PFunction with wrong type
        pfunc = PFunction(
            fn_type="wrong.type",
            ins_info=(),
            outs_info=(),
            fn_name="test",
            fn_text="dummy",
        )

        with pytest.raises(ValueError, match="jax exec kernel received wrong fn_type"):
            _jax_exec(pfunc)

    def test_error_handling_missing_fn_text(self):
        """Test error handling for missing function text."""
        pfunc = PFunction(
            fn_type="jax.exec", ins_info=(), outs_info=(), fn_name="test", fn_text=None
        )

        with pytest.raises(ValueError, match="jax exec kernel missing fn_text"):
            _jax_exec(pfunc)

    def test_error_handling_invalid_base64(self):
        """Test error handling for invalid base64 data."""
        pfunc = PFunction(
            fn_type="jax.exec",
            ins_info=(),
            outs_info=(),
            fn_name="test",
            fn_text="invalid_base64!",
        )

        with pytest.raises(ValueError, match="Failed to decode base64 export data"):
            _jax_exec(pfunc)

    def test_error_handling_invalid_serialized_data(self):
        """Test error handling for invalid serialized JAX data."""
        # Create valid base64 but invalid JAX data
        invalid_data = base64.b64encode(b"invalid_jax_data").decode()
        pfunc = PFunction(
            fn_type="jax.exec",
            ins_info=(),
            outs_info=(),
            fn_name="test",
            fn_text=invalid_data,
        )

        with pytest.raises(ValueError, match="Failed to deserialize JAX export"):
            _jax_exec(pfunc)

    def test_error_handling_invalid_argument_type(self):
        """Test error handling for invalid argument types."""

        def simple_fn(x):
            return x + 1

        x = jnp.array(1.0)
        pfunc = self._create_test_pfunction(simple_fn, x)

        # Pass invalid argument
        with pytest.raises(ValueError, match="Cannot convert argument 0 of type"):
            _jax_exec(pfunc, "invalid_argument")

    def test_argument_conversion_numpy_array(self):
        """Test that numpy arrays are properly converted."""

        def add_fn(x, y):
            return x + y

        # Create test data as numpy arrays
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        # Create PFunction
        pfunc = self._create_test_pfunction(add_fn, jnp.array(x), jnp.array(y))

        # Execute with numpy arrays
        result = _jax_exec(pfunc, self._tensor_value(x), self._tensor_value(y))

        # Verify result
        expected = np.array([4.0, 6.0])
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_argument_conversion_jax_array(self):
        """Test that JAX arrays are properly converted."""

        def add_fn(x, y):
            return x + y

        # Create test data as JAX arrays
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        # Create PFunction
        pfunc = self._create_test_pfunction(add_fn, x, y)

        # Execute with JAX arrays converted to TensorValue
        result = _jax_exec(pfunc, self._tensor_value(x), self._tensor_value(y))

        # Verify result
        expected = np.array([4.0, 6.0])
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_pfunction_properties(self):
        """Test that PFunction has correct properties after creation."""

        def add_fn(x, y):
            return x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        # Create PFunction
        pfunc = self._create_test_pfunction(add_fn, x, y)

        # Verify properties
        assert pfunc.fn_type == "jax.exec"
        assert pfunc.fn_name == "add_fn"
        assert len(pfunc.ins_info) == 2
        assert len(pfunc.outs_info) == 1
        assert pfunc.fn_text is not None
        assert len(pfunc.fn_text) > 0

        # Verify input tensor info
        assert pfunc.ins_info[0].shape == x.shape
        assert pfunc.ins_info[1].shape == y.shape

    @pytest.mark.parametrize("shape", [(1,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
    def test_various_shapes(self, shape):
        """Test execution with various tensor shapes."""

        def identity_fn(x):
            return x

        # Create test data with specified shape
        x = jnp.ones(shape)

        # Create PFunction
        pfunc = self._create_test_pfunction(identity_fn, x)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x))

        # Verify result
        assert isinstance(result, TensorValue)
        assert result.to_numpy().shape == shape
        np.testing.assert_array_equal(result.to_numpy(), np.ones(shape))

    def test_empty_arrays(self):
        """Test execution with empty arrays."""

        def empty_fn(x):
            return jnp.sum(x)

        # Create empty array
        x = jnp.array([])

        # Create PFunction
        pfunc = self._create_test_pfunction(empty_fn, x)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x))

        # Verify result (sum of empty array should be 0.0)
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), np.array(0.0))

    def test_large_arrays(self):
        """Test execution with large arrays."""

        def large_sum_fn(x):
            return jnp.sum(x)

        # Create large array
        x = jnp.ones(10000)

        # Create PFunction
        pfunc = self._create_test_pfunction(large_sum_fn, x)

        # Execute through kernel
        result = _jax_exec(pfunc, self._tensor_value(x))

        # Verify result
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), np.array(10000.0))


class TestJaxExportFullPipeline:
    """Test suite for complete frontend-to-backend pipeline."""

    def test_full_pipeline_simple_function(self):
        """Test complete pipeline with a simple function."""

        def add_fn(x, y):
            return x + y

        # Frontend: Generate PFunction using jax_export
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])

        is_variable = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _in_vars, _out_tree = jax_export(is_variable, add_fn, x, y)

        # Backend: Execute using _jax_exec
        result = _jax_exec(pfunc, TensorValue(np.array(x)), TensorValue(np.array(y)))

        # Verify result
        expected = np.array([5.0, 7.0, 9.0])
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_full_pipeline_complex_function(self):
        """Test complete pipeline with a complex function."""

        def complex_fn(x, y):
            return jnp.sin(x) * jnp.cos(x) + jnp.sum(x**2 + y**2)

        # Frontend: Generate PFunction
        x = jnp.array([0.5, 1.0, 1.5])
        y = jnp.array([0.3, 0.7, 1.1])

        is_variable = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _in_vars, _out_tree = jax_export(is_variable, complex_fn, x, y)

        # Backend: Execute
        result = _jax_exec(pfunc, TensorValue(np.array(x)), TensorValue(np.array(y)))

        # Verify result
        expected = np.sin(x) * np.cos(x) + np.sum(x**2 + y**2)
        assert isinstance(result, TensorValue)
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_full_pipeline_multiple_outputs(self):
        """Test complete pipeline with multiple outputs."""

        def multi_output_fn(x, y):
            return x + y, x - y, x * y

        # Frontend: Generate PFunction
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        is_variable = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _in_vars, _out_tree = jax_export(is_variable, multi_output_fn, x, y)

        # Backend: Execute
        result = _jax_exec(pfunc, TensorValue(np.array(x)), TensorValue(np.array(y)))

        # Verify multiple outputs
        assert isinstance(result, tuple)
        assert len(result) == 3

        expected_sum = np.array([4.0, 6.0])
        expected_diff = np.array([-2.0, -2.0])
        expected_prod = np.array([3.0, 8.0])

        np.testing.assert_array_equal(result[0].to_numpy(), expected_sum)
        np.testing.assert_array_equal(result[1].to_numpy(), expected_diff)
        np.testing.assert_array_equal(result[2].to_numpy(), expected_prod)

    def test_full_pipeline_with_constants(self):
        """Test complete pipeline with constants captured during compilation."""

        def scale_and_shift(x, scale=2.0, shift=1.0):
            return x * scale + shift

        # Frontend: Generate PFunction with constants
        x = jnp.array([1.0, 2.0, 3.0])

        is_variable = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _in_vars, _out_tree = jax_export(
            is_variable, scale_and_shift, x, scale=5.0, shift=10.0
        )

        # Backend: Execute (only need to provide variable arguments)
        result = _jax_exec(pfunc, TensorValue(np.array(x)))

        # Verify result with constants applied
        expected = x * 5.0 + 10.0
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_full_pipeline_high_dimensional(self):
        """Test complete pipeline with high-dimensional arrays."""

        def tensor_operation(x):
            # Reduce along specific axes and compute statistics
            mean_val = jnp.mean(x, axis=(1, 2))
            std_val = jnp.std(x, axis=(1, 2))
            return mean_val, std_val

        # Frontend: Generate PFunction
        x = jnp.ones((2, 3, 4, 5))

        is_variable = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _in_vars, _out_tree = jax_export(is_variable, tensor_operation, x)

        # Backend: Execute
        result = _jax_exec(pfunc, TensorValue(np.array(x)))

        # Verify results
        assert isinstance(result, tuple)
        assert len(result) == 2

        # For ones array, mean should be 1.0, std should be 0.0
        expected_mean = np.ones((2, 5))
        expected_std = np.zeros((2, 5))

        np.testing.assert_allclose(result[0].to_numpy(), expected_mean, atol=1e-6)
        np.testing.assert_allclose(result[1].to_numpy(), expected_std, atol=1e-6)

    def test_full_pipeline_serialization_roundtrip(self):
        """Test that PFunction properties survive basic attribute access."""

        def simple_fn(x):
            return jnp.sum(x)

        # Frontend: Generate PFunction
        x = jnp.array([1.0, 2.0, 3.0])

        is_variable = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _, _ = jax_export(is_variable, simple_fn, x)

        # Test that key properties are accessible (partial serialization test)
        fn_type = pfunc.fn_type
        fn_name = pfunc.fn_name
        fn_text = pfunc.fn_text
        ins_info = pfunc.ins_info
        outs_info = pfunc.outs_info

        # Verify properties are preserved
        assert fn_type == "jax.exec"
        assert fn_name == "simple_fn"
        assert fn_text is not None
        assert len(ins_info) == 1
        assert len(outs_info) == 1

        # Backend: Execute with original PFunction
        result = _jax_exec(pfunc, TensorValue(np.array(x)))

        # Verify result
        expected = np.array(6.0)
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_full_pipeline_robustness(self):
        """Test that the pipeline handles edge cases gracefully."""

        def edge_case_fn(x):
            # Test with operations that might be problematic
            return jnp.sqrt(jnp.abs(x))  # sqrt of absolute value (always safe)

        # Frontend: Generate PFunction
        x = jnp.array([-1.0, 0.0, 1.0])

        is_variable = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _, _ = jax_export(is_variable, edge_case_fn, x)

        # Backend: Execute should work fine
        result = _jax_exec(pfunc, TensorValue(np.array(x)))

        # Verify result - sqrt of absolute values
        expected = np.sqrt(np.abs([-1.0, 0.0, 1.0]))
        assert isinstance(result, TensorValue)
        np.testing.assert_allclose(result.to_numpy(), expected)
