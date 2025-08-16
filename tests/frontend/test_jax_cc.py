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

import jax
import jax.numpy as jnp
import pytest

from mplang.frontend import jax_cc

# Enable 64-bit precision in JAX for testing different dtypes
jax.config.update("jax_enable_x64", True)


class TestJax2StableHLO:
    """Test suite for JAX to StableHLO MLIR compilation functionality."""

    def _compile_with_transformer(self, fn, *args, **kwargs):
        """Compile function using JAX-to-StableHLO transformation pipeline.

        Args:
            fn: Function to compile
            *args, **kwargs: Function arguments for compilation context

        Returns:
            tuple[PFunction, PyTreeDef]: Compiled function and output structure
        """
        # Predicate: treat tensor-like objects as variables, others as constants
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _, out_tree = jax_cc.jax2stablehlo(is_var, fn, *args, **kwargs)
        return pfunc, out_tree

    @pytest.mark.parametrize(
        "test_function, inputs, kwargs, expected_ins_count, expected_outs_count, test_content",
        [
            # simple_function_compilation
            (
                lambda x, y: x + y,
                (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])),
                {},
                2,
                1,
                None,
            ),
            # function_with_kwargs
            (
                lambda x, y, scale=2.0: x * scale + y,
                (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
                {"scale": 3.0},
                2,
                1,
                None,
            ),
            # complex_function_compilation
            (
                lambda x, y: jnp.sum(x + y * jnp.sin(x + y)),
                (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])),
                {},
                2,
                1,
                ["add", "sine"],
            ),
            # mlir_text_generation
            (
                lambda x, y: x + y,
                (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
                {},
                2,
                1,
                ["func.func"],
            ),
            # tensor_info_accuracy
            (
                lambda x: jnp.sum(jnp.reshape(x, (-1,))),
                (jnp.array([[1.0, 2.0], [3.0, 4.0]]),),
                {},
                1,
                1,
                None,
            ),
            # function_with_no_variables
            (
                lambda: jnp.array([1.0, 2.0, 3.0]),
                (),
                {},
                0,
                1,
                None,
            ),
            # different_dtypes
            (
                lambda x, y: x.astype(jnp.float32) + y.astype(jnp.int32),
                (
                    jnp.array([1.0, 2.0], dtype=jnp.float64),
                    jnp.array([3, 4], dtype=jnp.int64),
                ),
                {},
                2,
                1,
                None,
            ),
            # nested_function_calls
            (
                lambda x, y: jnp.square(x) + jnp.square(y),
                (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
                {},
                2,
                1,
                None,
            ),
        ],
    )
    def test_function_compilation(
        self,
        test_function,
        inputs,
        kwargs,
        expected_ins_count,
        expected_outs_count,
        test_content,
    ):
        """Test compilation of various functions."""
        cfunc, _out_tree = self._compile_with_transformer(
            test_function, *inputs, **kwargs
        )

        # Check basic properties
        assert cfunc.fn_type == "mlir.stablehlo"
        assert cfunc.fn_name is not None
        assert len(cfunc.ins_info) == expected_ins_count
        assert len(cfunc.outs_info) == expected_outs_count

        # Check input tensor info
        for i, input_tensor in enumerate(inputs):
            assert cfunc.ins_info[i].shape == input_tensor.shape
            assert cfunc.ins_info[i].dtype.name == str(input_tensor.dtype)

        # Check that MLIR text is generated
        assert cfunc.fn_text is not None
        assert len(cfunc.fn_text) > 0
        assert isinstance(cfunc.fn_text, str)

        # Check specific content if provided
        if test_content:
            text_str = str(cfunc.fn_text)
            for content in test_content:
                assert content in text_str or any(alt in text_str for alt in [content])

    def test_error_handling_invalid_function(self):
        """Test error handling for invalid functions."""

        def invalid_func(x):
            # This should cause an error during compilation
            return x + "invalid"

        x = jnp.array([1.0, 2.0])

        with pytest.raises((ValueError, TypeError)):
            self._compile_with_transformer(invalid_func, x)

    def test_large_arrays(self):
        """Test compilation with larger arrays."""

        def matrix_mul(x, y):
            return jnp.dot(x, y)

        x = jnp.ones((100, 50))
        y = jnp.ones((50, 75))

        cfunc, _out_tree = self._compile_with_transformer(matrix_mul, x, y)

        assert cfunc.ins_info[0].shape == x.shape
        assert cfunc.ins_info[1].shape == y.shape
        assert cfunc.outs_info[0].shape == (100, 75)

        # Check that compilation succeeded
        assert cfunc.fn_text is not None

    def test_partial_function_compilation(self):
        """Test compilation of partial functions."""

        def multiply_add(x, y, multiplier, addend):
            return x * multiplier + y + addend

        # Create a partial function and test it
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        cfunc, _out_tree = self._compile_with_transformer(
            multiply_add, x, y, multiplier=2.0, addend=5.0
        )

        # The function name should be extracted correctly
        assert "multiply_add" in cfunc.fn_name or cfunc.fn_name == "multiply_add"
        assert len(cfunc.ins_info) == 2

        # Check that compilation succeeded
        assert cfunc.fn_text is not None

    def test_compilation_deterministic(self):
        """Test that compilation is deterministic."""

        def simple_func(x):
            return x * 2 + 1

        x = jnp.array([1.0, 2.0, 3.0])

        # Compile the same function twice
        cfunc1, _out_tree1 = self._compile_with_transformer(simple_func, x)
        cfunc2, _out_tree2 = self._compile_with_transformer(simple_func, x)

        # Should produce identical results
        assert cfunc1.fn_type == cfunc2.fn_type
        assert cfunc1.fn_name == cfunc2.fn_name
        assert cfunc1.ins_info == cfunc2.ins_info
        assert cfunc1.outs_info == cfunc2.outs_info
        # Note: MLIR text might differ due to internal naming, so we don't check that

    def test_high_dimensional_tensors(self):
        """Test compilation with high-dimensional tensors."""

        def tensor_ops(x):
            return jnp.sum(x, axis=(1, 3))

        # 4D tensor
        x = jnp.ones((2, 3, 4, 5))

        cfunc, _out_tree = self._compile_with_transformer(tensor_ops, x)

        assert cfunc.ins_info[0].shape == x.shape
        assert cfunc.outs_info[0].shape == (2, 4)
        assert cfunc.fn_text is not None

    def test_multiple_outputs(self):
        """Test compilation with multiple outputs."""

        def multi_output(x, y):
            return x + y, x - y, x * y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        cfunc, _out_tree = self._compile_with_transformer(multi_output, x, y)

        assert len(cfunc.ins_info) == 2
        assert len(cfunc.outs_info) == 3

        # All outputs should have the same shape as inputs
        for out_info in cfunc.outs_info:
            assert out_info.shape == x.shape

        assert cfunc.fn_text is not None
