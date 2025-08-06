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
from jax.tree_util import tree_flatten, tree_unflatten

# Enable 64-bit precision in JAX for testing different dtypes
jax.config.update("jax_enable_x64", True)

from mplang.core.base import TensorInfo
from mplang.core.pfunc import PFunction
from mplang.plib import jax2stablehlo
from mplang.plib.stablehlo_handler import StablehloHandler


class TestStablehloHandler:
    """Test suite for StableHLO MLIR execution runtime functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Initialize and cleanup StableHLO runtime for each test case."""
        self.runtime = StablehloHandler()
        self.runtime.setup()
        yield
        self.runtime.teardown()

    @pytest.mark.parametrize(
        "test_function, inputs",
        [
            # simple_function
            (
                lambda x, y: x + y * 2,
                (jnp.array([1.0, 2.0, 3.0]), jnp.array([0.5, 1.5, 2.5])),
            ),
            # multi_output_function
            (
                lambda x, y: (x + y, x * y),
                (jnp.array([2.0, 3.0]), jnp.array([1.0, 4.0])),
            ),
            # complex_function_with_math_ops
            (
                lambda x, y, z: (
                    (
                        c := jnp.matmul(
                            (x + y).reshape(-1, 1), jnp.sin(z).reshape(1, -1)
                        )
                    ),
                    jnp.sum(c),
                ),
                (
                    jnp.array([1.0, 2.0]),
                    jnp.array([0.5, 1.5]),
                    jnp.array([0.1, 0.2]),
                ),
            ),
            # compilation_preserves_function_signature
            (
                lambda a, b, c: a * b + c,
                (
                    jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                    jnp.array([[0.5, 1.5], [2.5, 3.5]]),
                    jnp.array([[0.1, 0.2], [0.3, 0.4]]),
                ),
            ),
            # empty_input_function
            (lambda: jnp.array([1.0, 2.0, 3.0]), ()),
            # scalar_function
            (
                lambda x, y: jnp.sum(x * y),
                (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])),
            ),
            # matrix_operations
            (
                lambda a, b: jnp.transpose(jnp.matmul(a, b)) + 1.0,
                (
                    jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                    jnp.array([[5.0, 6.0], [7.0, 8.0]]),
                ),
            ),
            # different_dtypes
            (
                lambda x, y: x + y,
                (
                    jnp.array([1.0, 2.0], dtype=jnp.float32),
                    jnp.array([3.0, 4.0], dtype=jnp.float64),
                ),
            ),
        ],
    )
    def test_function_execution(self, test_function, inputs):
        """Verify end-to-end compilation and execution pipeline for diverse function types."""
        # Establish ground truth via local JAX execution
        expected = test_function(*inputs)

        # Generate tensor metadata for runtime input validation
        inputs_info = [TensorInfo(shape=x.shape, dtype=x.dtype) for x in inputs]

        # Compile function to portable StableHLO MLIR representation
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, out_tree = jax2stablehlo.compile(is_var, test_function, *inputs)

        # Execute compiled function on StableHLO runtime
        result_flat = self.runtime.execute(cfunc, inputs)

        # Reconstruct nested output structure from flattened results
        result = tree_unflatten(out_tree, result_flat)

        # Validate numerical correctness within tolerance bounds
        expected_flat, _ = tree_flatten(expected)
        result_flat_from_tree, _ = tree_flatten(result)
        assert len(result_flat_from_tree) == len(expected_flat)
        for res, exp in zip(result_flat_from_tree, expected_flat):
            assert jnp.allclose(jnp.asarray(res), jnp.asarray(exp), rtol=1e-5)

    def test_invalid_format_execution(self):
        """Verify runtime error handling for unsupported PFunction format types."""
        # Construct malformed PFunction with unsupported format identifier
        invalid_pfunc = PFunction(
            fn_name="test",
            fn_type="invalid_format",
            fn_text="invalid_text",
            fn_body=None,
            ins_info=(),
            outs_info=(),
            attrs={},
        )

        # Assert runtime rejects invalid format with descriptive error
        with pytest.raises(ValueError, match="Unsupported format"):
            self.runtime.execute(invalid_pfunc, [])
