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
import numpy as np
import pytest
from jax.tree_util import tree_flatten, tree_unflatten

from mplang.v1.core.expr.evaluator import create_evaluator
from mplang.v1.core.pfunc import PFunction
from mplang.v1.core.tensor import TensorType
from mplang.v1.kernels import stablehlo  # noqa: F401
from mplang.v1.kernels.context import RuntimeContext
from mplang.v1.kernels.value import TensorValue
from mplang.v1.ops import jax_cc

# Enable 64-bit precision in JAX for testing different dtypes
jax.config.update("jax_enable_x64", True)


class TestStablehloKernel:
    """Tests for mlir.stablehlo flat backend kernel."""

    @pytest.fixture(autouse=True)
    def _evaluator(self):
        # Minimal single-rank communicator (no actual messaging needed for local kernel)
        from mplang.v1.core.comm import CommunicatorBase

        class _SingleComm(CommunicatorBase):  # type: ignore[misc]
            def send(self, to: int, key: str, data):  # pragma: no cover - unused
                raise RuntimeError("send should not be called in single-rank test")

        comm = _SingleComm(rank=0, world_size=1)
        runtime = RuntimeContext(rank=0, world_size=1)
        ev = create_evaluator(rank=0, env={}, comm=comm, runtime=runtime)
        self.ev = ev
        yield

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
        [TensorType(shape=x.shape, dtype=x.dtype) for x in inputs]

        # Compile function to portable StableHLO MLIR representation
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, out_tree = jax_cc.jax2stablehlo(is_var, test_function, *inputs)

        # Execute via evaluator (dispatches kernel by fn_type)
        tensor_inputs = [TensorValue(np.asarray(x)) for x in inputs]
        result_flat = self.ev._exec_pfunc(cfunc, tensor_inputs)  # type: ignore[attr-defined]

        assert all(isinstance(val, TensorValue) for val in result_flat)
        result_arrays = [val.to_numpy() for val in result_flat]

        # Reconstruct nested output structure from flattened results
        result = tree_unflatten(out_tree, result_arrays)

        # Validate numerical correctness within tolerance bounds
        expected_flat, _ = tree_flatten(expected)
        result_flat_from_tree, _ = tree_flatten(result)
        assert len(result_flat_from_tree) == len(expected_flat)
        for res, exp in zip(result_flat_from_tree, expected_flat, strict=False):
            assert jnp.allclose(jnp.asarray(res), jnp.asarray(exp), rtol=1e-5)

    def test_invalid_format_execution(self):
        invalid_pfunc = PFunction(
            fn_type="invalid_format",
            ins_info=(),
            outs_info=(),
            fn_name="test",
            fn_text="invalid_text",
        )
        from mplang.v1.core.comm import CommunicatorBase

        class _SingleComm(CommunicatorBase):  # type: ignore[misc]
            def send(self, to: int, key: str, data):  # pragma: no cover - unused
                raise RuntimeError("send should not be called in single-rank test")

        comm = _SingleComm(rank=0, world_size=1)
        runtime = RuntimeContext(rank=0, world_size=1)
        ev = create_evaluator(0, {}, comm, runtime)
        with pytest.raises(NotImplementedError):
            ev._exec_pfunc(invalid_pfunc, [])  # type: ignore[attr-defined]
