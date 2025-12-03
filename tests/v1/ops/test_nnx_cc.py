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
from flax import nnx

from mplang.v1.ops import jax_cc, nnx_cc

# Enable 64-bit precision in JAX for testing different dtypes
jax.config.update("jax_enable_x64", True)


class TestNnx2StableHLO:
    """Test suite for NNX to StableHLO MLIR compilation functionality."""

    def _compile_with_transformer(self, fn, *args, **kwargs):
        """Compile function using NNX-to-StableHLO transformation pipeline.

        Args:
            fn: Function to compile
            *args, **kwargs: Function arguments for compilation context

        Returns:
            tuple[PFunction, PyTreeDef]: Compiled function and output structure
        """
        # Predicate: treat tensor-like objects as variables, others as constants
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        pfunc, _, out_tree = nnx_cc.nnx2stablehlo(is_var, fn, *args, **kwargs)
        return pfunc, out_tree

    @pytest.mark.parametrize(
        "test_function, inputs, kwargs, expected_ins_count, expected_outs_count, test_content",
        [
            # simple_function_compilation
            (
                lambda x, y: x + y,
                (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
                {},
                2,  # Expected number of inputs
                1,  # Expected number of outputs
                "simple addition function compilation",
            ),
            # array_manipulation
            (
                lambda arr: jnp.sum(arr, axis=0),
                (jnp.array([[1.0, 2.0], [3.0, 4.0]]),),
                {},
                1,
                1,
                "array sum with axis specification",
            ),
            # nnx_function_with_constants
            (
                lambda x, scale: x * scale,
                (jnp.array([1.0, 2.0, 3.0]), 2.5),
                {},
                1,  # Only x is a variable, scale is a constant
                1,
                "function with constant parameter optimization",
            ),
        ],
    )
    def test_basic_compilation_cases(
        self,
        test_function,
        inputs,
        kwargs,
        expected_ins_count,
        expected_outs_count,
        test_content,
    ):
        """Test basic NNX function compilation scenarios."""
        pfunc, _ = self._compile_with_transformer(test_function, *inputs, **kwargs)

        # Validate PFunction properties
        assert pfunc.fn_type == "mlir.stablehlo"
        assert len(pfunc.ins_info) == expected_ins_count
        assert len(pfunc.outs_info) == expected_outs_count
        assert isinstance(pfunc.fn_text, str)
        assert len(pfunc.fn_text) > 0

        # Validate MLIR content contains expected patterns
        mlir_text = pfunc.fn_text
        assert "func.func" in mlir_text  # MLIR function declaration
        assert "return" in mlir_text  # MLIR return statement

    def test_nnx_activation_function(self):
        """Test compilation of NNX activation functions."""

        def relu_activation(x):
            return nnx.relu(x)

        input_tensor = jnp.array([-1.0, 0.0, 1.0, 2.0])
        pfunc, _ = self._compile_with_transformer(relu_activation, input_tensor)

        # Verify compilation success
        assert pfunc.fn_type == "mlir.stablehlo"
        assert len(pfunc.ins_info) == 1
        assert len(pfunc.outs_info) == 1

    def test_nnx_simple_mlp_compilation(self):
        """Test compilation of a simple MLP model using NNX Linear layers."""

        class SimpleMLP(nnx.Module):
            """Simple Multi-Layer Perceptron using NNX Linear layers."""

            def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
                self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
                self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

            def __call__(self, x: jax.Array) -> jax.Array:
                # First layer with activation
                x = nnx.relu(self.linear1(x))
                # Second layer
                x = self.linear2(x)
                return x

        # Create model instance
        model = SimpleMLP(2, 16, 5, rngs=nnx.Rngs(0))

        # Test input
        x_input = jnp.ones((3, 2))

        # Define a function that uses the model
        def model_forward(x):
            return model(x)

        # Test compilation
        pfunc, _ = self._compile_with_transformer(model_forward, x_input)

        # Validate compilation properties
        assert pfunc.fn_type == "mlir.stablehlo"
        assert len(pfunc.ins_info) == 1  # Only x_input is variable
        assert len(pfunc.outs_info) == 1
        assert "func.func" in pfunc.fn_text

        # Verify the MLIR contains neural network operations
        mlir_text = pfunc.fn_text.lower()
        # Should contain matrix operations from linear layers
        assert any(op in mlir_text for op in ["dot", "add", "mul"])

    def test_nnx_stateful_model_with_functional_api(self):
        """Test compilation of stateful NNX model using Functional API."""

        class Count(nnx.Variable):
            """Custom counter variable."""

        class StatefulLinear(nnx.Module):
            """Linear layer with state counter."""

            def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
                self.w = nnx.Param(
                    nnx.initializers.lecun_normal()(rngs.params(), (din, dout))
                )
                self.b = nnx.Param(jnp.zeros((dout,)))
                self.count = Count(jnp.array(0, dtype=jnp.int32))

            def __call__(self, x: jax.Array):
                # Note: In a real stateful scenario, we'd increment count
                # but for compilation testing, we keep it functional
                return x @ self.w + self.b

        # Create stateful model
        model = StatefulLinear(din=4, dout=6, rngs=nnx.Rngs(0))

        # Test input
        x_input = jnp.ones((2, 4))

        # Use Functional API to handle stateful model
        graphdef, state = nnx.split(model)

        def functional_forward(graphdef, state, x):
            """Forward function using functional API."""
            # Merge model inside the function
            model = nnx.merge(graphdef, state)
            # Call the model
            y = model(x)
            # Split again to get updated state (if any)
            _, new_state = nnx.split(model)
            return y, new_state

        # Define compilation function
        def compile_forward(x):
            # For compilation, we treat graphdef and state as constants
            y, _ = functional_forward(graphdef, state, x)
            return y

        # Test compilation
        pfunc, _ = self._compile_with_transformer(compile_forward, x_input)

        # Validate compilation
        assert pfunc.fn_type == "mlir.stablehlo"
        assert len(pfunc.ins_info) == 1  # Only x is variable
        assert len(pfunc.outs_info) == 1
        assert "func.func" in pfunc.fn_text

    def test_nnx_vs_jax_comparison(self):
        """Compare NNX and JAX compilation for the same function."""

        def simple_function(x):
            return x * 2 + 1

        x_data = jnp.array([1.0, 2.0, 3.0])

        def is_variable(arg):
            return hasattr(arg, "dtype") and hasattr(arg, "shape")

        # Test JAX compilation
        jax_pfunc, _, _ = jax_cc.jax2stablehlo(is_variable, simple_function, x_data)

        # Test NNX compilation
        nnx_pfunc, _, _ = nnx_cc.nnx2stablehlo(is_variable, simple_function, x_data)

        # Both should produce similar results for simple functions
        assert jax_pfunc.fn_type == nnx_pfunc.fn_type == "mlir.stablehlo"
        assert len(jax_pfunc.ins_info) == len(nnx_pfunc.ins_info)
        assert len(jax_pfunc.outs_info) == len(nnx_pfunc.outs_info)


class TestNnxRunner:
    """Test suite for the NnxRunner operation class."""

    def test_run_nnx_module_creation(self):
        """Test that run_nnx module is properly created."""
        assert hasattr(nnx_cc, "run_nnx")
        assert isinstance(nnx_cc.run_nnx, nnx_cc.NnxRunner)
        # Check the stateless module was created correctly
        assert nnx_cc.run_nnx.name == "run"

    def test_nnx_runner_trace_method(self):
        """Test NnxRunner.trace method with mock inputs."""

        def simple_nnx_fn(x, y):
            return nnx.relu(x + y)

        runner = nnx_cc.NnxRunner(None, "test")

        # Test with regular JAX arrays - these won't be treated as variables by the runner
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        # The runner uses is_variable that only returns True for MPObject instances
        # So with regular JAX arrays, we should get 0 variables (all constants)
        pfunc, in_vars, _ = runner.trace(simple_nnx_fn, x, y)

        # Validate basic properties
        assert pfunc.fn_type == "mlir.stablehlo"
        # Regular JAX arrays are treated as constants by the runner
        assert len(in_vars) == 0  # No MPObjects means no variables
        assert len(pfunc.ins_info) == 0  # All inputs were constants
