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
Tests for simulation module.

This test suite tests the Simulator class by:
1. Creating expressions using trace/primitive operations
2. Creating a Simulator instance
3. Evaluating expressions and checking results
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mplang.v1 as mp
from mplang.v1.core import (
    FLOAT32,
    INT32,
    ClusterSpec,
    Mask,
    MPObject,
    MPType,
    Rank,
    TraceContext,
    TraceVar,
    pshfl_s,
    trace,
    uniform_cond,
    while_loop,
    with_ctx,
)
from mplang.v1.runtime.simulation import Simulator, SimVar
from mplang.v1.simp.api import constant, prank, set_mask

# Enable JAX x64 mode to match type expectations
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def mask_2p():
    """Mask for 2-party computation."""
    return Mask(3)  # 0b11


@pytest.fixture
def trace_context(mask_2p):
    """Create a trace context for testing."""
    cluster_spec = ClusterSpec.simple(world_size=2)
    return TraceContext(cluster_spec=cluster_spec, mask=mask_2p)


@pytest.fixture
def simulator():
    """Create a simulator for testing."""
    return Simulator.simple(world_size=2)


class TestSimVar:
    """Test SimVar class."""

    def test_simvar_creation(self, simulator):
        """Test SimVar creation and properties."""
        # Create a simple MPType
        mptype = MPType.tensor(dtype=INT32, shape=(2, 3), pmask=Mask(3))

        # Create values for both parties
        values = [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32),
        ]

        simvar = SimVar(simulator, mptype, values)

        assert simvar.ctx is simulator
        assert simvar.mptype is mptype
        assert len(simvar.values) == 2
        np.testing.assert_array_equal(simvar.values[0], values[0])
        np.testing.assert_array_equal(simvar.values[1], values[1])

    def test_simvar_repr(self, simulator):
        """Test SimVar string representation."""
        mptype = MPType.tensor(FLOAT32, (2,), Mask(3))
        values = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        simvar = SimVar(simulator, mptype, values)
        repr_str = repr(simvar)

        assert "SimVar" in repr_str
        assert str(mptype) in repr_str


class TestSimulator:
    """Test Simulator class."""

    def test_simulator_creation(self):
        """Test Simulator creation."""
        sim = Simulator.simple(world_size=3)
        assert sim.world_size() == 3
        assert len(sim._comms) == 3
        # persistent runtimes after refactor
        assert len(sim._runtimes) == 3

        # Check that communicators are properly connected
        for i, comm in enumerate(sim._comms):
            assert comm.rank == i
            assert comm.world_size == 3
            assert len(comm.peers) == 3

    def test_evaluate_constant(self, simulator, trace_context):
        """Test evaluating a constant expression."""

        # Create a traced function that returns a constant
        def const_func():
            return constant(42)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, const_func)

        # Get the function definition expression and extract the body
        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body  # Get the function body for evaluation

        # Evaluate with empty bindings
        results = simulator.evaluate(expr, {})

        # Check results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert results[0].ctx is simulator

        # Both parties should have the same constant value
        assert len(results[0].values) == 2
        assert results[0].values[0] == 42  # Party 0 value
        assert results[0].values[1] == 42  # Party 1 value

    def test_evaluate_prank(self, simulator, trace_context):
        """Test evaluating a prank expression."""
        # Create a traced function that returns rank
        with with_ctx(trace_context):
            traced_fn = trace(trace_context, prank)

        # Get the function definition expression and extract the body
        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body  # Get the function body for evaluation

        # Evaluate with empty bindings
        results = simulator.evaluate(expr, {})

        # Check results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert results[0].ctx is simulator

        # Each party should have its own rank
        assert len(results[0].values) == 2
        assert results[0].values[0] == 0  # Party 0's rank
        assert results[0].values[1] == 1  # Party 1's rank

    def test_evaluate_with_bindings(self, simulator, trace_context):
        """Test evaluating an expression with variable bindings."""

        # Create a traced function that uses a variable
        def func_with_var(x):
            return x

        # Create input variable
        mptype = MPType.tensor(INT32, (2,), Mask(3))
        input_values = [
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
        ]
        input_var = SimVar(simulator, mptype, input_values)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, func_with_var, input_var)

        # Get the function definition expression and extract the body
        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body  # Get the function body for evaluation

        # Evaluate with bindings - need to use the traced parameter name
        param_names = traced_fn.in_names()
        bindings = {param_names[0]: input_var} if param_names else {}
        results = simulator.evaluate(expr, bindings)

        # Check results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert results[0].ctx is simulator

        # Should return some valid values (the exact format depends on evaluator implementation)
        assert len(results[0].values) == 2
        assert results[0].values[0] is not None
        assert results[0].values[1] is not None
        # Verify that the values are reasonable arrays from our input
        assert isinstance(results[0].values[0], np.ndarray)
        assert isinstance(results[0].values[1], np.ndarray)
        np.testing.assert_array_equal(results[0].values[0], [1, 2])
        np.testing.assert_array_equal(results[0].values[1], [3, 4])

    def test_evaluate_multi_output(self, simulator, trace_context):
        """Test evaluating an expression with multiple outputs."""

        # Create a function that returns multiple values
        def multi_output_func():
            x = constant(10)
            y = constant(20)
            return x, y

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, multi_output_func)

        # Get the function definition expression and extract the body
        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body  # Get the function body for evaluation

        # Evaluate with empty bindings
        results = simulator.evaluate(expr, {})

        # Check results
        assert len(results) == 2
        assert all(isinstance(r, SimVar) for r in results)
        assert all(r.ctx is simulator for r in results)

        # Check first output (10)
        assert len(results[0].values) == 2
        assert results[0].values[0] == 10  # First output value
        assert results[0].values[1] == 10  # First output value

        # Check second output (20)
        assert len(results[1].values) == 2
        assert results[1].values[0] == 20  # Second output value
        assert results[1].values[1] == 20  # Second output value

    def test_evaluate_wrong_context(self, trace_context):
        """Test that evaluation fails with variables from wrong context."""
        sim1 = Simulator.simple(world_size=2)
        sim2 = Simulator.simple(world_size=2)

        # Create a variable in sim1
        mptype = MPType.tensor(INT32, (1,), Mask(3))
        var_sim1 = SimVar(sim1, mptype, [np.array([1]), np.array([2])])

        # Create a simple expression
        def const_func():
            return constant(42)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, const_func)

        expr = traced_fn.make_expr()
        assert expr is not None

        # Try to evaluate in sim2 with variable from sim1
        bindings: dict[str, MPObject] = {"x": var_sim1}

        with pytest.raises(ValueError, match=r"Variable .* not in this context"):
            sim2.evaluate(expr, bindings)

    def test_empty_bindings(self, simulator, trace_context):
        """Test evaluation with empty bindings."""

        def simple_func():
            return constant(123)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, simple_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body  # Get the function body for evaluation
        results = simulator.evaluate(expr, {})

        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert results[0].values[0] == 123  # Constant value
        assert results[0].values[1] == 123  # Constant value


class TestSimulatorIntegration:
    """Integration tests for Simulator."""

    def test_chain_evaluations(self, simulator, trace_context):
        """Test chaining multiple evaluations."""

        # First evaluation: create a constant
        def const_func():
            return constant(5)

        with with_ctx(trace_context):
            traced_fn1 = trace(trace_context, const_func)

        func_expr1 = traced_fn1.make_expr()
        assert func_expr1 is not None
        expr1 = func_expr1.body  # Get the function body for evaluation
        results1 = simulator.evaluate(expr1, {})

        # Second evaluation: use the result from first evaluation
        def identity_func(x):
            return x

        with with_ctx(trace_context):
            traced_fn2 = trace(trace_context, identity_func, results1[0])

        func_expr2 = traced_fn2.make_expr()
        assert func_expr2 is not None
        expr2 = func_expr2.body  # Get the function body for evaluation
        param_names2 = traced_fn2.in_names()
        bindings2 = {param_names2[0]: results1[0]} if param_names2 else {}
        results2 = simulator.evaluate(expr2, bindings2)

        # Check that the value is preserved through the chain
        assert len(results2) == 1
        assert results2[0].values[0] == 5  # Constant value
        assert results2[0].values[1] == 5  # Constant value

    def test_multiple_simulator_independence(self, trace_context):
        """Test that multiple simulators are independent."""
        sim1 = Simulator.simple(world_size=2)
        sim2 = Simulator.simple(world_size=2)

        def const_func():
            return constant(100)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, const_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body  # Get the function body for evaluation

        # Evaluate in both simulators
        results1 = sim1.evaluate(expr, {})
        results2 = sim2.evaluate(expr, {})

        # Results should be the same but from different contexts
        assert results1[0].ctx is sim1
        assert results2[0].ctx is sim2
        # Need to cast to SimVar to access values
        simvar1 = results1[0]
        simvar2 = results2[0]
        assert isinstance(simvar1, SimVar)
        assert isinstance(simvar2, SimVar)
        assert simvar1.values == simvar2.values


class TestComplexCond:
    """Test complex conditional expressions with JAX functional branches.

    This test suite focuses on "complex" conditional functions where:
    1. Branches are real functional JAX computations (not simple constants)
    2. Parameter passing and variable capture scenarios are thoroughly tested
    3. Different parties may execute different branches based on rank predicates

    Test Categories:
    - Basic JAX operations in branches
    - Parameter passing between functions
    - Variable capture from outer scope
    - Advanced computational scenarios (neural networks, optimizers)
    """

    # ===== Helper JAX Functions =====
    # These helper functions make tests more readable by extracting
    # common JAX computation patterns

    @staticmethod
    def simple_arithmetic_ops():
        """Simple JAX arithmetic operations for basic testing."""

        def square_plus_one(x):
            return jnp.square(x) + 1.0

        def multiply_and_add(x):
            return x * 2.0 + 5.0

        return square_plus_one, multiply_and_add

    @staticmethod
    def matrix_operations():
        """JAX matrix operations for more complex scenarios."""

        def matrix_vector_op(x, matrix):
            return jnp.dot(matrix, x) + jnp.sum(matrix)

        def element_wise_with_stats(x, vector):
            return x * vector + jnp.mean(vector)

        return matrix_vector_op, element_wise_with_stats

    @staticmethod
    def neural_network_ops():
        """Simple neural network-like operations."""

        def forward_pass(x, weights, bias):
            hidden = jnp.dot(weights.T, x) + bias
            activated = jnp.tanh(hidden)
            return jnp.array([jnp.sum(activated), jnp.sum(activated)])

        def statistical_transform(x, weights):
            mean_x = jnp.mean(x)
            std_x = jnp.std(x) + 1e-8
            normalized = (x - mean_x) / std_x
            return jnp.dot(weights, normalized.reshape(-1, 1)).flatten()

        return forward_pass, statistical_transform

    @staticmethod
    def optimizer_ops():
        """Optimizer-like operations for advanced scenarios."""

        def adam_step(grad, lr, momentum, eps, state):
            mean_grad = state[0] * momentum + (1 - momentum) * jnp.mean(grad)
            var_grad = state[1] * momentum + (1 - momentum) * jnp.var(grad)
            step = state[2] + 1
            corrected_mean = mean_grad / (1 - momentum**step)
            corrected_var = var_grad / (1 - momentum**step)
            update = lr * corrected_mean / (jnp.sqrt(corrected_var) + eps)
            return grad - update * jnp.ones_like(grad)

        def sgd_step(grad, lr):
            return grad - lr * grad

        return adam_step, sgd_step

    def test_rank_based_conditional(self, simulator, trace_context):
        """Test conditional with rank-based predicate where different parties execute different branches."""

        def rank_based_func():
            """Function that uses rank as predicate (divergent per-party) – should NOT use uniform_cond.

            Migrated to element-wise selection using jax.where semantics via mp.run.
            """
            r = prank()
            is_one = mp.run_jax(lambda v: v == 1, r)  # bool per-party
            val_then = constant(100)
            val_else = constant(200)
            # Element-wise select (both sides cheap); result structure matches prior expectations.
            result = mp.run_jax(jnp.where, is_one, val_then, val_else)
            return result

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, rank_based_func)

        # Get the function definition expression and extract the body
        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body  # Get the function body for evaluation

        # Evaluate with empty bindings
        results = simulator.evaluate(expr, {})

        # Check results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert results[0].ctx is simulator

        # Different parties execute different paths based on rank predicate
        assert len(results[0].values) == 2
        assert results[0].values[0] == 200  # Party 0 (rank=0, False) -> else_fn
        assert results[0].values[1] == 100  # Party 1 (rank=1, True) -> then_fn

    def test_nested_conditional_with_rank(self, simulator, trace_context):
        """Test two-level nested conditional structure with rank-based predicates."""

        def two_level_func():
            """Nested conditional demo updated:

            Outer predicate: still divergent (rank==1) -> use elementwise selection.
            Inner predicate: uniform False -> shows uniform_cond legitimate usage.
            """
            r = prank()
            pred1 = mp.run_jax(lambda v: v == 1, r)  # per-party bool

            def level1_then(val):  # executes only where pred1 True (conceptually)
                # Build a uniform predicate (constant False) - safe for uniform_cond
                pred2 = constant(False)

                def level2_then(v):
                    return constant(30)

                def level2_else(v):
                    return constant(40)

                # Uniform conditional (verify disabled until predicate aggregation logic richer)
                return uniform_cond(
                    pred2, level2_then, level2_else, val, verify_uniform=True
                )

            def level1_else(val):
                return constant(50)

            # Emulate outer branching via where: where pred1 then apply level1_then else level1_else
            # Both branches evaluated; acceptable for local divergent control.
            val0 = level1_then(constant(0))
            val1 = level1_else(constant(0))
            combined = mp.run_jax(jnp.where, pred1, val0, val1)
            return combined

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, two_level_func)

        # Get the function definition expression and extract the body
        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body  # Get the function body for evaluation

        # Evaluate with empty bindings
        results = simulator.evaluate(expr, {})

        # Check results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert results[0].ctx is simulator

        # Verify the computation paths based on rank:
        # Party 0: pred1=0 (False) -> level1_else -> return 50
        # Party 1: pred1=1 (True) -> level1_then -> pred2=False -> level2_else -> return 40
        assert len(results[0].values) == 2
        assert results[0].values[0] == 50  # Party 0: level1_else result
        assert results[0].values[1] == 40  # Party 1: level1_then -> level2_else result

    def test_cond_with_jax_functions_basic(self, simulator, trace_context):
        """Test: Basic JAX operations in conditional branches.

        Purpose: Verify that JAX functions work correctly in then/else branches
        Input: [2.0, 3.0] array
        Expected: Different JAX computations based on rank predicate
        """

        def cond_with_jax():
            # Divergent predicate -> switch to elementwise where
            pred = mp.run_jax(lambda v: v == 1, prank())
            input_data = constant(np.array([2.0, 3.0]))

            # Extract JAX operations for clarity
            square_plus_one, multiply_and_add = self.simple_arithmetic_ops()

            def then_fn(val):
                return mp.run_jax(square_plus_one, val)

            def else_fn(val):
                return mp.run_jax(multiply_and_add, val)

            # Evaluate both, elementwise select
            t_res = then_fn(input_data)
            f_res = else_fn(input_data)
            return mp.run_jax(jnp.where, pred, t_res, f_res)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, cond_with_jax)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Party 0 (else): [2,3] * 2 + 5 = [9, 11]
        # Party 1 (then): [2,3]^2 + 1 = [5, 10]
        np.testing.assert_array_almost_equal(results[0].values[0], [9.0, 11.0])
        np.testing.assert_array_almost_equal(results[0].values[1], [5.0, 10.0])

    def test_cond_with_parameter_passing(self, simulator, trace_context):
        """Test: Multiple parameter passing in conditional branches.

        Purpose: Demonstrate complex parameter passing scenarios
        Scenario: Pass 3 parameters (x, y, scale) to both branches
        """

        def cond_with_params():
            pred = mp.run_jax(lambda v: v == 1, prank())

            # Input parameters
            x = constant(np.array([1.0, 2.0]))
            y = constant(np.array([3.0, 4.0]))
            scale = constant(2.5)

            def then_fn(a, b, s):
                # Multiple parameters passed directly
                def matmul_scale(x_val, y_val, scale_val):
                    return x_val * y_val * scale_val + jnp.sum(x_val)

                return mp.run_jax(matmul_scale, a, b, s)

            def else_fn(a, b, s):
                def polynomial(x_val, y_val, scale_val):
                    return jnp.power(x_val, 2) + y_val / scale_val

                return mp.run_jax(polynomial, a, b, s)

            t_res = then_fn(x, y, scale)
            f_res = else_fn(x, y, scale)
            return mp.run_jax(jnp.where, pred, t_res, f_res)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, cond_with_params)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Expected calculations:
        # Party 0 (else): [1,2]^2 + [3,4]/2.5 = [1,4] + [1.2,1.6] = [2.2, 5.6]
        # Party 1 (then): [1,2]*[3,4]*2.5 + sum([1,2]) = [7.5,20] + 3 = [10.5, 23]
        np.testing.assert_array_almost_equal(results[0].values[0], [2.2, 5.6])
        np.testing.assert_array_almost_equal(results[0].values[1], [10.5, 23.0])

    def test_cond_with_captured_variables(self, simulator, trace_context):
        """Test: Variable capture from outer scope.

        Purpose: Show how branches can capture and use outer scope variables
        Scenario: Capture matrix and vector from outer scope in different branches
        """

        def cond_with_capture():
            pred = mp.run_jax(lambda v: v == 1, prank())

            # Variables to be captured
            outer_matrix = constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
            outer_vector = constant(np.array([0.5, 1.5]))
            input_data = constant(np.array([2.0, 3.0]))

            # Extract matrix operations for clarity
            matrix_op, element_op = self.matrix_operations()

            def then_fn(input_val):
                return mp.run_jax(matrix_op, input_val, outer_matrix)

            def else_fn(input_val):
                return mp.run_jax(element_op, input_val, outer_vector)

            t_res = then_fn(input_data)
            f_res = else_fn(input_data)
            return mp.run_jax(jnp.where, pred, t_res, f_res)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, cond_with_capture)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Expected calculations:
        # Party 0 (else): [2,3] * [0.5,1.5] + mean([0.5,1.5]) = [1,4.5] + 1 = [2, 5.5]
        # Party 1 (then): dot([[1,2],[3,4]], [2,3]) + sum(matrix) = [8,18] + 10 = [18, 28]
        np.testing.assert_array_almost_equal(results[0].values[0], [2.0, 5.5])
        np.testing.assert_array_almost_equal(results[0].values[1], [18.0, 28.0])

    def test_cond_with_neural_network_simulation(self, simulator, trace_context):
        """Test: Neural network-like operations in conditional branches.

        Purpose: Demonstrate advanced computational scenarios
        Scenario: Simple neural network forward pass vs statistical analysis
        """

        def neural_network_cond():
            pred = mp.run_jax(lambda v: v == 1, prank())

            # Network parameters
            weights = constant(np.array([[0.5, 1.0], [1.5, 2.0]]))
            bias = constant(np.array([0.1, 0.2]))
            input_data = constant(np.array([1.0, 2.0]))

            # Extract neural network operations
            forward_pass, statistical_transform = self.neural_network_ops()

            def then_fn(input_val):
                return mp.run_jax(forward_pass, input_val, weights, bias)

            def else_fn(input_val):
                return mp.run_jax(statistical_transform, input_val, weights)

            t_res = then_fn(input_data)
            f_res = else_fn(input_data)
            return mp.run_jax(jnp.where, pred, t_res, f_res)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, neural_network_cond)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Both branches return 2-element arrays
        assert len(results[0].values[0]) == 2  # statistical_transform result
        assert len(results[0].values[1]) == 2  # forward_pass result

        # Verify finite results
        assert jnp.all(jnp.isfinite(results[0].values[0]))
        assert jnp.all(jnp.isfinite(results[0].values[1]))

    def test_cond_with_optimizer_simulation(self, simulator, trace_context):
        """Test: Optimizer-like operations in conditional branches.

        Purpose: Show state-dependent computations (Adam vs SGD)
        Scenario: Different optimization algorithms based on rank
        """

        def optimizer_cond():
            pred = mp.run_jax(lambda v: v == 1, prank())

            # Optimizer parameters
            learning_rate = constant(0.01)
            momentum = constant(0.9)
            epsilon = constant(1e-8)
            state_vector = constant(np.array([0.5, 1.0, 1.5, 2.0]))
            gradients = constant(np.array([0.1, 0.2, 0.3, 0.4]))

            # Extract optimizer operations
            adam_step, sgd_step = self.optimizer_ops()

            def adam_optimizer(grad, lr, mom, eps, state):
                return mp.run_jax(adam_step, grad, lr, mom, eps, state)

            def sgd_optimizer(grad, lr, mom, eps, state):
                # Only use first two parameters for SGD
                return mp.run_jax(sgd_step, grad, lr)

            adam_res = adam_optimizer(
                gradients, learning_rate, momentum, epsilon, state_vector
            )
            sgd_res = sgd_optimizer(
                gradients, learning_rate, momentum, epsilon, state_vector
            )
            return mp.run_jax(jnp.where, pred, adam_res, sgd_res)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, optimizer_cond)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Both results should be 4-element arrays
        assert len(results[0].values[0]) == 4  # sgd result
        assert len(results[0].values[1]) == 4  # adam result

        # Verify finite results
        assert jnp.all(jnp.isfinite(results[0].values[0]))
        assert jnp.all(jnp.isfinite(results[0].values[1]))

        # SGD result: input * (1 - lr) = [0.1, 0.2, 0.3, 0.4] * 0.99
        expected_sgd = np.array([0.1, 0.2, 0.3, 0.4]) * 0.99
        np.testing.assert_array_almost_equal(
            results[0].values[0], expected_sgd, decimal=5
        )

    def test_cond_matrix_reshape_operations(self, simulator, trace_context):
        """Test: Matrix reshaping and transformations.

        Purpose: Test different output structures but compatible shapes
        Scenario: Matrix operations vs element-wise operations
        """

        def matrix_reshape_cond():
            pred = mp.run_jax(lambda v: v == 1, prank())
            input_data = constant(np.array([1.0, 2.0, 3.0, 4.0]))  # Can reshape to 2x2

            def matrix_branch(val):
                def matrix_transform(x):
                    matrix = jnp.reshape(x, (2, 2))
                    result_matrix = jnp.transpose(matrix) + jnp.eye(2)
                    return jnp.ravel(result_matrix)  # Flatten back to 1D

                return mp.run_jax(matrix_transform, val)

            def element_branch(val):
                def element_transform(x):
                    return jnp.roll(x, shift=1) * 2 + jnp.ones_like(x)

                return mp.run_jax(element_transform, val)

            t_res = matrix_branch(input_data)
            f_res = element_branch(input_data)
            return mp.run_jax(jnp.where, pred, t_res, f_res)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, matrix_reshape_cond)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Both branches return 4-element arrays
        assert len(results[0].values[0]) == 4  # element_branch result
        assert len(results[0].values[1]) == 4  # matrix_branch result

        # Expected results:
        # Party 0 (element): roll([1,2,3,4], 1) * 2 + [1,1,1,1] = [4,1,2,3] * 2 + [1,1,1,1] = [9,3,5,7]
        # Party 1 (matrix): reshape->transpose->add_eye->flatten = [2,3,2,5]
        np.testing.assert_array_almost_equal(results[0].values[0], [9.0, 3.0, 5.0, 7.0])
        np.testing.assert_array_almost_equal(results[0].values[1], [2.0, 3.0, 2.0, 5.0])


class TestWhileLoop:
    """Test while_loop functionality in simulation.

    This test suite focuses on while_loop evaluation with different scenarios:
    1. Expression generation and validation (works)
    2. Simulation execution (currently has bugs - marked as skipped)

    Note: The simulator's visit_while implementation currently has bugs where it
    tries to create CallExpr with evaluated values instead of expressions.
    These tests are marked as skipped until the simulator is fixed.
    """

    def test_while_loop_expression_generation(self, trace_context):
        """Test: While_loop expression generation (without simulation).

        Purpose: Verify that while_loop can generate correct expressions
        Scenario: Simple counting loop pattern with matching types
        Expected: Correct expression structure is generated
        """

        def while_func():
            # Use explicit numpy types to ensure type consistency
            init_val = constant(np.int64(0))
            counter_max = constant(np.int64(5))  # Will be captured by cond_fn
            increment = constant(np.int64(1))  # Will be captured by body_fn

            def cond_fn(x):
                # Captures counter_max from outer scope
                return mp.run_jax(lambda a, b: a < b, x, counter_max)

            def body_fn(x):
                # Captures increment from outer scope
                return mp.run_jax(lambda a, b: a + b, x, increment)

            return while_loop(cond_fn, body_fn, init_val)

        traced_fn = trace(trace_context, while_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

        # Check that the expression contains a while loop
        from mplang.v1.core.expr.printer import Printer

        printer = Printer()
        expr_str = printer.print_expr(func_expr)

        # Should contain pwhile operation
        assert "pwhile" in expr_str
        assert "cond_fn:" in expr_str
        assert "body_fn:" in expr_str

    def test_simple_constant_loop(self, simulator, trace_context):
        """Test: Basic while_loop with constant condition (always false).

        Purpose: Verify basic while_loop functionality with immediate termination
        Scenario: Loop that never executes because condition is always false
        Expected: Returns initial value unchanged

        BUG: Currently fails because visit_while creates CallExpr([42]) instead of CallExpr(expr)
        """

        def constant_loop():
            init_val = constant(42)

            def cond_fn(x):
                # Always return False to terminate immediately
                return constant(False)

            def body_fn(x):
                # This should never execute
                return x

            return while_loop(cond_fn, body_fn, init_val)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, constant_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Since condition is always false, should return initial value
        assert results[0].values[0] == 42  # Party 0 result
        assert results[0].values[1] == 42  # Party 1 result

    def test_rank_based_termination(self, simulator, trace_context):
        """Test: While_loop with rank-based termination.

        Purpose: Verify that different parties can have different loop behaviors
        Scenario: Condition depends on party rank
        Expected: Different parties may loop different numbers of times

        BUG: Same visit_while issue prevents execution
        """

        def rank_loop():
            init_val = constant(0)

            def cond_fn(counter):
                # Party 0: always false (no iterations)
                # Party 1: check if counter < 1 (one iteration)
                rank = prank()

                def check(cnt, r):
                    # If rank is 0, return False (no loop)
                    # If rank is 1, return cnt < 1 (loop once)
                    import jax.numpy as jnp

                    return jnp.where(r == 0, False, cnt < 1)

                return mp.run_jax(check, counter, rank)

            def body_fn(counter):
                # Increment by 10
                def increment(cnt):
                    return cnt + 10

                return mp.run_jax(increment, counter)

            return while_loop(cond_fn, body_fn, init_val)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, rank_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Party 0: condition always false, so returns initial value 0
        # Party 1: condition true once (0 < 1), so executes body once: 0 + 10 = 10
        assert results[0].values[0] == 0  # Party 0: no iterations
        assert results[0].values[1] == 10  # Party 1: one iteration

    def test_counting_loop_with_captures(self, simulator, trace_context):
        """Test: While_loop with variable capture from outer scope.

        Purpose: Test the fixed simulator with the original pattern
        Scenario: Count from 0 to 5 using captured variables
        Expected: Both parties reach 5
        """

        def counting_loop():
            # Use explicit numpy types to ensure type consistency
            init_val = constant(np.int64(0))
            counter_max = constant(np.int64(5))  # Will be captured by cond_fn
            increment = constant(np.int64(1))  # Will be captured by body_fn

            def cond_fn(x):
                # Captures counter_max from outer scope
                return mp.run_jax(lambda a, b: a < b, x, counter_max)

            def body_fn(x):
                # Captures increment from outer scope
                return mp.run_jax(lambda a, b: a + b, x, increment)

            return while_loop(cond_fn, body_fn, init_val)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, counting_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Both parties: 0 -> 1 -> 2 -> 3 -> 4 -> 5 (stops when counter >= 5)
        assert results[0].values[0] == 5  # Party 0 result
        assert results[0].values[1] == 5  # Party 1 result

    def test_array_accumulation_loop(self, simulator, trace_context):
        """Test: While_loop with array state accumulation.

        Purpose: Test while_loop with more complex state (arrays)
        Scenario: Accumulate array values until sum exceeds threshold
        Expected: Both parties reach the same final array state
        """

        def accumulation_loop():
            # Start with array [1.0, 2.0]
            init_state = constant(np.array([1.0, 2.0]))
            threshold = constant(15.0)

            def cond_fn(state):
                # Continue while sum of state < threshold
                def check_sum(arr, thresh):
                    return jnp.sum(arr) < thresh

                return mp.run_jax(check_sum, state, threshold)

            def body_fn(state):
                # Add [1.0, 1.0] to current state
                increment = constant(np.array([1.0, 1.0]))

                def add_increment(arr, inc):
                    return arr + inc

                return mp.run_jax(add_increment, state, increment)

            return while_loop(cond_fn, body_fn, init_state)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, accumulation_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Both parties execute the same logic:
        # Iteration 0: [1,2] -> sum=3 < 15, continue
        # Iteration 1: [1,2] + [1,1] = [2,3] -> sum=5 < 15, continue
        # Iteration 2: [2,3] + [1,1] = [3,4] -> sum=7 < 15, continue
        # Iteration 3: [3,4] + [1,1] = [4,5] -> sum=9 < 15, continue
        # Iteration 4: [4,5] + [1,1] = [5,6] -> sum=11 < 15, continue
        # Iteration 5: [5,6] + [1,1] = [6,7] -> sum=13 < 15, continue
        # Iteration 6: [6,7] + [1,1] = [7,8] -> sum=15 >= 15, stop
        expected_result = np.array([7.0, 8.0])

        np.testing.assert_array_almost_equal(results[0].values[0], expected_result)
        np.testing.assert_array_almost_equal(results[0].values[1], expected_result)

    def test_while_loop_type_validation(self, trace_context):
        """Test: While_loop type validation during expression generation.

        Purpose: Verify that while_loop properly validates function signatures
        Scenario: Test both valid and invalid type combinations
        Expected: Proper validation errors for type mismatches
        """

        # Test: Valid case - body returns same type as init
        def valid_loop():
            init_val = constant(0)

            def cond_fn(x):
                return constant(False)  # Always false for quick termination

            def body_fn(x):
                return x  # Returns same type as input

            return while_loop(cond_fn, body_fn, init_val)

        with with_ctx(trace_context):
            # This should work without errors
            traced_fn = trace(trace_context, valid_loop)
            assert traced_fn is not None

        # Test: Invalid case - body returns wrong type
        def invalid_loop():
            init_val = constant(0)  # int type

            def cond_fn(x):
                return constant(False)

            def body_fn(x):
                return constant(3.14)  # float type - different from init

            return while_loop(cond_fn, body_fn, init_val)

        with pytest.raises(
            (ValueError, TypeError), match=r"Body output leaf 0 type mismatch: .*"
        ):
            with with_ctx(trace_context):
                trace(trace_context, invalid_loop)

    def test_while_loop_with_complex_capture(self, trace_context):
        """Test: While_loop with complex variable capture patterns.

        Purpose: Test advanced capture scenarios without simulation
        Scenario: Multiple captured variables from different scopes
        Expected: Correct expression generation with all captures
        """

        def complex_capture_loop():
            # Multiple captured variables
            init_val = constant(1)
            multiplier = constant(2)
            max_value = constant(8)

            # Nested function with additional captures
            def create_condition():
                threshold = constant(10)  # Additional capture

                def cond_fn(x):
                    # Use both max_value and threshold
                    def check(val, max_val, thresh):
                        import jax.numpy as jnp

                        return jnp.logical_and(val < max_val, val < thresh)

                    return mp.run_jax(check, x, max_value, threshold)

                return cond_fn

            def create_body():
                step = constant(1)  # Another capture

                def body_fn(x):
                    # Use both multiplier and step
                    def transform(val, mult, st):
                        return val * mult + st

                    return mp.run_jax(transform, x, multiplier, step)

                return body_fn

            cond_fn = create_condition()
            body_fn = create_body()

            return while_loop(cond_fn, body_fn, init_val)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, complex_capture_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        # Check expression structure
        from mplang.v1.core.expr.printer import Printer

        printer = Printer()
        expr_str = printer.print_expr(func_expr)

        # Should contain the while loop with multiple arguments (captures)
        assert "pwhile" in expr_str
        assert "cond_fn:" in expr_str
        assert "body_fn:" in expr_str

    def test_accumulation_loop(self, simulator, trace_context):
        """Test: Accumulation pattern with array operations.

        Purpose: Test while_loop with array state and accumulation
        Scenario: Accumulate sum of squares until sum exceeds threshold
        """

        def accumulation_loop():
            # Start with array [1.0, 2.0]
            init_state = constant(np.array([1.0, 2.0]))
            threshold = constant(50.0)

            def cond_fn(state):
                # Continue while sum of state < threshold
                def check_sum(arr, thresh):
                    return jnp.sum(arr) < thresh

                return mp.run_jax(check_sum, state, threshold)

            def body_fn(state):
                # Add squares of current state to itself
                def accumulate_squares(arr):
                    return arr + jnp.square(arr)

                return mp.run_jax(accumulate_squares, state)

            return while_loop(cond_fn, body_fn, init_state)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, accumulation_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Both parties execute the same logic:
        # Iteration 0: [1, 2] -> sum=3 < 50, continue
        # Iteration 1: [1,2] + [1,4] = [2,6] -> sum=8 < 50, continue
        # Iteration 2: [2,6] + [4,36] = [6,42] -> sum=48 < 50, continue
        # Iteration 3: [6,42] + [36,1764] = [42,1806] -> sum=1848 >= 50, stop
        expected_result = np.array([42.0, 1806.0])

        np.testing.assert_array_almost_equal(results[0].values[0], expected_result)
        np.testing.assert_array_almost_equal(results[0].values[1], expected_result)

    def test_rank_dependent_termination(self, simulator, trace_context):
        """Test: Different termination conditions based on party rank.

        Purpose: Verify that different parties can have different loop behaviors
        Scenario: Party 0 multiplies by 2, Party 1 multiplies by 3, different limits
        """

        def rank_dependent_loop():
            init_val = constant(1)
            rank = prank()

            def cond_fn(state):
                # Party 0: continue while state < 16
                # Party 1: continue while state < 27
                def check_limit(val, party_rank):
                    limit = jnp.where(party_rank == 0, 16, 27)
                    return val < limit

                return mp.run_jax(check_limit, state, rank)

            def body_fn(state):
                # Party 0: multiply by 2
                # Party 1: multiply by 3
                def transform(val, party_rank):
                    multiplier = jnp.where(party_rank == 0, 2, 3)
                    return val * multiplier

                return mp.run_jax(transform, state, rank)

            return while_loop(cond_fn, body_fn, init_val)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, rank_dependent_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Party 0: 1 -> 2 -> 4 -> 8 -> 16 (stops when >= 16)
        # Party 1: 1 -> 3 -> 9 -> 27 (stops when >= 27)
        assert results[0].values[0] == 16  # Party 0 result
        assert results[0].values[1] == 27  # Party 1 result

    def test_matrix_transformation_loop(self, simulator, trace_context):
        """Test: Matrix operations in while_loop with complex state.

        Purpose: Test while_loop with matrix state and transformations
        Scenario: Iteratively transform a 2x2 matrix until diagonal sum exceeds threshold
        """

        def matrix_loop():
            # Initialize 2x2 matrix
            init_matrix = constant(np.array([[1.0, 0.5], [0.5, 1.0]]))
            threshold = constant(10.0)

            def cond_fn(matrix):
                # Continue while trace (diagonal sum) < threshold
                def check_trace(mat, thresh):
                    trace_val = jnp.trace(mat)
                    return trace_val < thresh

                return mp.run_jax(check_trace, matrix, threshold)

            def body_fn(matrix):
                # Transform: M' = M + 0.5 * M * M^T
                def transform_matrix(mat):
                    return mat + 0.5 * jnp.dot(mat, mat.T)

                return mp.run_jax(transform_matrix, matrix)

            return while_loop(cond_fn, body_fn, init_matrix)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, matrix_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Both parties execute the same matrix transformation
        # The result should be a 2x2 matrix with trace >= 10
        result_matrix_0 = results[0].values[0]
        result_matrix_1 = results[0].values[1]

        assert result_matrix_0.shape == (2, 2)
        assert result_matrix_1.shape == (2, 2)

        # Check that trace condition is satisfied
        trace_0 = np.trace(result_matrix_0)
        trace_1 = np.trace(result_matrix_1)

        assert trace_0 >= 10.0
        assert trace_1 >= 10.0

        # Both parties should have the same result
        np.testing.assert_array_almost_equal(result_matrix_0, result_matrix_1)

    def test_early_termination_with_capture(self, simulator, trace_context):
        """Test: Early termination with captured variables.

        Purpose: Test variable capture in while_loop with early termination
        Scenario: Use captured constants for different termination logic
        """

        def early_termination_loop():
            init_val = constant(0)
            step_size = constant(2)  # Captured by body_fn
            max_iterations = constant(5)  # Captured by cond_fn
            rank = prank()

            def cond_fn(counter):
                # Different termination for different parties using captured variables
                def should_continue(cnt, max_iter, party_rank):
                    # Party 0: stop at max_iterations
                    # Party 1: stop at max_iterations - 1
                    limit = jnp.where(party_rank == 0, max_iter, max_iter - 1)
                    return cnt < limit

                return mp.run_jax(should_continue, counter, max_iterations, rank)

            def body_fn(counter):
                # Use captured step_size
                def increment_by_step(cnt, step):
                    return cnt + step

                return mp.run_jax(increment_by_step, counter, step_size)

            return while_loop(cond_fn, body_fn, init_val)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, early_termination_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Party 0: 0 -> 2 -> 4 -> 6 (stops when counter >= 5)
        # Party 1: 0 -> 2 -> 4 (stops when counter >= 4)
        assert results[0].values[0] == 6  # Party 0: 3 iterations
        assert results[0].values[1] == 4  # Party 1: 2 iterations

    def test_while_loop_subset_state_mask(self):
        """Loop state and control stay on subset of parties."""

        cluster_spec = ClusterSpec.simple(world_size=3)
        full_mask = Mask(0b111)
        subset_mask = Mask(0b011)
        trace_ctx = TraceContext(cluster_spec=cluster_spec, mask=full_mask)
        simulator = Simulator.simple(world_size=3)

        def subset_loop():
            init_state = set_mask(constant(np.int64(0)), subset_mask)
            threshold = set_mask(constant(np.int64(3)), subset_mask)
            step = set_mask(constant(np.int64(1)), subset_mask)

            def cond_fn(state):
                subset_pred = mp.run_jax(
                    lambda val, limit: val < limit, state, threshold
                )
                return pshfl_s(subset_pred, full_mask, [Rank(0), Rank(0), Rank(0)])

            def body_fn(state):
                return mp.run_jax(lambda val, inc: val + inc, state, step)

            return while_loop(cond_fn, body_fn, init_state)

        with with_ctx(trace_ctx):
            traced_fn = trace(trace_ctx, subset_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        assert len(results) == 1
        sim_var = results[0]
        assert isinstance(sim_var, SimVar)
        assert sim_var.mptype.pmask == subset_mask

        values = sim_var.values
        assert len(values) == 3
        assert values[0] == 3
        assert values[1] == 3
        assert values[2] is None

    def test_while_loop_subset_context_mask_success(self):
        """Trace under subset context mask; predicate pmask==context mask so no broadcast needed.

        Ensures static pmask validation (design A) does NOT raise when the trace context
        itself is the subset. Predicate pmask equals the context mask.
        """
        # Use a 2-party cluster because only parties 0 and 1 participate.
        cluster_spec = ClusterSpec.simple(world_size=2)
        subset_mask = Mask(0b11)  # parties 0 and 1
        trace_ctx = TraceContext(cluster_spec=cluster_spec, mask=subset_mask)
        simulator = Simulator.simple(world_size=2)

        def subset_loop():
            init_state = set_mask(constant(np.int64(0)), subset_mask)
            threshold = set_mask(constant(np.int64(3)), subset_mask)
            step = set_mask(constant(np.int64(1)), subset_mask)

            def cond_fn(state):
                # Returns bool with pmask=subset_mask (no broadcast)
                return mp.run_jax(lambda val, limit: val < limit, state, threshold)

            def body_fn(state):
                return mp.run_jax(lambda val, inc: val + inc, state, step)

            return while_loop(cond_fn, body_fn, init_state)

        with with_ctx(trace_ctx):
            traced_fn = trace(trace_ctx, subset_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        assert len(results) == 1
        sim_var = results[0]
        assert isinstance(sim_var, SimVar)
        assert sim_var.mptype.pmask == subset_mask
        values = sim_var.values
        assert len(values) == 2
        assert values[0] == 3
        assert values[1] == 3

    def test_while_loop_predicate_static_pmask_mismatch_error(self):
        """Full context mask but predicate has smaller static pmask -> trace-time ValueError.

        We purposely do NOT broadcast the subset predicate to full mask, expecting the
        new static pmask validation in while_loop to raise.
        """
        cluster_spec = ClusterSpec.simple(world_size=3)
        full_mask = Mask(0b111)
        subset_mask = Mask(0b011)
        trace_ctx = TraceContext(cluster_spec=cluster_spec, mask=full_mask)

        def bad_loop():
            init_state = set_mask(constant(np.int64(0)), subset_mask)
            threshold = set_mask(constant(np.int64(2)), subset_mask)
            step = set_mask(constant(np.int64(1)), subset_mask)

            def cond_fn(state):
                # Returns bool with pmask=subset_mask only; no broadcast.
                return mp.run_jax(lambda val, limit: val < limit, state, threshold)

            def body_fn(state):
                return mp.run_jax(lambda val, inc: val + inc, state, step)

            return while_loop(cond_fn, body_fn, init_state)

        with with_ctx(trace_ctx):
            with pytest.raises(
                ValueError, match=r"while_loop predicate static pmask mismatch"
            ):
                trace(trace_ctx, bad_loop)

    def test_while_loop_cond_body_with_aux_party(self):
        """Loop state on subset while cond/body still invoke a third party."""

        cluster_spec = ClusterSpec.simple(world_size=3)
        full_mask = Mask(0b111)
        subset_mask = Mask(0b011)
        aux_mask = Mask(0b100)
        trace_ctx = TraceContext(cluster_spec=cluster_spec, mask=full_mask)
        simulator = Simulator.simple(world_size=3)

        def cooperative_loop():
            subset_state = set_mask(constant(np.int64(0)), subset_mask)
            aux_state = set_mask(constant(np.int64(0)), aux_mask)

            subset_limit = set_mask(constant(np.int64(6)), subset_mask)
            subset_step = set_mask(constant(np.int64(2)), subset_mask)
            aux_step = set_mask(constant(np.int64(1)), aux_mask)

            def cond_fn(states):
                sub_val, aux_val = states

                # Auxiliary party executes a helper kernel (result ignored by others)
                _ = mp.run_jax(lambda val, inc: val + inc, aux_val, aux_step)
                subset_pred = mp.run_jax(
                    lambda val, limit: val < limit, sub_val, subset_limit
                )
                # Broadcast predicate so every party observes the same boolean
                return pshfl_s(subset_pred, full_mask, [Rank(0), Rank(0), Rank(0)])

            def body_fn(states):
                sub_val, aux_val = states

                next_sub = mp.run_jax(
                    lambda val, step: val + step, sub_val, subset_step
                )
                next_aux = mp.run_jax(lambda val, inc: val + inc, aux_val, aux_step)

                return (next_sub, next_aux)

            return while_loop(cond_fn, body_fn, (subset_state, aux_state))

        with with_ctx(trace_ctx):
            traced_fn = trace(trace_ctx, cooperative_loop)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        assert len(results) == 2
        subset_result, aux_result = results

        assert isinstance(subset_result, SimVar)
        assert subset_result.mptype.pmask == subset_mask
        assert subset_result.values == [6, 6, None]

        assert isinstance(aux_result, SimVar)
        assert aux_result.mptype.pmask == aux_mask
        assert aux_result.values == [None, None, 3]

    def test_nested_while_with_conditional(self, simulator, trace_context):
        """Test: While_loop containing conditional operations.

        Purpose: Test interaction between while_loop and cond primitives
        Scenario: Loop that uses conditional logic inside the body
        """

        def nested_while_cond():
            init_val = constant(1)
            target = constant(20)

            def cond_fn(state):
                # Continue while state < target
                def check_target(val, tgt):
                    return val < tgt

                return mp.run_jax(check_target, state, target)

            def body_fn(state):
                # Use conditional logic based on current state value
                even_check = mp.run_jax(lambda x: x % 2 == 0, state)

                def then_fn(val):
                    # If even: multiply by 2
                    return mp.run_jax(lambda x: x * 2, val)

                def else_fn(val):
                    # If odd: add 3
                    return mp.run_jax(lambda x: x + 3, val)

                return uniform_cond(even_check, then_fn, else_fn, state)

            return while_loop(cond_fn, body_fn, init_val)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, nested_while_cond)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Both parties follow same logic:
        # 1 (odd) -> 1+3=4 (even) -> 4*2=8 (even) -> 8*2=16 (even) -> 16*2=32 >= 20, stop
        expected_result = 32

        assert results[0].values[0] == expected_result
        assert results[0].values[1] == expected_result


class TestPShflS:
    """Test pshfl_s (static shuffle) functionality in simulation.

    This test suite focuses on static shuffle operations with different scenarios:
    1. Basic shuffle operations with simple mappings
    2. Complex mappings with multiple source ranks
    3. Edge cases and validation scenarios
    4. Integration with other primitives
    """

    def test_basic_pshfl_s(self, simulator, trace_context):
        """Test: Basic pshfl_s operation.

        Purpose: Verify basic static shuffle functionality
        Scenario: Simple 1-to-1 mapping from source ranks to receivers
        Expected: Each receiver gets data from specified source rank
        """

        def basic_shuffle():
            # Create source data at party 0
            src_data = constant(42)

            # Shuffle to both parties: both get data from rank 0
            pmask = Mask(3)  # 0b11 - both parties receive
            src_ranks = [Rank(0), Rank(0)]  # Both receive from party 0

            return pshfl_s(src_data, pmask, src_ranks)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, basic_shuffle)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Both parties should receive the same value (42) from party 0
        assert results[0].values[0] == 42  # Party 0 result
        assert results[0].values[1] == 42  # Party 1 result

    def test_different_source_mapping(self, simulator, trace_context):
        """Test: pshfl_s with different source ranks.

        Purpose: Test shuffle with different sources for different receivers
        Scenario: Party 0 receives from rank 1, Party 1 receives from rank 0
        Expected: Cross-communication pattern
        """

        def cross_shuffle():
            # Create different data at each party based on rank
            rank_data = prank()  # Party 0 has 0, Party 1 has 1
            multiplied_data = mp.run_jax(lambda x: x * 10 + 100, rank_data)

            # Cross shuffle: party 0 gets from rank 1, party 1 gets from rank 0
            pmask = Mask(3)  # 0b11 - both parties receive
            src_ranks = [Rank(1), Rank(0)]  # Party 0 <- rank 1, Party 1 <- rank 0

            return pshfl_s(multiplied_data, pmask, src_ranks)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, cross_shuffle)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Original data: Party 0 has 100 (0*10+100), Party 1 has 110 (1*10+100)
        # After shuffle: Party 0 gets 110 (from rank 1), Party 1 gets 100 (from rank 0)
        assert results[0].values[0] == 110  # Party 0 receives from rank 1
        assert results[0].values[1] == 100  # Party 1 receives from rank 0

    def test_partial_shuffle(self, simulator, trace_context):
        """Test: pshfl_s with partial party participation.

        Purpose: Test shuffle where only some parties receive data
        Scenario: Only party 1 receives data (pmask = 0b10)
        Expected: Only party 1 has valid result
        """

        def partial_shuffle():
            # Create source data
            src_data = constant(777)

            # Only party 1 receives data
            pmask = Mask(2)  # 0b10 - only party 1 receives
            src_ranks = [Rank(0)]  # Party 1 receives from rank 0

            return pshfl_s(src_data, pmask, src_ranks)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, partial_shuffle)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Only party 1 should have received data
        # Party 0 should have some default/undefined value (implementation dependent)
        # Party 1 should have the source value (777)
        assert results[0].values[1] == 777  # Party 1 receives from rank 0

    def test_array_shuffle(self, simulator, trace_context):
        """Test: pshfl_s with array data.

        Purpose: Test shuffle with more complex data types (arrays)
        Scenario: Shuffle arrays between parties
        Expected: Arrays are correctly transferred
        """

        def array_shuffle():
            # Create array data based on rank
            rank = prank()
            array_data = mp.run_jax(
                lambda r: jnp.array([r * 10 + 1, r * 10 + 2, r * 10 + 3]),
                rank,
            )

            # Cross shuffle
            pmask = Mask(3)  # Both parties receive
            src_ranks = [Rank(1), Rank(0)]  # Cross pattern

            return pshfl_s(array_data, pmask, src_ranks)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, array_shuffle)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Original arrays: Party 0 has [1,2,3], Party 1 has [11,12,13]
        # After shuffle: Party 0 gets [11,12,13], Party 1 gets [1,2,3]
        np.testing.assert_array_equal(results[0].values[0], [11, 12, 13])
        np.testing.assert_array_equal(results[0].values[1], [1, 2, 3])

    def test_same_rank_mapping(self, simulator, trace_context):
        """Test: pshfl_s where multiple parties receive from same source.

        Purpose: Test the current semantic where multiple receivers can get from same sender
        Scenario: Both parties receive from rank 0
        Expected: Both parties get the same data from rank 0
        """

        def same_source_shuffle():
            # Create unique data at each party
            rank = prank()
            unique_data = mp.run_jax(lambda r: (r + 1) * 100, rank)

            # Both parties receive from rank 0
            pmask = Mask(3)  # Both parties receive
            src_ranks = [Rank(0), Rank(0)]  # Both from rank 0

            return pshfl_s(unique_data, pmask, src_ranks)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, same_source_shuffle)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Original data: Party 0 has 100, Party 1 has 200
        # After shuffle: Both parties get 100 (from rank 0)
        assert results[0].values[0] == 100  # Party 0 gets from rank 0
        assert results[0].values[1] == 100  # Party 1 gets from rank 0

    def test_shuffle_with_computation(self, simulator, trace_context):
        """Test: pshfl_s integrated with other computations.

        Purpose: Test shuffle as part of larger computation pipeline
        Scenario: Compute -> Shuffle -> Compute pattern
        Expected: Correct data flow through the pipeline
        """

        def computation_shuffle():
            # Step 1: Initial computation
            rank = prank()
            initial_data = mp.run_jax(lambda r: r * 5 + 10, rank)

            # Step 2: Shuffle the computed data
            pmask = Mask(3)
            src_ranks = [Rank(1), Rank(0)]  # Cross pattern
            shuffled_data = pshfl_s(initial_data, pmask, src_ranks)

            # Step 3: Further computation on shuffled data
            final_result = mp.run_jax(lambda x: x * 2 + 1, shuffled_data)

            return final_result

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, computation_shuffle)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Step-by-step calculation:
        # Initial: Party 0 has 10 (0*5+10), Party 1 has 15 (1*5+10)
        # Shuffle: Party 0 gets 15, Party 1 gets 10
        # Final: Party 0 gets 15*2+1=31, Party 1 gets 10*2+1=21
        assert results[0].values[0] == 31  # Party 0 final result
        assert results[0].values[1] == 21  # Party 1 final result

    def test_matrix_shuffle(self, simulator, trace_context):
        """Test: pshfl_s with matrix data.

        Purpose: Test shuffle with 2D array (matrix) data
        Scenario: Create different matrices at each party and shuffle them
        Expected: Matrices are correctly transferred between parties
        """

        def matrix_shuffle():
            # Create different matrices at each party
            rank = prank()
            matrix_data = mp.run_jax(
                lambda r: jnp.array([[r + 1, r + 2], [r + 3, r + 4]]),
                rank,
            )

            # Shuffle matrices
            pmask = Mask(3)
            src_ranks = [Rank(1), Rank(0)]  # Cross pattern

            return pshfl_s(matrix_data, pmask, src_ranks)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, matrix_shuffle)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Original matrices:
        # Party 0: [[1,2],[3,4]]
        # Party 1: [[2,3],[4,5]]
        # After shuffle:
        # Party 0 gets Party 1's matrix: [[2,3],[4,5]]
        # Party 1 gets Party 0's matrix: [[1,2],[3,4]]

        expected_matrix_0 = np.array([[2, 3], [4, 5]])
        expected_matrix_1 = np.array([[1, 2], [3, 4]])

        np.testing.assert_array_equal(results[0].values[0], expected_matrix_0)
        np.testing.assert_array_equal(results[0].values[1], expected_matrix_1)

    def test_shuffle_chain(self, simulator, trace_context):
        """Test: Multiple pshfl_s operations in sequence.

        Purpose: Test chaining multiple shuffle operations
        Scenario: Shuffle -> Transform -> Shuffle again
        Expected: Correct data flow through multiple shuffles
        """

        def shuffle_chain():
            # Initial data
            rank = prank()
            initial = mp.run_jax(lambda r: (r + 1) * 10, rank)

            # First shuffle: cross pattern
            pmask = Mask(3)
            first_shuffle = pshfl_s(initial, pmask, [Rank(1), Rank(0)])

            # Transform the shuffled data
            transformed = mp.run_jax(lambda x: x + 5, first_shuffle)

            # Second shuffle: back to original (reverse cross)
            second_shuffle = pshfl_s(transformed, pmask, [Rank(1), Rank(0)])

            return second_shuffle

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, shuffle_chain)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Step-by-step:
        # Initial: Party 0 has 10, Party 1 has 20
        # First shuffle: Party 0 gets 20, Party 1 gets 10
        # Transform: Party 0 has 25, Party 1 has 15
        # Second shuffle: Party 0 gets 15, Party 1 gets 25
        assert results[0].values[0] == 15  # Party 0 final
        assert results[0].values[1] == 25  # Party 1 final

    def test_shuffle_with_conditional(self, simulator, trace_context):
        """Test: pshfl_s integrated with conditional operations.

        Purpose: Test shuffle combined with conditional logic
        Scenario: Use conditional to determine shuffle source data
        Expected: Different parties may shuffle different data
        """

        def conditional_shuffle():
            # Use rank as predicate for conditional
            rank = prank()
            pred = mp.run_jax(lambda r: r == 0, rank)

            def then_fn(r):  # local cheap
                return constant(100)

            def else_fn(r):  # local cheap
                return constant(200)

            # Divergent predicate (rank==0) → use elementwise selection instead of uniform_cond
            t_val = then_fn(rank)
            f_val = else_fn(rank)
            cond_result = mp.run_jax(jnp.where, pred, t_val, f_val)

            # Shuffle the conditional result
            pmask = Mask(3)
            src_ranks = [Rank(0), Rank(1)]  # Party 0 <- rank 0, Party 1 <- rank 1

            return pshfl_s(cond_result, pmask, src_ranks)

        with with_ctx(trace_context):
            traced_fn = trace(trace_context, conditional_shuffle)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        expr = func_expr.body
        results = simulator.evaluate(expr, {})

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SimVar)
        assert len(results[0].values) == 2

        # Conditional results: Party 0 has 100 (rank==0 is true), Party 1 has 200 (rank==0 is false)
        # Shuffle: Party 0 gets from rank 0 (100), Party 1 gets from rank 1 (200)
        assert results[0].values[0] == 100  # Party 0 gets from rank 0
        assert results[0].values[1] == 200  # Party 1 gets from rank 1


class TestPShfl:
    """Test pshfl (dynamic shuffle) related functions."""

    def test_pshfl_basic(self):
        """Test basic pshfl functionality with random permutation."""
        num_parties = 10
        sim = Simulator.simple(num_parties)

        @mp.function
        def create_test_data():
            # Create random source data and permutation index
            src = mp.prandint(0, 100)
            key = mp.ukey(42)
            index = mp.pperm(key)
            return src, index

        @mp.function
        def shuffle_data(src, index):
            # Shuffle data with permutation index
            shuffled = mp.pshfl(src, index)
            return shuffled

        # Generate test data
        src, index = mp.evaluate(sim, create_test_data)

        # Perform shuffle
        shuffled = mp.evaluate(sim, shuffle_data, src, index)

        # Fetch results
        data, index_vals, shuffled_vals = mp.fetch(sim, (src, index, shuffled))

        # Convert to arrays for comparison
        data_arr = np.stack(data)
        index_arr = np.stack(index_vals)
        shuffled_arr = np.stack(shuffled_vals)

        # Verify: shuffled[i] should equal data[index[i]]
        np.testing.assert_array_equal(data_arr[index_arr], shuffled_arr)
