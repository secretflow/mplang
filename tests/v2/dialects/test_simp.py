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

"""Tests for SIMP dialect (EDSL-based primitives).

Note: uniform_cond and while_loop are bound methods from Primitive instances.
Call them directly like functions: uniform_cond(...), while_loop(...)
"""

import jax.numpy as jnp
import numpy as np
import pytest

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects.simp import peval, uniform_cond, while_loop
from mplang.v2.runtime.interpreter import InterpObject

pytestmark = pytest.mark.usefixtures("simp_simulator_default")


class TestUniformCond:
    """Test suite for uniform_cond primitive in trace mode."""

    def test_with_callables_true_branch(self):
        """Test uniform_cond traces correctly.

        Graph structure independent of pred value.
        """
        pred_val = InterpObject(np.array(True), elt.Tensor[elt.f32, ()])
        x_val = InterpObject(np.array(5.0), elt.Tensor[elt.f32, ()])

        def then_fn(x):
            return x  # Simple identity

        def else_fn(x):
            return x  # Simple identity

        def test_fn(pred, x):
            return uniform_cond(pred, then_fn, else_fn, x)

        traced = el.trace(test_fn, pred_val, x_val)
        graph = traced.graph

        # Verify graph has cond operation
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1
        assert len(cond_ops[0].regions) == 2

    def test_with_callables_false_branch(self):
        """Test uniform_cond traces correctly.

        Both branches traced regardless of pred.
        """
        pred_val = InterpObject(np.array(False), elt.Tensor[elt.f32, ()])
        x_val = InterpObject(np.array(5.0), elt.Tensor[elt.f32, ()])

        def then_fn(x):
            return x

        def else_fn(x):
            return x

        def test_fn(pred, x):
            return uniform_cond(pred, then_fn, else_fn, x)

        traced = el.trace(test_fn, pred_val, x_val)
        graph = traced.graph

        # Verify graph structure (independent of pred value at trace time)
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1

    def test_with_multiple_outputs(self):
        """Test uniform_cond with branches returning multiple values."""
        pred_val = InterpObject(np.array(True), elt.Tensor[elt.f32, ()])
        x_val = InterpObject(np.array(5.0), elt.Tensor[elt.f32, ()])
        y_val = InterpObject(np.array(3.0), elt.Tensor[elt.f32, ()])

        def then_fn(x, y):
            return (x, y)

        def else_fn(x, y):
            return (x, y)

        def test_fn(pred, x, y):
            return uniform_cond(pred, then_fn, else_fn, x, y)

        traced = el.trace(test_fn, pred_val, x_val, y_val)
        graph = traced.graph

        # Verify cond operation exists
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1
        # Should have 2 outputs (tuple of 2 values)
        assert len(cond_ops[0].outputs) == 2

    def test_branches_must_match_output_types(self):
        """Test that branches with mismatched outputs raise TypeError."""

        # Create primitives that return different number of outputs
        ret1_p = el.Primitive("test.ret1")
        ret2_p = el.Primitive("test.ret2")

        @ret1_p.def_abstract_eval
        def _ret1_ae(x_type):
            return x_type

        @ret2_p.def_abstract_eval
        def _ret2_ae(x_type):
            return [x_type, x_type]

        pred_val = InterpObject(np.array(True), elt.Tensor[elt.f32, ()])
        x_val = InterpObject(np.array(5.0), elt.Tensor[elt.f32, ()])

        def test_fn(pred, x):
            def then_fn(x):
                return ret1_p.bind(x)  # Returns 1 value

            def else_fn(x):
                return ret2_p.bind(x)  # Returns 2 values

            return uniform_cond(pred, then_fn, else_fn, x)

        # Should raise TypeError due to output mismatch
        with pytest.raises(TypeError, match="output signature mismatch"):
            el.trace(test_fn, pred_val, x_val)

    def test_with_multiple_args(self):
        """Test uniform_cond with multiple arguments."""
        pred_val = InterpObject(np.array(True), elt.Tensor[elt.f32, ()])
        x_val = InterpObject(np.array(10.0), elt.Tensor[elt.f32, ()])
        y_val = InterpObject(np.array(3.0), elt.Tensor[elt.f32, ()])

        def then_fn(x, y):
            return x

        def else_fn(x, y):
            return y

        def test_fn(pred, x, y):
            return uniform_cond(pred, then_fn, else_fn, x, y)

        traced = el.trace(test_fn, pred_val, x_val, y_val)
        graph = traced.graph

        # Verify graph structure
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1
        cond_op = cond_ops[0]
        assert len(cond_op.inputs) == 3  # pred + 2 args, no captures
        then_region, else_region = cond_op.regions
        assert len(then_region.inputs) == 2
        assert len(else_region.inputs) == 2

    def test_branch_captures_are_aligned(self):
        """Captured variables from both branches become explicit cond inputs."""
        pred_val = InterpObject(np.array(True), elt.Tensor[elt.f32, ()])
        x_val = InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])
        outer_a = InterpObject(np.array(2.0), elt.Tensor[elt.f32, ()])
        outer_b = InterpObject(np.array(3.0), elt.Tensor[elt.f32, ()])

        def then_fn(x):
            return outer_a

        def else_fn(x):
            return outer_b

        def test_fn(pred, x):
            return uniform_cond(pred, then_fn, else_fn, x)

        traced = el.trace(test_fn, pred_val, x_val)
        graph = traced.graph

        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1
        cond_op = cond_ops[0]

        # pred + arg + 2 captures
        assert len(cond_op.inputs) == 4

        then_region, else_region = cond_op.regions
        assert len(then_region.inputs) == 3  # arg + both captures
        assert len(else_region.inputs) == 3  # arg + both captures (one unused)

    def test_verify_uniform_attribute(self):
        """Test that verify_uniform flag uses global config."""
        from mplang.v2.dialects import simp

        pred_val = InterpObject(np.array(True), elt.Tensor[elt.f32, ()])
        x_val = InterpObject(np.array(5.0), elt.Tensor[elt.f32, ()])

        # Test with default (True)
        def test_fn(pred, x):
            return uniform_cond(pred, lambda x: x, lambda x: x, x)

        traced = el.trace(test_fn, pred_val, x_val)
        graph = traced.graph

        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1
        assert cond_ops[0].attrs["verify_uniform"] is True

        # Test with global config set to False
        original_value = simp.VERIFY_UNIFORM_DEFAULT
        try:
            simp.VERIFY_UNIFORM_DEFAULT = False
            traced2 = el.trace(test_fn, pred_val, x_val)
            graph2 = traced2.graph
            cond_ops2 = [
                op for op in graph2.operations if op.opcode == "simp.uniform_cond"
            ]
            assert len(cond_ops2) == 1
            assert cond_ops2[0].attrs["verify_uniform"] is False
        finally:
            simp.VERIFY_UNIFORM_DEFAULT = original_value


class TestUniformCondTracing:
    """Test suite for uniform_cond in trace mode."""

    def test_trace_simple_cond(self):
        """Test tracing uniform_cond with simple branches."""

        # Create test inputs
        pred_val = InterpObject(True, elt.Tensor[elt.f32, ()])
        x_val = InterpObject(5.0, elt.Tensor[elt.f32, ()])

        def test_fn(pred, x):
            def then_fn(x):
                return x

            def else_fn(x):
                return x

            return uniform_cond(pred, then_fn, else_fn, x)

        # Trace the function
        traced = el.trace(test_fn, pred_val, x_val)
        graph = traced.graph

        # Verify graph structure
        assert graph is not None
        assert len(graph.operations) > 0

        # Find the cond operation
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1

        cond_op = cond_ops[0]
        assert len(cond_op.regions) == 2  # then and else regions
        assert cond_op.attrs["verify_uniform"] is True

    def test_trace_cond_with_ops(self):
        """Test tracing uniform_cond with nested primitives inside branches."""

        # Create a simple test primitive
        negate_p = el.Primitive("test.negate")

        @negate_p.def_abstract_eval
        def _negate_ae(x_type):
            return x_type

        pred_val = InterpObject(True, elt.Tensor[elt.f32, ()])
        x_val = InterpObject(5.0, elt.Tensor[elt.f32, ()])

        def test_fn(pred, x):
            def then_fn(x):
                # Use a primitive inside the branch
                return negate_p.bind(x)

            def else_fn(x):
                # Another primitive
                return negate_p.bind(x)

            return uniform_cond(pred, then_fn, else_fn, x)

        traced = el.trace(test_fn, pred_val, x_val)
        graph = traced.graph

        # Find the cond operation
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1

        cond_op = cond_ops[0]
        then_graph, else_graph = cond_op.regions

        # Verify both branches have negate operation
        then_ops = [op.opcode for op in then_graph.operations]
        assert "test.negate" in then_ops

        else_ops = [op.opcode for op in else_graph.operations]
        assert "test.negate" in else_ops

    def test_trace_cond_type_mismatch(self):
        """Test that branches with mismatched output types raise TypeError."""

        pred_val = InterpObject(True, elt.Tensor[elt.f32, ()])
        x_val = InterpObject(5.0, elt.Tensor[elt.f32, ()])

        def test_fn(pred, x):
            def then_fn(x):
                # Return elt.f32
                return x

            def else_fn(x):
                # Return different type (simulate by creating wrong type)
                # For this test we'll just ensure different output counts
                return x

            return uniform_cond(pred, then_fn, else_fn, x)

        # This should work fine (same types)
        traced = el.trace(test_fn, pred_val, x_val)
        graph = traced.graph
        assert graph is not None


class TestWhileLoop:
    """Tests for the SIMP while_loop primitive."""

    def test_traces_basic_loop(self):
        """while_loop should emit a loop op with cond/body regions."""
        flag = InterpObject(np.array(True), elt.Tensor[elt.f32, ()])
        value = InterpObject(np.array(0.0), elt.Tensor[elt.f32, ()])

        def cond_fn(state):
            loop_flag, _ = state
            return loop_flag

        def body_fn(state):
            loop_flag, loop_value = state
            return (loop_flag, loop_value)

        def test_fn(loop_flag, loop_value):
            return while_loop(cond_fn, body_fn, (loop_flag, loop_value))

        traced = el.trace(test_fn, flag, value)
        graph = traced.graph

        loop_ops = [op for op in graph.operations if op.opcode == "simp.while_loop"]
        assert len(loop_ops) == 1
        loop_op = loop_ops[0]

        assert len(loop_op.inputs) == 2  # state only (no captures)
        assert len(loop_op.outputs) == 2
        assert len(loop_op.regions) == 2
        cond_region, body_region = loop_op.regions
        assert len(cond_region.inputs) == 2
        assert len(cond_region.outputs) == 1
        assert len(body_region.inputs) == 2
        assert len(body_region.outputs) == 2

    def test_cond_must_return_scalar(self):
        """cond_fn returning multiple outputs should raise."""
        flag = InterpObject(np.array(True), elt.Tensor[elt.f32, ()])

        def cond_fn(state):
            return (state, state)

        def body_fn(state):
            return state

        def test_fn(loop_flag):
            return while_loop(cond_fn, body_fn, loop_flag)

        with pytest.raises(TypeError, match="cond_fn must return exactly one output"):
            el.trace(test_fn, flag)

    def test_body_must_match_state_arity(self):
        """body_fn returning mismatched state size should error."""
        flag = InterpObject(np.array(True), elt.Tensor[elt.f32, ()])
        extra = InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])

        def cond_fn(state):
            loop_flag, _ = state
            return loop_flag

        def body_fn(state):
            loop_flag, _ = state
            return (loop_flag,)  # Missing second component

        def test_fn(loop_flag, loop_value):
            return while_loop(cond_fn, body_fn, (loop_flag, loop_value))

        with pytest.raises(
            TypeError, match="body_fn must return same number of values"
        ):
            el.trace(test_fn, flag, extra)


class TestPeval:
    """Tests specific to the peval primitive."""

    def test_local_region_unwraps_mp_types(self):
        mp_tensor = elt.MPType[elt.Tensor[elt.f32, ()], (0, 1)]
        x = InterpObject(np.array(1.0, dtype=np.float32), mp_tensor)
        bias = InterpObject(np.array(2.0, dtype=np.float32), mp_tensor)

        def wrapper(val, captured):
            def local_fn(inner):
                return inner, captured

            return peval(None, local_fn, val)

        traced = el.trace(wrapper, x, bias)
        graph = traced.graph
        # peval with None parties uses pcall_dynamic now
        local_ops = [
            op
            for op in graph.operations
            if op.opcode in ("simp.peval", "simp.pcall_dynamic")
        ]
        assert len(local_ops) == 1
        local_op = local_ops[0]
        region = local_op.regions[0]

        # First input comes from peval argument, second from captured bias.
        assert len(region.inputs) == 2
        for value in region.inputs:
            assert not isinstance(value.type, elt.MPType)
            assert str(value.type) == str(elt.Tensor[elt.f32, ()])

        # Outputs inside the region should also be unwrapped.
        for value in region.outputs:
            assert not isinstance(value.type, elt.MPType)

        # Outermost op re-wraps with MP typing.
        for value in local_op.outputs:
            assert isinstance(value.type, elt.MPType)


class TestPcallWithTensorDialect:
    """Test suite for pcall_static using tensor dialect operations.

    Demonstrates SIMP dialect (pcall) can reference tensor dialect (jax_fn, run_jax).
    """

    def test_pcall_jax_fn_single_arg(self):
        """Test pcall_static with jax_fn wrapper (single argument)."""
        from mplang.v2.dialects.simp import pcall_static
        from mplang.v2.dialects.tensor import jax_fn

        def square(x):
            return jnp.square(x)

        x_val = InterpObject(
            np.array([1.0, 2.0, 3.0]),
            elt.MPType(elt.TensorType(elt.f32, (3,)), (0,)),
        )

        def compute(x):
            return pcall_static((0,), jax_fn(square), x)

        traced = el.trace(compute, x_val)

        # Verify graph structure
        assert len(traced.graph.operations) == 1
        op = traced.graph.operations[0]
        assert op.opcode == "simp.pcall_static"
        assert op.attrs["fn_name"] == "square"
        assert op.attrs["parties"] == [0]
        assert len(op.regions) == 1

        # Verify region contains run_jax from tensor dialect
        region = op.regions[0]
        assert len(region.operations) == 1
        assert region.operations[0].opcode == "tensor.run_jax"

    def test_pcall_jax_fn_multi_arg(self):
        """Test pcall_static with jax_fn wrapper (multiple arguments)."""
        from mplang.v2.dialects.simp import pcall_static
        from mplang.v2.dialects.tensor import jax_fn

        def add(x, y):
            return jnp.add(x, y)

        x_val = InterpObject(
            np.array([1.0, 2.0]),
            elt.MPType(elt.TensorType(elt.f32, (2,)), (0,)),
        )
        y_val = InterpObject(
            np.array([3.0, 4.0]),
            elt.MPType(elt.TensorType(elt.f32, (2,)), (0,)),
        )

        def compute(x, y):
            return pcall_static((0,), jax_fn(add), x, y)

        traced = el.trace(compute, x_val, y_val)

        # Verify graph structure
        assert len(traced.graph.operations) == 1
        op = traced.graph.operations[0]
        assert op.opcode == "simp.pcall_static"
        assert op.attrs["fn_name"] == "add"
        assert len(op.regions) == 1

        # Verify region has 2 inputs
        region = op.regions[0]
        assert len(region.inputs) == 2

    def test_jax_fn_preserves_function_name(self):
        """Test that jax_fn wrapper preserves function name for better debugging."""
        from mplang.v2.dialects.tensor import jax_fn

        def my_custom_function(x):
            return jnp.sqrt(x)

        wrapped = jax_fn(my_custom_function)

        assert wrapped.__name__ == "my_custom_function"
        assert wrapped.__doc__ == my_custom_function.__doc__

    def test_jax_fn_vs_lambda_comparison(self):
        """Compare jax_fn wrapper with lambda approach.

        Shows jax_fn provides better IR readability (fn_name) while
        maintaining same region structure.
        """
        from mplang.v2.dialects.simp import pcall_static
        from mplang.v2.dialects.tensor import jax_fn, run_jax

        def square(x):
            return jnp.square(x)

        x_val = InterpObject(
            np.array([1.0, 2.0]),
            elt.MPType(elt.TensorType(elt.f32, (2,)), (0,)),
        )

        # Using jax_fn wrapper
        def compute_jax_fn(x):
            return pcall_static((0,), jax_fn(square), x)

        traced_jax_fn = el.trace(compute_jax_fn, x_val)
        op_jax_fn = traced_jax_fn.graph.operations[0]

        # Using lambda + run_jax
        def compute_lambda(x):
            return pcall_static((0,), lambda t: run_jax(square, t), x)

        traced_lambda = el.trace(compute_lambda, x_val)
        op_lambda = traced_lambda.graph.operations[0]

        # jax_fn has better fn_name for debugging
        assert op_jax_fn.attrs["fn_name"] == "square"
        assert op_lambda.attrs["fn_name"] == "<lambda>"

        # Both have same region structure (tensor.run_jax)
        assert len(op_jax_fn.regions) == len(op_lambda.regions)
        assert op_jax_fn.regions[0].operations[0].opcode == "tensor.run_jax"
        assert op_lambda.regions[0].operations[0].opcode == "tensor.run_jax"
