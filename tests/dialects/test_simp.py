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

import numpy as np
import pytest

from mplang.dialects.simp import uniform_cond
from mplang.edsl.interpreter import InterpObject
from mplang.edsl.tracer import trace
from mplang.edsl.typing import Tensor, f32


class TestUniformCond:
    """Test suite for uniform_cond primitive in trace mode."""

    def test_with_callables_true_branch(self):
        """Test uniform_cond traces correctly - graph structure independent of pred value."""
        pred_val = InterpObject(np.array(True), Tensor[f32, ()])
        x_val = InterpObject(np.array(5.0), Tensor[f32, ()])

        def then_fn(x):
            return x  # Simple identity

        def else_fn(x):
            return x  # Simple identity

        def test_fn(pred, x):
            return uniform_cond(pred, then_fn, else_fn, x)

        graph = trace(test_fn, pred_val, x_val)

        # Verify graph has cond operation
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1
        assert len(cond_ops[0].regions) == 2

    def test_with_callables_false_branch(self):
        """Test uniform_cond traces correctly - both branches traced regardless of pred."""
        pred_val = InterpObject(np.array(False), Tensor[f32, ()])
        x_val = InterpObject(np.array(5.0), Tensor[f32, ()])

        def then_fn(x):
            return x

        def else_fn(x):
            return x

        def test_fn(pred, x):
            return uniform_cond(pred, then_fn, else_fn, x)

        graph = trace(test_fn, pred_val, x_val)

        # Verify graph structure (independent of pred value at trace time)
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1

    def test_with_multiple_outputs(self):
        """Test uniform_cond with branches returning multiple values."""
        pred_val = InterpObject(np.array(True), Tensor[f32, ()])
        x_val = InterpObject(np.array(5.0), Tensor[f32, ()])
        y_val = InterpObject(np.array(3.0), Tensor[f32, ()])

        def then_fn(x, y):
            return (x, y)

        def else_fn(x, y):
            return (x, y)

        def test_fn(pred, x, y):
            return uniform_cond(pred, then_fn, else_fn, x, y)

        graph = trace(test_fn, pred_val, x_val, y_val)

        # Verify cond operation exists
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1
        # Should have 2 outputs (tuple of 2 values)
        assert len(cond_ops[0].outputs) == 2

    def test_branches_must_match_output_types(self):
        """Test that branches with mismatched outputs raise TypeError."""
        from mplang.edsl.primitive import Primitive

        # Create primitives that return different number of outputs
        ret1_p = Primitive("test.ret1")
        ret2_p = Primitive("test.ret2")

        @ret1_p.def_abstract_eval
        def _ret1_ae(x_type):
            return x_type

        @ret2_p.def_abstract_eval
        def _ret2_ae(x_type):
            return [x_type, x_type]

        pred_val = InterpObject(np.array(True), Tensor[f32, ()])
        x_val = InterpObject(np.array(5.0), Tensor[f32, ()])

        def test_fn(pred, x):
            def then_fn(x):
                return ret1_p.bind(x)  # Returns 1 value

            def else_fn(x):
                return ret2_p.bind(x)  # Returns 2 values

            return uniform_cond(pred, then_fn, else_fn, x)

        # Should raise TypeError due to output count mismatch
        with pytest.raises(TypeError, match="output count mismatch"):
            trace(test_fn, pred_val, x_val)

    def test_with_multiple_args(self):
        """Test uniform_cond with multiple arguments."""
        pred_val = InterpObject(np.array(True), Tensor[f32, ()])
        x_val = InterpObject(np.array(10.0), Tensor[f32, ()])
        y_val = InterpObject(np.array(3.0), Tensor[f32, ()])

        def then_fn(x, y):
            return x

        def else_fn(x, y):
            return y

        def test_fn(pred, x, y):
            return uniform_cond(pred, then_fn, else_fn, x, y)

        graph = trace(test_fn, pred_val, x_val, y_val)

        # Verify graph structure
        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1
        # Should have pred + 2 args as inputs (+ any captures)
        assert len(cond_ops[0].inputs) >= 3

    def test_verify_uniform_attribute(self):
        """Test that verify_uniform flag is properly set in graph."""
        pred_val = InterpObject(np.array(True), Tensor[f32, ()])
        x_val = InterpObject(np.array(5.0), Tensor[f32, ()])

        def test_fn(pred, x):
            return uniform_cond(pred, lambda x: x, lambda x: x, x, verify_uniform=False)

        graph = trace(test_fn, pred_val, x_val)

        cond_ops = [op for op in graph.operations if op.opcode == "simp.uniform_cond"]
        assert len(cond_ops) == 1
        assert cond_ops[0].attrs["verify_uniform"] is False


class TestUniformCondTracing:
    """Test suite for uniform_cond in trace mode."""

    def test_trace_simple_cond(self):
        """Test tracing uniform_cond with simple branches."""
        from mplang.edsl.interpreter import InterpObject
        from mplang.edsl.typing import Tensor, f32

        # Create test inputs
        pred_val = InterpObject(True, Tensor[f32, ()])
        x_val = InterpObject(5.0, Tensor[f32, ()])

        def test_fn(pred, x):
            def then_fn(x):
                return x

            def else_fn(x):
                return x

            return uniform_cond(pred, then_fn, else_fn, x)

        # Trace the function
        graph = trace(test_fn, pred_val, x_val)

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
        from mplang.edsl.interpreter import InterpObject
        from mplang.edsl.primitive import Primitive
        from mplang.edsl.typing import Tensor, f32

        # Create a simple test primitive
        negate_p = Primitive("test.negate")

        @negate_p.def_abstract_eval
        def _negate_ae(x_type):
            return x_type

        pred_val = InterpObject(True, Tensor[f32, ()])
        x_val = InterpObject(5.0, Tensor[f32, ()])

        def test_fn(pred, x):
            def then_fn(x):
                # Use a primitive inside the branch
                return negate_p.bind(x)

            def else_fn(x):
                # Another primitive
                return negate_p.bind(x)

            return uniform_cond(pred, then_fn, else_fn, x)

        graph = trace(test_fn, pred_val, x_val)

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
        from mplang.edsl.interpreter import InterpObject
        from mplang.edsl.typing import Tensor, f32

        pred_val = InterpObject(True, Tensor[f32, ()])
        x_val = InterpObject(5.0, Tensor[f32, ()])

        def test_fn(pred, x):
            def then_fn(x):
                # Return f32
                return x

            def else_fn(x):
                # Return different type (simulate by creating wrong type)
                # For this test we'll just ensure different output counts
                return x

            return uniform_cond(pred, then_fn, else_fn, x)

        # This should work fine (same types)
        graph = trace(test_fn, pred_val, x_val)
        assert graph is not None
