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

"""Tests for mplang.edsl.primitive module."""

import numpy as np
import pytest

from mplang.edsl.context import pop_context, push_context
from mplang.edsl.interpreter import InterpObject
from mplang.edsl.primitive import Primitive, add_p, primitive
from mplang.edsl.tracer import TraceObject, Tracer
from mplang.edsl.typing import Tensor, f32


class TestPrimitiveBasics:
    """Test basic Primitive functionality."""

    def test_create_primitive(self):
        """Test creating a Primitive."""
        p = Primitive("test_op")
        assert p.name == "test_op"
        assert p._abstract_eval is None
        assert p._trace is None

    def test_def_abstract_eval(self):
        """Test defining abstract_eval rule."""
        p = Primitive("test_op")

        @p.def_abstract_eval
        def test_abstract(x_type):
            return x_type

        assert p._abstract_eval is test_abstract

    def test_def_trace(self):
        """Test defining trace logic."""
        p = Primitive("test_op")

        @p.def_trace
        def test_trace(x):
            return x

        assert p._trace is test_trace


class TestPrimitiveTraceMode:
    """Test Primitive in trace mode (with Object operands)."""

    def test_bind_in_trace_mode_basic(self):
        """Test primitive.bind() in trace mode with Object operands."""
        # Define a simple add primitive
        my_add_p = Primitive("my_add")

        @my_add_p.def_abstract_eval
        def my_add_abstract(x_type, y_type):
            assert x_type == y_type, "Types must match"
            return x_type

        # Create tracer and InterpObjects
        tracer = Tracer()
        x_data = np.array([1.0, 2.0, 3.0])
        y_data = np.array([4.0, 5.0, 6.0])
        x = InterpObject(x_data, Tensor[f32, (3,)])
        y = InterpObject(y_data, Tensor[f32, (3,)])

        # Enter trace mode
        push_context(tracer)
        try:
            # Call primitive.bind()
            z = my_add_p.bind(x, y)

            # Verify result is TraceObject
            assert isinstance(z, TraceObject)
            assert z.type == Tensor[f32, (3,)]

            # Verify operation was added to graph
            assert len(tracer.graph.operations) == 1
            op = tracer.graph.operations[0]
            assert op.opcode == "my_add"
            assert len(op.inputs) == 2
            assert len(op.outputs) == 1
        finally:
            pop_context()

    def test_bind_in_trace_mode_with_attrs(self):
        """Test primitive.bind() with attrs (kwargs) in trace mode."""
        # Define primitive with attributes
        scaled_add_p = Primitive("scaled_add")

        @scaled_add_p.def_abstract_eval
        def scaled_add_abstract(x_type, y_type, *, scale=1.0):
            # scale is an attribute, not used in type inference
            assert x_type == y_type
            return x_type

        # Create tracer and objects
        tracer = Tracer()
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        push_context(tracer)
        try:
            # Call with attrs
            z = scaled_add_p.bind(x, y, scale=2.5)

            # Verify operation has attrs
            assert len(tracer.graph.operations) == 1
            op = tracer.graph.operations[0]
            assert op.opcode == "scaled_add"
            assert op.attrs == {"scale": 2.5}
            assert isinstance(z, TraceObject)
        finally:
            pop_context()

    def test_bind_rejects_object_in_kwargs(self):
        """Test that primitive.bind() rejects Object in kwargs."""
        my_op_p = Primitive("my_op")

        @my_op_p.def_abstract_eval
        def my_op_abstract(x_type):
            return x_type

        tracer = Tracer()
        x = InterpObject(np.array([1.0]), Tensor[f32, (1,)])
        y = InterpObject(np.array([2.0]), Tensor[f32, (1,)])

        push_context(tracer)
        try:
            # Try to pass Object as kwarg (should fail)
            with pytest.raises(TypeError, match="cannot be an Object"):
                my_op_p.bind(x, other=y)  # y is Object, not allowed in kwargs
        finally:
            pop_context()

    def test_bind_without_abstract_eval_fails(self):
        """Test that bind() fails in trace mode without abstract_eval or trace."""
        p = Primitive("no_abstract")
        # No abstract_eval or trace defined

        tracer = Tracer()
        x = InterpObject(np.array([1.0]), Tensor[f32, (1,)])

        push_context(tracer)
        try:
            with pytest.raises(RuntimeError, match="neither trace nor abstract_eval"):
                p.bind(x)
        finally:
            pop_context()


class TestPrimitiveInterpMode:
    """Test Primitive in interp mode (eager execution)."""

    def test_bind_in_interp_mode_basic(self):
        """Test primitive.bind() in interp mode requires def_trace."""
        # Define primitive with trace
        my_mul_p = Primitive("my_mul")

        @my_mul_p.def_abstract_eval
        def my_mul_abstract(x_type, y_type):
            return x_type

        @my_mul_p.def_trace
        def my_mul_trace(x, y):
            # Execute on runtime objects
            result_data = x.runtime_obj * y.runtime_obj
            return InterpObject(result_data, x.type)

        # Create InterpObjects
        x = InterpObject(np.array([2.0, 3.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([4.0, 5.0]), Tensor[f32, (2,)])

        # Call in eager mode (no tracer)
        z = my_mul_p.bind(x, y)

        # Verify result is InterpObject with correct data
        assert isinstance(z, InterpObject)
        assert z.type == Tensor[f32, (2,)]
        np.testing.assert_array_equal(z.runtime_obj, np.array([8.0, 15.0]))

    def test_bind_in_interp_mode_with_attrs(self):
        """Test primitive.bind() with attrs in interp mode."""
        # Define primitive that uses attrs
        power_p = Primitive("power")

        @power_p.def_abstract_eval
        def power_abstract(x_type, *, exponent=2):
            return x_type

        @power_p.def_trace
        def power_trace(x, *, exponent=2):
            result_data = x.runtime_obj**exponent
            return InterpObject(result_data, x.type)

        # Execute with attrs
        x = InterpObject(np.array([2.0, 3.0]), Tensor[f32, (2,)])
        z = power_p.bind(x, exponent=3)

        # Verify result
        assert isinstance(z, InterpObject)
        np.testing.assert_array_equal(z.runtime_obj, np.array([8.0, 27.0]))

    def test_bind_without_trace_fails(self):
        """Test that bind() fails in interp mode without trace."""
        p = Primitive("no_trace")

        @p.def_abstract_eval
        def abstract(x_type):
            return x_type

        # No trace defined - should fail in interp mode
        x = InterpObject(np.array([1.0]), Tensor[f32, (1,)])

        with pytest.raises(NotImplementedError, match="Graph IR not yet implemented"):
            p.bind(x)


class TestPrimitiveDecorator:
    """Test @primitive decorator."""

    def test_primitive_decorator(self):
        """Test @primitive decorator creates Primitive."""

        @primitive("custom_neg")
        def custom_neg_abstract(x_type):
            return x_type

        # Verify it returns a Primitive
        assert isinstance(custom_neg_abstract, Primitive)
        assert custom_neg_abstract.name == "custom_neg"
        assert custom_neg_abstract._abstract_eval is not None

    def test_primitive_decorator_with_trace(self):
        """Test using @primitive decorator and then adding trace."""

        @primitive("custom_sqrt")
        def custom_sqrt_abstract(x_type):
            return x_type

        @custom_sqrt_abstract.def_trace
        def custom_sqrt_trace(x):
            result_data = np.sqrt(x.runtime_obj)
            return InterpObject(result_data, x.type)

        # Test in interp mode
        x = InterpObject(np.array([4.0, 9.0, 16.0]), Tensor[f32, (3,)])
        z = custom_sqrt_abstract.bind(x)

        assert isinstance(z, InterpObject)
        np.testing.assert_array_almost_equal(z.runtime_obj, np.array([2.0, 3.0, 4.0]))


class TestPredefinedPrimitives:
    """Test pre-defined primitives."""

    def test_add_p_exists(self):
        """Test that add_p is pre-defined."""
        assert isinstance(add_p, Primitive)
        assert add_p.name == "add"

    def test_predefined_primitives_have_abstract_eval(self):
        """Test that pre-defined primitives have abstract_eval defined."""
        # add_p has abstract_eval but not trace (relies on Graph IR execution)
        assert add_p._abstract_eval is not None
        # No def_trace needed - works via default Graph IR path

    def test_add_p_works_in_trace_mode(self):
        """Test that add_p works correctly in trace mode."""
        from mplang.edsl.tracer import Tracer

        tracer = Tracer()
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        push_context(tracer)
        try:
            result = add_p.bind(x, y)
            assert isinstance(result, TraceObject)
            assert len(tracer.graph.operations) == 1
            assert tracer.graph.operations[0].opcode == "add"
        finally:
            pop_context()

    def test_add_p_works_in_interp_mode(self):
        """Test that add_p raises NotImplementedError in interp mode (Graph IR not ready)."""
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        # add_p doesn't have def_trace, so execution via Graph IR is not yet implemented
        with pytest.raises(NotImplementedError, match="Graph IR not yet implemented"):
            add_p.bind(x, y)


class TestPrimitiveComplexScenarios:
    """Test complex scenarios combining trace and interp."""

    def test_multiple_operations_in_trace(self):
        """Test multiple primitive operations in one trace."""
        # Define two primitives
        add_p = Primitive("add")
        mul_p = Primitive("mul")

        @add_p.def_abstract_eval
        def add_abstract(x_type, y_type):
            return x_type

        @mul_p.def_abstract_eval
        def mul_abstract(x_type, y_type):
            return x_type

        # Trace a function using both
        tracer = Tracer()
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])
        z = InterpObject(np.array([5.0, 6.0]), Tensor[f32, (2,)])

        push_context(tracer)
        try:
            # (x + y) * z
            temp = add_p.bind(x, y)
            result = mul_p.bind(temp, z)

            # Verify two operations in graph
            assert len(tracer.graph.operations) == 2
            assert tracer.graph.operations[0].opcode == "add"
            assert tracer.graph.operations[1].opcode == "mul"
            assert isinstance(result, TraceObject)
        finally:
            pop_context()

    def test_primitive_with_multiple_attrs(self):
        """Test primitive with multiple attributes."""
        conv_p = Primitive("conv")

        @conv_p.def_abstract_eval
        def conv_abstract(x_type, kernel_type, *, stride=1, padding=0, groups=1):
            # Simplified type inference
            return x_type

        @conv_p.def_trace
        def conv_trace(x, kernel, *, stride=1, padding=0, groups=1):
            # Dummy implementation - just return x as placeholder
            # (In real code, def_trace would use primitives to build the graph)
            return x

        # Test in trace mode
        tracer = Tracer()
        x = InterpObject(np.random.randn(1, 3, 32, 32), Tensor[f32, (1, 3, 32, 32)])
        kernel = InterpObject(np.random.randn(16, 3, 3, 3), Tensor[f32, (16, 3, 3, 3)])

        push_context(tracer)
        try:
            result = conv_p.bind(x, kernel, stride=2, padding=1, groups=1)

            # When using def_trace, attrs include morph info
            op = tracer.graph.operations[0]
            assert "_in_morph" in op.attrs  # Morph info is stored
            assert "_out_tree" in op.attrs
            assert isinstance(result, TraceObject)
        finally:
            pop_context()

        # Test in interp mode
        result2 = conv_p.bind(x, kernel, stride=2, padding=1, groups=1)
        assert isinstance(result2, InterpObject)

    def test_trace_primitive_directly(self):
        """Test tracing a Primitive directly (not a lambda)."""
        from mplang.edsl.tracer import trace

        # Define a primitive
        my_add_p = Primitive("my_add")

        @my_add_p.def_abstract_eval
        def my_add_abstract(x_type, y_type):
            return x_type

        # Trace the primitive directly
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        graph = trace(my_add_p, x, y)

        # Verify graph structure
        assert len(graph.operations) == 1
        assert graph.operations[0].opcode == "my_add"
        assert len(graph.inputs) == 2
        assert len(graph.outputs) == 1

    def test_add_operator_uses_primitive(self):
        """Test that __add__ operator uses add_p primitive."""
        from mplang.edsl.tracer import Tracer

        tracer = Tracer()
        x_interp = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y_interp = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        push_context(tracer)
        try:
            # Use __add__ operator
            result = x_interp + y_interp

            # Should create TraceObject via add_p primitive
            assert isinstance(result, TraceObject)
            assert len(tracer.graph.operations) == 1
            assert tracer.graph.operations[0].opcode == "add"
        finally:
            pop_context()

        # Eager mode not supported without backend
        # (add_p has no def_trace, only def_abstract_eval)
