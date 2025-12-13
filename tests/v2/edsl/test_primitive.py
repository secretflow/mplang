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

from mplang.v2.edsl.context import pop_context, push_context
from mplang.v2.edsl.primitive import Primitive, primitive
from mplang.v2.edsl.tracer import TraceObject, Tracer
from mplang.v2.edsl.typing import Tensor, f32
from mplang.v2.runtime.interpreter import InterpObject


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


# TestPredefinedPrimitives removed - primitives should be defined in dialects/backends


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

        # Test in trace mode with def_abstract_eval
        tracer = Tracer()
        x = InterpObject(np.random.randn(1, 3, 32, 32), Tensor[f32, (1, 3, 32, 32)])
        kernel = InterpObject(np.random.randn(16, 3, 3, 3), Tensor[f32, (16, 3, 3, 3)])

        push_context(tracer)
        try:
            result = conv_p.bind(x, kernel, stride=2, padding=1, groups=1)

            # def_abstract_eval mode: Tracer builds the operation
            assert len(tracer.graph.operations) == 1
            op = tracer.graph.operations[0]
            assert op.opcode == "conv"
            # Attributes are stored
            assert op.attrs["stride"] == 2
            assert op.attrs["padding"] == 1
            assert op.attrs["groups"] == 1
            assert isinstance(result, TraceObject)
        finally:
            pop_context()

    def test_trace_primitive_directly(self):
        """Test tracing a Primitive directly (not a lambda)."""
        from mplang.v2.edsl.tracer import trace

        # Define a primitive
        my_add_p = Primitive("my_add")

        @my_add_p.def_abstract_eval
        def my_add_abstract(x_type, y_type):
            return x_type

        # Trace the primitive directly
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        traced = trace(my_add_p, x, y)

        # Verify graph structure
        assert len(traced.graph.operations) == 1
        assert traced.graph.operations[0].opcode == "my_add"
        assert len(traced.graph.inputs) == 2
        assert len(traced.graph.outputs) == 1

    # test_add_operator_uses_primitive removed - operator overloading moved to future dispatch module
