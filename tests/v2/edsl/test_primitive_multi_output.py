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

"""Tests for multi-output primitives and def_trace."""

import numpy as np

from mplang.v2.edsl.context import pop_context, push_context
from mplang.v2.edsl.primitive import Primitive
from mplang.v2.edsl.tracer import Tracer
from mplang.v2.edsl.typing import Tensor, f32
from mplang.v2.runtime.interpreter import InterpObject

# Test helper primitives (for testing purposes only)
add_p = Primitive("add")


@add_p.def_abstract_eval
def _add_abstract(x_type, y_type):
    return x_type


mul_p = Primitive("mul")


@mul_p.def_abstract_eval
def _mul_abstract(x_type, y_type):
    return x_type


class TestMultiOutputPrimitives:
    """Test primitives with multiple outputs."""

    def test_def_abstract_eval_multi_output_positional(self):
        """Test def_abstract_eval with multi-output (positional style)."""
        split_p = Primitive("split")

        @split_p.def_abstract_eval
        def split_abstract(x_type, *, num_splits: int):
            # Return list of types
            return [x_type] * num_splits

        # Trace mode
        tracer = Tracer()
        x = InterpObject(np.array([1.0, 2.0, 3.0]), Tensor[f32, (3,)])

        push_context(tracer)
        try:
            results = split_p.bind(x, num_splits=3)

            # Should return list of TraceObjects
            assert isinstance(results, list)
            assert len(results) == 3
            for r in results:
                from mplang.v2.edsl.tracer import TraceObject

                assert isinstance(r, TraceObject)
                assert r.type == Tensor[f32, (3,)]

            # Verify graph
            assert len(tracer.graph.operations) == 1
            op = tracer.graph.operations[0]
            assert op.opcode == "split"
            assert len(op.outputs) == 3
            assert op.attrs == {"num_splits": 3}
        finally:
            pop_context()

    def test_def_abstract_eval_multi_output_flat(self):
        """Test def_abstract_eval with flat signature."""
        concat_p = Primitive("concat")

        @concat_p.def_abstract_eval
        def concat_abstract(in_types: list, **attrs):
            # Flat signature: (in_types, attrs) -> list[BaseType]
            _ = attrs.get("axis", 0)  # unused in this simplified example
            # Simplified: just return first input type
            return [in_types[0]]

        # Trace mode
        tracer = Tracer()
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        push_context(tracer)
        try:
            result = concat_p.bind(x, y, axis=0)

            # Single output (list of 1)
            from mplang.v2.edsl.tracer import TraceObject

            assert isinstance(result, TraceObject)
            assert result.type == Tensor[f32, (2,)]

            # Verify graph
            assert len(tracer.graph.operations) == 1
            op = tracer.graph.operations[0]
            assert op.opcode == "concat"
            assert len(op.inputs) == 2
            assert op.attrs == {"axis": 0}
        finally:
            pop_context()


class TestDefTrace:
    """Test def_trace for custom trace logic."""

    def test_def_trace_basic(self):
        """Test basic def_trace usage."""
        custom_p = Primitive("custom_op")

        @custom_p.def_trace
        def custom_trace(x, y, *, factor: int):
            # Custom trace: call other primitives
            temp = add_p.bind(x, y)
            # Note: mul with scalar not implemented, simplified here
            return temp

        # Trace mode
        tracer = Tracer()
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        push_context(tracer)
        try:
            result = custom_p.bind(x, y, factor=2)

            from mplang.v2.edsl.tracer import TraceObject

            assert isinstance(result, TraceObject)

            # def_trace mode: only operations from called primitives appear
            # custom_p calls add_p.bind(), so only "add" operation is in graph
            opcodes = [op.opcode for op in tracer.graph.operations]
            assert "add" in opcodes
            assert "custom_op" not in opcodes  # def_trace doesn't create its own op
        finally:
            pop_context()

    def test_def_trace_with_pytree_output(self):
        """Test def_trace with PyTree output."""
        pytree_p = Primitive("pytree_op")

        @pytree_p.def_trace
        def pytree_trace(x, y):
            sum_result = add_p.bind(x, y)
            prod_result = mul_p.bind(x, y)

            # Return dict (PyTree)
            return {"sum": sum_result, "prod": prod_result}

        # Trace mode
        tracer = Tracer()
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        push_context(tracer)
        try:
            result = pytree_p.bind(x, y)

            # Result should be a dict with TraceObjects
            assert isinstance(result, dict)
            assert "sum" in result
            assert "prod" in result

            from mplang.v2.edsl.tracer import TraceObject

            assert isinstance(result["sum"], TraceObject)
            assert isinstance(result["prod"], TraceObject)

            # def_trace mode: only operations from called primitives appear
            opcodes = [op.opcode for op in tracer.graph.operations]
            assert "add" in opcodes
            assert "mul" in opcodes
            assert "pytree_op" not in opcodes  # def_trace doesn't create its own op
        finally:
            pop_context()

    def test_def_trace_with_mixed_args(self):
        """Test def_trace with mixed Object and constant args."""
        mixed_p = Primitive("mixed_op")

        @mixed_p.def_trace
        def mixed_trace(fn, x, const_val, *, k):
            # fn is a callable, x is Object, const_val is constant
            # This demonstrates mixing args/kwargs
            # Simplified: just add x to itself
            return add_p.bind(x, x)

        # Trace mode
        tracer = Tracer()
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])

        def dummy_fn(a, b):
            return a + b

        push_context(tracer)
        try:
            result = mixed_p.bind(dummy_fn, x, 42, k=3.14)

            from mplang.v2.edsl.tracer import TraceObject

            assert isinstance(result, TraceObject)

            # def_trace mode: only operations from called primitives appear
            opcodes = [op.opcode for op in tracer.graph.operations]
            assert "add" in opcodes
            assert "mixed_op" not in opcodes  # def_trace doesn't create its own op
        finally:
            pop_context()

    def test_def_trace_eager_mode(self):
        """Test def_trace in eager execution mode."""
        eager_p = Primitive("eager_op")

        @eager_p.def_trace
        def eager_trace(x, y):
            # In eager mode, this works directly
            return add_p.bind(x, y)

        # Eager mode (no tracer) - will fail since add_p doesn't work in eager
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        # Should fail because add_p.bind calls add_p._trace which raises NotImplementedError
        import pytest

        with pytest.raises(NotImplementedError):
            eager_p.bind(x, y)


class TestBackwardCompatibility:
    """Test that existing code still works."""

    def test_single_output_trace_mode(self):
        """Test that single-output primitives work in trace mode."""
        simple_p = Primitive("simple_add")

        @simple_p.def_abstract_eval
        def simple_abstract(x_type, y_type):
            return x_type

        # Trace mode
        tracer = Tracer()
        x = InterpObject(np.array([1.0]), Tensor[f32, (1,)])
        y = InterpObject(np.array([2.0]), Tensor[f32, (1,)])

        push_context(tracer)
        try:
            result = simple_p.bind(x, y)

            from mplang.v2.edsl.tracer import TraceObject

            # Should return single TraceObject (not list)
            assert isinstance(result, TraceObject)
            assert not isinstance(result, list)
        finally:
            pop_context()
