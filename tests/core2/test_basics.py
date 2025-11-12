"""Basic tests for mplang.core2 module.

These tests verify the fundamental functionality of the new core2 system.
"""

import pytest

from mplang.core2.context import get_context
from mplang.core2.object import InterpObject, Object, TraceObject
from mplang.core2.tracer import Tracer
from mplang.edsl.graph import Graph
from mplang.edsl.typing import Tensor, f32


class TestExecutionContext:
    """Test ExecutionContext functionality."""

    def test_context_singleton(self):
        """Test that get_context returns the same instance."""
        ctx1 = get_context()
        ctx2 = get_context()
        assert ctx1 is ctx2

    def test_default_mode_is_eager(self):
        """Test that default execution mode is eager."""
        ctx = get_context()
        assert not ctx.is_tracing
        assert ctx.current_tracer is None

    def test_enter_exit_tracing(self):
        """Test entering and exiting tracing mode."""
        ctx = get_context()
        tracer = Tracer()

        # Enter tracing
        ctx.enter_context(tracer)
        assert ctx.is_tracing
        assert ctx.current_tracer is tracer

        # Exit tracing
        ctx.exit_context()
        assert not ctx.is_tracing
        assert ctx.current_tracer is None


class TestInterpObject:
    """Test InterpObject functionality."""

    def test_create_interp_object_with_simple_data(self):
        """Test creating an InterpObject with simple runtime data."""
        import numpy as np

        # Simulate FHE backend: runtime_obj is a local numpy array (could be ciphertext)
        runtime_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        obj = InterpObject(runtime_data, Tensor[f32, (5,)])

        assert obj.type == Tensor[f32, (5,)]
        assert obj.runtime_obj is runtime_data

    def test_interp_object_repr(self):
        """Test InterpObject string representation."""
        # Test with simple data
        runtime_data = "some_runtime_data"
        obj = InterpObject(runtime_data, Tensor[f32, (5,)])

        repr_str = repr(obj)
        assert "InterpObject" in repr_str
        assert "some_runtime_data" in repr_str


class TestTraceObject:
    """Test TraceObject functionality."""

    def test_create_trace_object(self):
        """Test creating a TraceObject."""
        tracer = Tracer()
        graph_value = tracer.graph.add_input("x", Tensor[f32, (10,)])
        trace_obj = TraceObject(graph_value, tracer)

        assert trace_obj.type == Tensor[f32, (10,)]
        assert trace_obj._graph_value is graph_value
        assert trace_obj._tracer is tracer


class TestTracer:
    """Test Tracer functionality."""

    def test_create_tracer(self):
        """Test creating a Tracer."""
        tracer = Tracer()

        assert isinstance(tracer.graph, Graph)
        assert len(tracer._captured_vars) == 0
        assert len(tracer._params) == 0

    def test_make_constant_scalar(self):
        """Test creating constant TraceObject from scalar."""
        tracer = Tracer()
        trace_obj = tracer.make_constant(3.14)

        assert isinstance(trace_obj, TraceObject)
        assert trace_obj.type == Tensor[f32, ()]

    def test_make_constant_array(self):
        """Test creating constant TraceObject from numpy array."""
        import numpy as np

        tracer = Tracer()
        arr = np.array([1.0, 2.0, 3.0])
        trace_obj = tracer.make_constant(arr)

        assert isinstance(trace_obj, TraceObject)
        assert trace_obj.type == Tensor[f32, (3,)]

    def test_promote_interp_to_trace(self):
        """Test promoting InterpObject to TraceObject."""
        import numpy as np

        tracer = Tracer()

        # Create InterpObject with simple runtime data (e.g., FHE backend ciphertext)
        runtime_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        interp_obj = InterpObject(runtime_data, Tensor[f32, (5,)])

        # Promote
        trace_obj = tracer.promote(interp_obj)

        assert isinstance(trace_obj, TraceObject)
        assert trace_obj.type == Tensor[f32, (5,)]
        assert id(interp_obj) in tracer._captured_vars

        # Promote again - should return same graph value
        trace_obj2 = tracer.promote(interp_obj)
        assert trace_obj2._graph_value is trace_obj._graph_value


class TestObjectHierarchy:
    """Test Object hierarchy and polymorphism."""

    def test_object_is_abstract(self):
        """Test that Object is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            Object()  # Should fail - abstract class

    def test_trace_object_is_object(self):
        """Test that TraceObject is a subclass of Object."""
        tracer = Tracer()
        graph_value = tracer.graph.add_input("x", Tensor[f32, ()])
        trace_obj = TraceObject(graph_value, tracer)

        assert isinstance(trace_obj, Object)

    def test_interp_object_is_object(self):
        """Test that InterpObject is a subclass of Object."""
        # Simple runtime data (e.g., FHE backend)
        import numpy as np

        runtime_data = np.array([1.0, 2.0, 3.0])
        interp_obj = InterpObject(runtime_data, Tensor[f32, (3,)])

        assert isinstance(interp_obj, Object)
