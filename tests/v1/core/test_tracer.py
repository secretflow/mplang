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
Tests for trace module.
"""

from unittest.mock import Mock

import pytest

from mplang.v1.core.cluster import ClusterSpec
from mplang.v1.core.dtypes import FLOAT32, INT32
from mplang.v1.core.expr.ast import VariableExpr
from mplang.v1.core.mask import Mask
from mplang.v1.core.mpobject import MPContext, MPObject
from mplang.v1.core.mptype import MPType
from mplang.v1.core.tensor import TensorType
from mplang.v1.core.tracer import (
    TraceContext,
    TracedFunction,
    TraceVar,
    VarNamer,
    trace,
)


class MockMPObject(MPObject):
    """Mock MPObject for testing."""

    def __init__(
        self, mptype: MPType, name: str = "mock", ctx: MPContext | None = None
    ):
        self._mptype = mptype
        self._name = name
        self._ctx = ctx

    @property
    def mptype(self) -> MPType:
        return self._mptype

    @property
    def ctx(self) -> MPContext:
        if self._ctx is None:
            # Return a mock context for testing
            mock_ctx = Mock(spec=MPContext)
            mock_ctx.psize.return_value = 2
            mock_ctx.attrs.return_value = {}
            return mock_ctx
        return self._ctx


@pytest.fixture
def tensor_info_float():
    """Float tensor info for testing."""
    return TensorType(FLOAT32, (2, 3))


@pytest.fixture
def tensor_info_int():
    """Int tensor info for testing."""
    return TensorType(INT32, (4,))


@pytest.fixture
def mask_2p():
    """Mask for 2-party computation."""
    return Mask(3)  # 0b11


@pytest.fixture
def cluster_spec_2p():
    """Create a 2-party cluster spec for testing."""
    return ClusterSpec.simple(2)


@pytest.fixture
def trace_context(mask_2p, cluster_spec_2p):
    """Create a trace context for testing."""
    return TraceContext(cluster_spec_2p, mask=mask_2p)


@pytest.fixture
def mock_mpobject(tensor_info_float, mask_2p):
    """Create a mock MPObject for testing."""
    mptype = MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p)
    return MockMPObject(mptype, "test_obj")


class TestVarNamer:
    """Test the VarNamer utility class."""

    def test_default_prefix(self):
        """Test VarNamer with default prefix."""
        namer = VarNamer()
        assert namer.next_name() == "$0"
        assert namer.next_name() == "$1"
        assert namer.next_name() == "$2"

    def test_custom_prefix(self):
        """Test VarNamer with custom prefix."""
        namer = VarNamer("var")
        assert namer.next_name() == "var0"
        assert namer.next_name() == "var1"

    def test_sequential_naming(self):
        """Test that names are generated sequentially."""
        namer = VarNamer("x")
        names = [namer.next_name() for _ in range(10)]
        expected = [f"x{i}" for i in range(10)]
        assert names == expected


class TestTraceContext:
    """Test the TraceContext class."""

    def test_initialization(self, mask_2p, cluster_spec_2p):
        """Test TraceContext initialization."""
        ctx = TraceContext(cluster_spec_2p, mask=mask_2p)

        assert ctx.world_size() == 2
        assert ctx.mask == mask_2p
        assert ctx.cluster_spec == cluster_spec_2p
        assert ctx._captures == {}

    def test_initialization_default_mask(self, cluster_spec_2p):
        """Test TraceContext initialization with default mask."""
        ctx = TraceContext(cluster_spec_2p)

        assert ctx.world_size() == 2
        assert ctx.mask == Mask.all(2)  # Should default to all parties

    def test_name_generation(self, trace_context):
        """Test unique name generation."""
        name1 = trace_context._gen_name()
        name2 = trace_context._gen_name()

        assert name1 != name2
        assert name1.startswith("$")
        assert name2.startswith("$")

    def test_capture_new_object(self, trace_context, mock_mpobject):
        """Test capturing a new MPObject."""
        captured = trace_context.capture(mock_mpobject)

        assert isinstance(captured, TraceVar)
        assert captured.mptype == mock_mpobject.mptype
        assert captured._ctx == trace_context

        # Should return the same TraceVar for the same object
        captured2 = trace_context.capture(mock_mpobject)
        assert captured is captured2

    def test_capture_different_objects(
        self, trace_context, tensor_info_float, tensor_info_int, mask_2p
    ):
        """Test capturing different MPObjects."""
        obj1 = MockMPObject(
            MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p),
            "obj1",
        )
        obj2 = MockMPObject(
            MPType.tensor(tensor_info_int.dtype, tensor_info_int.shape, mask_2p), "obj2"
        )

        captured1 = trace_context.capture(obj1)
        captured2 = trace_context.capture(obj2)

        assert captured1 is not captured2
        assert captured1.mptype != captured2.mptype

    def test_get_captures_empty(self, trace_context):
        """Test get_captures when no objects are captured."""
        captures = trace_context.get_captures()
        assert captures == {}

    def test_get_captures_with_objects(self, trace_context, mock_mpobject):
        """Test get_captures with captured objects."""
        captured = trace_context.capture(mock_mpobject)
        captures = trace_context.get_captures()

        assert len(captures) == 1
        assert mock_mpobject in captures.keys()
        assert captured in captures.values()


class TestTraceVar:
    """Test the TraceVar class."""

    def test_initialization_single_output(
        self, trace_context, tensor_info_float, mask_2p
    ):
        """Test TraceVar initialization with single-output expression."""
        expr = VariableExpr(
            "test_var",
            MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p),
        )
        trace_var = TraceVar(trace_context, expr)

        assert trace_var._ctx == trace_context
        assert trace_var._expr == expr
        assert trace_var.mptype == expr.mptype
        assert trace_var.ctx == trace_context
        assert trace_var.expr == expr

    def test_initialization_multi_output_fails(
        self, trace_context, tensor_info_float, mask_2p
    ):
        """Test TraceVar initialization fails with multi-output expression."""
        # Create a mock expression with multiple outputs
        expr = Mock()
        expr.mptypes = [
            MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p),
            MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p),
        ]

        with pytest.raises(
            ValueError, match="TraceVar requires single-output expression"
        ):
            TraceVar(trace_context, expr)

    def test_repr(self, trace_context, tensor_info_float, mask_2p):
        """Test TraceVar string representation."""
        expr = VariableExpr(
            "test_var",
            MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p),
        )
        trace_var = TraceVar(trace_context, expr)

        # Since name property raises NotImplementedError, repr should handle it
        repr_str = repr(trace_var)
        assert "TraceVar" in repr_str


class TestTracedFunction:
    """Test the TracedFunction class."""

    def test_initialization(self, trace_context, mock_mpobject):
        """Test TracedFunction initialization."""
        from mplang.v1.core.expr.ast import VariableExpr
        from mplang.v1.utils.func_utils import var_morph

        # Create mock data for TracedFunction
        func_name = "test_func"

        # Create input variables
        in_vars = [
            TraceVar(trace_context, VariableExpr("param1", mock_mpobject.mptype))
        ]

        # Create mock structures using proper var_morph
        test_input = ([mock_mpobject],)
        _, in_imms, in_struct = var_morph(
            test_input, lambda x: isinstance(x, MockMPObject)
        )

        test_output = [mock_mpobject]
        _, out_imms, out_struct = var_morph(
            test_output, lambda x: isinstance(x, MockMPObject)
        )
        out_vars = [TraceVar(trace_context, VariableExpr("out1", mock_mpobject.mptype))]

        capture_map = {}

        traced_fn = TracedFunction(
            func_name=func_name,
            in_vars=in_vars,
            in_struct=in_struct,
            in_imms=in_imms,
            capture_map=capture_map,
            out_vars=out_vars,
            out_struct=out_struct,
            out_imms=out_imms,
        )

        assert traced_fn.func_name == func_name
        assert traced_fn.in_vars == in_vars
        assert traced_fn.capture_map == capture_map
        assert traced_fn.in_struct == in_struct
        assert traced_fn.out_struct == out_struct

    def test_in_names(self, trace_context, mock_mpobject):
        """Test in_names method."""
        from mplang.v1.core.expr.ast import VariableExpr
        from mplang.v1.utils.func_utils import var_morph

        in_vars = [
            TraceVar(trace_context, VariableExpr("param1", mock_mpobject.mptype)),
            TraceVar(trace_context, VariableExpr("param2", mock_mpobject.mptype)),
        ]

        test_input = ([mock_mpobject, mock_mpobject],)
        _, in_imms, in_struct = var_morph(
            test_input, lambda x: isinstance(x, MockMPObject)
        )

        _, out_imms, out_struct = var_morph([], lambda x: False)

        traced_fn = TracedFunction(
            func_name="test",
            in_vars=in_vars,
            in_struct=in_struct,
            in_imms=in_imms,
            capture_map={},
            out_vars=[],
            out_struct=out_struct,
            out_imms=out_imms,
        )

        names = traced_fn.in_names()
        assert names == ["param1", "param2"]

    def test_capture_names(self, trace_context, mock_mpobject):
        """Test capture_names method."""
        from mplang.v1.core.expr.ast import VariableExpr
        from mplang.v1.utils.func_utils import var_morph

        captured_var = TraceVar(
            trace_context, VariableExpr("captured_var", mock_mpobject.mptype)
        )
        capture_map = {mock_mpobject: captured_var}

        _, in_imms, in_struct = var_morph([], lambda x: False)
        _, out_imms, out_struct = var_morph([], lambda x: False)

        traced_fn = TracedFunction(
            func_name="test",
            in_vars=[],
            in_struct=in_struct,
            in_imms=in_imms,
            capture_map=capture_map,
            out_vars=[],
            out_struct=out_struct,
            out_imms=out_imms,
        )

        capture_names = traced_fn.capture_names()
        assert capture_names == ["captured_var"]

    def test_make_expr_single_output(self, trace_context, mock_mpobject):
        """Test make_expr method with single output."""
        from mplang.v1.core.expr.ast import FuncDefExpr, VariableExpr
        from mplang.v1.utils.func_utils import var_morph

        in_var = TraceVar(trace_context, VariableExpr("param1", mock_mpobject.mptype))
        out_var = TraceVar(trace_context, VariableExpr("out1", mock_mpobject.mptype))

        _, in_imms, in_struct = var_morph([], lambda x: False)
        _, out_imms, out_struct = var_morph([], lambda x: False)

        traced_fn = TracedFunction(
            func_name="test",
            in_vars=[in_var],
            in_struct=in_struct,
            in_imms=in_imms,
            capture_map={},
            out_vars=[out_var],
            out_struct=out_struct,
            out_imms=out_imms,
        )

        func_expr = traced_fn.make_expr()
        assert isinstance(func_expr, FuncDefExpr)
        assert func_expr.params == ["param1"]

    def test_make_expr_no_outputs(self, trace_context, mock_mpobject):
        """Test make_expr method with no outputs."""
        from mplang.v1.core.expr.ast import FuncDefExpr, TupleExpr, VariableExpr
        from mplang.v1.utils.func_utils import var_morph

        in_var = TraceVar(trace_context, VariableExpr("param1", mock_mpobject.mptype))

        _, in_imms, in_struct = var_morph([], lambda x: False)
        _, out_imms, out_struct = var_morph([], lambda x: False)

        traced_fn = TracedFunction(
            func_name="test",
            in_vars=[in_var],
            in_struct=in_struct,
            in_imms=in_imms,
            capture_map={},
            out_vars=[],
            out_struct=out_struct,
            out_imms=out_imms,
        )

        func_expr = traced_fn.make_expr()
        # Should always return a FuncDefExpr, even for no outputs
        assert isinstance(func_expr, FuncDefExpr)
        # Body should be an empty tuple for no outputs
        assert isinstance(func_expr.body, TupleExpr)
        assert len(func_expr.body.args) == 0


class TestTrace:
    """Test the trace function."""

    def test_trace_simple_function(self, trace_context, mock_mpobject):
        """Test tracing a simple function with MPObject inputs."""

        def simple_func(x):
            return x

        traced_fn = trace(trace_context, simple_func, mock_mpobject)

        assert isinstance(traced_fn, TracedFunction)
        assert traced_fn.func_name == "simple_func"
        assert len(traced_fn.in_vars) == 1
        assert len(traced_fn.out_vars) == 1
        assert traced_fn.out_vars[0].mptype == mock_mpobject.mptype

    def test_trace_function_with_immediate_values(self, trace_context, mock_mpobject):
        """Test tracing a function with both MPObjects and immediate values."""

        def mixed_func(x, imm_val):
            # In a real implementation, this would perform some operation
            # For testing, we just return the MPObject
            return x

        traced_fn = trace(trace_context, mixed_func, mock_mpobject, 42)

        assert isinstance(traced_fn, TracedFunction)
        assert len(traced_fn.in_vars) == 1  # Only MPObject parameters are counted
        assert len(traced_fn.in_imms) > 0  # Immediate values are stored separately

    def test_trace_function_with_kwargs(self, trace_context, mock_mpobject):
        """Test tracing a function with keyword arguments."""

        def kwargs_func(x, y=None):
            return x

        traced_fn = trace(trace_context, kwargs_func, mock_mpobject, y=100)

        assert isinstance(traced_fn, TracedFunction)
        assert traced_fn.func_name == "kwargs_func"

    def test_trace_function_no_outputs(self, trace_context):
        """Test tracing a function that returns no MPObjects."""

        def no_output_func():
            return 42  # Immediate value, not an MPObject

        traced_fn = trace(trace_context, no_output_func)

        assert isinstance(traced_fn, TracedFunction)
        func_expr = traced_fn.make_expr()
        # Should always return a FuncDefExpr, even for no MPObject outputs
        from mplang.v1.core.expr.ast import FuncDefExpr, TupleExpr

        assert isinstance(func_expr, FuncDefExpr)
        # Body should be an empty tuple for no MPObject outputs
        assert isinstance(func_expr.body, TupleExpr)
        assert len(func_expr.body.args) == 0

    def test_trace_function_multiple_outputs(
        self, trace_context, tensor_info_float, mask_2p
    ):
        """Test tracing a function with multiple MPObject outputs."""
        # Create multiple mock objects
        obj1 = MockMPObject(
            MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p),
            "obj1",
        )
        obj2 = MockMPObject(
            MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p),
            "obj2",
        )

        def multi_output_func(x, y):
            # In a real scenario, these would be TraceVars created by operations
            # For testing, we need to create TraceVars in the trace context
            var1 = trace_context.capture(obj1)
            var2 = trace_context.capture(obj2)
            return (var1, var2)

        traced_fn = trace(trace_context, multi_output_func, obj1, obj2)

        assert isinstance(traced_fn, TracedFunction)
        func_expr = traced_fn.make_expr()
        assert func_expr is not None  # Should have function expression
        assert len(traced_fn.out_vars) == 2

    def test_trace_invalid_tracer_type(self, mock_mpobject):
        """Test trace function with invalid tracer type."""

        def simple_func(x):
            return x

        # Use type: ignore to suppress the type checker for this intentional error test
        with pytest.raises(AssertionError, match="Expect TraceContext"):
            trace("not_a_tracer", simple_func, mock_mpobject)  # type: ignore

    def test_trace_function_with_captures(
        self, trace_context, tensor_info_float, mask_2p
    ):
        """Test tracing a function that captures external variables."""
        # Create an external variable
        external_obj = MockMPObject(
            MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p),
            "external",
        )

        def capturing_func(x):
            # This function would capture the external_obj in a real scenario
            # For testing, we simulate this by manually capturing
            captured = trace_context.capture(external_obj)
            return captured

        input_obj = MockMPObject(
            MPType.tensor(tensor_info_float.dtype, tensor_info_float.shape, mask_2p),
            "input",
        )
        traced_fn = trace(trace_context, capturing_func, input_obj)

        assert isinstance(traced_fn, TracedFunction)
        assert len(traced_fn.capture_map) > 0  # Should have captured variables
        assert external_obj in traced_fn.capture_map.keys()

    def test_trace_preserves_input_structure(self, trace_context, mock_mpobject):
        """Test that trace preserves complex input structures."""

        def structured_func(args_tuple, kwargs_dict):
            x, _y = args_tuple
            kwargs_dict["key"]
            return x  # Return first argument for simplicity

        args = (mock_mpobject, 42)
        kwargs = {"key": "value"}

        traced_fn = trace(trace_context, structured_func, args, kwargs)

        assert isinstance(traced_fn, TracedFunction)
        # The structure should be preserved in in_struct
        assert traced_fn.in_struct is not None
        assert traced_fn.in_imms is not None
