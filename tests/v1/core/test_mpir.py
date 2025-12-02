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


import numpy as np
import pytest

from mplang.v1.core.dtypes import FLOAT32, INT32, DType
from mplang.v1.core.expr import Expr
from mplang.v1.core.expr.ast import (
    AccessExpr,
    CallExpr,
    CondExpr,
    ConvExpr,
    EvalExpr,
    FuncDefExpr,
    ShflExpr,
    ShflSExpr,
    TupleExpr,
    VariableExpr,
    WhileExpr,
)
from mplang.v1.core.mask import Mask
from mplang.v1.core.mpir import IrReader, IrWriter
from mplang.v1.core.mptype import MPType
from mplang.v1.core.pfunc import PFunction
from mplang.v1.core.tensor import TensorType
from mplang.v1.protos.v1alpha1 import mpir_pb2


class TestBasicExpressions:
    """Test basic expression serialization/deserialization."""

    def test_tuple_expr_roundtrip(self):
        """Test TupleExpr roundtrip."""
        # Use VariableExpr instead of removed expressions
        var1 = VariableExpr("x", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        var2 = VariableExpr("y", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))

        original = TupleExpr([var1, var2])

        writer = IrWriter()
        proto = writer.dumps(original)

        reader = IrReader()
        result = reader.loads(proto)

        assert isinstance(result, TupleExpr)
        assert len(result.args) == 2

    def test_access_expr_roundtrip(self):
        """Test AccessExpr roundtrip."""
        var1 = VariableExpr("x", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        var2 = VariableExpr("y", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        tuple_expr = TupleExpr([var1, var2])

        original = AccessExpr(tuple_expr, 1)

        writer = IrWriter()
        proto = writer.dumps(original)

        reader = IrReader()
        result = reader.loads(proto)

        assert isinstance(result, AccessExpr)
        assert result.index == 1

    def test_variable_expr_roundtrip(self):
        """Test VariableExpr roundtrip."""
        mptype = MPType.tensor(FLOAT32, (3,), pmask=Mask(7))
        original = VariableExpr("test_var", mptype)

        writer = IrWriter()
        proto = writer.dumps(original)

        reader = IrReader()
        result = reader.loads(proto)

        assert isinstance(result, VariableExpr)
        assert result.name == "test_var"


class TestFunctionExpressions:
    """Test function-related expressions (basic tests only)."""

    def test_eval_expr_serialization(self):
        """Test EvalExpr serialization (write only)."""
        var_expr = VariableExpr("x", MPType.tensor(FLOAT32, (3,), pmask=Mask(7)))

        pfunc = PFunction(
            fn_type="builtin",
            ins_info=[TensorType(FLOAT32, (3,)), TensorType(FLOAT32, (3,))],
            outs_info=[TensorType(FLOAT32, (3,))],
            fn_name="add",
        )

        original = EvalExpr(pfunc, [var_expr, var_expr], rmask=None)

        writer = IrWriter()
        proto = writer.dumps(original)

        # Just verify serialization works
        assert len(proto.nodes) >= 2  # At least const + eval nodes
        eval_node = None
        for node in proto.nodes:
            if node.op_type == "eval":
                eval_node = node
                break
        assert eval_node is not None
        assert "pfunc" in eval_node.attrs

    def test_func_def_expr_serialization(self):
        """Test FuncDefExpr serialization (write only)."""
        body = VariableExpr("result", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))

        params = ["x", "y"]
        original = FuncDefExpr(params, body)

        writer = IrWriter()
        proto = writer.dumps(original)

        # Just verify serialization works
        assert len(proto.nodes) >= 2  # At least body + func_def nodes
        func_node = None
        for node in proto.nodes:
            if node.op_type == "func_def":
                func_node = node
                break
        assert func_node is not None
        assert "params" in func_node.attrs


class TestUtilityFunctions:
    """Test utility functions in mpir module."""

    def test_dtype_to_proto_conversion(self):
        """Test dtype to protobuf conversion."""
        from mplang.v1.core.mpir import dtype_to_proto

        # Test DType conversion
        dtype = DType.from_numpy(np.float32)
        result = dtype_to_proto(dtype)
        assert result == mpir_pb2.DataType.F32

        # Test numpy dtype conversion
        result = dtype_to_proto(np.float32)
        assert result == mpir_pb2.DataType.F32

        # Test unsupported dtype
        with pytest.raises(ValueError, match="Invalid dtype"):
            dtype_to_proto("invalid_dtype")

    def test_proto_to_dtype_conversion(self):
        """Test protobuf to dtype conversion."""
        from mplang.v1.core.mpir import proto_to_dtype

        # Test basic conversion
        result = proto_to_dtype(mpir_pb2.DataType.F32)
        assert result.name == "float32"

        # Test unsupported enum
        with pytest.raises(ValueError, match="Unsupported dtype enum"):
            proto_to_dtype(999)  # Invalid enum value

    def test_attr_to_proto_conversions(self):
        """Test attribute to protobuf conversions."""
        from mplang.v1.core.mpir import attr_to_proto

        # Test int
        attr = attr_to_proto(42)
        assert attr.type == mpir_pb2.AttrProto.INT
        assert attr.i == 42

        # Test float
        attr = attr_to_proto(3.14)
        assert attr.type == mpir_pb2.AttrProto.FLOAT
        assert abs(attr.f - 3.14) < 1e-6  # Use approximate comparison for float

        # Test string
        attr = attr_to_proto("test")
        assert attr.type == mpir_pb2.AttrProto.STRING
        assert attr.s == "test"

        # Test bytes
        test_bytes = b"test_bytes"
        attr = attr_to_proto(test_bytes)
        assert attr.type == mpir_pb2.AttrProto.BYTES
        assert attr.raw_bytes == test_bytes

        # Test list of ints
        attr = attr_to_proto([1, 2, 3])
        assert attr.type == mpir_pb2.AttrProto.INTS
        assert list(attr.ints) == [1, 2, 3]

        # Test list of floats
        attr = attr_to_proto([1.0, 2.0, 3.0])
        assert attr.type == mpir_pb2.AttrProto.FLOATS
        assert list(attr.floats) == [1.0, 2.0, 3.0]

        # Test list of strings
        attr = attr_to_proto(["a", "b", "c"])
        assert attr.type == mpir_pb2.AttrProto.STRINGS
        assert list(attr.strs) == ["a", "b", "c"]

        # Test PFunction
        pfunc = PFunction(
            fn_type="builtin",
            ins_info=[],
            outs_info=[],
            fn_name="add",
        )
        attr = attr_to_proto(pfunc)
        assert attr.type == mpir_pb2.AttrProto.FUNCTION
        assert attr.func.type == "builtin"
        assert attr.func.name == "add"

        # Test unsupported type
        with pytest.raises(TypeError, match="Unsupported attribute type"):
            attr_to_proto(object())


class TestComplexExpressions:
    """Test complex expression serialization/deserialization."""

    def test_cond_expr_serialization(self):
        """Test CondExpr serialization (write only)."""
        pred = VariableExpr(
            "pred", MPType.tensor(DType.from_numpy(np.bool_), (), pmask=Mask(7))
        )

        # Create simple function bodies for then/else branches
        then_body = VariableExpr(
            "then_result", MPType.tensor(FLOAT32, (2,), pmask=Mask(7))
        )
        else_body = VariableExpr(
            "else_result", MPType.tensor(FLOAT32, (2,), pmask=Mask(7))
        )

        then_fn = FuncDefExpr([], then_body)
        else_fn = FuncDefExpr([], else_body)

        args: list[Expr] = [
            VariableExpr("arg", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        ]
        original = CondExpr(pred, then_fn, else_fn, args)

        writer = IrWriter()
        proto = writer.dumps(original)

        # Verify serialization works
        assert len(proto.nodes) >= 3  # pred, args, cond (functions are embedded)
        cond_node = None
        for node in proto.nodes:
            if node.op_type == "cond":
                cond_node = node
                break
        assert cond_node is not None
        assert "then_fn" in cond_node.attrs
        assert "else_fn" in cond_node.attrs

    def test_while_expr_serialization(self):
        """Test WhileExpr serialization (write only)."""
        # Create simple function bodies for condition and loop body using VariableExpr
        cond_body = VariableExpr(
            "cond_result", MPType.tensor(DType.from_numpy(np.bool_), (), pmask=Mask(7))
        )
        loop_body = VariableExpr(
            "loop_result", MPType.tensor(FLOAT32, (2,), pmask=Mask(7))
        )

        cond_fn = FuncDefExpr(["x"], cond_body)
        body_fn = FuncDefExpr(["x"], loop_body)

        args: list[Expr] = [
            VariableExpr("init_arg", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        ]
        original = WhileExpr(cond_fn, body_fn, args)

        writer = IrWriter()
        proto = writer.dumps(original)

        # Verify serialization works
        assert len(proto.nodes) >= 2  # args, while (functions are embedded)
        while_node = None
        for node in proto.nodes:
            if node.op_type == "while":
                while_node = node
                break
        assert while_node is not None
        assert "cond_fn" in while_node.attrs
        assert "body_fn" in while_node.attrs

    def test_conv_expr_serialization(self):
        """Test ConvExpr serialization (write only)."""
        mptype1 = MPType.tensor(
            DType.from_numpy(np.float32), (2,), pmask=Mask(3)
        )  # Party 0 and 1
        mptype2 = MPType.tensor(
            DType.from_numpy(np.float32), (2,), pmask=Mask(4)
        )  # Party 2
        var1 = VariableExpr("x", mptype1)
        var2 = VariableExpr("y", mptype2)

        original = ConvExpr([var1, var2])

        writer = IrWriter()
        proto = writer.dumps(original)

        # Verify serialization works
        assert len(proto.nodes) >= 3  # var1, var2, conv
        conv_node = None
        for node in proto.nodes:
            if node.op_type == "conv":
                conv_node = node
                break
        assert conv_node is not None

    def test_shfl_s_expr_serialization(self):
        """Test ShflSExpr serialization (write only)."""
        src_val = VariableExpr("src", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))

        original = ShflSExpr(src_val, pmask=Mask(3), src_ranks=[0, 1])

        writer = IrWriter()
        proto = writer.dumps(original)

        # Verify serialization works
        assert len(proto.nodes) >= 2  # src_val, shfl_s
        shfl_node = None
        for node in proto.nodes:
            if node.op_type == "shfl_s":
                shfl_node = node
                break
        assert shfl_node is not None
        assert "pmask" in shfl_node.attrs
        assert "src_ranks" in shfl_node.attrs

    def test_shfl_expr_serialization(self):
        """Test ShflExpr serialization (write only)."""
        src = VariableExpr("src", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        index = VariableExpr("index", MPType.tensor(INT32, (), pmask=Mask(7)))

        original = ShflExpr(src, index)

        writer = IrWriter()
        proto = writer.dumps(original)

        # Verify serialization works
        assert len(proto.nodes) >= 3  # src, index, shfl
        shfl_node = None
        for node in proto.nodes:
            if node.op_type == "shfl":
                shfl_node = node
                break
        assert shfl_node is not None

    def test_call_expr_serialization(self):
        """Test CallExpr serialization (write only)."""
        # Create function definition using VariableExpr
        body = VariableExpr("result", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        fn = FuncDefExpr(["x"], body)

        # Create arguments using VariableExpr
        arg = VariableExpr("input", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))

        original = CallExpr("original", fn, [arg])

        writer = IrWriter()
        proto = writer.dumps(original)

        # Verify serialization works
        assert len(proto.nodes) >= 4  # body, fn, arg, call
        call_node = None
        for node in proto.nodes:
            if node.op_type == "call":
                call_node = node
                break
        assert call_node is not None


class TestWriterReader:
    """Test Writer and Reader classes."""

    def test_writer_reset(self):
        """Test Writer reset functionality."""
        writer = IrWriter()

        # Create some expression using VariableExpr

        expr = VariableExpr("test", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))

        # Process expression
        writer.dumps(expr)
        assert len(writer._nodes) > 0
        assert len(writer._expr_ids) > 0

        # Reset and verify
        writer.reset()
        assert len(writer._nodes) == 0
        assert len(writer._expr_ids) == 0
        assert writer._counter == 0

    def test_writer_expr_naming(self):
        """Test Writer expression naming."""
        writer = IrWriter()

        expr1 = VariableExpr("x", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        expr2 = VariableExpr("y", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))

        name1 = writer.expr_name(expr1)
        name2 = writer.expr_name(expr2)

        assert name1 != name2
        assert name1.startswith("%")
        assert name2.startswith("%")

        # Same expression should get same name
        name1_again = writer.expr_name(expr1)
        assert name1 == name1_again

    def test_writer_value_naming(self):
        """Test Writer value naming for multi-output expressions."""
        writer = IrWriter()

        expr1 = VariableExpr("x", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        expr2 = VariableExpr("y", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        tuple_expr = TupleExpr([expr1, expr2])

        # Single output should not have index
        value_name = writer.value_name(expr1)
        assert ":" not in value_name

        # Multi-output should have index
        value_name_0 = writer.value_name(tuple_expr, 0)
        value_name_1 = writer.value_name(tuple_expr, 1)
        assert ":0" in value_name_0
        assert ":1" in value_name_1

    def test_reader_empty_graph(self):
        """Test Reader with empty graph."""

        reader = IrReader()
        empty_graph = mpir_pb2.GraphProto()

        result = reader.loads(empty_graph)
        assert result is None

    def test_reader_invalid_output(self):
        """Test Reader with invalid output reference."""

        reader = IrReader()
        graph = mpir_pb2.GraphProto()
        graph.outputs.append("nonexistent_node")

        with pytest.raises(ValueError, match=r"Output .* not found"):
            reader.loads(graph)

    def test_reader_unsupported_node_type(self):
        """Test Reader with unsupported node type."""

        # Create a graph with unsupported node type
        graph = mpir_pb2.GraphProto()
        node = graph.nodes.add()
        node.name = "%0"
        node.op_type = "unsupported_op"
        graph.outputs.append("%0")

        reader = IrReader()
        with pytest.raises(ValueError, match="Unsupported node type"):
            reader.loads(graph)


class TestErrorHandling:
    """Test error handling in mpir module."""

    def test_invalid_dtype_conversion(self):
        """Test error handling for invalid dtype conversions."""
        from mplang.v1.core.mpir import dtype_to_proto

        with pytest.raises(ValueError, match="Invalid dtype"):
            dtype_to_proto(object())

    def test_unsupported_attribute_type(self):
        """Test error handling for unsupported attribute types."""
        from mplang.v1.core.mpir import attr_to_proto

        with pytest.raises(TypeError, match="Unsupported attribute type"):
            attr_to_proto(set())  # Unsupported type

    def test_mixed_type_lists(self):
        """Test error handling for mixed type lists in attributes."""
        from mplang.v1.core.mpir import attr_to_proto

        with pytest.raises(TypeError, match="Unsupported tuple/list type"):
            attr_to_proto([1, "string", 3.14])  # Mixed types

    def test_reader_missing_input(self):
        """Test Reader error when input is missing."""

        # Create graph with missing dependency
        graph = mpir_pb2.GraphProto()

        # Add a tuple node that references non-existent input
        node = graph.nodes.add()
        node.name = "%0"
        node.op_type = "tuple"
        node.inputs.append("nonexistent")  # Missing input
        graph.outputs.append("%0")

        reader = IrReader()
        with pytest.raises(ValueError, match=r"Input .* not found"):
            reader.loads(graph)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_tuple_expr(self):
        """Test TupleExpr with empty args - serialization only."""
        original = TupleExpr([])

        writer = IrWriter()
        proto = writer.dumps(original)

        # Empty tuple expr produces a graph with 1 node (the tuple itself)
        assert len(proto.nodes) == 1
        assert proto.nodes[0].op_type == "tuple"
        assert len(proto.nodes[0].inputs) == 0

        # Note: Empty tuple has no outputs, so reader returns None
        # This is expected behavior for an empty tuple

    def test_complex_nested_structure(self):
        """Test complex nested expression structure."""

        # Create nested structure: tuple(var, access(tuple(var, var), 1))
        var1 = VariableExpr("x", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        var2 = VariableExpr("y", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        var3 = VariableExpr("z", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))

        inner_tuple = TupleExpr([var2, var3])
        access_expr = AccessExpr(inner_tuple, 1)
        outer_tuple = TupleExpr([var1, access_expr])

        writer = IrWriter()
        proto = writer.dumps(outer_tuple)

        reader = IrReader()
        result = reader.loads(proto)

        assert isinstance(result, TupleExpr)
        assert len(result.args) == 2
        assert isinstance(result.args[1], AccessExpr)

    def test_multiple_output_expression_naming(self):
        """Test expression naming with multiple outputs."""
        writer = IrWriter()

        var1 = VariableExpr("x", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        var2 = VariableExpr("y", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))
        tuple_expr = TupleExpr([var1, var2])

        # Multiple outputs should have different names
        names = [
            writer.value_name(tuple_expr, i) for i in range(len(tuple_expr.mptypes))
        ]
        assert len(set(names)) == len(names)  # All names should be unique
        assert all(":" in name for name in names)  # Should have output indices

    def test_writer_with_duplicate_expressions(self):
        """Test that Writer properly handles duplicate expression references."""
        writer = IrWriter()

        shared_var = VariableExpr("shared", MPType.tensor(FLOAT32, (2,), pmask=Mask(7)))

        # Create multiple references to the same expression
        tuple1 = TupleExpr([shared_var])
        tuple2 = TupleExpr([shared_var])
        final_tuple = TupleExpr([tuple1, tuple2])

        proto = writer.dumps(final_tuple)

        # Note: The current implementation creates separate nodes for each
        # expression instance, not sharing them. This test verifies current behavior.
        var_nodes = [node for node in proto.nodes if node.op_type == "variable"]
        assert len(var_nodes) >= 1  # At least one var node exists

        # Verify the structure is correct
        tuple_nodes = [node for node in proto.nodes if node.op_type == "tuple"]
        assert len(tuple_nodes) == 3  # tuple1, tuple2, final_tuple

    def test_large_shape_tensor(self):
        """Test handling of tensors with large shapes."""
        large_shape = (100, 200, 50)
        # Use VariableExpr to represent a tensor with the given shape and pmask
        original = VariableExpr(
            "large_tensor",
            MPType.tensor(FLOAT32, large_shape, pmask=Mask(1)),
        )

        writer = IrWriter()
        proto = writer.dumps(original)

        reader = IrReader()
        result = reader.loads(proto)
        assert isinstance(result, VariableExpr)
        assert result.mptypes[0].shape == large_shape

    def test_zero_pmask(self):
        """Test expressions with pmask = 0."""

        original = VariableExpr(
            "zero_pmask",
            MPType.tensor(FLOAT32, (2,), pmask=Mask(0)),
        )

        writer = IrWriter()
        proto = writer.dumps(original)

        reader = IrReader()
        result = reader.loads(proto)
        assert isinstance(result, VariableExpr)
        assert result.mptype.pmask == Mask(0)

    def test_dynamic_pmask(self):
        """Test MPTypeProto serialization/deserialization with dynamic pmask (None)."""
        from mplang.v1.core.dtypes import INT64, STRING
        from mplang.v1.core.expr.ast import VariableExpr
        from mplang.v1.core.mpir import IrReader, IrWriter
        from mplang.v1.core.table import TableType

        # Create table type with dynamic pmask (None)
        schema = TableType.from_dict({
            "id": INT64,
            "name": STRING,
        })

        original_mptype = MPType.table(schema, pmask=None)  # Dynamic pmask
        var_expr = VariableExpr("test_dynamic", original_mptype)

        # Serialize and deserialize
        writer = IrWriter()
        proto = writer.dumps(var_expr)

        reader = IrReader()
        deserialized_expr = reader.loads(proto)

        # Verify
        assert isinstance(deserialized_expr, VariableExpr)
        assert deserialized_expr.name == "test_dynamic"

        # Check that the dynamic pmask is preserved
        recovered_mptype = deserialized_expr.mptype
        assert recovered_mptype.is_table
        assert recovered_mptype.pmask is None  # Should be None for dynamic mask
        assert recovered_mptype.schema.num_columns() == 2

    def test_pmask_proto_encoding(self):
        """Test that pmask values are correctly encoded in protobuf."""

        # Test static pmask values
        proto = mpir_pb2.MPTypeProto()
        proto.pmask = 0  # Mask 0
        assert proto.pmask == 0

        proto.pmask = 1  # Mask 1 (party 0)
        assert proto.pmask == 1

        proto.pmask = 0b1101  # Mask for parties 0, 2, 3
        assert proto.pmask == 13

        # Test dynamic pmask
        proto.pmask = -1  # Dynamic mask
        assert proto.pmask == -1

        # Test large mask values
        proto.pmask = 0xFFFFFFFFFFFFFFFF >> 1  # Max positive int64
        assert proto.pmask == 0x7FFFFFFFFFFFFFFF

    def test_expression_dynamic_pmask(self):
        """Test that Expression-level objects can handle dynamic pmask correctly."""
        from mplang.v1.core.mpir import IrReader, IrWriter

        # Test VariableExpr with dynamic pmask
        const_expr = VariableExpr(
            "dyn_pmask",
            MPType.tensor(FLOAT32, (2,), pmask=None),
        )

        # Verify the expression was created correctly
        assert const_expr.mptype.pmask is None  # Dynamic pmask is allowed

        # Test serialization/deserialization
        writer = IrWriter()
        proto = writer.dumps(const_expr)

        reader = IrReader()
        result = reader.loads(proto)
        assert isinstance(result, VariableExpr)
        assert result.mptype.pmask is None  # Dynamic pmask should be preserved

    def test_reader_proto_to_attr_edge_cases(self):
        """Test Reader._proto_to_attr with edge cases."""

        reader = IrReader()

        # Test empty lists
        attr_proto = mpir_pb2.AttrProto()
        attr_proto.type = mpir_pb2.AttrProto.INTS
        # ints list is empty by default
        result = reader._proto_to_attr(attr_proto)
        assert result == []

        # Test empty strings list
        attr_proto = mpir_pb2.AttrProto()
        attr_proto.type = mpir_pb2.AttrProto.STRINGS
        result = reader._proto_to_attr(attr_proto)
        assert result == []

    def test_attr_to_proto_edge_cases(self):
        """Test attr_to_proto with edge cases."""
        from mplang.v1.core.mpir import attr_to_proto

        # Test empty list - should work and create INTS type
        attr = attr_to_proto([])
        assert attr.type == mpir_pb2.AttrProto.INTS
        assert list(attr.ints) == []

        # Test mixed type list - should fail
        with pytest.raises(TypeError, match="Unsupported tuple/list type"):
            attr_to_proto([1, "string", 3.14])

    def test_writer_counter_overflow_simulation(self):
        """Test that writer can handle many expressions."""
        writer = IrWriter()

        # Create many expressions to test counter behavior
        expressions = []
        for i in range(100):
            expr = VariableExpr(
                f"v{i}",
                MPType.tensor(FLOAT32, (2,), pmask=Mask(7)),
            )
            expressions.append(expr)

        # Create a tuple with all expressions
        big_tuple = TupleExpr(expressions)

        proto = writer.dumps(big_tuple)
        # Should have 101 nodes (100 variables + 1 tuple)
        assert len(proto.nodes) == 101

        # All names should be unique
        names = [node.name for node in proto.nodes]
        assert len(set(names)) == len(names)


class TestComplexExpressionRoundtrip:
    """Test complex expression roundtrip to expose serialization/deserialization bugs."""

    def test_cond_expr_roundtrip(self):
        """Test CondExpr roundtrip - this should expose the CallExpr fn type issue."""

        pred = VariableExpr(
            "pred",
            MPType.tensor(DType.from_numpy(np.bool_), (), pmask=Mask(7)),
        )

        # Create simple function bodies for then/else branches
        then_body = VariableExpr(
            "then_result",
            MPType.tensor(FLOAT32, (2,), pmask=Mask(7)),
        )
        else_body = VariableExpr(
            "else_result",
            MPType.tensor(FLOAT32, (2,), pmask=Mask(7)),
        )

        then_fn = FuncDefExpr([], then_body)
        else_fn = FuncDefExpr([], else_body)

        args: list[Expr] = [
            VariableExpr(
                "arg",
                MPType.tensor(FLOAT32, (2,), pmask=Mask(7)),
            )
        ]
        original = CondExpr(pred, then_fn, else_fn, args)

        # Test roundtrip
        writer = IrWriter()
        proto = writer.dumps(original)

        reader = IrReader()
        result = reader.loads(proto)

        # Verify basic properties
        assert isinstance(result, CondExpr)
        assert isinstance(result.pred, VariableExpr)

        # This is where we expect to find the issue:
        # The then_fn and else_fn should be FuncDefExpr but might be something else
        print(f"Original then_fn type: {type(original.then_fn)}")
        print(f"Deserialized then_fn type: {type(result.then_fn)}")
        print(f"Original else_fn type: {type(original.else_fn)}")
        print(
            f"Deserialized else_fn type: {type(result.else_fn)}"
        )  # These assertions should expose the bug
        assert isinstance(result.then_fn, FuncDefExpr), (
            f"Expected FuncDefExpr, got {type(result.then_fn)}"
        )
        assert isinstance(result.else_fn, FuncDefExpr), (
            f"Expected FuncDefExpr, got {type(result.else_fn)}"
        )

    def test_while_expr_roundtrip(self):
        """Test WhileExpr roundtrip - this should expose similar issues."""

        # Create simple function bodies for condition and loop body
        cond_body = VariableExpr(
            "cond_result",
            MPType.tensor(DType.from_numpy(np.bool_), (), pmask=Mask(7)),
        )
        loop_body = VariableExpr(
            "loop_result",
            MPType.tensor(FLOAT32, (2,), pmask=Mask(7)),
        )

        cond_fn = FuncDefExpr(["x"], cond_body)
        body_fn = FuncDefExpr(["x"], loop_body)

        args: list[Expr] = [
            VariableExpr(
                "init_arg",
                MPType.tensor(FLOAT32, (2,), pmask=Mask(7)),
            )
        ]
        original = WhileExpr(cond_fn, body_fn, args)

        # Test roundtrip
        writer = IrWriter()
        proto = writer.dumps(original)

        reader = IrReader()
        result = reader.loads(proto)

        # Verify basic properties and expose issues
        assert isinstance(result, WhileExpr)

        print(f"Original cond_fn type: {type(original.cond_fn)}")
        print(f"Deserialized cond_fn type: {type(result.cond_fn)}")
        print(f"Original body_fn type: {type(original.body_fn)}")
        print(f"Deserialized body_fn type: {type(result.body_fn)}")

        # These assertions should expose the bug
        assert isinstance(result.cond_fn, FuncDefExpr), (
            f"Expected FuncDefExpr, got {type(result.cond_fn)}"
        )
        assert isinstance(result.body_fn, FuncDefExpr), (
            f"Expected FuncDefExpr, got {type(result.body_fn)}"
        )

    def test_call_expr_roundtrip(self):
        """Test CallExpr roundtrip - this should expose the evaluator assertion error."""

        # Create function definition
        body = VariableExpr(
            "result",
            MPType.tensor(FLOAT32, (2,), pmask=Mask(7)),
        )
        fn = FuncDefExpr(["x"], body)

        # Create arguments
        arg = VariableExpr(
            "input",
            MPType.tensor(FLOAT32, (2,), pmask=Mask(7)),
        )

        original = CallExpr("original", fn, [arg])

        # Test roundtrip
        writer = IrWriter()
        proto = writer.dumps(original)

        reader = IrReader()
        result = reader.loads(proto)

        # Verify basic properties and expose issues
        assert isinstance(result, CallExpr)

        print(f"Original fn type: {type(original.fn)}")
        print(f"Deserialized fn type: {type(result.fn)}")

        # This assertion should expose the bug that causes the evaluator to fail
        assert isinstance(result.fn, FuncDefExpr), (
            f"Expected FuncDefExpr, got {type(result.fn)}"
        )


class TestRelationTypeSupport:
    """Test RelationType support in mpir serialization/deserialization."""

    def test_relation_dtype_conversion(self):
        """Test dtype conversion for relation-only types."""
        from mplang.v1.core.dtypes import DATE, JSON, STRING, TIME, TIMESTAMP
        from mplang.v1.core.mpir import dtype_to_proto, proto_to_dtype

        # Test relation-only dtypes to proto
        test_cases = [
            (STRING, mpir_pb2.DataType.STRING),
            (DATE, mpir_pb2.DataType.DATE),
            (TIME, mpir_pb2.DataType.TIME),
            (TIMESTAMP, mpir_pb2.DataType.TIMESTAMP),
            (JSON, mpir_pb2.DataType.JSON),
        ]

        for dtype, expected_proto in test_cases:
            # Test dtype_to_proto
            proto_result = dtype_to_proto(dtype)
            assert proto_result == expected_proto, f"Failed for {dtype.name}"

            # Test proto_to_dtype
            dtype_result = proto_to_dtype(expected_proto)
            assert dtype_result == dtype, f"Failed round-trip for {dtype.name}"

    def test_mptype_proto_tensor_type(self):
        """Test MPTypeProto with tensor type."""

        # Create tensor type proto
        mp_proto = mpir_pb2.MPTypeProto()
        mp_proto.tensor_type.dtype = mpir_pb2.DataType.F32
        mp_proto.tensor_type.shape_dims.extend([3, 4, 5])
        mp_proto.pmask = 123

        # Verify structure
        assert mp_proto.HasField("tensor_type")
        assert not mp_proto.HasField("table_type")
        assert mp_proto.tensor_type.dtype == mpir_pb2.DataType.F32
        assert list(mp_proto.tensor_type.shape_dims) == [3, 4, 5]
        assert mp_proto.pmask == 123

    def test_mptype_proto_table_type(self):
        """Test MPTypeProto with table type."""

        # Create table type proto
        mp_proto = mpir_pb2.MPTypeProto()

        # Add columns
        col1 = mp_proto.table_type.columns.add()
        col1.name = "id"
        col1.dtype = mpir_pb2.DataType.I64

        col2 = mp_proto.table_type.columns.add()
        col2.name = "name"
        col2.dtype = mpir_pb2.DataType.STRING

        col3 = mp_proto.table_type.columns.add()
        col3.name = "created_at"
        col3.dtype = mpir_pb2.DataType.DATE

        mp_proto.pmask = 456

        # Verify structure
        assert not mp_proto.HasField("tensor_type")
        assert mp_proto.HasField("table_type")
        assert len(mp_proto.table_type.columns) == 3
        assert mp_proto.pmask == 456

        # Verify columns
        columns = mp_proto.table_type.columns
        assert columns[0].name == "id"
        assert columns[0].dtype == mpir_pb2.DataType.I64
        assert columns[1].name == "name"
        assert columns[1].dtype == mpir_pb2.DataType.STRING
        assert columns[2].name == "created_at"
        assert columns[2].dtype == mpir_pb2.DataType.DATE

    def test_mptype_proto_serialization(self):
        """Test MPTypeProto serialization/deserialization."""

        # Create and serialize table type
        mp_proto = mpir_pb2.MPTypeProto()
        col = mp_proto.table_type.columns.add()
        col.name = "test_col"
        col.dtype = mpir_pb2.DataType.JSON
        mp_proto.pmask = 789

        # Serialize
        serialized = mp_proto.SerializeToString()

        # Deserialize
        mp_proto2 = mpir_pb2.MPTypeProto()
        mp_proto2.ParseFromString(serialized)

        # Verify
        assert mp_proto2.HasField("table_type")
        assert len(mp_proto2.table_type.columns) == 1
        assert mp_proto2.table_type.columns[0].name == "test_col"
        assert mp_proto2.table_type.columns[0].dtype == mpir_pb2.DataType.JSON
        assert mp_proto2.pmask == 789

    def test_mptype_with_table_type(self):
        """Test MPType integration with RelationType (table type in proto)."""
        from mplang.v1.core.dtypes import DATE, INT64, STRING
        from mplang.v1.core.table import TableType

        # Create table type
        schema = TableType.from_dict({
            "user_id": INT64,
            "username": STRING,
            "signup_date": DATE,
        })

        # Create MPType with table
        pmask = Mask(0b1101)  # parties 0, 2, 3
        mptype = MPType.table(schema, pmask)

        # Verify properties
        assert mptype.is_table
        assert not mptype.is_tensor
        assert mptype.schema == schema
        assert mptype.pmask == pmask

        # Verify schema access
        assert mptype.schema.num_columns() == 3
        assert mptype.schema.has_column("user_id")
        assert mptype.schema.has_column("username")
        assert mptype.schema.has_column("signup_date")
        assert mptype.schema.get_column_type("user_id") == INT64
        assert mptype.schema.get_column_type("username") == STRING
        assert mptype.schema.get_column_type("signup_date") == DATE

    def test_mptype_proto_conversion_round_trip(self):
        """Test full round-trip conversion between MPType and MPTypeProto."""
        from mplang.v1.core.dtypes import INT64, JSON, STRING, TIMESTAMP
        from mplang.v1.core.expr.ast import VariableExpr
        from mplang.v1.core.mpir import IrReader, IrWriter
        from mplang.v1.core.table import TableType

        # Create complex table type
        schema = TableType.from_dict({
            "id": INT64,
            "name": STRING,
            "created_at": TIMESTAMP,
            "metadata": JSON,
        })

        pmask = Mask(0b11110000)  # parties 4, 5, 6, 7
        original_mptype = MPType.table(schema, pmask)

        # Create a simple expression with table type
        var_expr = VariableExpr("test_table", original_mptype)

        # Serialize and deserialize
        writer = IrWriter()
        proto = writer.dumps(var_expr)

        reader = IrReader()
        deserialized_expr = reader.loads(proto)

        # Verify - the deserialized expression should be a VariableExpr
        assert isinstance(deserialized_expr, VariableExpr)
        assert deserialized_expr.name == "test_table"

        # Check the type
        recovered_mptype = deserialized_expr.mptype

        assert recovered_mptype.is_table
        assert recovered_mptype.pmask == pmask
        assert recovered_mptype.schema.num_columns() == 4

        # Verify all columns
        for (orig_name, orig_dtype), (rec_name, rec_dtype) in zip(
            original_mptype.schema.columns, recovered_mptype.schema.columns, strict=True
        ):
            assert orig_name == rec_name
            assert orig_dtype == rec_dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
