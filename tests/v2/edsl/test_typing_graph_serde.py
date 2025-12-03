# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for typing and graph serde."""

from __future__ import annotations

from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Graph
from mplang.v2.edsl.typing import (
    ComplexType,
    CustomType,
    FloatType,
    IntegerType,
    MPType,
    SSType,
    TableType,
    TensorType,
    VectorType,
    f32,
    f64,
    i32,
    i64,
)

# =============================================================================
# Tests: Scalar Types
# =============================================================================


class TestScalarTypeSerde:
    """Test serialization of scalar types."""

    def test_integer_type(self):
        t = IntegerType(bitwidth=64, signed=True)
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, IntegerType)
        assert result.bitwidth == 64
        assert result.signed is True

    def test_integer_type_unsigned(self):
        t = IntegerType(bitwidth=32, signed=False)
        result = serde.from_json(serde.to_json(t))
        assert result.bitwidth == 32
        assert result.signed is False

    def test_float_type(self):
        t = FloatType(bitwidth=32)
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, FloatType)
        assert result.bitwidth == 32

    def test_complex_type(self):
        t = ComplexType(inner_type=f64)
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, ComplexType)
        assert result.inner_type == f64

    def test_predefined_types(self):
        """Test that predefined type instances serialize correctly."""
        for t in [i32, i64, f32, f64]:
            result = serde.from_json(serde.to_json(t))
            assert result == t


# =============================================================================
# Tests: Layout Types
# =============================================================================


class TestLayoutTypeSerde:
    """Test serialization of layout types."""

    def test_tensor_type(self):
        t = TensorType(f32, (10, 20))
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, TensorType)
        assert result.element_type == f32
        assert result.shape == (10, 20)

    def test_tensor_type_scalar(self):
        t = TensorType(i32, ())
        result = serde.from_json(serde.to_json(t))
        assert result.shape == ()

    def test_tensor_type_dynamic(self):
        t = TensorType(f32, (-1, 10))
        result = serde.from_json(serde.to_json(t))
        assert result.shape == (-1, 10)

    def test_tensor_subscript_syntax(self):
        """Test Tensor[dtype, shape] syntax."""
        t = TensorType[f32, (5, 5)]
        result = serde.from_json(serde.to_json(t))
        assert result == t

    def test_vector_type(self):
        t = VectorType(f32, 4096)
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, VectorType)
        assert result.element_type == f32
        assert result.size == 4096

    def test_table_type(self):
        t = TableType({"id": i64, "value": f32})
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, TableType)
        assert result.schema["id"] == i64
        assert result.schema["value"] == f32

    def test_custom_type(self):
        t = CustomType("EncryptionKey")
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, CustomType)
        assert result.kind == "EncryptionKey"


# =============================================================================
# Tests: Encryption Types
# =============================================================================


class TestEncryptionTypeSerde:
    """Test serialization of encryption types."""

    def test_ss_type(self):
        inner = TensorType(f32, (10,))
        t = SSType(inner)
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, SSType)
        assert result.pt_type == inner

    def test_ss_type_nested(self):
        """Test SS with nested tensor."""
        t = SSType[TensorType[i32, (5, 5)]]
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, SSType)
        assert isinstance(result.pt_type, TensorType)


# =============================================================================
# Tests: Distribution Types
# =============================================================================


class TestDistributionTypeSerde:
    """Test serialization of distribution types."""

    def test_mp_type(self):
        inner = TensorType(f32, (10,))
        t = MPType(inner, (0, 1))
        result = serde.from_json(serde.to_json(t))
        assert isinstance(result, MPType)
        assert result.value_type == inner
        assert result.parties == (0, 1)

    def test_mp_type_dynamic_parties(self):
        inner = TensorType(f32, (10,))
        t = MPType(inner, None)
        result = serde.from_json(serde.to_json(t))
        assert result.parties is None


# =============================================================================
# Tests: Nested/Composed Types
# =============================================================================


class TestNestedTypeSerde:
    """Test serialization of complex nested types."""

    def test_mp_ss_tensor(self):
        """Test MP[SS[Tensor[f32, (10,)]], (0, 1)]"""
        tensor = TensorType(f32, (10,))
        ss = SSType(tensor)
        mp = MPType(ss, (0, 1))

        result = serde.from_json(serde.to_json(mp))

        assert isinstance(result, MPType)
        assert result.parties == (0, 1)
        assert isinstance(result.value_type, SSType)
        assert isinstance(result.value_type.pt_type, TensorType)
        assert result.value_type.pt_type.element_type == f32

    def test_table_with_custom_types(self):
        """Test Table with CustomType columns."""
        from mplang.v2.edsl.typing import STRING, TIMESTAMP

        t = TableType({"name": STRING, "created_at": TIMESTAMP, "count": i64})
        result = serde.from_json(serde.to_json(t))
        assert result.schema["name"] == STRING
        assert result.schema["created_at"] == TIMESTAMP
        assert result.schema["count"] == i64


# =============================================================================
# Tests: Graph Serde
# =============================================================================


class TestGraphSerde:
    """Test serialization of Graph IR."""

    def test_simple_graph(self):
        """Test a simple graph with one operation."""
        graph = Graph()
        x = graph.add_input("x", TensorType(f32, (10,)))
        (y,) = graph.add_op("tensor.neg", [x], output_types=[TensorType(f32, (10,))])
        graph.add_output(y)

        result = serde.from_json(serde.to_json(graph))

        assert isinstance(result, Graph)
        assert len(result.inputs) == 1
        assert len(result.operations) == 1
        assert len(result.outputs) == 1
        assert result.inputs[0].name == "x"
        assert result.operations[0].opcode == "tensor.neg"

    def test_graph_with_attrs(self):
        """Test graph with operation attributes."""
        graph = Graph()
        (c,) = graph.add_op(
            "tensor.constant",
            [],
            output_types=[TensorType(f32, ())],
            attrs={"value": 3.14, "name": "pi"},
        )
        graph.add_output(c)

        result = serde.from_json(serde.to_json(graph))

        assert result.operations[0].attrs["value"] == 3.14
        assert result.operations[0].attrs["name"] == "pi"

    def test_graph_with_multiple_ops(self):
        """Test graph with multiple chained operations."""
        graph = Graph()
        x = graph.add_input("x", TensorType(f32, (10,)))
        y = graph.add_input("y", TensorType(f32, (10,)))
        (z,) = graph.add_op("tensor.add", [x, y])
        (w,) = graph.add_op("tensor.neg", [z])
        graph.add_output(w)

        result = serde.from_json(serde.to_json(graph))

        assert len(result.inputs) == 2
        assert len(result.operations) == 2
        # Check that the operations are connected correctly
        add_op = result.operations[0]
        neg_op = result.operations[1]
        assert add_op.opcode == "tensor.add"
        assert neg_op.opcode == "tensor.neg"
        assert neg_op.inputs[0] is add_op.outputs[0]

    def test_graph_with_regions(self):
        """Test graph with nested regions (for control flow)."""
        # Create a simple true branch
        true_branch = Graph()
        _ = true_branch.add_input("x", TensorType(f32, ()))
        (true_result,) = true_branch.add_op(
            "tensor.constant",
            [],
            output_types=[TensorType(f32, ())],
            attrs={"value": 1.0},
        )
        true_branch.add_output(true_result)

        # Create main graph with region
        graph = Graph()
        cond = graph.add_input("cond", TensorType(IntegerType(bitwidth=1), ()))
        x = graph.add_input("x", TensorType(f32, ()))
        (result,) = graph.add_op(
            "cond",
            [cond, x],
            output_types=[TensorType(f32, ())],
            regions=[true_branch, true_branch],  # true and false branches
        )
        graph.add_output(result)

        serialized = serde.to_json(graph)
        result = serde.from_json(serialized)

        assert len(result.operations) == 1
        assert result.operations[0].opcode == "cond"
        assert len(result.operations[0].regions) == 2

    def test_graph_roundtrip_preserves_names(self):
        """Test that SSA names are preserved through serialization."""
        graph = Graph()
        x = graph.add_input("input_x", TensorType(f32, (5,)))
        (y,) = graph.add_op("tensor.neg", [x])
        graph.add_output(y)

        result = serde.from_json(serde.to_json(graph))

        assert result.inputs[0].name == "input_x"
        # Output names are auto-generated but should be preserved
        assert result.outputs[0].name == y.name


# =============================================================================
# Tests: Wire Format for Graph
# =============================================================================


class TestGraphWireFormat:
    """Test bytes serialization for graphs."""

    def test_dumps_loads(self):
        """Test dumps/loads roundtrip."""
        graph = Graph()
        x = graph.add_input("x", TensorType(f32, (10,)))
        (y,) = graph.add_op("tensor.neg", [x])
        graph.add_output(y)

        serialized = serde.dumps(graph)
        result = serde.loads(serialized)

        assert isinstance(result, Graph)
        assert len(result.operations) == 1

    def test_b64_roundtrip(self):
        """Test base64 string roundtrip (for HTTP transport)."""
        graph = Graph()
        x = graph.add_input("x", TensorType(i32, (5, 5)))
        graph.add_output(x)

        b64_str = serde.dumps_b64(graph)
        assert isinstance(b64_str, str)

        result = serde.loads_b64(b64_str)
        assert isinstance(result, Graph)
        assert result.inputs[0].type == TensorType(i32, (5, 5))
