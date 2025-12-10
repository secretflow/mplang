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
Tests for the Graph IR module.

This test suite validates the modern Operation List IR implementation,
covering SSA value management, operation creation, and IR generation.
"""

import pytest

from mplang.v2.edsl.graph import Graph, Operation, Value
from mplang.v2.edsl.typing import Custom, Tensor, Vector, f32, i32

# ==============================================================================
# --- Test Value (SSA Values)
# ==============================================================================


class TestValue:
    """Test SSA value creation and use-def chain management."""

    def test_value_creation(self):
        """Test basic value creation."""
        v = Value("%0", f32)
        assert v.name == "%0"
        assert v.type == f32
        assert v.defining_op is None
        assert v.uses == {}

    def test_value_str_representation(self):
        """Test value string representation."""
        v = Value("%42", Tensor[i32, (10,)])
        assert str(v) == "%42"
        assert v.name in repr(v)
        assert "Tensor" in repr(v)

    def test_value_add_use(self):
        """Test adding operations that use a value."""
        v = Value("%0", f32)
        op1 = Operation("add", [], [])
        op2 = Operation("mul", [], [])

        v.add_use(op1)
        assert v.num_uses == 1
        assert op1 in v.uses

        v.add_use(op2)
        assert v.num_uses == 2
        assert op2 in v.uses

    def test_value_add_use_idempotent(self):
        """Test that adding the same use twice doesn't duplicate."""
        v = Value("%0", f32)
        op = Operation("add", [], [])

        v.add_use(op)
        v.add_use(op)  # Add again
        assert v.num_uses == 1

    def test_value_remove_use(self):
        """Test removing operations from use list."""
        v = Value("%0", f32)
        op1 = Operation("add", [], [])
        op2 = Operation("mul", [], [])

        v.add_use(op1)
        v.add_use(op2)
        assert v.num_uses == 2

        v.remove_use(op1)
        assert v.num_uses == 1
        assert op1 not in v.uses
        assert op2 in v.uses

    def test_value_is_dead(self):
        """Test dead code detection."""
        v1 = Value("%0", f32)
        v2 = Value("%1", f32)
        op = Operation("constant", [], [v1])

        # v1 has defining_op but no uses -> dead
        v1.defining_op = op
        assert v1.is_dead

        # v2 has no defining_op (input) -> not considered dead
        assert not v2.is_dead

        # Add a use to v1 -> not dead anymore
        v1.add_use(Operation("add", [], []))
        assert not v1.is_dead


# ==============================================================================
# --- Test Operation
# ==============================================================================


class TestOperation:
    """Test operation creation and use-def chain updates."""

    def test_operation_creation(self):
        """Test basic operation creation."""
        v1 = Value("%0", f32)
        v2 = Value("%1", f32)
        v_out = Value("%2", f32)

        op = Operation("add", [v1, v2], [v_out])

        assert op.opcode == "add"
        assert op.inputs == [v1, v2]
        assert op.outputs == [v_out]
        assert op.attrs == {}
        assert op.regions == []

    def test_operation_post_init_updates_use_def(self):
        """Test that __post_init__ registers use-def relationships."""
        v1 = Value("%0", f32)
        v2 = Value("%1", f32)
        v_out = Value("%2", f32)

        assert v1.num_uses == 0
        assert v_out.defining_op is None

        op = Operation("add", [v1, v2], [v_out])

        # Inputs should be registered as used
        assert v1.num_uses == 1
        assert v2.num_uses == 1
        assert op in v1.uses
        assert op in v2.uses

        # Output should have defining_op set
        assert v_out.defining_op is op

    def test_operation_with_attributes(self):
        """Test operation with attributes."""
        v_out = Value("%0", f32)
        op = Operation(
            "constant",
            inputs=[],
            outputs=[v_out],
            attrs={"value": 3.14},
        )

        assert op.attrs["value"] == 3.14

    def test_operation_with_regions(self):
        """Test operation with nested regions (control flow)."""
        graph1 = Graph()
        graph2 = Graph()

        v_pred = Value("%pred", i32)
        v_out = Value("%out", f32)

        op = Operation(
            "cond",
            inputs=[v_pred],
            outputs=[v_out],
            regions=[graph1, graph2],
        )

        assert len(op.regions) == 2
        assert op.regions[0] is graph1
        assert op.regions[1] is graph2

    def test_operation_replace_input(self):
        """Test replacing an input value."""
        v1 = Value("%0", f32)
        v2 = Value("%1", f32)
        v_new = Value("%new", f32)
        v_out = Value("%out", f32)

        op = Operation("add", [v1, v2], [v_out])

        assert v1.num_uses == 1
        assert v_new.num_uses == 0

        # Replace v1 with v_new
        op.replace_input(v1, v_new)

        assert op.inputs[0] is v_new
        assert v1.num_uses == 0
        assert v_new.num_uses == 1

    def test_operation_erase(self):
        """Test erasing an operation (cleanup use-def)."""
        v1 = Value("%0", f32)
        v2 = Value("%1", f32)
        v_out = Value("%2", f32)

        op = Operation("add", [v1, v2], [v_out])

        assert v1.num_uses == 1
        assert v_out.defining_op is op

        # Erase operation
        op.erase()

        assert v1.num_uses == 0
        assert v2.num_uses == 0
        assert v_out.defining_op is None

    def test_operation_repr(self):
        """Test operation string representation."""
        v1 = Value("%0", f32)
        v_out = Value("%1", f32)
        op = Operation("negate", [v1], [v_out])

        r = repr(op)
        assert "negate" in r
        assert "%0" in r
        assert "%1" in r


# ==============================================================================
# --- Test Graph
# ==============================================================================


class TestGraph:
    """Test graph construction and IR generation."""

    def test_graph_creation(self):
        """Test basic graph creation."""
        g = Graph()
        assert g.operations == []
        assert g.values == {}
        assert g.inputs == []
        assert g.outputs == []

    def test_graph_add_value(self):
        """Test adding values to the graph."""
        g = Graph()

        v1 = g.add_value(f32)
        assert v1.name == "%0"
        assert v1.type == f32
        assert v1.name in g.values

        v2 = g.add_value(i32, name="custom")
        assert v2.name == "custom"
        assert "custom" in g.values

    def test_graph_add_value_duplicate_name(self):
        """Test that duplicate value names raise error."""
        g = Graph()
        g.add_value(f32, name="x")

        with pytest.raises(ValueError, match="Value x already exists"):
            g.add_value(f32, name="x")

    def test_graph_add_input(self):
        """Test adding graph inputs."""
        g = Graph()

        x = g.add_input("x", Tensor[f32, (10,)])
        y = g.add_input("y", Tensor[f32, (10,)])

        assert len(g.inputs) == 2
        assert x in g.inputs
        assert y in g.inputs
        assert x.name == "x"
        assert y.name == "y"

    def test_graph_add_constant(self):
        """Test adding constant values."""
        g = Graph()

        c1 = g.add_op("constant", [], output_types=[i32], attrs={"value": 42})[0]
        c2 = g.add_op("constant", [], output_types=[f32], attrs={"value": 3.14})[0]

        assert len(g.operations) == 2

        # Check constant operations
        op1 = g.operations[0]
        assert op1.opcode == "constant"
        assert op1.attrs["value"] == 42
        assert op1.outputs[0] is c1

        op2 = g.operations[1]
        assert op2.attrs["value"] == 3.14
        assert op2.outputs[0] is c2

    def test_graph_add_op_single_output(self):
        """Test adding operation with single output."""
        g = Graph()

        x = g.add_input("x", f32)
        y = g.add_input("y", f32)

        z_list = g.add_op("add", [x, y])
        assert isinstance(z_list, list)
        assert len(z_list) == 1
        z = z_list[0]
        assert isinstance(z, Value)
        assert len(g.operations) == 1

        op = g.operations[0]
        assert op.opcode == "add"
        assert op.inputs == [x, y]
        assert op.outputs == [z]

    def test_graph_add_op_multiple_outputs(self):
        """Test adding operation with multiple outputs."""
        g = Graph()

        x = g.add_input("x", Tensor[f32, (10, 10)])

        # Operation with 2 outputs (e.g., qr decomposition)
        outputs = g.add_op(
            "qr",
            [x],
            output_types=[Tensor[f32, (10, 10)], Tensor[f32, (10, 10)]],
        )

        assert isinstance(outputs, list)
        assert len(outputs) == 2

        op = g.operations[0]
        assert op.opcode == "qr"
        assert len(op.outputs) == 2

    def test_graph_add_op_with_attributes(self):
        """Test adding operation with attributes."""
        g = Graph()

        x = g.add_input("x", Tensor[f32, (10, 20)])

        [_y] = g.add_op("reshape", [x], attrs={"new_shape": (20, 10)})

        op = g.operations[0]
        assert op.attrs["new_shape"] == (20, 10)

    def test_graph_add_op_with_regions(self):
        """Test adding operation with nested regions."""
        g = Graph()
        then_graph = Graph()
        else_graph = Graph()

        pred = g.add_input("pred", i32)

        g.add_op(
            "cond",
            [pred],
            output_types=[f32],
            regions=[then_graph, else_graph],
        )

        op = g.operations[0]
        assert len(op.regions) == 2
        assert op.regions[0] is then_graph
        assert op.regions[1] is else_graph

    def test_graph_add_op_type_inference_from_input(self):
        """Test automatic type inference from input."""
        g = Graph()

        x = g.add_input("x", Tensor[f32, (10,)])

        # Don't specify output_types -> infer from first input
        [y] = g.add_op("negate", [x])

        assert y.type == x.type

    def test_graph_add_op_no_inputs_requires_output_types(self):
        """Test that operations with no inputs require explicit output types."""
        g = Graph()

        with pytest.raises(ValueError, match="Cannot infer type"):
            g.add_op("random", [])  # No inputs, no output_types

    def test_graph_add_output(self):
        """Test marking values as graph outputs."""
        g = Graph()

        x = g.add_input("x", f32)
        y = g.add_op("constant", [], output_types=[f32], attrs={"value": 1.0})[0]
        [z] = g.add_op("add", [x, y])

        g.add_output(z)

        assert len(g.outputs) == 1
        assert z in g.outputs

    def test_graph_add_output_invalid_value(self):
        """Test that adding non-existent value as output raises error."""
        g1 = Graph()
        g2 = Graph()

        v = g2.add_input("x", f32)

        with pytest.raises(ValueError, match=r"Value .* not in graph"):
            g1.add_output(v)

    def test_graph_to_string_simple(self):
        """Test IR string generation for simple graph."""
        g = Graph()

        x = g.add_input("x", f32)
        y = g.add_input("y", f32)
        [z] = g.add_op("add", [x, y])
        g.add_output(z)

        ir = g.to_string()

        assert "x = input" in ir
        assert "y = input" in ir
        assert "add(x, y)" in ir
        assert "return" in ir

    def test_graph_to_string_with_constants(self):
        """Test IR string generation with constants."""
        g = Graph()

        c = g.add_op("constant", [], output_types=[f32], attrs={"value": 42.0})[0]
        g.add_output(c)

        ir = g.to_string()

        assert "constant 42.0" in ir

    def test_graph_to_string_verbose(self):
        """Test IR string generation with type annotations."""
        g = Graph()

        x = g.add_input("x", Tensor[f32, (10,)])
        g.add_output(x)

        ir = g.to_string(verbose=True)

        assert ": Tensor[f32, (10)]" in ir

    def test_graph_to_string_multiple_outputs(self):
        """Test IR string generation with multiple outputs."""
        g = Graph()

        x = g.add_input("x", f32)
        outputs = g.add_op("split", [x], output_types=[f32, f32], attrs={"axis": 0})
        g.add_output(outputs[0])
        g.add_output(outputs[1])

        ir = g.to_string()

        assert "split" in ir
        assert "return" in ir
        # Should have both outputs
        assert outputs[0].name in ir
        assert outputs[1].name in ir

    def test_graph_repr(self):
        """Test graph repr."""
        g = Graph()
        g.add_input("x", f32)
        g.add_op("negate", [g.values["x"]])

        r = repr(g)
        assert "Graph" in r
        assert "1 ops" in r
        assert "2 values" in r  # x + output


# ==============================================================================
# --- Integration Tests
# ==============================================================================


class TestGraphIntegration:
    """Integration tests for realistic graph construction scenarios."""

    def test_simple_computation_graph(self):
        """Test building a simple computation: z = (x + y) * 2."""
        g = Graph()

        x = g.add_input("x", Tensor[f32, (10,)])
        y = g.add_input("y", Tensor[f32, (10,)])

        [sum_result] = g.add_op("add", [x, y])
        const_2 = g.add_op("constant", [], output_types=[f32], attrs={"value": 2.0})[0]
        [result] = g.add_op("mul", [sum_result, const_2])

        g.add_output(result)

        # Verify structure
        assert len(g.inputs) == 2
        assert len(g.operations) == 3  # constant, add, mul
        assert len(g.outputs) == 1

        # Verify IR
        ir = g.to_string()
        assert "add(x, y)" in ir
        assert "constant 2.0" in ir
        assert "mul" in ir

    def test_linear_layer_graph(self):
        """Test building a linear layer: y = matmul(x, W) + b."""
        g = Graph()

        x = g.add_input("x", Tensor[f32, (1, 784)])  # Batch size 1
        W = g.add_input("W", Tensor[f32, (784, 128)])
        b = g.add_input("b", Tensor[f32, (128,)])

        [mm_result] = g.add_op("matmul", [x, W], output_types=[Tensor[f32, (1, 128)]])
        [y] = g.add_op("add", [mm_result, b], output_types=[Tensor[f32, (1, 128)]])

        g.add_output(y)

        ir = g.to_string()
        assert "matmul(x, W)" in ir
        assert "add" in ir

    def test_conditional_graph(self):
        """Test building a conditional graph with nested regions."""
        g = Graph()

        # Main graph
        pred = g.add_input("pred", i32)
        x = g.add_input("x", f32)

        # Then branch: return x + 1
        then_graph = Graph()
        then_x = then_graph.add_input("x", f32)
        then_const = then_graph.add_op(
            "constant", [], output_types=[f32], attrs={"value": 1.0}
        )[0]
        [then_result] = then_graph.add_op("add", [then_x, then_const])
        then_graph.add_output(then_result)

        # Else branch: return x - 1
        else_graph = Graph()
        else_x = else_graph.add_input("x", f32)
        else_const = else_graph.add_op(
            "constant", [], output_types=[f32], attrs={"value": 1.0}
        )[0]
        [else_result] = else_graph.add_op("sub", [else_x, else_const])
        else_graph.add_output(else_result)

        # Main graph cond operation
        [result] = g.add_op(
            "cond",
            [pred, x],
            output_types=[f32],
            regions=[then_graph, else_graph],
        )
        g.add_output(result)

        # Verify nested structure
        assert len(g.operations) == 1
        cond_op = g.operations[0]
        assert cond_op.opcode == "cond"
        assert len(cond_op.regions) == 2

        # Verify branches
        assert len(then_graph.operations) == 2  # constant, add
        assert len(else_graph.operations) == 2  # constant, sub

    def test_encrypted_computation_graph(self):
        """Test building a graph with encrypted types."""
        g = Graph()

        # Inputs
        plaintext = g.add_input("plaintext", Tensor[f32, (100,)])
        key = g.add_input("key", Custom["crypto.key"])

        # Encrypt
        [ciphertext] = g.add_op(
            "encrypt",
            [plaintext, key],
            output_types=[Vector[f32, 100]],
        )

        # Homomorphic add
        const_encrypted = g.add_op(
            "constant", [], output_types=[Vector[f32, 100]], attrs={"value": 1.0}
        )[0]
        [result_encrypted] = g.add_op(
            "he_add",
            [ciphertext, const_encrypted],
            output_types=[Vector[f32, 100]],
        )

        # Decrypt
        [result] = g.add_op(
            "decrypt",
            [result_encrypted, key],
            output_types=[Tensor[f32, (100,)]],
        )

        g.add_output(result)

        # Verify encrypted types
        ir = g.to_string(verbose=True)
        assert "Vector" in ir
        assert "encrypt" in ir
        assert "decrypt" in ir

    def test_dag_with_shared_values(self):
        """Test building a DAG with shared subexpressions."""
        g = Graph()

        x = g.add_input("x", f32)

        # Shared computation: x + 1
        const_1 = g.add_op("constant", [], output_types=[f32], attrs={"value": 1.0})[0]
        [x_plus_1] = g.add_op("add", [x, const_1])

        # Two paths use x_plus_1
        [y] = g.add_op("mul", [x_plus_1, const_1])
        [z] = g.add_op("sub", [x_plus_1, const_1])

        # Combine results
        [result] = g.add_op("add", [y, z])
        g.add_output(result)

        # x_plus_1 should have 2 uses
        assert x_plus_1.num_uses == 2

        ir = g.to_string()
        # Should not duplicate x + 1 computation
        assert ir.count("add(x,") == 1


# ==============================================================================
# --- Example from module docstring
# ==============================================================================


def test_example_from_docstring():
    """Test the example from the graph.py module docstring."""
    graph = Graph()

    # Create values
    x = graph.add_input("x", Tensor[f32, (10,)])
    y = graph.add_input("y", Tensor[f32, (10,)])

    # Add operations
    [z] = graph.add_op("add", [x, y])
    const_2 = graph.add_op("constant", [], output_types=[f32], attrs={"value": 2.0})[0]
    [result] = graph.add_op("mul", [z, const_2])

    # Mark outputs
    graph.add_output(result)

    # Print IR
    ir_str = graph.to_string()

    # Verify expected structure
    assert "input" in ir_str
    assert "constant" in ir_str
    assert "add" in ir_str
    assert "mul" in ir_str
    assert "return" in ir_str
