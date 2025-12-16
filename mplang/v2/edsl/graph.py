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
Graph IR: Operation List + SSA Values.

This module implements a modern, flat IR representation inspired by torch.fx
and JAX jaxpr, replacing the tree-based Expr system.

Key Design Principles:
----------------------
1. **Flat Structure**: Operations in a list, not a tree
2. **SSA Form**: Each value defined once, use-def chains explicit
3. **Easy Traversal**: No visitor pattern needed
4. **Optimization-Friendly**: Dead code elimination, fusion, etc.

Example:
--------
    from mplang.v2.edsl.graph import Graph, Operation, Value
    from mplang.v2.edsl.typing import Tensor, f32

    graph = Graph()

    # Create values
    x = graph.add_input("x", Tensor[f32, (10,)])
    y = graph.add_input("y", Tensor[f32, (10,)])

    # Add operations
    z, = graph.add_op("add", [x, y])
    scale, = graph.add_op("tensor.constant", [], output_types=[f32], attrs={"data": 2.0})
    result, = graph.add_op("mul", [z, scale])

    # Mark outputs
    graph.add_output(result)

    # Print IR
    print(graph.to_string())
    # Output:
    # %0 = input "x" : Tensor[f32, (10,)]
    # %1 = input "y" : Tensor[f32, (10,)]
    # %2 = tensor.constant {data=2.0} : f32
    # %3 = add %0, %1 : Tensor[f32, (10,)]
    # %4 = mul %3, %2 : Tensor[f32, (10,)]
    # return %4
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from mplang.v2.edsl import serde
from mplang.v2.edsl.typing import BaseType


@dataclass
class Value:
    """SSA value in the IR.

    Each value is defined exactly once by an operation (or is an input).
    Values track their uses and defining operation for def-use chain analysis.

    Attributes:
        name: Unique SSA name (e.g., "%0", "%1", ...)
        type: Type of this value (from mplang.v2.edsl.typing)
        defining_op: Operation that produces this value (None for inputs)
        uses: List of operations that consume this value
    """

    name: str
    type: BaseType
    defining_op: Operation | None = None
    uses: dict[Operation, None] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Value({self.name}: {self.type})"

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def add_use(self, op: Operation) -> None:
        """Register an operation that uses this value."""
        self.uses[op] = None

    def remove_use(self, op: Operation) -> None:
        """Unregister an operation that uses this value."""
        if op in self.uses:
            del self.uses[op]

    @property
    def num_uses(self) -> int:
        """Number of operations using this value."""
        return len(self.uses)

    @property
    def is_dead(self) -> bool:
        """True if this value is never used (dead code)."""
        return self.num_uses == 0 and self.defining_op is not None

    @property
    def is_bound(self) -> bool:
        """True if this value is bound (defined by an operation)."""
        return self.defining_op is not None

    @property
    def is_free(self) -> bool:
        """True if this value is free (graph input, not defined by operation)."""
        return self.defining_op is None


@dataclass
class Operation:
    """Single operation in the IR.

    Operations represent computations in the graph. They consume input values
    and produce output values.

    Attributes:
        opcode: Operation name (e.g., "add", "mul", "cond")
        inputs: Input values consumed by this operation
        outputs: Output values produced by this operation
        attrs: Additional attributes (e.g., shape, dtype, backend-specific)
        regions: Nested graphs (for control flow: cond, while)
    """

    opcode: str
    inputs: list[Value]
    outputs: list[Value]
    attrs: dict[str, Any] = field(default_factory=dict)
    regions: list[Graph] = field(default_factory=list)
    name: str = field(default="")

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __post_init__(self) -> None:
        """Register this operation as the definer and user of values."""
        # Register as defining op for outputs
        for output in self.outputs:
            output.defining_op = self

        # Register as user for inputs
        for input_val in self.inputs:
            input_val.add_use(self)

    def __repr__(self) -> str:
        inputs_str = ", ".join(str(v) for v in self.inputs)
        outputs_str = ", ".join(str(v) for v in self.outputs)
        return f"Operation({self.opcode}: {inputs_str} -> {outputs_str})"

    def replace_input(self, old: Value, new: Value) -> None:
        """Replace an input value (updates use-def chains)."""
        for i, inp in enumerate(self.inputs):
            if inp is old:
                self.inputs[i] = new
                old.remove_use(self)
                new.add_use(self)

    def erase(self) -> None:
        """Remove this operation (updates use-def chains)."""
        for inp in self.inputs:
            inp.remove_use(self)
        for out in self.outputs:
            out.defining_op = None


@serde.register_class
class Graph:
    """Computation graph as a flat list of operations.

    A graph contains:
    - Inputs: Named input values
    - Operations: Flat list of computations
    - Outputs: Values returned from the graph
    - Values: All SSA values in the graph

    Example:
        graph = Graph()
        x = graph.add_input("x", Tensor[f32, (10,)])
        y, = graph.add_op("tensor.constant", [], output_types=[f32], attrs={"data": 1.0})
        z, = graph.add_op("add", [x, y])
        graph.add_output(z)
    """

    _serde_kind: ClassVar[str] = "mplang.Graph"

    def __init__(self) -> None:
        self.operations: list[Operation] = []
        self.values: dict[str, Value] = {}
        self.inputs: list[Value] = []
        self.outputs: list[Value] = []
        self._value_counter = 0
        self._op_counter = 0

    def _gen_value_name(self) -> str:
        """Generate a unique SSA value name."""
        name = f"%{self._value_counter}"
        self._value_counter += 1
        return name

    def add_value(self, type: BaseType, name: str | None = None) -> Value:
        """Create a new SSA value.

        Args:
            type: Type of the value
            name: Optional custom name (auto-generated if None)

        Returns:
            New Value instance
        """
        if name is None:
            name = self._gen_value_name()

        if name in self.values:
            raise ValueError(f"Value {name} already exists")

        value = Value(name, type)
        self.values[name] = value
        return value

    def add_input(self, name: str, type: BaseType) -> Value:
        """Add a graph input.

        Args:
            name: Input parameter name
            type: Type of the input

        Returns:
            Input value
        """
        value = self.add_value(type, name=name)
        self.inputs.append(value)
        return value

    def add_op(
        self,
        opcode: str,
        inputs: list[Value],
        output_types: Sequence[BaseType] | None = None,
        attrs: dict[str, Any] | None = None,
        regions: list[Graph] | None = None,
    ) -> list[Value]:
        """Add an operation to the graph.

        Args:
            opcode: Operation name
            inputs: Input values
            output_types: Types of outputs (inferred if None)
            attrs: Additional attributes
            regions: Nested graphs (for control flow)

        Returns:
            List of output values (one entry per output)
        """
        # Type inference (placeholder - should be backend-specific)
        if output_types is None:
            # Simple rule: inherit from first input
            if inputs:
                output_types = [inputs[0].type]
            else:
                raise ValueError(f"Cannot infer type for {opcode} with no inputs")

        # Create output values
        outputs = [self.add_value(t) for t in output_types]

        # Create operation
        op_name = f"op{self._op_counter}"
        self._op_counter += 1
        op = Operation(
            opcode=opcode,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs or {},
            regions=regions or [],
            name=op_name,
        )
        self.operations.append(op)

        return outputs

    def add_output(self, value: Value) -> None:
        """Mark a value as a graph output.

        Args:
            value: Value to be returned from the graph
        """
        if value not in self.values.values():
            raise ValueError(f"Value {value} not in graph")
        self.outputs.append(value)

    def to_string(self, verbose: bool = False) -> str:
        """Generate human-readable IR representation.

        Args:
            verbose: Include type annotations

        Returns:
            String representation of the graph
        """
        lines = []

        # Print inputs
        for inp in self.inputs:
            type_str = f" : {inp.type}" if verbose else ""
            lines.append(f"{inp.name} = input{type_str}")

        # Print operations
        for op in self.operations:
            if op.opcode == "constant":
                value_str = op.attrs.get("value", "?")
                type_str = f" : {op.outputs[0].type}" if verbose else ""
                lines.append(f"{op.outputs[0].name} = constant {value_str}{type_str}")
            else:
                inputs_str = ", ".join(str(v) for v in op.inputs)
                outputs_str = ", ".join(str(v) for v in op.outputs)

                # Handle single vs multiple outputs
                if len(op.outputs) == 1:
                    lhs = str(op.outputs[0])
                else:
                    lhs = f"[{outputs_str}]"

                type_str = f" : {op.outputs[0].type}" if verbose and op.outputs else ""

                if op.attrs:
                    attrs_str = ", ".join(f"{k}={v}" for k, v in op.attrs.items())
                    lines.append(
                        f"{lhs} = {op.opcode}({inputs_str}) {{{attrs_str}}}{type_str}"
                    )
                else:
                    lines.append(f"{lhs} = {op.opcode}({inputs_str}){type_str}")

        # Print outputs
        if self.outputs:
            outputs_str = ", ".join(str(v) for v in self.outputs)
            lines.append(f"return {outputs_str}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Graph({len(self.operations)} ops, {len(self.values)} values)"

    def __str__(self) -> str:
        return self.to_string()

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_json(self) -> dict:
        """Serialize graph to JSON-compatible dict."""

        def _type_to_json(t: BaseType) -> dict:
            return serde.to_json(t)

        def _attr_to_json(value: Any) -> dict:
            return serde.to_json(value)

        def _attrs_to_json(attrs: dict[str, Any]) -> dict[str, Any]:
            return {k: _attr_to_json(v) for k, v in attrs.items()}

        return {
            "inputs": [
                {"name": v.name, "type": _type_to_json(v.type)} for v in self.inputs
            ],
            "operations": [
                {
                    "opcode": op.opcode,
                    "inputs": [v.name for v in op.inputs],
                    "outputs": [
                        {"name": v.name, "type": _type_to_json(v.type)}
                        for v in op.outputs
                    ],
                    "attrs": _attrs_to_json(op.attrs),
                    "regions": [serde.to_json(r) for r in op.regions],
                    "name": op.name,
                }
                for op in self.operations
            ],
            "outputs": [v.name for v in self.outputs],
        }

    @classmethod
    def from_json(cls, data: dict) -> Graph:
        """Deserialize graph from JSON-compatible dict."""

        def _type_from_json(d: dict) -> BaseType:
            result = serde.from_json(d)
            if not isinstance(result, BaseType):
                raise TypeError(f"Expected BaseType, got {type(result)}")
            return result

        def _attr_from_json(value: dict) -> Any:
            return serde.from_json(value)

        def _attrs_from_json(attrs: dict[str, Any]) -> dict[str, Any]:
            return {k: _attr_from_json(v) for k, v in attrs.items()}

        graph = cls()

        # Reconstruct inputs
        for inp_data in data["inputs"]:
            graph.add_input(inp_data["name"], _type_from_json(inp_data["type"]))

        # Reconstruct operations
        for op_data in data["operations"]:
            # Resolve input values by name
            inputs = [graph.values[name] for name in op_data["inputs"]]

            # Get output types
            output_types = [_type_from_json(out["type"]) for out in op_data["outputs"]]

            # Deserialize nested graphs (regions)
            regions = [serde.from_json(r) for r in op_data.get("regions", [])]

            # Add operation
            outputs = graph.add_op(
                op_data["opcode"],
                inputs,
                output_types=output_types,
                attrs=_attrs_from_json(op_data.get("attrs", {})),
                regions=regions,
            )

            # Rename outputs to match original names
            for out_val, out_data in zip(outputs, op_data["outputs"], strict=False):
                original_name = out_data["name"]
                if out_val.name != original_name:
                    # Update the values dict with the original name
                    del graph.values[out_val.name]
                    out_val.name = original_name
                    graph.values[original_name] = out_val

            # Set operation name if provided
            if op_data.get("name"):
                graph.operations[-1].name = op_data["name"]

        # Reconstruct outputs
        for name in data["outputs"]:
            graph.add_output(graph.values[name])

        return graph
