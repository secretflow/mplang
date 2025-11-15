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
    from mplang.edsl.graph import Graph, Operation, Value
    from mplang.edsl.typing import Tensor, f32

    graph = Graph()

    # Create values
    x = graph.add_input("x", Tensor[f32, (10,)])
    y = graph.add_input("y", Tensor[f32, (10,)])

    # Add operations
    z, = graph.add_op("add", [x, y])
    result, = graph.add_op("mul", [z, graph.add_constant(2.0)])

    # Mark outputs
    graph.add_output(result)

    # Print IR
    print(graph.to_string())
    # Output:
    # %0 = input "x" : Tensor[f32, (10,)]
    # %1 = input "y" : Tensor[f32, (10,)]
    # %2 = constant 2.0 : f32
    # %3 = add %0, %1 : Tensor[f32, (10,)]
    # %4 = mul %3, %2 : Tensor[f32, (10,)]
    # return %4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mplang.edsl.typing import BaseType


@dataclass
class Value:
    """SSA value in the IR.

    Each value is defined exactly once by an operation (or is an input/constant).
    Values track their uses and defining operation for def-use chain analysis.

    Attributes:
        name: Unique SSA name (e.g., "%0", "%1", ...)
        type: Type of this value (from mplang.edsl.typing)
        defining_op: Operation that produces this value (None for inputs)
        uses: List of operations that consume this value
    """

    name: str
    type: BaseType
    defining_op: Operation | None = None
    uses: list[Operation] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Value({self.name}: {self.type})"

    def __str__(self) -> str:
        return self.name

    def add_use(self, op: Operation) -> None:
        """Register an operation that uses this value."""
        if op not in self.uses:
            self.uses.append(op)

    def remove_use(self, op: Operation) -> None:
        """Unregister an operation that uses this value."""
        if op in self.uses:
            self.uses.remove(op)

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
        y = graph.add_constant(1.0, f32)
        z = graph.add_op("add", [x, y])
        graph.add_output(z)
    """

    def __init__(self) -> None:
        self.operations: list[Operation] = []
        self.values: dict[str, Value] = {}
        self.inputs: list[Value] = []
        self.outputs: list[Value] = []
        self._value_counter = 0

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

    def add_constant(self, data: Any, type: BaseType) -> Value:
        """Add a constant value.

        Args:
            data: Constant data
            type: Type of the constant

        Returns:
            Constant value
        """
        value = self.add_value(type)
        op = Operation(
            opcode="constant",
            inputs=[],
            outputs=[value],
            attrs={"value": data},
        )
        self.operations.append(op)
        return value

    def add_op(
        self,
        opcode: str,
        inputs: list[Value],
        output_types: list[BaseType] | None = None,
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
        op = Operation(
            opcode=opcode,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs or {},
            regions=regions or [],
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

                # Attrs
                if op.attrs:
                    attrs_str = ", ".join(f"{k}={v}" for k, v in op.attrs.items())
                    lines.append(
                        f"{lhs} = {op.opcode}({inputs_str}, {attrs_str}){type_str}"
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


# Example usage
if __name__ == "__main__":
    from mplang.edsl.typing import Tensor, f32

    print("=== Graph IR Example ===\n")

    graph = Graph()

    # Build a simple computation: z = (x + y) * 2
    x = graph.add_input("x", Tensor[f32, (10,)])
    y = graph.add_input("y", Tensor[f32, (10,)])

    (sum_result,) = graph.add_op("add", [x, y])
    const_2 = graph.add_constant(2.0, f32)

    (result,) = graph.add_op("mul", [sum_result, const_2])
    graph.add_output(result)

    print("Simple IR:")
    print(graph.to_string())
    print()

    print("Verbose IR (with types):")
    print(graph.to_string(verbose=True))
    print()

    print(f"Graph summary: {graph}")
    print(f"Number of inputs: {len(graph.inputs)}")
    print(f"Number of operations: {len(graph.operations)}")
    print(f"Number of outputs: {len(graph.outputs)}")
