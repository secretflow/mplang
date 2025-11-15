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

"""Pretty printer for the EDSL Graph IR."""

from __future__ import annotations

from typing import Any

from mplang.edsl.graph import Graph, Operation, Value


class GraphPrinter:
    """Format Graph IR in a readable, MLIR-like style."""

    def __init__(
        self,
        *,
        indent_size: int = 2,
        show_types: bool = True,
        show_attrs: bool = True,
    ):
        self.indent_size = indent_size
        self.show_types = show_types
        self.show_attrs = show_attrs

    def format(self, graph: Graph) -> str:
        """Return a formatted string representation of `graph`."""
        lines: list[str] = []
        self._format_graph(graph, lines, indent_level=0)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write(self, lines: list[str], indent_level: int, text: str) -> None:
        indent = " " * (indent_level * self.indent_size)
        lines.append(f"{indent}{text}")

    def _format_graph(self, graph: Graph, lines: list[str], indent_level: int) -> None:
        for value in graph.inputs:
            self._write(lines, indent_level, self._format_input(value))

        for op in graph.operations:
            self._write(lines, indent_level, self._format_operation(op))
            for region_idx, region in enumerate(op.regions):
                self._write(lines, indent_level + 1, f"region {region_idx} {{")
                self._format_graph(region, lines, indent_level + 2)
                self._write(lines, indent_level + 1, "}")

        if graph.outputs:
            out_names = ", ".join(val.name for val in graph.outputs)
            self._write(lines, indent_level, f"return {out_names}")

    def _format_input(self, value: Value) -> str:
        type_str = f" : {value.type}" if self.show_types else ""
        return f"{value.name} = input{type_str}"

    def _format_operation(self, op: Operation) -> str:
        lhs = self._format_outputs(op.outputs)
        inputs_str = ", ".join(val.name for val in op.inputs)
        attrs_str = self._format_attrs(op.attrs)
        type_str = self._format_output_types(op.outputs)
        return f"{lhs} = {op.opcode}({inputs_str}){attrs_str}{type_str}"

    def _format_outputs(self, outputs: list[Value]) -> str:
        if not outputs:
            return "[]"
        if len(outputs) == 1:
            return outputs[0].name
        return "[" + ", ".join(val.name for val in outputs) + "]"

    def _format_attrs(self, attrs: dict[str, Any]) -> str:
        if not self.show_attrs or not attrs:
            return ""
        parts = [f"{key}={attrs[key]!r}" for key in sorted(attrs)]
        return " {" + ", ".join(parts) + "}"

    def _format_output_types(self, outputs: list[Value]) -> str:
        if not self.show_types or not outputs:
            return ""
        type_strings = [str(val.type) for val in outputs]
        if len(type_strings) == 1:
            return f" : {type_strings[0]}"
        return " : (" + ", ".join(type_strings) + ")"


def format_graph(graph: Graph, **kwargs: Any) -> str:
    """Convenience helper that returns `GraphPrinter(**kwargs).format(graph)`."""
    return GraphPrinter(**kwargs).format(graph)
