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

from mplang.v2.edsl.graph import Graph, Operation, Value


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
        self._format_graph(graph, lines, indent_level=0, heading=None)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write(self, lines: list[str], indent_level: int, text: str) -> None:
        indent = " " * (indent_level * self.indent_size)
        lines.append(f"{indent}{text}")

    def _format_graph(
        self, graph: Graph, lines: list[str], indent_level: int, heading: str | None
    ) -> None:
        header_prefix = f"{heading}" if heading else ""
        params_str = self._format_params(graph.inputs)
        self._write(lines, indent_level, f"{header_prefix}{params_str} {{")

        for op in graph.operations:
            self._format_operation(op, lines, indent_level + 1)

        if graph.outputs:
            out_names = ", ".join(val.name for val in graph.outputs)
            self._write(lines, indent_level + 1, f"return {out_names}")

        self._write(lines, indent_level, "}")

    def _format_params(self, inputs: list[Value]) -> str:
        if not inputs:
            return "()"
        parts: list[str] = []
        for value in inputs:
            if self.show_types:
                parts.append(f"{value.name}: {value.type}")
            else:
                parts.append(f"{value.name}")
        joined = ", ".join(parts)
        return f"({joined})"

    def _format_operation(
        self, op: Operation, lines: list[str], indent_level: int
    ) -> None:
        lhs = self._format_outputs(op.outputs)
        inputs_str = ", ".join(val.name for val in op.inputs)
        attrs_str = self._format_attrs(op.attrs)
        type_str = self._format_output_types(op.outputs)
        op_line = f"{lhs} = {op.opcode}({inputs_str}){attrs_str}{type_str}"
        if op.regions:
            self._write(lines, indent_level, f"{op_line} {{")
            for region in op.regions:
                self._format_graph(region, lines, indent_level + 1, heading=None)
            self._write(lines, indent_level, "}")
        else:
            self._write(lines, indent_level, op_line)

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
