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

"""Generic kernel implementations for the `func` dialect.

Design: Function as Value
- func.func impl: Returns a FunctionValue wrapping the traced graph.
- func.call impl: Executes the function graph with provided arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from mplang.v2.dialects.func import call_p, func_def_p
from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Graph, Operation
from mplang.v2.runtime.value import Value

if TYPE_CHECKING:
    from typing import Self


@serde.register_class
class FunctionValue(Value):
    """Runtime representation of a traced function.

    This is a first-class runtime Value that wraps a Graph.
    Produced by func.func, consumed by func.call.

    Semantic rationale:
        In the interpreter's computation model, Values are data that flow
        between Operations. A function (Graph) is just another kind of data
        that can be passed around, stored, and invoked - hence it's a Value.
    """

    _serde_kind: ClassVar[str] = "func.FunctionValue"

    def __init__(self, graph: Graph, name: str = "anonymous") -> None:
        self._graph = graph
        self._name = name

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"FunctionValue({self._name!r}, ops={len(self._graph.operations)})"

    def to_json(self) -> dict[str, Any]:
        """Serialize function to JSON."""
        return {
            "graph": serde.to_json(self._graph),
            "name": self._name,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """Deserialize function from JSON."""
        graph = serde.from_json(data["graph"])
        return cls(graph=graph, name=data.get("name", "anonymous"))


@func_def_p.def_impl
def _func_def_impl(interpreter: Any, op: Operation, *args: Any) -> FunctionValue:
    """Implementation of func.func: return a FunctionValue wrapping the body graph."""
    if not op.regions:
        raise ValueError("func.func operation missing body region")

    name = op.attrs.get("sym_name", "anonymous")
    return FunctionValue(graph=op.regions[0], name=name)


@call_p.def_impl
def _call_impl(
    interpreter: Any, op: Operation, fn_obj: FunctionValue, *args: Any
) -> Any:
    """Implementation of func.call: execute the function graph.

    Args:
        interpreter: The interpreter instance.
        op: The func.call operation.
        fn_obj: The FunctionValue returned by func.func.
        *args: Arguments to pass to the function.
    """
    if not isinstance(fn_obj, FunctionValue):
        raise TypeError(f"func.call expects FunctionValue, got {type(fn_obj)}")

    call_args = list(args)
    result = interpreter.evaluate_graph(fn_obj.graph, call_args)
    # Return single value or list based on graph outputs
    return result[0] if len(fn_obj.graph.outputs) == 1 else result
