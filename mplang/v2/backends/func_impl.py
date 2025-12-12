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
- func.func impl: Returns a FunctionObject wrapping the traced graph.
- func.call impl: Executes the function graph with provided arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mplang.v2.dialects.func import call_p, func_def_p
from mplang.v2.edsl.graph import Graph, Operation


@dataclass
class FunctionObject:
    """Runtime representation of a traced function.

    Wraps a Graph that can be invoked via func.call.
    """
    graph: Graph
    name: str


@func_def_p.def_impl
def _func_def_impl(interpreter: Any, op: Operation, *args: Any) -> FunctionObject:
    """Implementation of func.func: return a FunctionObject wrapping the body graph."""
    if not op.regions:
        raise ValueError("func.func operation missing body region")

    name = op.attrs.get("sym_name", "anonymous")
    return FunctionObject(graph=op.regions[0], name=name)


@call_p.def_impl
def _call_impl(interpreter: Any, op: Operation, fn_obj: FunctionObject, *args: Any) -> Any:
    """Implementation of func.call: execute the function graph.

    Args:
        interpreter: The interpreter instance.
        op: The func.call operation.
        fn_obj: The FunctionObject returned by func.func.
        *args: Arguments to pass to the function.
    """
    if not isinstance(fn_obj, FunctionObject):
        raise TypeError(f"func.call expects FunctionObject, got {type(fn_obj)}")

    call_args = list(args)
    return interpreter.evaluate_graph(fn_obj.graph, call_args)
