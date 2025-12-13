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

"""Function dialect: generic region-based call + definition primitives.

Design: Function as Value
- func.func: Defines a function and returns a function handle (TraceObject).
- func.call: Invokes a function using the handle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt

func_def_p = el.Primitive[el.TraceObject]("func.func")
call_p = el.Primitive[Any]("func.call")
FuncType = elt.CustomType("function")


def _current_tracer() -> el.Tracer:
    ctx = el.get_current_context()
    if not isinstance(ctx, el.Tracer):
        raise TypeError(f"Expected Tracer context, got {type(ctx)}")
    return ctx


@func_def_p.def_trace
def _func_trace(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> el.TraceObject:
    """Trace func.func: Define a function and return a handle.

    Returns a TraceObject representing the function, which can be passed to func.call.
    """
    tracer = _current_tracer()
    traced = el.trace(fn, *args, **kwargs)

    attrs = {
        "sym_name": traced.name,
        "in_var_pos": traced.in_var_pos,
        "in_imms": traced.in_imms,
        "in_tree": traced.in_tree,
        "out_var_pos": traced.out_var_pos,
        "out_imms": traced.out_imms,
        "out_tree": traced.out_tree,
        "output_types": [val.type for val in traced.graph.outputs],
    }

    # func.func returns a function handle (single output of FuncType)
    result_values = tracer.graph.add_op(
        opcode="func.func",
        inputs=[],
        output_types=[FuncType],
        attrs=attrs,
        regions=[traced.graph],
    )

    return el.TraceObject(result_values[0], tracer)


@call_p.def_trace
def _call_trace(fn_handle: el.TraceObject, *args: Any) -> Any:
    """Trace func.call: Invoke a function using its handle.

    The function handle carries output type and PyTree information.
    """
    tracer = _current_tracer()

    if not isinstance(fn_handle, el.TraceObject):
        raise TypeError(
            f"func.call expects TraceObject as function handle, got {type(fn_handle)}"
        )
    if not all(isinstance(arg, el.TraceObject) for arg in args):
        raise TypeError("func.call arguments must be TraceObjects")

    # Get output types and PyTree from the func.func operation that produced fn_handle
    fn_op = fn_handle._graph_value.defining_op
    if fn_op is None:
        raise ValueError("Function handle has no defining operation")
    output_types = fn_op.attrs.get("output_types", [elt.TensorType(elt.i64, ())])
    out_tree = fn_op.attrs.get("out_tree")

    result_values = tracer.graph.add_op(
        opcode="func.call",
        inputs=[fn_handle._graph_value] + [arg._graph_value for arg in args],
        output_types=output_types,
        attrs={},
        regions=[],
    )

    traced_results = [el.TraceObject(v, tracer) for v in result_values]

    # Restructure outputs using PyTree if available
    if out_tree is not None:
        try:
            return out_tree.unflatten(traced_results)
        except ValueError as e:
            import warnings

            warnings.warn(
                f"Failed to unflatten PyTree for func.call: {e}", stacklevel=2
            )

    # Single result: return directly
    if len(result_values) == 1:
        return traced_results[0]
    return traced_results


def func(
    fn: Callable[..., Any], *args: el.TraceObject, **kwargs: Any
) -> el.TraceObject:
    """Define a function and return its handle."""
    return func_def_p.bind(fn, *args, **kwargs)


def call(fn_handle: el.TraceObject, *args: el.TraceObject) -> Any:
    """Call a function using its handle."""
    return call_p.bind(fn_handle, *args)


__all__ = ["call", "call_p", "func", "func_def_p"]
