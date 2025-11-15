"""Function dialect: generic region-based call + definition primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mplang.edsl as el
import mplang.edsl.typing as elt

func_def_p = el.Primitive("func.func")
call_p = el.Primitive("func.call")
FUNCTION_TYPE = elt.CustomType("function")


def _current_tracer() -> el.Tracer:
    ctx = el.get_current_context()
    if not isinstance(ctx, el.Tracer):
        raise TypeError(f"Expected Tracer context, got {type(ctx)}")
    return ctx


@func_def_p.def_trace
def _func_trace(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> el.TraceObject:
    tracer = _current_tracer()
    traced = el.trace(fn, *args, **kwargs)
    attrs = {
        "sym_name": traced.name,
        "in_var_pos": traced.in_var_pos,
        "in_imms": traced.in_imms,
        "out_var_pos": traced.out_var_pos,
        "out_imms": traced.out_imms,
        "in_tree": traced.in_tree,
        "out_tree": traced.out_tree,
        "output_types": [val.type for val in traced.graph.outputs],
    }
    [value] = tracer.graph.add_op(
        opcode="func.func",
        inputs=[],
        output_types=[FUNCTION_TYPE],
        attrs=attrs,
        regions=[traced.graph],
    )
    return el.TraceObject(value, tracer)


@call_p.def_trace
def _call_trace(
    fn_value: el.TraceObject, *args: Any
) -> el.TraceObject | list[el.TraceObject]:
    tracer = _current_tracer()
    if not isinstance(fn_value, el.TraceObject) or fn_value.type != FUNCTION_TYPE:
        raise TypeError("func.call expects the callee TraceObject as first argument")
    if not all(isinstance(arg, el.TraceObject) for arg in args):
        raise TypeError("func.call arguments must be TraceObjects")

    defining_op = fn_value._graph_value.defining_op
    if defining_op is None or defining_op.attrs.get("sym_name") is None:
        raise ValueError("Function value must originate from func.func")
    out_types = defining_op.attrs.get("output_types", [])

    result_values = tracer.graph.add_op(
        opcode="func.call",
        inputs=[fn_value._graph_value, *[arg._graph_value for arg in args]],
        output_types=out_types,
        attrs={"callee": defining_op.attrs["sym_name"]},
        regions=[],
    )
    return tracer.reconstruct_outputs(
        defining_op.attrs["out_var_pos"],
        defining_op.attrs["out_imms"],
        defining_op.attrs["out_tree"],
        result_values,
    )


def func(
    fn: Callable[..., Any], *args: el.TraceObject, **kwargs: Any
) -> el.TraceObject:
    return func_def_p.bind(fn, *args, **kwargs)


def call(fn_value: el.TraceObject, *args: el.TraceObject):
    return call_p.bind(fn_value, *args)


__all__ = ["call", "call_p", "func", "func_def_p"]
