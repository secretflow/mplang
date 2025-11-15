"""Function dialect: generic region-based call + definition primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mplang.edsl.context import get_current_context
from mplang.edsl.primitive import Primitive
from mplang.edsl.tracer import TraceObject, Tracer, reconstruct_outputs
from mplang.edsl.tracer import trace as trace_fn
from mplang.edsl.typing import CustomType

func_def_p = Primitive("func.func")
call_p = Primitive("func.call")
FUNCTION_TYPE = CustomType("function")


def _current_tracer() -> Tracer:
    ctx = get_current_context()
    if not isinstance(ctx, Tracer):
        raise TypeError(f"Expected Tracer context, got {type(ctx)}")
    return ctx


@func_def_p.def_trace
def _func_trace(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> TraceObject:
    tracer = _current_tracer()
    traced = trace_fn(fn, *args, **kwargs)
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
    return TraceObject(value, tracer)


def _reconstruct_outputs(
    tracer: Tracer,
    values: list[Any],
    out_var_pos: list[int],
    out_imms: list[Any],
    out_tree: Any,
) -> Any:
    var_objs = [TraceObject(val, tracer) for val in values]
    total_len = len(out_imms) + len(out_var_pos)
    flat_out: list[Any] = []
    var_iter = iter(var_objs)
    var_positions = iter(out_var_pos)
    next_var_pos = next(var_positions, None)
    imm_idx = 0
    for idx in range(total_len):
        if next_var_pos is not None and idx == next_var_pos:
            flat_out.append(next(var_iter))
            next_var_pos = next(var_positions, None)
        else:
            flat_out.append(out_imms[imm_idx])
            imm_idx += 1
    return out_tree.unflatten(flat_out)


@call_p.def_trace
def _call_trace(fn_value: TraceObject, *args: Any) -> TraceObject | list[TraceObject]:
    tracer = _current_tracer()
    if not isinstance(fn_value, TraceObject) or fn_value.type != FUNCTION_TYPE:
        raise TypeError("func.call expects the callee TraceObject as first argument")
    if not all(isinstance(arg, TraceObject) for arg in args):
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
    return reconstruct_outputs(
        tracer,
        result_values,
        defining_op.attrs["out_var_pos"],
        defining_op.attrs["out_imms"],
        defining_op.attrs["out_tree"],
    )


def func(fn: Callable[..., Any], *args: TraceObject, **kwargs: Any) -> TraceObject:
    return func_def_p.bind(fn, *args, **kwargs)


def call(fn_value: TraceObject, *args: TraceObject):
    return call_p.bind(fn_value, *args)


__all__ = ["call", "call_p", "func", "func_def_p"]
