"""Tensor dialect: tensor ops backed by plaintext/private JAX execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from mplang.edsl.context import get_current_context
from mplang.edsl.primitive import Primitive
from mplang.edsl.tracer import TraceObject, Tracer
from mplang.edsl.typing import BaseType

run_jax_p = Primitive("tensor.run_jax")


def _current_tracer() -> Tracer:
    ctx = get_current_context()
    if not isinstance(ctx, Tracer):
        raise TypeError(f"Expected Tracer context, got {type(ctx)}")
    return ctx


def _normalize_out_types(out_types: BaseType | Sequence[BaseType]) -> list[BaseType]:
    if isinstance(out_types, BaseType):
        return [out_types]
    if isinstance(out_types, Sequence):
        normalized = list(out_types)
        if not normalized or not all(isinstance(t, BaseType) for t in normalized):
            raise TypeError("out_types sequence must contain BaseType entries")
        return normalized
    raise TypeError("out_types must be BaseType or sequence of BaseType")


def _qualname(fn: Callable[..., Any]) -> str:
    module = getattr(fn, "__module__", "__main__")
    qualname = getattr(fn, "__qualname__", fn.__name__)
    return f"{module}:{qualname}"


@run_jax_p.def_trace
def _run_jax_trace(
    fn: Callable[..., Any],
    *args: TraceObject,
    out_types: BaseType | Sequence[BaseType],
    backend: str = "plaintext",
    **static_kwargs: Any,
) -> TraceObject | list[TraceObject]:
    if not callable(fn):
        raise TypeError(f"run_jax expects callable, got {type(fn)}")
    tracer = _current_tracer()
    if not args:
        raise TypeError("run_jax requires at least one argument")
    if not all(isinstance(arg, TraceObject) for arg in args):
        raise TypeError("run_jax arguments must be TraceObject instances")

    input_values = [arg._graph_value for arg in args]
    output_types = _normalize_out_types(out_types)
    result_values = tracer.graph.add_op(
        opcode="tensor.run_jax",
        inputs=input_values,
        output_types=output_types,
        attrs={
            "fn": _qualname(fn),
            "backend": backend,
            "static_kwargs": static_kwargs,
        },
    )
    outputs = [TraceObject(val, tracer) for val in result_values]
    return outputs[0] if len(outputs) == 1 else outputs


def run_jax(
    fn: Callable[..., Any],
    *args: TraceObject,
    out_types: BaseType | Sequence[BaseType],
    backend: str = "plaintext",
    **static_kwargs: Any,
) -> TraceObject | list[TraceObject]:
    """Trace a tensor JAX function as a graph op."""

    return run_jax_p.bind(
        fn, *args, out_types=out_types, backend=backend, **static_kwargs
    )


__all__ = ["run_jax", "run_jax_p"]
