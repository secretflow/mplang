"""Tensor dialect: tensor ops backed by plaintext/private JAX execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Any

import jax
import numpy as np
from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef

from mplang.edsl.context import get_current_context
from mplang.edsl.primitive import Primitive
from mplang.edsl.tracer import TraceObject, Tracer
from mplang.edsl.typing import BaseType, ScalarType
from mplang.edsl.typing import TensorType as EDSLTensorType
from mplang.edsl.typing import f32 as edsl_f32
from mplang.edsl.typing import f64 as edsl_f64
from mplang.edsl.typing import i32 as edsl_i32
from mplang.edsl.typing import i64 as edsl_i64
from mplang.utils.func_utils import normalize_fn

run_jax_p = Primitive("tensor.run_jax")

_SCALAR_TO_NP_DTYPE = {
    "f32": np.dtype("float32"),
    "f64": np.dtype("float64"),
    "i32": np.dtype("int32"),
    "i64": np.dtype("int64"),
}

_NP_DTYPE_NAME_TO_SCALAR = {
    "float32": edsl_f32,
    "float64": edsl_f64,
    "int32": edsl_i32,
    "int64": edsl_i64,
}


@dataclass
class _RunJaxCompilation:
    """Book-keeping for compiled tensor.run_jax functions."""

    fn: Callable[..., Any]
    backend: str
    stablehlo: str
    out_tree: PyTreeDef
    output_types: list[BaseType]


@dataclass(frozen=True)
class RunJaxCompilationInfo:
    """Public view of a compiled tensor.run_jax function."""

    stablehlo: str
    out_tree: PyTreeDef
    output_types: list[BaseType]


_RUN_JAX_REGISTRY: dict[str, _RunJaxCompilation] = {}
_RUN_JAX_ID_GENERATOR = count()


def _current_tracer() -> Tracer:
    ctx = get_current_context()
    if not isinstance(ctx, Tracer):
        raise TypeError(f"Expected Tracer context, got {type(ctx)}")
    return ctx


def _scalar_to_numpy_dtype(scalar: ScalarType) -> np.dtype:
    dtype = _SCALAR_TO_NP_DTYPE.get(str(scalar))
    if dtype is None:
        raise TypeError(f"Unsupported scalar type '{scalar}' for tensor.run_jax")
    return dtype


def _numpy_dtype_to_scalar(dtype: Any) -> ScalarType:
    np_dtype = np.dtype(dtype)
    scalar = _NP_DTYPE_NAME_TO_SCALAR.get(np_dtype.name)
    if scalar is None:
        raise TypeError(f"tensor.run_jax received unsupported dtype '{np_dtype.name}'")
    return scalar


def _tensor_type_to_placeholder(tensor_type: EDSLTensorType) -> ShapeDtypeStruct:
    if tensor_type.shape is None:
        raise TypeError("tensor.run_jax requires fully-ranked tensor shapes")
    normalized_shape: list[int] = []
    for idx, dim in enumerate(tensor_type.shape):
        if dim is None:
            raise TypeError(
                f"tensor.run_jax argument dimension {idx} is None; "
                "please provide a static dimension."
            )
        if dim == -1:
            raise TypeError(
                "tensor.run_jax does not yet support dynamic (-1) dimensions"
            )
        if dim <= 0 and dim != 0:
            raise ValueError(f"Invalid tensor dimension {dim}")
        normalized_shape.append(dim)
    dtype = _scalar_to_numpy_dtype(tensor_type.element_type)
    return ShapeDtypeStruct(tuple(normalized_shape), dtype)


def _out_info_to_edsl(out_info: Any) -> EDSLTensorType:
    scalar = _numpy_dtype_to_scalar(out_info.dtype)
    shape = tuple(out_info.shape)
    return EDSLTensorType(scalar, shape)


def _register_compilation(compilation: _RunJaxCompilation) -> str:
    compilation_id = f"tensor.run_jax::{next(_RUN_JAX_ID_GENERATOR)}"
    _RUN_JAX_REGISTRY[compilation_id] = compilation
    return compilation_id


def get_run_jax_compilation(compilation_id: str) -> RunJaxCompilationInfo:
    try:
        record = _RUN_JAX_REGISTRY[compilation_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown tensor.run_jax compilation id '{compilation_id}'"
        ) from exc
    return RunJaxCompilationInfo(
        stablehlo=record.stablehlo,
        out_tree=record.out_tree,
        output_types=list(record.output_types),
    )


def _compile_run_jax(
    fn: Callable[..., Any],
    normalized_fn: Callable[..., Any],
    placeholders: list[ShapeDtypeStruct],
    backend: str,
) -> tuple[_RunJaxCompilation, str]:
    jit_backend = None if backend in (None, "", "plaintext") else backend
    jitted = jax.jit(normalized_fn, backend=jit_backend)
    lowered = jitted.lower(*placeholders)
    stablehlo_text = str(lowered.compiler_ir("stablehlo"))
    output_types = [_out_info_to_edsl(info) for info in lowered.out_info]
    compilation = _RunJaxCompilation(
        fn=fn,
        backend=backend,
        stablehlo=stablehlo_text,
        out_tree=lowered.out_tree,
        output_types=output_types,
    )
    compilation_id = _register_compilation(compilation)
    return compilation, compilation_id


def _prepare_run_jax_arguments(
    fn: Callable[..., Any],
    call_args: tuple[Any, ...],
    user_kwargs: dict[str, Any],
) -> tuple[Callable[..., Any], list[ShapeDtypeStruct], list[TraceObject]]:
    def _is_trace_object(value: Any) -> bool:
        return isinstance(value, TraceObject)

    normalized_fn, variables = normalize_fn(
        fn, call_args, user_kwargs, _is_trace_object
    )

    trace_objects: list[TraceObject] = []
    placeholders: list[ShapeDtypeStruct] = []
    for var in variables:
        if not isinstance(var, TraceObject):
            raise TypeError(
                f"tensor.run_jax expected TraceObject variables, got {type(var)}"
            )
        arg_type = var.type
        if not isinstance(arg_type, EDSLTensorType):
            raise TypeError(
                "tensor.run_jax only supports Tensor arguments; "
                f"got {arg_type} for argument {var}"
            )
        trace_objects.append(var)
        placeholders.append(_tensor_type_to_placeholder(arg_type))

    def normalized_wrapped(*dynamic_args: Any) -> Any:
        return normalized_fn(list(dynamic_args))

    return normalized_wrapped, placeholders, trace_objects


def _run_jax_trace(
    fn: Callable[..., Any],
    *call_args: Any,
    backend: str = "plaintext",
    _user_kwargs: dict[str, Any] | None = None,
) -> TraceObject | list[TraceObject]:
    if not callable(fn):
        raise TypeError(f"run_jax expects callable, got {type(fn)}")
    tracer = _current_tracer()

    normalized_fn, placeholders, dynamic_trace_objects = _prepare_run_jax_arguments(
        fn, tuple(call_args), dict(_user_kwargs or {})
    )

    if not dynamic_trace_objects:
        raise TypeError("tensor.run_jax requires at least one Tensor argument")

    compilation, text_ref = _compile_run_jax(fn, normalized_fn, placeholders, backend)

    input_values = [arg._graph_value for arg in dynamic_trace_objects]
    result_values = tracer.graph.add_op(
        opcode="tensor.run_jax",
        inputs=input_values,
        output_types=compilation.output_types,
        attrs={
            "ir_type": "stablehlo",
            "text_ref": text_ref,
        },
    )
    outputs = [TraceObject(val, tracer) for val in result_values]
    return outputs[0] if len(outputs) == 1 else outputs


def run_jax(
    fn: Callable[..., Any],
    *args: Any,
    backend: str = "plaintext",
    **kwargs: Any,
) -> TraceObject | list[TraceObject]:
    """Trace a tensor JAX function as a graph op.

    Args:
        fn: Callable that accepts JAX-compatible tensors.
        *args: Positional arguments to the callable. TraceObjects are treated
            as dynamic tensors, while non-Object values become static parameters.
        backend: Execution backend identifier.
        **kwargs: Keyword arguments for the callable. TraceObjects are treated
            as dynamic tensors, while non-Object values become static parameters.
    """

    return run_jax_p.bind(fn, *args, backend=backend, _user_kwargs=kwargs)


__all__ = ["RunJaxCompilationInfo", "get_run_jax_compilation", "run_jax", "run_jax_p"]
