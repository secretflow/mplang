"""Tensor dialect: tensor ops backed by plaintext/private JAX execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Any, cast

import jax
import numpy as np
from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef

import mplang2.edsl as el
import mplang2.edsl.typing as elt
from mplang.utils.func_utils import normalize_fn

run_jax_p = el.Primitive("tensor.run_jax")

_SCALAR_TO_NP_DTYPE = {
    "f32": np.dtype("float32"),
    "f64": np.dtype("float64"),
    "i32": np.dtype("int32"),
    "i64": np.dtype("int64"),
}

_NP_DTYPE_NAME_TO_SCALAR = {
    "float32": elt.f32,
    "float64": elt.f64,
    "int32": elt.i32,
    "int64": elt.i64,
}


@dataclass
class RunJaxCompilation:
    """Compilation record for tensor.run_jax functions.

    Stores both the compilation artifacts (StableHLO MLIR, types, tree structure)
    and metadata needed for execution (arg_keep_map for JAX's unused param elimination).
    """

    fn: Callable[..., Any]
    stablehlo: str
    out_tree: PyTreeDef
    output_types: list[elt.BaseType]
    arg_keep_map: list[int] | None = None


_RUN_JAX_REGISTRY: dict[str, RunJaxCompilation] = {}
_RUN_JAX_ID_GENERATOR = count()


def _current_tracer() -> el.Tracer:
    ctx = el.get_current_context()
    if not isinstance(ctx, el.Tracer):
        raise TypeError(f"Expected Tracer context, got {type(ctx)}")
    return ctx


def _scalar_to_numpy_dtype(scalar: elt.ScalarType) -> np.dtype[np.generic]:
    dtype = _SCALAR_TO_NP_DTYPE.get(str(scalar))
    if dtype is None:
        raise TypeError(f"Unsupported scalar type '{scalar}' for tensor.run_jax")
    return cast(np.dtype[np.generic], dtype)


def _numpy_dtype_to_scalar(dtype: Any) -> elt.ScalarType:
    np_dtype = np.dtype(dtype)
    scalar = _NP_DTYPE_NAME_TO_SCALAR.get(np_dtype.name)
    if scalar is None:
        raise TypeError(f"tensor.run_jax received unsupported dtype '{np_dtype.name}'")
    return scalar


def _tensor_type_to_placeholder(tensor_type: elt.TensorType) -> ShapeDtypeStruct:
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
    # element_type is ScalarTrait, need to convert to ScalarType for _scalar_to_numpy_dtype
    if not isinstance(tensor_type.element_type, elt.ScalarType):
        raise TypeError(
            f"Expected ScalarType element, got {type(tensor_type.element_type)}"
        )
    dtype = _scalar_to_numpy_dtype(tensor_type.element_type)
    return ShapeDtypeStruct(tuple(normalized_shape), dtype)


def _out_info_to_edsl(out_info: Any) -> elt.TensorType:
    scalar = _numpy_dtype_to_scalar(out_info.dtype)
    shape = tuple(out_info.shape)
    return elt.TensorType(scalar, shape)


def _register_compilation(compilation: RunJaxCompilation) -> str:
    compilation_id = f"tensor.run_jax::{next(_RUN_JAX_ID_GENERATOR)}"
    _RUN_JAX_REGISTRY[compilation_id] = compilation
    return compilation_id


def get_run_jax_compilation(compilation_id: str) -> RunJaxCompilation:
    """Get compilation record by ID.

    Returns:
        The compilation record containing StableHLO MLIR, types, and metadata.
    """
    try:
        return _RUN_JAX_REGISTRY[compilation_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown tensor.run_jax compilation id '{compilation_id}'"
        ) from exc


def _compile_run_jax(
    fn: Callable[..., Any],
    normalized_fn: Callable[..., Any],
    placeholders: list[ShapeDtypeStruct],
) -> tuple[RunJaxCompilation, str]:
    """Compile JAX function to StableHLO MLIR.

    Pipeline: jit → lower → StableHLO MLIR

    Args:
        fn: Original JAX function
        normalized_fn: Function accepting list of variables (for JAX lower API)
        placeholders: JAX ShapeDtypeStruct list for lowering

    Returns:
        Tuple of (compilation record, compilation_id)
    """
    jitted = jax.jit(normalized_fn)
    lowered = jitted.lower(placeholders)
    stablehlo_text = str(lowered.compiler_ir("stablehlo"))

    # Handle JAX's unused parameter elimination
    arg_keep_map: list[int] | None = None
    try:
        compile_args = lowered._lowering.compile_args
        kept_var_idx = compile_args["kept_var_idx"]
        kept_indices = sorted(kept_var_idx)
        if len(kept_indices) < len(placeholders):
            arg_keep_map = kept_indices
    except (AttributeError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"Cannot access JAX's kept_var_idx for unused parameter handling. "
            f"JAX may have optimized away unused parameters. Error: {e}"
        ) from e

    # Convert output info to EDSL types
    output_types: list[elt.BaseType]
    if isinstance(lowered.out_info, tuple):
        output_types = [_out_info_to_edsl(info) for info in lowered.out_info]
    else:
        output_types = [_out_info_to_edsl(lowered.out_info)]

    compilation = RunJaxCompilation(
        fn=fn,
        stablehlo=stablehlo_text,
        out_tree=lowered.out_tree,
        output_types=output_types,
        arg_keep_map=arg_keep_map,
    )
    compilation_id = _register_compilation(compilation)
    return compilation, compilation_id


@run_jax_p.def_trace
def _run_jax_trace(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Trace tensor.run_jax primitive.

    Compiles JAX function to StableHLO and emits graph operation.

    Args:
        fn: JAX-compatible callable
        *args: Positional arguments (TraceObjects become dynamic, others static)
        **kwargs: Keyword arguments (TraceObjects become dynamic, others static)

    Returns:
        PyTree of TraceObjects matching fn's output structure
    """
    if not callable(fn):
        raise TypeError(f"run_jax expects callable, got {type(fn)}")

    tracer = _current_tracer()

    # Extract TraceObjects (dynamic args) from args/kwargs
    def _is_trace_object(value: Any) -> bool:
        return isinstance(value, el.TraceObject)

    normalized_fn, variables = normalize_fn(fn, args, kwargs, _is_trace_object)

    if not variables:
        raise TypeError("tensor.run_jax requires at least one Tensor argument")

    # Convert TraceObjects to JAX placeholders for compilation
    placeholders: list[ShapeDtypeStruct] = []
    for var in variables:
        if not isinstance(var, el.TraceObject):
            raise TypeError(f"Expected TraceObject, got {type(var)}")
        if not isinstance(var.type, elt.TensorType):
            raise TypeError(f"run_jax only supports Tensors, got {var.type}")
        placeholders.append(_tensor_type_to_placeholder(var.type))

    # Compile to StableHLO
    compilation, text_ref = _compile_run_jax(fn, normalized_fn, placeholders)

    # Emit graph operation
    input_values = [var._graph_value for var in variables]
    result_values = tracer.graph.add_op(
        opcode="tensor.run_jax",
        inputs=input_values,
        output_types=compilation.output_types,
        attrs={"ir_type": "stablehlo", "text_ref": text_ref},
    )

    # Reconstruct output PyTree (JAX outputs are all variables)
    out_var_pos = list(range(len(result_values)))
    return tracer.reconstruct_outputs(
        out_var_pos, [], compilation.out_tree, result_values
    )


def run_jax(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Trace a tensor JAX function as a graph op.

    Args:
        fn: Callable that accepts JAX-compatible tensors.
        *args: Positional arguments to the callable. TraceObjects are treated
            as dynamic tensors, while non-Object values become static parameters.
        **kwargs: Keyword arguments for the callable. TraceObjects are treated
            as dynamic tensors, while non-Object values become static parameters.

    Returns:
        PyTree of TraceObjects with the same structure as fn's output.
    """

    return run_jax_p.bind(fn, *args, **kwargs)


__all__ = ["RunJaxCompilation", "get_run_jax_compilation", "run_jax", "run_jax_p"]
