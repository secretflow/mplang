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

"""Tensor dialect: tensor ops backed by plaintext/private JAX execution.

Design Philosophy
-----------------
This dialect is intentionally *lightweight* — it focuses on **structural/shape
operations** (slice, reshape, transpose, gather, scatter, concat) rather than
full-fledged element-wise arithmetic.

Why not add bitwise_and / bitwise_or / arithmetic primitives here?

1. **Shape Dialect**: The primitives defined here perform *index arithmetic* on
   tensor metadata (offsets, strides, dim sizes). They don't interpret element
   values — that's left to the backend (JAX/XLA).

2. **Delegate to run_jax**: For element-wise logic (bitwise ops, arithmetic),
   use `tensor.run_jax(jnp.bitwise_xor, a, b)`. This leverages JAX's mature XLA
   backend without duplicating op definitions or abstract_eval rules for every
   possible JAX op.

3. **Type Preservation**: `run_jax` infers output types from JAX's shape/dtype
   inference, avoiding the need to re-implement type rules for hundreds of ops.

For domain-specific ops (GF(2^128) mul, AES expand), use dedicated dialects
like `field` which have optimized C++ kernel backends.

Helper Functions
----------------
- `bitcast(x, dtype)`: Type reinterpretation (SSA-safe, same bytes).
- For random tensor generation, see `crypto.random_tensor`.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Any, cast
from weakref import WeakKeyDictionary

import jax
import numpy as np
from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef, tree_flatten

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v1.utils.func_utils import normalize_fn
from mplang.v2.dialects import dtypes

run_jax_p = el.Primitive[Any]("tensor.run_jax")
constant_p = el.Primitive[el.Object]("tensor.constant")


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
    return np.dtype(dtypes.to_jax(scalar))  # type: ignore[no-any-return]


def _numpy_dtype_to_scalar(dtype: Any) -> elt.ScalarType:
    return dtypes.from_dtype(dtype)


def _tensor_type_to_placeholder(
    tensor_type: elt.TensorType | elt.ScalarType,
) -> ShapeDtypeStruct:
    if isinstance(tensor_type, elt.ScalarType):
        # Treat scalar as rank-0 tensor
        dtype = _scalar_to_numpy_dtype(tensor_type)
        return ShapeDtypeStruct((), dtype)

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
    # element_type must be ScalarType for conversion to numpy dtype
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
        if not isinstance(var.type, (elt.TensorType, elt.ScalarType)):
            raise TypeError(f"run_jax only supports Tensors/Scalars, got {var.type}")
        placeholders.append(_tensor_type_to_placeholder(var.type))

    # Compile to StableHLO
    compilation, text_ref = _compile_run_jax(fn, normalized_fn, placeholders)

    # Emit graph operation
    input_values = [var._graph_value for var in variables]
    result_values = tracer.graph.add_op(
        opcode="tensor.run_jax",
        inputs=input_values,
        output_types=compilation.output_types,
        attrs={
            "ir_type": "stablehlo",
            "text_ref": text_ref,
            "stablehlo_code": compilation.stablehlo,
            "arg_keep_map": compilation.arg_keep_map,
        },
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


def jax_fn(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a JAX function for use with pcall.

    This creates a callable that can be passed to pcall primitives,
    providing a cleaner user interface:

    Instead of:
        pcall_static((0,), lambda x, y: run_jax(native_fn, x, y), x_p0, y_p0)

    You can write:
        pcall_static((0,), jax_fn(native_fn), x_p0, y_p0)

    Args:
        fn: JAX function to wrap

    Returns:
        Wrapped function that calls run_jax when invoked

    Example:
        >>> def square(x):
        ...     return jnp.square(x)
        >>> wrapped = jax_fn(square)
        >>> result = pcall_static((0,), wrapped, x_p0)
    """

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return run_jax(fn, *args, **kwargs)

    # Preserve function name for better IR readability
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


@constant_p.def_trace
def _constant_trace(data: Any) -> el.TraceObject:
    """Create constant tensor from data.

    Args:
        data: Scalar, numpy array, or array-like object

    Returns:
        TraceObject with inferred tensor type

    Raises:
        TypeError: If data cannot be converted to a tensor
    """
    tracer = _current_tracer()

    # Unified numpy conversion for all data types
    np_array = np.array(data)
    dtype = _numpy_dtype_to_scalar(np_array.dtype)
    shape = tuple(np_array.shape)
    output_type: elt.TensorType = elt.TensorType(dtype, shape)

    # Emit graph operation with data as attribute
    # Use base64 encoded bytes for efficiency and precision
    data_b64 = base64.b64encode(np_array.tobytes()).decode("ascii")

    [value] = tracer.graph.add_op(
        opcode="tensor.constant",
        inputs=[],
        output_types=[output_type],
        attrs={
            "value_b64": data_b64,
        },
        regions=[],
    )

    return el.TraceObject(value, tracer)


# Constant cache: Tracer -> { (dtype, shape, bytes) -> Object }
_CONSTANT_CACHE: WeakKeyDictionary[
    el.Tracer, dict[tuple[str, tuple[int, ...], bytes], el.Object]
] = WeakKeyDictionary()


def constant(data: Any) -> el.Object:
    """Create a tensor constant value.

    This creates a constant tensor that can be used in tensor computations.
    The constant value is embedded directly into the computation graph.
    Duplicate constants (same data and shape) are cached per-Tracer to
    minimize graph size.

    Args:
        data: Constant data. Can be:
            - A scalar value (int, float, bool, complex)
            - A numpy array
            - Any array-like object that can be converted to numpy

    Returns:
        Object representing the constant tensor

    Raises:
        TypeError: If data cannot be converted to a tensor

    Example:
        >>> x = constant(3.14)  # Scalar constant
        >>> y = constant(np.array([1, 2, 3]))  # Array constant
        >>> z = constant([[1, 2], [3, 4]])  # Nested list constant
    """
    # Normalize data to numpy
    np_array = np.array(data)

    # Ensure canonical form for cache key
    key_shape = tuple(np_array.shape)
    key_dtype = np_array.dtype
    # Use simple bytes for cache key. For very large constants this might
    # be expensive, but typically constants in MPC are small (params, masks).
    key_bytes = np_array.tobytes()

    try:
        tracer = _current_tracer()
    except TypeError:
        # If no tracer is active (e.g. eager execution), skip caching logic
        # and fall back to standard bind which will handle eager/trace check.
        return cast(el.Object, constant_p.bind(np_array))

    inner_key = (str(key_dtype), key_shape, key_bytes)

    tracer_cache: dict[tuple[str, tuple[int, ...], bytes], el.Object] = (
        _CONSTANT_CACHE.setdefault(tracer, {})
    )
    if inner_key in tracer_cache:
        return tracer_cache[inner_key]

    # Create new constant
    obj = cast(el.Object, constant_p.bind(np_array))

    # Store in cache
    tracer_cache[inner_key] = obj
    return obj


# ==============================================================================
# --- Tensor Structural Operations (Element-type agnostic)
# ==============================================================================

transpose_p = el.Primitive[el.Object]("tensor.transpose")
reshape_p = el.Primitive[el.Object]("tensor.reshape")
concat_p = el.Primitive[el.Object]("tensor.concat")
gather_p = el.Primitive[el.Object]("tensor.gather")
scatter_p = el.Primitive[el.Object]("tensor.scatter")
slice_p = el.Primitive[el.Object]("tensor.slice")
elementwise_p = el.Primitive[el.Object]("tensor.elementwise")


class _ElementwiseTracer(el.Tracer):
    """Tracer for element-wise function body.

    Unwraps TensorType→element type during lift, enabling the traced function
    to operate on scalar element types instead of full tensors. Non-tensor
    arguments (scalars, custom types) are passed through unchanged.

    Validates that all tensor inputs have the same shape, tracking the first
    tensor's shape in _tensor_shape for result type construction.
    """

    def __init__(self) -> None:
        """Initialize elementwise tracer."""
        super().__init__()
        self._tensor_shape: tuple[int, ...] | None = None

    def _lift_type(self, obj: el.Object) -> elt.BaseType:
        """Override to unwrap Tensor→element type, keep scalar as-is.

        Args:
            obj: Object to lift (can be Tensor or Scalar typed)

        Returns:
            element type (for Tensor) or original type (for Scalar)

        Raises:
            ValueError: If tensor shapes don't match
        """
        obj_type = obj.type

        if isinstance(obj_type, elt.TensorType):
            # Validate and track shape
            new_shape = obj_type.shape
            if self._tensor_shape is None:
                self._tensor_shape = new_shape
            elif self._tensor_shape == new_shape:
                pass  # Shapes match
            elif self._tensor_shape == ():
                # Upgrade tracked shape from scalar to tensor
                self._tensor_shape = new_shape
            elif new_shape == ():
                # Input is scalar, broadcasts to tracked shape
                pass
            else:
                raise ValueError(
                    f"All tensor arguments must have the same shape. "
                    f"Expected {self._tensor_shape}, got {obj_type.shape}"
                )

            # Unwrap to element type
            return cast(elt.BaseType, obj_type.element_type)
        else:
            # Non-tensor (scalar, custom type) - keep as-is
            return cast(elt.BaseType, obj_type)


@elementwise_p.def_trace
def _elementwise_trace(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Apply element-wise operation to tensor elements.

    This primitive maps an element-level callable to tensor elements while
    preserving shape. All tensor arguments must have the same shape.
    Supports mixing tensor and scalar arguments (scalars passed unchanged to each element).

    Args:
        fn: Callable/traceable function operating on scalar elements.
            Must NOT capture any variables (closure-free).
        *args: Arguments to pass to fn (can be Tensor or Scalar types)
        **kwargs: Keyword arguments to pass to fn

    Returns:
        PyTree whose leaves are TraceObjects with tensor types.
        Each output tensor has the same shape as input tensors,
        with element types determined by tracing fn.

    Raises:
        ValueError: If fn captures variables or tensor shapes don't match
        TypeError: If outputs contain non-scalar types
    """
    tracer = _current_tracer()

    # Trace fn with element inputs using custom tracer
    # The tracer will automatically:
    # 1. Unwrap Tensor→element, keep Scalar as-is
    # 2. Validate all tensors have the same shape
    # 3. Track the tensor shape in _tensor_shape
    element_tracer = _ElementwiseTracer()
    traced_fn = element_tracer.run(fn, *args, **kwargs)

    # Get result shape from the tracer (set by first tensor in _lift)
    if element_tracer._tensor_shape is None:
        # If no tensor arguments were found, it means we only had
        # non-tensor arguments (scalars/custom types).
        # Degrade to scalar operation (shape ()).
        result_shape: tuple[int, ...] = ()
    else:
        result_shape = element_tracer._tensor_shape

    # Check that fn doesn't capture variables (closure-free requirement)
    if traced_fn.captured:
        captured_names = [f"{type(obj).__name__}" for obj in traced_fn.captured]
        raise ValueError(
            f"elementwise function must not capture variables. "
            f"Found {len(traced_fn.captured)} captured object(s): {captured_names}. "
            f"Pass all dependencies as explicit arguments."
        )

    # Get output type from traced graph
    if not traced_fn.graph.outputs:
        raise TypeError("elementwise function must return a value, got empty outputs")

    if traced_fn.out_imms:
        raise TypeError(
            "elementwise function outputs must be TraceObjects (no pure Python constants)"
        )

    output_types: list[elt.BaseType] = []
    for idx, output_value in enumerate(traced_fn.graph.outputs):
        output_element_type = output_value.type
        # Allow rank-0 tensors as scalars (produced by run_jax)
        if (
            isinstance(output_element_type, elt.TensorType)
            and output_element_type.shape == ()
        ):
            output_element_type = output_element_type.element_type

        if not isinstance(output_element_type, elt.BaseType):
            raise TypeError(
                "elementwise function must return BaseType leaves, "
                f"got {type(output_element_type).__name__} at output index {idx}. "
                "Elementwise only supports operations producing valid MPLang types."
            )
        output_types.append(elt.TensorType(output_element_type, result_shape))
    flat_inputs, _ = tree_flatten((args, kwargs))
    input_values = [
        value._graph_value for value in flat_inputs if isinstance(value, el.TraceObject)
    ]

    # Emit graph operation with traced subgraph as region
    result_values = tracer.graph.add_op(
        opcode="tensor.elementwise",
        inputs=input_values,
        output_types=output_types,
        attrs={},
        regions=[traced_fn.graph],
    )

    return tracer.reconstruct_outputs(
        traced_fn.out_var_pos,
        traced_fn.out_imms,
        traced_fn.out_tree,
        result_values,
    )


def elementwise(fn: Callable[..., Any], *inputs: el.Object, **kwargs: Any) -> el.Object:
    """Apply element-wise operation to tensor elements.

    Maps an element-level callable to tensor elements while preserving shape.
    All tensor arguments must have the same shape. Allows mixing tensor and
    scalar arguments (scalars are passed unchanged to fn for each element).

    The function `fn` must be closure-free (no captured variables) - all
    dependencies must be passed as explicit arguments. This ensures the
    computation graph captures all data dependencies.

    Type Promotion Rule:
        If all arguments are scalars, the result will be lifted to a rank-0 tensor (shape=()).

    Args:
        fn: Callable/traceable function operating on scalar elements.
            Can be a lambda, regular function, or Primitive.bind.
            Must not capture variables (closure-free).
            Must return ScalarType values - no tensor nesting.
        *inputs: Tensor or Scalar arguments to pass to fn.
            All tensor inputs must have the same shape.
        **kwargs: Keyword arguments to pass to fn

    Returns:
        PyTree whose leaves are Tensors with the same shape as the input tensors.
        The PyTree structure matches the return value of `fn`.
        Each leaf has element type determined by fn's corresponding output.

    Raises:
        ValueError: If fn captures variables or tensor shapes don't match
        TypeError: If fn returns non-scalar types

    Example:
        >>> # Element-wise addition with lambda
        >>> t1 = ...  # Tensor[f32, (10,)]
        >>> t2 = ...  # Tensor[f32, (10,)]
        >>> result = elementwise(lambda x, y: x + y, t1, t2)
        >>> # result: Tensor[f32, (10,)]
        >>>
        >>> # PHE encryption: mixing tensor and scalar (key)
        >>> plaintext = ...  # Tensor[f32, (10,)]
        >>> public_key = ...  # PHEPublicKey (scalar)
        >>> ciphertext = elementwise(phe.encrypt, plaintext, public_key)
        >>> # ciphertext: Tensor[HE[f32], (10,)]
        >>>
        >>> # Multiple tensors with same shape
        >>> t1 = ...  # Tensor[f32, (3, 4)]
        >>> t2 = ...  # Tensor[f32, (3, 4)]
        >>> result = elementwise(lambda x, y: x * y, t1, t2)
        >>> # result: Tensor[f32, (3, 4)]
        >>>
        >>> # Tensor-scalar operation
        >>> tensor = ...  # Tensor[f32, (10,)]
        >>> scalar = ...  # f32
        >>> result = elementwise(lambda x, s: x * s, tensor, scalar)
        >>> # result: Tensor[f32, (10,)]
    """
    return elementwise_p.bind(fn, *inputs, **kwargs)  # type: ignore[no-any-return]


@transpose_p.def_abstract_eval
def _transpose_ae(input: elt.TensorType, *, perm: tuple[int, ...]) -> elt.TensorType:
    """Transpose tensor dimensions.

    Args:
        input: Input tensor type
        perm: Permutation of dimensions (e.g., (1, 0) for 2D transpose)

    Returns:
        Tensor type with permuted shape

    Raises:
        TypeError: If input is not a TensorType
        ValueError: If permutation is invalid
    """
    if not isinstance(input, elt.TensorType):
        raise TypeError(f"transpose expects TensorType, got {type(input)}")

    # Shape is always a tuple (TensorType enforces ranked tensors)
    rank = len(input.shape)
    if len(perm) != rank:
        raise ValueError(
            f"Permutation length {len(perm)} doesn't match tensor rank {rank}"
        )

    if set(perm) != set(range(rank)):
        raise ValueError(
            f"Invalid permutation {perm}, expected permutation of 0..{rank - 1}"
        )

    # Apply permutation to shape
    new_shape = tuple(input.shape[i] for i in perm)
    return elt.TensorType(input.element_type, new_shape)


@reshape_p.def_abstract_eval
def _reshape_ae(input: elt.TensorType, new_shape: tuple[int, ...]) -> elt.TensorType:
    """Reshape tensor to new shape.

    Args:
        tensor_type: Input tensor type
        new_shape: Target shape (can contain -1 for inferred dimension)

    Returns:
        Tensor type with new shape

    Raises:
        TypeError: If input is not a TensorType
        ValueError: If reshape is invalid
    """
    if not isinstance(input, elt.TensorType):
        raise TypeError(f"reshape expects TensorType, got {type(input)}")

    # Validate new_shape
    if not isinstance(new_shape, tuple):
        raise TypeError(f"new_shape must be tuple, got {type(new_shape)}")

    neg_one_count = sum(1 for d in new_shape if d == -1)
    if neg_one_count > 1:
        raise ValueError("new_shape can contain at most one -1 dimension")

    # Compute output shape
    if input.is_fully_static:
        # Input size is known - we can infer or validate
        input_size = math.prod(input.shape)

        if neg_one_count == 0:
            # No -1: validate total size matches
            new_size = math.prod(new_shape)
            if input_size != new_size:
                raise ValueError(
                    f"Cannot reshape tensor of size {input_size} to shape {new_shape} (size {new_size})"
                )
            output_shape = new_shape
        else:
            # One -1: infer that dimension
            known_size = math.prod(d for d in new_shape if d != -1)
            if known_size == 0:
                raise ValueError("Cannot reshape: new_shape has zero-size dimensions")
            if input_size % known_size != 0:
                raise ValueError(
                    f"Cannot infer dimension: {input_size} is not divisible by {known_size}"
                )
            inferred_dim = input_size // known_size
            output_shape = tuple(inferred_dim if d == -1 else d for d in new_shape)
    else:
        # Input has dynamic dims - output inherits uncertainty
        # Keep -1 in output (we cannot infer at trace time)
        output_shape = new_shape

    return elt.TensorType(input.element_type, output_shape)


@concat_p.def_abstract_eval
def _concat_ae(in_types: list[elt.BaseType], *, axis: int = 0) -> elt.TensorType:
    """Concatenate tensors along axis.

    Args:
        in_types: List of input tensor types
        axis: Axis along which to concatenate (default: 0)

    Returns:
        Concatenated tensor type

    Raises:
        TypeError: If inputs are not TensorTypes
        ValueError: If shapes are incompatible
    """
    if not in_types:
        raise ValueError("concat requires at least one input tensor")

    # Verify all inputs are TensorType
    for i, t in enumerate(in_types):
        if not isinstance(t, elt.TensorType):
            raise TypeError(f"Input {i} is not TensorType: {type(t)}")

    tensor_types = cast(list[elt.TensorType], in_types)

    # Check element types match
    element_type = tensor_types[0].element_type
    for i, t in enumerate(tensor_types[1:], 1):
        if t.element_type != element_type:
            raise TypeError(
                f"Element type mismatch: tensor 0 has {element_type}, "
                f"tensor {i} has {t.element_type}"
            )

    # All tensors are ranked (shape is always a tuple)
    first_shape = tensor_types[0].shape
    rank = len(first_shape)

    # Normalize negative axis
    normalized_axis = axis if axis >= 0 else rank + axis
    if normalized_axis < 0 or normalized_axis >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")

    # Check shape compatibility
    result_shape = list(first_shape)
    concat_dim_size = first_shape[normalized_axis]

    for i, t in enumerate(tensor_types[1:], 1):
        if len(t.shape) != rank:
            raise ValueError(
                f"Rank mismatch: tensor 0 has rank {rank}, tensor {i} has rank {len(t.shape)}"
            )

        for dim_idx in range(rank):
            if dim_idx == normalized_axis:
                # Concatenation dimension
                if concat_dim_size == -1 or t.shape[dim_idx] == -1:
                    concat_dim_size = -1  # Result is dynamic
                else:
                    concat_dim_size += t.shape[dim_idx]
            else:
                # Other dimensions must match (or be dynamic)
                if (
                    result_shape[dim_idx] != -1
                    and t.shape[dim_idx] != -1
                    and result_shape[dim_idx] != t.shape[dim_idx]
                ):
                    raise ValueError(
                        f"Dimension {dim_idx} mismatch: tensor 0 has {result_shape[dim_idx]}, "
                        f"tensor {i} has {t.shape[dim_idx]}"
                    )
                if t.shape[dim_idx] == -1:
                    result_shape[dim_idx] = -1

    result_shape[normalized_axis] = concat_dim_size
    return elt.TensorType(element_type, tuple(result_shape))


@gather_p.def_abstract_eval
def _gather_ae(
    input: elt.TensorType, index: elt.TensorType, *, axis: int = 0
) -> elt.TensorType:
    """Gather elements along axis using indices.

    Args:
        input: Input tensor type
        index: Integer indices tensor type
        axis: Axis along which to gather

    Returns:
        Tensor type with gathered elements

    Raises:
        TypeError: If inputs are not TensorTypes or indices are not integer
        ValueError: If axis is invalid
    """
    if not isinstance(input, elt.TensorType):
        raise TypeError(f"gather expects TensorType, got {type(input)}")
    if not isinstance(index, elt.TensorType):
        raise TypeError(f"indices must be TensorType, got {type(index)}")

    # Verify indices are integer type (ScalarType includes IntegerType)
    if not isinstance(index.element_type, elt.IntegerType):
        raise TypeError(
            f"indices must have IntegerType element, got {type(index.element_type).__name__}"
        )
    # Check for 32-bit or 64-bit integers
    if index.element_type.bitwidth not in (32, 64):
        raise TypeError(
            f"indices must be 32-bit or 64-bit integers (i32/i64/u32/u64), got {index.element_type}"
        )

    # Both inputs must be ranked (shape is always a tuple now)
    rank = len(input.shape)
    normalized_axis = axis if axis >= 0 else rank + axis
    if normalized_axis < 0 or normalized_axis >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")

    # Result shape: replace axis dimension with indices shape
    result_shape = (
        input.shape[:normalized_axis] + index.shape + input.shape[normalized_axis + 1 :]
    )
    return elt.TensorType(input.element_type, result_shape)


@scatter_p.def_abstract_eval
def _scatter_ae(
    tensor_type: elt.TensorType,
    indices_type: elt.TensorType,
    updates_type: elt.TensorType,
    axis: int = 0,
) -> elt.TensorType:
    """Scatter updates into tensor at indices.

    Args:
        tensor_type: Input tensor type
        indices_type: Integer indices tensor type
        updates_type: Updates tensor type
        axis: Axis along which to scatter

    Returns:
        Tensor type (same as input)

    Raises:
        TypeError: If inputs are not compatible
        ValueError: If shapes are incompatible
    """
    if not isinstance(tensor_type, elt.TensorType):
        raise TypeError(f"scatter expects TensorType, got {type(tensor_type)}")
    if not isinstance(indices_type, elt.TensorType):
        raise TypeError(f"indices must be TensorType, got {type(indices_type)}")
    if not isinstance(updates_type, elt.TensorType):
        raise TypeError(f"updates must be TensorType, got {type(updates_type)}")

    # Verify element types match
    if updates_type.element_type != tensor_type.element_type:
        raise TypeError(
            f"Element type mismatch: tensor has {tensor_type.element_type}, "
            f"updates has {updates_type.element_type}"
        )

    # Scatter returns same type as input
    return tensor_type


@slice_p.def_abstract_eval
def _slice_ae(
    tensor_type: elt.TensorType,
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
) -> elt.TensorType:
    """Slice tensor along dimensions.

    Args:
        tensor_type: Input tensor type
        starts: Start indices for each dimension
        ends: End indices for each dimension
        strides: Stride for each dimension (defaults to 1)

    Returns:
        Sliced tensor type

    Raises:
        TypeError: If input is not TensorType
        ValueError: If slice parameters are invalid
    """
    if not isinstance(tensor_type, elt.TensorType):
        raise TypeError(f"slice expects TensorType, got {type(tensor_type)}")

    # Tensor is always ranked (shape is always a tuple)
    rank = len(tensor_type.shape)
    if len(starts) != rank or len(ends) != rank:
        raise ValueError(
            f"starts and ends must have length {rank}, got {len(starts)} and {len(ends)}"
        )

    if strides is None:
        strides = tuple([1] * rank)
    elif len(strides) != rank:
        raise ValueError(f"strides must have length {rank}, got {len(strides)}")

    # Compute result shape
    result_shape = []
    for dim_idx in range(rank):
        dim_size = tensor_type.shape[dim_idx]
        if dim_size == -1:
            # Dynamic dimension - result is also dynamic
            result_shape.append(-1)
        else:
            # Static dimension - compute slice size
            start = starts[dim_idx]
            end = ends[dim_idx]
            stride = strides[dim_idx]

            if stride <= 0:
                raise ValueError(
                    f"stride must be positive, got {stride} at dim {dim_idx}"
                )

            # Handle negative indices
            if start < 0:
                start = max(0, dim_size + start)
            if end < 0:
                end = max(0, dim_size + end)

            # Clamp to valid range
            start = max(0, min(start, dim_size))
            end = max(0, min(end, dim_size))

            # Compute slice length
            if end <= start:
                slice_len = 0
            else:
                slice_len = (end - start + stride - 1) // stride

            result_shape.append(slice_len)

    return elt.TensorType(tensor_type.element_type, tuple(result_shape))


# User-facing API
def transpose(tensor: el.Object, perm: tuple[int, ...]) -> el.Object:
    """Transpose tensor dimensions.

    Args:
        tensor: Input tensor
        perm: Permutation of dimensions

    Returns:
        Transposed tensor

    Example:
        >>> x = constant([[1, 2], [3, 4]])  # shape (2, 2)
        >>> y = transpose(x, (1, 0))  # shape (2, 2), transposed
    """
    return transpose_p.bind(tensor, perm=perm)  # type: ignore[no-any-return]


def reshape(tensor: el.Object, new_shape: tuple[int, ...]) -> el.Object:
    """Reshape tensor to new shape.

    Args:
        tensor: Input tensor
        new_shape: Target shape (can contain -1 for inferred dimension)

    Returns:
        Reshaped tensor

    Example:
        >>> x = constant([1, 2, 3, 4, 5, 6])  # shape (6,)
        >>> y = reshape(x, (2, 3))  # shape (2, 3)
        >>> z = reshape(x, (2, -1))  # shape (2, 3), -1 inferred
    """
    return reshape_p.bind(tensor, new_shape=new_shape)  # type: ignore[no-any-return]


def concat(tensors: list[el.Object], axis: int = 0) -> el.Object:
    """Concatenate tensors along axis.

    Args:
        tensors: List of tensors to concatenate
        axis: Axis along which to concatenate

    Returns:
        Concatenated tensor

    Example:
        >>> x = constant([1, 2, 3])
        >>> y = constant([4, 5, 6])
        >>> z = concat([x, y], axis=0)  # [1, 2, 3, 4, 5, 6]
    """
    return concat_p.bind(*tensors, axis=axis)  # type: ignore[no-any-return]


def gather(tensor: el.Object, indices: el.Object, axis: int = 0) -> el.Object:
    """Gather elements along axis using indices.

    Args:
        tensor: Input tensor
        indices: Integer indices tensor
        axis: Axis along which to gather

    Returns:
        Gathered tensor

    Example:
        >>> x = constant([10, 20, 30, 40])
        >>> idx = constant([0, 2, 1])
        >>> y = gather(x, idx)  # [10, 30, 20]
    """
    return gather_p.bind(tensor, indices, axis=axis)  # type: ignore[no-any-return]


def scatter(
    tensor: el.Object,
    indices: el.Object,
    updates: el.Object,
    axis: int = 0,
) -> el.Object:
    """Scatter updates into tensor at indices.

    Args:
        tensor: Input tensor
        indices: Integer indices tensor
        updates: Updates tensor
        axis: Axis along which to scatter

    Returns:
        Updated tensor

    Example:
        >>> x = constant([1, 2, 3, 4])
        >>> idx = constant([0, 2])
        >>> updates = constant([10, 30])
        >>> y = scatter(x, idx, updates)  # [10, 2, 30, 4]
    """
    return scatter_p.bind(tensor, indices, updates, axis=axis)  # type: ignore[no-any-return]


def slice_tensor(
    tensor: el.Object,
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
) -> el.Object:
    """Slice tensor along dimensions.

    Args:
        tensor: Input tensor
        starts: Start indices for each dimension
        ends: End indices for each dimension
        strides: Stride for each dimension (defaults to 1)

    Returns:
        Sliced tensor

    Example:
        >>> x = constant([[1, 2, 3], [4, 5, 6]])
        >>> y = slice_tensor(x, (0, 1), (2, 3))  # [[2, 3], [5, 6]]
    """
    return slice_p.bind(tensor, starts=starts, ends=ends, strides=strides)  # type: ignore[no-any-return]


# ==============================================================================
# --- Type Reinterpretation (via run_jax)
# ==============================================================================


def bitcast(x: el.Object, dtype: elt.ScalarType) -> el.Object:
    """Reinterpret tensor bytes as a different dtype.

    This is a zero-copy (at execution time) type reinterpretation that views
    the same underlying bytes as a different element type. The total byte
    count must remain the same.

    This follows LLVM/MLIR `bitcast` semantics: the operation produces a new
    SSA value with different type but same bit representation.

    Args:
        x: Input tensor.
        dtype: Target element type (e.g., elt.u64, elt.u8, elt.i32).

    Returns:
        Tensor with same bytes reinterpreted as dtype.
        Shape changes to preserve total bytes.

    Example:
        >>> # Tensor[u8, (8,)] -> Tensor[u64, (1,)]
        >>> packed = tensor.bitcast(bytes_tensor, elt.u64)
        >>> # Tensor[u64, (10, 2)] -> Tensor[u8, (10, 16)]
        >>> unpacked = tensor.bitcast(u64_tensor, elt.u8)
    """
    from typing import cast

    jax_dtype = dtypes.to_jax(dtype)

    def _bitcast(arr: Any) -> Any:
        return arr.view(jax_dtype)

    return cast(el.Object, run_jax(_bitcast, x))


__all__ = [
    "RunJaxCompilation",
    "bitcast",
    "concat",
    "concat_p",
    "constant",
    "constant_p",
    "elementwise",
    "elementwise_p",
    "gather",
    "gather_p",
    "get_run_jax_compilation",
    "jax_fn",
    "reshape",
    "reshape_p",
    "run_jax",
    "run_jax_p",
    "scatter",
    "scatter_p",
    "slice_p",
    "slice_tensor",
    "transpose",
    "transpose_p",
]
