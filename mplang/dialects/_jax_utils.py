# Copyright 2026 Ant Group Co., Ltd.
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


from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import jax
from jax.tree_util import PyTreeDef

import mplang.edsl as el
import mplang.edsl.typing as elt
from mplang.dialects import dtypes


@dataclass
class JaxCompilation:
    """Compilation record for tensor.run_jax functions.

    Stores both the compilation artifacts (StableHLO MLIR, types, tree structure)
    and metadata needed for execution (arg_keep_map for JAX's unused param elimination).
    """

    stablehlo: str
    out_tree: PyTreeDef  # type: ignore
    output_types: list[elt.TensorType]
    arg_keep_map: list[int] | None = None
    has_dynamic_shape: bool = False


def _scalar_to_dtype(scalar: elt.ScalarType) -> Any:
    """Convert MPLang scalar type to JAX dtype.

    JAX ShapeDtypeStruct automatically handles conversion to numpy dtype internally.
    """
    return dtypes.to_jax(scalar)


def _dtype_to_scalar(dtype: Any) -> elt.ScalarType:
    return dtypes.from_dtype(dtype)


def _out_info_to_edsl(out_info: Any) -> elt.TensorType:
    scalar = _dtype_to_scalar(out_info.dtype)
    shape = [dim if isinstance(dim, int) else -1 for dim in out_info.shape]
    return elt.TensorType(scalar, tuple(shape))


def _get_symbolic_name(
    symbolic_shapes: Sequence[Sequence[str | None]] | None, obj_idx: int, dim_idx: int
) -> str:
    if (
        symbolic_shapes is not None
        and obj_idx < len(symbolic_shapes)
        and dim_idx < len(symbolic_shapes[obj_idx])
    ):
        name = symbolic_shapes[obj_idx][dim_idx]
        if not isinstance(name, str):
            raise ValueError(f"Symbolic shape name must be a string, got {type(name)}")
        return name

    # We use "n_rows" instead of generic pattern like f"arg_{obj_idx}_dim_{dim_idx}"
    # because our common use case is representing unknown row count in tabular data
    return "n_rows"


def _make_placeholders(
    variables: list[el.Object], symbolic_shapes: Sequence[Sequence[str | None]] | None
) -> list[jax.ShapeDtypeStruct]:
    # Build symbolic shapes and placeholders
    symbol_scope = jax.export.SymbolicScope()
    # Maps symbol name to SymbolicDimension
    symbol_map: dict[str, Any] = {}

    def _make_symbol(name: str) -> Any:
        if name in symbol_map:
            return symbol_map[name]
        symbolics = jax.export.symbolic_shape(name, scope=symbol_scope)
        symbol_map[name] = symbolics[0]
        return symbolics[0]

    placeholders: list[jax.ShapeDtypeStruct] = []
    for obj_idx, obj in enumerate(variables):
        # Extract the plain tensor type from SS (Secret Sharing) type if needed
        obj_type = obj.type.pt_type if isinstance(obj.type, elt.SSType) else obj.type

        if not isinstance(obj_type, (elt.TensorType, elt.ScalarType)):
            raise TypeError(f"run_jax only supports Tensors/Scalars, got {obj.type}")
        if isinstance(obj_type, elt.ScalarType):
            dtype = _scalar_to_dtype(obj_type)
            placeholders.append(jax.ShapeDtypeStruct((), dtype))
        else:
            # element_type must be ScalarType for conversion to numpy dtype
            if not isinstance(obj_type.element_type, elt.ScalarType):
                raise TypeError(
                    f"Expected ScalarType element, got {type(obj_type.element_type)}"
                )
            obj_shape = []
            for idx, dim in enumerate(obj_type.shape):
                if dim is None:
                    raise TypeError(
                        f"Argument dimension {idx} is None; "
                        "please provide a static dimension."
                    )
                elif dim < -1:
                    raise ValueError(f"Invalid tensor dimension {dim}")
                if dim == -1:
                    name = _get_symbolic_name(symbolic_shapes, obj_idx, idx)
                    symbol = _make_symbol(name)
                    obj_shape.append(symbol)
                else:
                    obj_shape.append(dim)
            dtype = _scalar_to_dtype(obj_type.element_type)
            placeholders.append(jax.ShapeDtypeStruct(obj_shape, dtype))

    return placeholders


def compile_jax(
    normalized_fn: Callable[..., Any],
    variables: list[el.Object],
    symbolic_shapes: Sequence[Sequence[str | None]] | None = None,
) -> JaxCompilation:
    """Compile JAX function to StableHLO MLIR.

    Pipeline: jit → export → StableHLO MLIR

    Args:
        normalized_fn: Function accepting list of variables (for JAX lower API)
        variables: List of MPLang objects representing function inputs
        symbolic_shapes: Optional sequence of symbolic shape names for dynamic dimensions.
            Each inner sequence corresponds to an input variable's dimensions, with None
            for static dimensions and string names for symbolic dimensions.

    Returns:
        JaxCompilation: Compilation record containing StableHLO MLIR, output types,
        tree structure, and metadata for execution including arg_keep_map for JAX's
        unused parameter elimination and dynamic shape flag.
    """

    placeholders = _make_placeholders(variables, symbolic_shapes)

    # Calculate has_dynamic_shape from placeholders
    # Check if any placeholder has symbolic dimensions (i.e., non-concrete shape)
    has_dynamic_shape = any(
        not all(isinstance(dim, int) for dim in placeholder.shape)
        for placeholder in placeholders
    )

    # Wrap normalized_fn to collect JAX's individual args into a list
    # This is needed because:
    # - normalized_fn expects: normalized([arg1, arg2, ...])
    # - But JAX export calls: wrapper_fn(arg1, arg2, ...)
    def wrapped_fn(*args: Any) -> Any:
        return normalized_fn(list(args))

    # Tip: Use `jax.config.update("jax_traceback_in_locations_limit", 0)` to reduce location information
    # in MLIR output. This disables JAX traceback locations, removing verbose source location
    # annotations (e.g., #loc = #loc1) from the generated MLIR, which makes the output more
    # readable and reduces serialization overhead.
    jitted = jax.jit(wrapped_fn)
    exported = jax.export.export(jitted)(*placeholders)
    stablehlo_text = str(exported.mlir_module())

    # Handle JAX's unused parameter elimination
    arg_keep_map: list[int] | None = None
    try:
        kept_var_idx = exported.module_kept_var_idx
        if len(kept_var_idx) < len(placeholders):
            arg_keep_map = list(kept_var_idx)
    except (AttributeError, TypeError) as e:
        raise RuntimeError(
            f"Cannot access JAX's module_kept_var_idx for unused parameter handling. "
            f"JAX may have optimized away unused parameters. Error: {e}"
        ) from e

    # Convert output info to EDSL types
    output_types: list[elt.TensorType]
    if isinstance(exported.out_avals, tuple):
        output_types = [_out_info_to_edsl(aval) for aval in exported.out_avals]
    else:
        output_types = [_out_info_to_edsl(exported.out_avals)]

    compilation = JaxCompilation(
        stablehlo=stablehlo_text,
        out_tree=exported.out_tree,
        output_types=output_types,
        arg_keep_map=arg_keep_map,
        has_dynamic_shape=has_dynamic_shape,
    )
    return compilation


@contextmanager
def patch_jax() -> Generator[None, None, None]:
    """Context manager that patches JAX for SPU compatibility.

    This patches JAX's lax operations to be compatible with SPU.
    The patches are automatically restored when exiting the context.

    Usage:
        with patch_jax():
            # JAX code that needs SPU compatibility
            result = jax.jit(fn)(args)
    """
    # Import here to avoid circular imports
    import spu.utils.frontend as spu_fe

    # Apply patches
    patches = spu_fe._patch_jax()

    try:
        yield
    finally:
        # Always restore the original JAX functions
        spu_fe._restore_jax_patch(patches)
