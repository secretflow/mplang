# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import export
from jax.tree_util import PyTreeDef, tree_flatten

from mplang.v1.core import MPObject, PFunction, TensorType, get_fn_name
from mplang.v1.ops.base import FeOperation, stateless_mod
from mplang.v1.utils.func_utils import normalize_fn

# Enable 64-bit precision for JAX to match tensor types
jax.config.update("jax_enable_x64", True)


def jax2stablehlo(
    is_variable: Callable[[Any], bool], flat_fn: Any, *args: Any, **kwargs: Any
) -> tuple[PFunction, list, PyTreeDef]:
    """Compile JAX function to StableHLO MLIR format for remote execution.

    Translates high-level JAX functions into StableHLO MLIR representations,
    enabling execution on JAX backends across different processes and platforms.
    Uses a hybrid approach: traditional JAX trace/lower for compilation compatibility,
    with stable jax.export API for parameter tracking.

    Args:
        is_variable: Predicate function to classify parameters as variables vs. constants.
                    Returns True for parameters that should be treated as PFunction inputs.
        flat_fn: JAX function to be compiled into StableHLO format
        *args: Positional arguments passed to the function during compilation
        **kwargs: Keyword arguments passed to the function during compilation

    Returns:
        tuple[PFunction, list, PyTreeDef]: Compilation artifacts containing:
            - PFunction: Serialized function with embedded MLIR text and type metadata
            - list: Extracted variable parameters (those satisfying is_variable predicate).
                   Non-variable parameters are captured as compile-time constants within
                   the PFunction body, while variables become runtime input parameters.
            - PyTreeDef: Tree structure template for reconstructing nested output values
    """
    # Flatten (args, kwargs) and capture immediates using the moved logic from primitive.py
    normalized_fn, in_vars = normalize_fn(flat_fn, args, kwargs, is_variable)

    # Convert TensorType in_vars to ShapeDtypeStruct for JAX tracing
    jax_params = [
        jax.ShapeDtypeStruct(arg.shape, jnp.dtype(arg.dtype.name)) for arg in in_vars
    ]

    # Hybrid approach: Use standard JAX trace/lower for compatibility, but jax.export for parameter tracking
    jitted_fn = jax.jit(normalized_fn)
    traced = jitted_fn.trace(jax_params)
    lowered = traced.lower()

    # Get StableHLO MLIR representation using traditional approach
    stablehlo_mlir = lowered.compiler_ir("stablehlo")
    mlir_text = str(stablehlo_mlir)

    # Get output info using traditional approach
    out_info_flat, out_tree = tree_flatten(lowered.out_info)
    out_info_flat = [TensorType.from_obj(info) for info in out_info_flat]

    # Extract argument keep mapping using stable jax.export API for parameter tracking
    # We use jax.export only for getting the kept_var_idx information, not for the main compilation
    arg_keep_map = None
    original_arg_count = len(in_vars)

    try:
        # Use jax.export just to get the stable parameter tracking information
        export_fn = export.export(jitted_fn)
        exported = export_fn(jax_params)
        kept_var_idx = exported.module_kept_var_idx
        if kept_var_idx is not None and len(kept_var_idx) < original_arg_count:
            # JAX eliminated some unused parameters during compilation
            # Keep the indices in sorted order for consistent mapping
            arg_keep_map = sorted(kept_var_idx)
    except Exception as e:
        # Fallback: if jax.export fails, we can still use the compiled result without parameter tracking
        # This ensures backward compatibility even if export has issues
        logging.warning(
            f"jax.export failed to get kept_var_idx, proceeding without it. Error: {e}"
        )

    # This format tells JaxRT how to handle the compiled result
    pfn_kwargs: dict[str, Any] = {
        "fn_type": "mlir.stablehlo",  # Key: specify StableHLO MLIR format
        "ins_info": tuple(TensorType.from_obj(x) for x in in_vars),
        "outs_info": tuple(out_info_flat),
        "fn_name": get_fn_name(flat_fn),
        "fn_text": mlir_text,  # MLIR text, serializable for transmission
    }

    if arg_keep_map is not None:
        pfn_kwargs["arg_keep_map"] = arg_keep_map

    pfn = PFunction(**pfn_kwargs)
    return pfn, in_vars, out_tree


class JaxRunner(FeOperation):
    """JAX function runner frontend operation."""

    def trace(
        self, jax_fn: Callable, *args: Any, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """
        JAX compilation helper function.

        Compiles a JAX function to StableHLO format and returns the PFunction
        along with variable arguments for evaluation.

        Args:
            jax_fn: The JAX function to compile
            *args: Positional arguments to the function
            **kwargs: Keyword arguments to the function

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: The compiled PFunction, input variables, and output tree
        """

        def is_variable(arg: Any) -> bool:
            return isinstance(arg, MPObject)

        pfunc, in_vars, out_tree = jax2stablehlo(is_variable, jax_fn, *args, **kwargs)
        return pfunc, in_vars, out_tree


_JAX_MOD = stateless_mod("jax")

run_jax = JaxRunner(_JAX_MOD, "run")
