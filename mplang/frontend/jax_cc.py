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

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction, get_fn_name
from mplang.core.tensor import TensorType
from mplang.frontend.base import FeOperation, stateless_mod
from mplang.utils.func_utils import normalize_fn

# Enable 64-bit precision for JAX to match tensor types
jax.config.update("jax_enable_x64", True)


def jax2stablehlo(
    is_variable: Callable[[Any], bool], flat_fn: Any, *args: Any, **kwargs: Any
) -> tuple[PFunction, list, PyTreeDef]:
    """Compile JAX function to StableHLO MLIR format for remote execution.

    Translates high-level JAX functions into StableHLO MLIR representations,
    enabling execution on JAX backends across different processes and platforms.
    Uses the standard JAX compilation pipeline: jit → trace → lower → StableHLO MLIR.

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

    Rationale:
        JAX Serialization Options Analysis:
        1. jax.export (JAX ≥0.4.35) - Official export API with StableHLO backend
        2. HLO protobuf - Raw XLA HloModule serialization
        3. HLO text - Human-readable HLO representation
        4. StableHLO MLIR - Portable intermediate representation
        5. JAX compiled object pickling - Limited to same-process execution

        Current Choice: StableHLO MLIR
        Advantages:
        - ✅ Available in current JAX version (0.4.34)
        - ✅ Cross-version compatibility guaranteed by StableHLO design
        - ✅ Direct compilation support via XLA client.compile(mlir_string)
        - ✅ Handles complex functions (multi-input/output, control flow)
        - ✅ Preserves numerical precision
        - ✅ Platform-independent representation

        Alternative Options Issues:
        - jax.export: Not available in JAX 0.4.34
        - HLO protobuf: Version compatibility issues with StableHLO parser
        - HLO text: Parser compatibility issues with XLA client
        - Pickle: Cannot serialize XLA LoadedExecutable objects

        Future Migration Path:
        - JAX ≥0.4.35: Migrate to jax.export.export() + jax.export.deserialize()
        - JAX ≥0.5.x: Consider new portable formats if available
        - Long-term: Adopt official JAX serialization standards as they mature
    """
    # Flatten (args, kwargs) and capture immediates using the moved logic from primitive.py
    normalized_fn, in_vars = normalize_fn(flat_fn, args, kwargs, is_variable)

    # Convert TensorType in_vars to ShapeDtypeStruct for JAX tracing
    jax_params = [
        jax.ShapeDtypeStruct(arg.shape, jnp.dtype(arg.dtype.name)) for arg in in_vars
    ]

    # Standard JAX serialization pipeline: jit → trace → lower → StableHLO MLIR
    jitted_fn = jax.jit(normalized_fn)
    traced = jitted_fn.trace(jax_params)
    lowered = traced.lower()

    # Get StableHLO MLIR representation - the portable format
    # compiler_ir("stablehlo") returns jaxlib.mlir.ir.Module object
    # str() converts to serializable text format
    stablehlo_mlir = lowered.compiler_ir("stablehlo")
    mlir_text = str(stablehlo_mlir)

    # Get output info and tree structure for result reconstruction after remote execution
    out_info_flat, out_tree = tree_flatten(lowered.out_info)
    out_info_flat = [TensorType.from_obj(info) for info in out_info_flat]

    # This format tells JaxRT how to handle the compiled result
    pfn = PFunction(
        fn_type="mlir.stablehlo",  # Key: specify StableHLO MLIR format
        ins_info=tuple(TensorType.from_obj(x) for x in in_vars),
        outs_info=tuple(out_info_flat),
        fn_name=get_fn_name(flat_fn),
        fn_text=mlir_text,  # MLIR text, serializable for transmission
    )
    return pfn, in_vars, out_tree


class JaxCompiler(FeOperation):
    """JAX compiler frontend operation."""

    def trace(
        self, func: Callable, *args: Any, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """
        JAX compilation helper function.

        Compiles a JAX function to StableHLO format and returns the PFunction
        along with variable arguments for evaluation.

        Args:
            func: The JAX function to compile
            *args: Positional arguments to the function
            **kwargs: Keyword arguments to the function

        Returns:
            tuple[PFunction, list[MPObject], Any]: The compiled PFunction, input variables, and output tree
        """

        def is_variable(arg: Any) -> bool:
            return isinstance(arg, MPObject)

        pfunc, in_vars, out_tree = jax2stablehlo(is_variable, func, *args, **kwargs)
        return pfunc, in_vars, out_tree


_JAX_MOD = stateless_mod("jax")

jax_compile = JaxCompiler(_JAX_MOD, "compile")
