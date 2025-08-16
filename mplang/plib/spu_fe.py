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

from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import jax.numpy as jnp
import spu.libspu as libspu
import spu.utils.frontend as spu_fe
from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.base import TensorInfo, TensorLike
from mplang.core.pfunc import PFunction, get_fn_name
from mplang.utils.func_utils import normalize_fn


class Visibility(Enum):
    """Visibility types for SPU shares."""

    SECRET = libspu.Visibility.VIS_SECRET
    PUBLIC = libspu.Visibility.VIS_PUBLIC
    PRIVATE = libspu.Visibility.VIS_PRIVATE


class SpuFE:
    """SPU Frontend for input/output operations."""

    def __init__(
        self,
        world_size: int,
        enable_private: bool = False,
        copts: Any = None,
    ) -> None:
        """Initialize the SPU frontend."""
        self.world_size = world_size
        self.enable_private = enable_private
        # Create a fresh CompilerOptions instance to avoid JAX compatibility issues
        self.copts = copts or libspu.CompilerOptions()

        # The (num_parties, protocol, field) are frontend transparency settings
        # The backend could change them on demand and the behavior does not change.

    def makeshares(
        self,
        data: TensorLike,
        visibility: Visibility = Visibility.SECRET,
        owner_rank: int = -1,
    ) -> PFunction:
        """Create a PFunction that generates SPU shares from input data.

        This function creates a PFunction that wraps the SPU share generation process,
        allowing input data to be converted into secret shares for secure computation.

        Args:
            data: Input tensor to be shared
            visibility: Visibility type for the shares (SECRET, PUBLIC, PRIVATE)
            owner_rank: Owner rank for PRIVATE visibility (-1 for all parties)

        Returns:
            PFunction that generates SPU shares when executed

        Example:
            >>> import jax.numpy as jnp
            >>> data = jnp.array([1.0, 2.0, 3.0])
            >>> share_fn = SpuFrontend.makeshares(data, Visibility.SECRET)
            >>> # This PFunction can be executed by SPU runtime to create shares
        """
        # Create input info for the data
        ins_info = [TensorInfo.from_obj(data)]

        # For makeshares, the output will be SPU shares - one for each party
        # Create one output tensor info for each party's share
        outs_info = [ins_info[0] for _ in range(self.world_size)]

        # Create metadata for the share generation
        attrs = {
            "visibility": visibility.value,  # Use the libspu enum value
            "owner_rank": owner_rank,
            "operation": "makeshares",
            "world_size": self.world_size,  # Add world_size to metadata
        }

        # Create the PFunction
        pfunc = PFunction(
            fn_type="spu.makeshares",
            ins_info=ins_info,
            outs_info=outs_info,
            fn_name="makeshares",
            fn_text=None,  # No serialized code needed
            attrs=attrs,
        )

        return pfunc

    def reconstruct(
        self,
        shares: Sequence[TensorLike],
    ) -> PFunction:
        """Create a PFunction that reconstructs plaintext data from SPU shares.

        This function creates a PFunction that wraps the SPU reconstruction process,
        allowing secret shares to be converted back into plaintext values.

        Args:
            shares: List of SPU shares to be reconstructed

        Returns:
            PFunction that reconstructs plaintext data when executed

        Example:
            >>> # Assuming we have SPU shares from previous computation
            >>> share_list = [share1, share2, share3]  # from different parties
            >>> reconstruct_fn = SpuFrontend.reconstruct(share_list)
            >>> # This PFunction can be executed by SPU runtime to get plaintext
        """
        # Create input info for the shares
        ins_info = [TensorInfo.from_obj(share) for share in shares]

        # Output will be a single plaintext tensor
        outs_info = [ins_info[0]] if ins_info else []

        attrs: dict[str, Any] = {}

        pfunc = PFunction(
            fn_type="spu.reconstruct",
            ins_info=ins_info,
            outs_info=outs_info,
            fn_name="reconstruct",
            fn_text=None,  # No serialized code needed
            attrs=attrs,
        )

        return pfunc

    def compile_jax(
        self,
        is_variable: Callable[[Any], bool],
        jax_fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[PFunction, list, PyTreeDef]:
        """Compile JAX function to SPU executable format for secure execution.

        This function translates high-level JAX functions into SPU-compatible
        representations, enabling secure multi-party computation.

        Args:
            is_variable: Predicate function to classify parameters as variables vs. constants.
                        Returns True for parameters that should be treated as PFunction inputs.
            flat_fn: JAX function to be compiled into SPU format
            *args: Positional arguments passed to the function during compilation
            **kwargs: Keyword arguments passed to the function during compilation

        Returns:
            tuple[PFunction, list, PyTreeDef]: Compilation artifacts containing:
                - PFunction: Serialized function with embedded SPU executable and metadata
                - list: Extracted variable parameters (those satisfying is_variable predicate).
                       Non-variable parameters are captured as compile-time constants within
                       the PFunction body, while variables become runtime input parameters.
                - PyTreeDef: Tree structure template for reconstructing nested output values

        Rationale:
            SPU Compilation Pipeline:
            1. Convert function inputs to JAX ShapeDtypeStruct for tracing
            2. Use spu.utils.frontend to compile the function with JAX frontend
            3. Serialize the executable and metadata for remote execution
            4. Handle visibility information for secure computation

            Key Components:
            - spu_fe.compile(): Compiles JAX functions to SPU executable format
            - Visibility handling: Determines which inputs are secret vs public
            - Executable serialization: Uses protobuf for cross-language transmission

            Technical Pipeline:
            function → jax tracing → spu_fe.compile → ExecutableProto → protobuf serialization
        """
        # Flatten (args, kwargs) and capture immediates using normalize_fn
        normalized_fn, in_vars = normalize_fn(jax_fn, args, kwargs, is_variable)

        # Convert TensorInfo in_vars to ShapeDtypeStruct for JAX tracing
        jax_params = [
            ShapeDtypeStruct(arg.shape, jnp.dtype(arg.dtype.name)) for arg in in_vars
        ]

        # Set up SPU compilation parameters
        in_vis = [libspu.Visibility.VIS_SECRET for _ in in_vars]

        # Note: in/out names are temporary within SPU runtime API lifecycle
        # Runtime uses setVar/run/getVar/clearVar as atomic transaction
        # where variable names exist only during the execution context
        in_names = [f"in{idx}" for idx in range(len(in_vars))]
        out_names_gen = lambda outs: [f"out{idx}" for idx in range(len(outs))]

        # Compile using SPU frontend
        executable, out_info = spu_fe.compile(
            spu_fe.Kind.JAX,
            normalized_fn,
            [jax_params],
            {},
            in_names,
            in_vis,
            out_names_gen,
            static_argnums=(),
            static_argnames=None,
            copts=self.copts,
        )

        # Extract output information
        out_info_flat, out_tree = tree_flatten(out_info)
        output_tensor_infos = [TensorInfo.from_obj(out) for out in out_info_flat]

        # Use MLIR code directly instead of protobuf serialization
        # This is more readable, compact, and closer to SPU's native representation
        executable_code = executable.code

        # Convert bytes to string for MLIR text
        assert isinstance(executable_code, bytes), (
            f"Expected bytes, got {type(executable_code)}"
        )
        executable_code = executable_code.decode("utf-8")

        # Create metadata for SPU execution
        spu_metadata = {
            "input_visibilities": in_vis,
            "input_names": list(executable.input_names),
            "output_names": list(executable.output_names),
            "executable_name": executable.name,
        }

        pfn = PFunction(
            fn_type="mlir.pphlo",
            ins_info=tuple(TensorInfo.from_obj(x) for x in in_vars),
            outs_info=tuple(output_tensor_infos),
            fn_name=get_fn_name(jax_fn),
            fn_text=executable_code,
            attrs=spu_metadata,
        )

        return pfn, in_vars, out_tree
