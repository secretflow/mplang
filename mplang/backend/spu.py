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

from dataclasses import dataclass
from typing import Any

import numpy as np
import spu.api as spu_api
import spu.libspu as libspu

from mplang.core.base import TensorLike
from mplang.core.pfunc import PFunction, PFunctionHandler
from mplang.runtime.grpc_comm import LinkCommunicator


def shape_spu_to_np(spu_shape: Any) -> tuple[int, ...]:
    """Convert SPU shape to numpy tuple."""
    return tuple(spu_shape.dims)


def dtype_spu_to_np(spu_dtype: Any) -> np.dtype:
    """Convert SPU dtype to numpy dtype."""
    MAP = {
        libspu.DataType.DT_F32: np.float32,
        libspu.DataType.DT_F64: np.float64,
        libspu.DataType.DT_I1: np.bool_,
        libspu.DataType.DT_I8: np.int8,
        libspu.DataType.DT_U8: np.uint8,
        libspu.DataType.DT_I16: np.int16,
        libspu.DataType.DT_U16: np.uint16,
        libspu.DataType.DT_I32: np.int32,
        libspu.DataType.DT_U32: np.uint32,
        libspu.DataType.DT_I64: np.int64,
        libspu.DataType.DT_U64: np.uint64,
    }
    return MAP[spu_dtype]  # type: ignore[return-value]


@dataclass
class SpuValue:
    """SPU value container for secure computation."""

    shape: tuple[int, ...]
    dtype: Any
    vtype: libspu.Visibility
    share: libspu.Share

    def __repr__(self) -> str:
        return f"SpuValue({self.shape},{self.dtype},{self.vtype})"


class SpuHandler(PFunctionHandler):
    """SPU (Secure Processing Unit) Handler for secure computation.

    Handler for loading and executing SPU functions compiled by SpuFrontend for
    secure multi-party computation.

    Together with SpuFrontend, provides complete SPU compilation and execution
    solution for privacy-preserving computations.
    """

    def __init__(
        self,
        world_size: int,
        spu_config: libspu.RuntimeConfig,
        link_comm: LinkCommunicator | None = None,
    ):
        """Initialize SPU handler for secure computation.

        Args:
            world_size: Total number of parties in the computation
            spu_config: SPU runtime configuration (protocol, field type, etc.)
            link_comm: Link communicator for this party. If None, indicates that
                      the current party is not participating in the SPU computation.
                      Note: Even without link_comm, the handler can still execute
                      IO operations (makeshares and reconstruct) using SPU IO.
        """
        self._world_size = world_size
        self._spu_config = spu_config
        self._link_comm = link_comm

    def set_link_context(self, link_context: Any) -> None:
        """Set the link communicator context for this party.

        This method allows flexible configuration of the link context,
        particularly useful for testing scenarios. Must be called before setup().

        Args:
            link_context: Link communicator context. If None, indicates this
                         party is not participating in the SPU computation.
        """
        # More flexible for testing, this should be called before setup()
        self._link_comm = link_context

    # override
    def setup(self) -> None:
        """Set up the SPU runtime environment.

        Creates the actual SPU runtime using the configuration and link context
        provided during initialization.

        Raises:
            RuntimeError: If SPU runtime creation fails
        """
        if self._link_comm is None:
            # This party may not be part of the SPU computation
            return
        else:
            # TODO(jint): setup the communicator
            pass

    # override
    def teardown(self) -> None:
        """Clean up the SPU runtime environment."""

    def list_fn_names(self) -> list[str]:
        """List function names that this handler can execute."""
        return [
            "mlir.pphlo",
            "spu.makeshares",
            "spu.reconstruct",
        ]

    # override
    def execute(
        self,
        pfunc: PFunction,
        args: list[TensorLike],
    ) -> list[TensorLike]:
        if pfunc.fn_type == "mlir.pphlo":
            return self.do_run(pfunc, args)
        elif pfunc.fn_type == "spu.makeshares":
            return self.do_makeshares(pfunc, args)
        elif pfunc.fn_type == "spu.reconstruct":
            return self.do_reconstruct(pfunc, args)
        else:
            raise ValueError(f"Unsupported function type: {pfunc.fn_type}")

    def do_makeshares(
        self,
        pfunc: PFunction,
        args: list[TensorLike],
    ) -> list[TensorLike]:
        """Create SPU shares from input data.

        Args:
            pfunc: PFunction containing makeshares metadata
            args: Input data to be shared (single tensor)

        Returns:
            List of SPU shares, one for each party

        Note:
            This operation can be performed even without link_comm (i.e., without runtime),
            as it only requires SPU IO for share generation.
        """
        assert len(args) == 1

        # Extract metadata from pfunc.attrs
        visibility_value = pfunc.attrs.get(
            "visibility", libspu.Visibility.VIS_SECRET.value
        )
        if isinstance(visibility_value, int):
            # Convert from integer value to enum
            visibility = libspu.Visibility(visibility_value)
        else:
            visibility = visibility_value

        # Convert input to numpy array (SPU make_shares expects np.array)
        arg = np.array(args[0], copy=False)

        # Create SPU IO - can be done without runtime/link_comm
        spu_io = spu_api.Io(self._world_size, self._spu_config)

        # Generate shares
        shares = spu_io.make_shares(arg, visibility)
        assert len(shares) == self._world_size, (
            f"Expected {self._world_size} shares, got {len(shares)}"
        )

        # Return list of SpuValue objects (one per party)
        return [
            SpuValue(
                shape=arg.shape,
                dtype=arg.dtype,
                vtype=visibility,
                share=share,
            )
            for share in shares
        ]

    def do_reconstruct(
        self,
        pfunc: PFunction,
        args: list[TensorLike],
    ) -> list[TensorLike]:
        """Reconstruct plaintext data from SPU shares.

        Args:
            pfunc: PFunction containing reconstruction metadata
            args: List of SPU shares to be reconstructed

        Returns:
            List containing the reconstructed plaintext tensor

        Note:
            This operation can be performed even without link_comm (i.e., without runtime),
            as it only requires SPU IO for reconstruction.
        """
        # Validate that we have the expected number of shares (should equal world_size)
        assert len(args) == self._world_size, (
            f"Expected {self._world_size} shares, got {len(args)}"
        )

        # Validate that all inputs are SpuValue objects
        for i, arg in enumerate(args):
            if not isinstance(arg, SpuValue):
                raise ValueError(
                    f"Input {i} must be SpuValue, got {type(arg)}. "
                    f"Reconstruction requires SPU shares as input."
                )

        # Cast for type checking (we've validated above)
        spu_args: list[SpuValue] = args  # type: ignore

        # Extract shares from SpuValue objects
        shares = [spu_arg.share for spu_arg in spu_args]

        # Create SPU IO - can be done without runtime/link_comm
        spu_io = spu_api.Io(self._world_size, self._spu_config)

        # Reconstruct the plaintext data
        reconstructed = spu_io.reconstruct(shares)

        # Return as a list (consistent with PFunction interface)
        return [reconstructed]

    def do_run(
        self,
        pfunc: PFunction,
        args: list[TensorLike],
    ) -> list[TensorLike]:
        """Execute compiled SPU function for secure computation.

        Implementation Notes:
        Uses the following SPU execution pipeline:
        1. Deserialize the SPU executable from PFunction
        2. Set input SPU values as variables in SPU runtime
        3. Execute using SPU runtime
        4. Retrieve output SPU values

        Note: In real SPU environments, all inputs must be SpuValue objects.

        Key Components:
        - SpuValue: SPU tensor data with metadata (shape, dtype, visibility, share)
        - spu_runtime.set_var/get_var: Variable management in SPU context
        - spu_runtime.run: Execute the compiled SPU code

        Args:
            pfunc: PFunction containing SPU executable
            args: Input arguments as SpuValue objects (TensorLike compatible)

        Returns:
            List of output SpuValue objects from secure computation

        Raises:
            ValueError: Unsupported format or invalid input types
            RuntimeError: Execution failure
            RuntimeError: Missing SPU runtime setup
        """
        # Validate format: only SPU protobuf supported
        if pfunc.fn_type != "mlir.pphlo":
            raise ValueError(
                f"Unsupported format: {pfunc.fn_type}. Expected 'mlir.pphlo'"
            )

        if self._link_comm is None:
            raise RuntimeError(
                "Link context not set, please check if this party is part of the SPU computation."
            )

        # Create the real SPU runtime
        spu_rt = spu_api.Runtime(self._link_comm.get_lctx(), self._spu_config)

        # Check runtime setup
        if spu_rt is None:
            raise RuntimeError("SPU runtime not set up. Call setup() first.")

        # Validate that all inputs are SpuValue objects
        for i, arg in enumerate(args):
            if not isinstance(arg, SpuValue):
                raise ValueError(
                    f"Input {i} must be SpuValue, got {type(arg)}. "
                    f"In real SPU environments, all inputs must be SpuValue objects."
                )

        # Cast for type checking (we've validated above)
        spu_args: list[SpuValue] = args  # type: ignore

        # Reconstruct SPU executable from MLIR code and metadata
        if pfunc.fn_text is None:
            raise ValueError("PFunction does not contain executable data")
        if not isinstance(pfunc.fn_text, str):
            raise ValueError(f"Expected str, got {type(pfunc.fn_text)}")

        # Extract metadata for executable reconstruction
        attrs: dict[str, Any] = dict(pfunc.attrs or {})
        input_names = attrs.get("input_names", [])
        output_names = attrs.get("output_names", [])
        executable_name = attrs.get("executable_name", pfunc.fn_name)

        # Create executable from MLIR code and metadata
        executable = libspu.Executable(
            name=executable_name,
            input_names=input_names,
            output_names=output_names,
            code=pfunc.fn_text,
        )

        # Set input variables in SPU runtime
        for idx, spu_arg in enumerate(spu_args):
            spu_rt.set_var(input_names[idx], spu_arg.share)

        # Execute the compiled function
        spu_rt.run(executable)

        # Retrieve output variables
        shares = [spu_rt.get_var(out_name) for out_name in output_names]
        metas = [spu_rt.get_var_meta(out_name) for out_name in output_names]

        # Create result SpuValues
        results: list[TensorLike] = [
            SpuValue(
                shape=shape_spu_to_np(meta.shape),
                dtype=dtype_spu_to_np(meta.data_type),
                vtype=meta.visibility,
                share=shares[idx],
            )
            for idx, meta in enumerate(metas)
        ]

        return results
