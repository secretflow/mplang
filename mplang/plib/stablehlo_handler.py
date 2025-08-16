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

import jax
import jax.numpy as jnp
from jax._src import xla_bridge
from jax.lib import xla_client as xc

from mplang.core.base import TensorLike
from mplang.core.pfunc import PFunction, PFunctionHandler


class StablehloHandler(PFunctionHandler):
    """StableHLO Handler for remote execution.

    Runtime for loading and executing JAX functions serialized by jax2stablehlo using
    StableHLO MLIR intermediate representation.

    Together with jax2stablehlo, provides complete StableHLO compilation and
    execution solution.
    """

    # override
    def setup(self) -> None:
        """Set up the runtime environment."""
        # StableHLO handler doesn't need special setup

    # override
    def teardown(self) -> None:
        """Clean up the runtime environment."""
        # StableHLO handler doesn't need special teardown

    def list_fn_names(self) -> list[str]:
        """List function names that this handler can execute."""
        return ["mlir.stablehlo"]

    # override
    def execute(
        self,
        pfunc: PFunction,
        args: list[TensorLike],
    ) -> list[TensorLike]:
        """Execute compiled function containing StableHLO MLIR.

        Implementation Notes:
        Uses the following verified execution pipeline:
        1. Extract MLIR text from PFunction
        2. Compile MLIR text to LoadedExecutable via XLA client
        3. Convert inputs to JAX device arrays
        4. Execute using execute_sharded
        5. Extract results using disassemble_into_single_device_arrays

        Key Findings:
        - XLA client.compile() accepts MLIR strings directly
        - execute_sharded() more stable than execute()
        - Results must be properly extracted from ExecuteResults

        Args:
            pfunc: PFunction containing MLIR StableHLO text
            args: Input arguments as TensorLike objects

        Returns:
            List of output tensors

        Raises:
            ValueError: Unsupported format
            RuntimeError: Compilation or execution failure
        """
        # Validate format: only StableHLO MLIR supported
        if pfunc.fn_type != "mlir.stablehlo":
            raise ValueError(
                f"Unsupported format: {pfunc.fn_type}. Expected 'mlir.stablehlo'"
            )

        # Extract MLIR text from compiled function
        mlir_text = pfunc.fn_text
        if mlir_text is None:
            raise ValueError("PFunction does not contain MLIR text")

        # Convert to string if it's bytes
        if isinstance(mlir_text, bytes):
            mlir_text = mlir_text.decode("utf-8")

        # Get JAX backend and compile MLIR
        # Key finding: XLA client.compile() accepts MLIR strings directly
        backend = jax.default_backend()
        client = xla_bridge.get_backend(backend)
        compile_options = xc.CompileOptions()

        compiled_executable = client.compile(mlir_text, compile_options)

        # Convert input args to JAX arrays and put on device
        # This is the correct input format discovered in research
        # TODO(jint): Do we have a data copy issue here?
        jax_args = []
        for arg in args:
            if hasattr(arg, "numpy"):
                # Convert from MPLang tensor to numpy then to JAX
                jax_arg = jnp.array(arg.numpy())  # type: ignore
            else:
                # Assume it's already array-like
                jax_arg = jnp.array(arg)
            jax_args.append(jax.device_put(jax_arg))

        # Execute compiled function
        # Key finding: execute_sharded is the most stable execution method
        try:
            result = compiled_executable.execute_sharded(jax_args)

            # Extract results
            # Key finding: must use disassemble_into_single_device_arrays() to extract results
            arrays = result.disassemble_into_single_device_arrays()

            # Convert back to expected format
            # Note: arrays is nested list, need to flatten appropriately
            output_tensors = []
            for array_list in arrays:
                if isinstance(array_list, list) and len(array_list) == 1:
                    # Single output case - extract the array
                    output_tensors.append(jnp.array(array_list[0]))
                else:
                    # Multiple outputs or other cases
                    output_tensors.extend([jnp.array(arr) for arr in array_list])

            return output_tensors  # type: ignore[return-value]

        except Exception as e:
            raise RuntimeError(f"Failed to execute compiled function: {e}") from e
