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

import os

import numpy as np

from mplang.core.mptype import TensorLike
from mplang.core.pfunc import PFunction, TensorHandler
from mplang.core.table import TableType


class BuiltinHandler(TensorHandler):
    """Handler for builtin operations including identity and standard I/O operations.

    This handler provides:
    1. Identity operation for pass-through functionality
    2. Read and write operations for tensor data using numpy as intermediate data format

    The I/O operations support reading from and writing to .npy files, with automatic
    conversion between different tensor formats (JAX arrays, PyTorch tensors, numpy arrays)
    and numpy arrays.
    """

    # override
    def setup(self, rank: int) -> None:
        """Set up the runtime environment."""
        self._rank = rank

    # override
    def teardown(self) -> None:
        """Clean up the runtime environment."""

    def list_fn_names(self) -> list[str]:
        """List function names that this handler can execute."""
        return [
            "builtin.identity",
            "builtin.read",
            "builtin.write",
            "builtin.constant",
            "builtin.rank",
            "builtin.prand",
        ]

    def _convert_to_numpy(self, obj: TensorLike) -> np.ndarray:
        """Convert a TensorLike object to numpy array.

        Args:
            obj: TensorLike object to convert

        Returns:
            np.ndarray: Converted numpy array

        Raises:
            Exception: If conversion fails
        """
        # Already a numpy array - use asarray to avoid unnecessary copies
        if isinstance(obj, np.ndarray):
            return obj

        # Try to use .numpy() method if available (e.g., for JAX/PyTorch tensors).
        if hasattr(obj, "numpy"):
            numpy_method = getattr(obj, "numpy", None)
            if callable(numpy_method):
                try:
                    # Use asarray to avoid a copy if the result is already a numpy array.
                    return np.asarray(numpy_method())
                except Exception:
                    # If .numpy() fails, fall through to the general conversion.
                    pass

        # Fallback for objects without a .numpy() method or if it fails.
        return np.asarray(obj)

    # override
    def execute(
        self,
        pfunc: PFunction,
        args: list[TensorLike],
    ) -> list[TensorLike]:
        """Execute builtin operations.

        Args:
            pfunc: PFunction containing operation type and attributes
            args: Input arguments - varies by operation type

        Returns:
            list[TensorLike]: Results based on operation type:
                             - Identity: the input argument
                             - Read: list containing loaded data
                             - Write: list containing the original object that was written
                             - Constant: list containing the constant data
                             - Rank: list containing the rank value
                             - Prand: list containing random data

        Raises:
            ValueError: If required attributes are missing or wrong number of args
            RuntimeError: If file I/O operations fail
        """
        if pfunc.fn_type == "builtin.identity":
            if len(args) != 1:
                raise ValueError("Identity expects exactly one argument.")
            return args

        elif pfunc.fn_type == "builtin.read":
            path = pfunc.attrs.get("path")
            if path is None:
                raise ValueError("Read function requires 'path' attribute.")
            if len(args) != 0:
                raise ValueError("Read expects no arguments.")

            # Read numpy array from file
            try:
                data = np.load(path)
                return [data]
            except Exception as e:
                raise RuntimeError(f"Failed to read from {path}: {e}") from e

        elif pfunc.fn_type == "builtin.write":
            path = pfunc.attrs.get("path")
            if path is None:
                raise ValueError("Write function requires 'path' attribute.")
            if len(args) != 1:
                raise ValueError("Write expects exactly one argument.")

            obj = args[0]

            # Convert TensorLike object to numpy array and write to file
            try:
                # Create directory if it doesn't exist
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                np_array = self._convert_to_numpy(obj)
                np.save(path, np_array)
                return [obj]  # Return the original object
            except Exception as e:
                raise RuntimeError(f"Failed to write to {path}: {e}") from e

        elif pfunc.fn_type == "builtin.constant":
            if len(args) != 0:
                raise ValueError("Constant expects no arguments.")

            data_bytes = pfunc.attrs.get("data_bytes")

            if data_bytes is None:
                raise ValueError("Constant function requires 'data_bytes' attribute.")

            output_type = pfunc.outs_info[0]

            if isinstance(output_type, TableType):
                # For table constants, parse JSON data
                import json

                table_data = json.loads(data_bytes.decode("utf-8"))
                return [table_data]
            else:
                # Handle tensor constants - get info from output_type
                shape = output_type.shape
                dtype = output_type.dtype.numpy_dtype()

                if shape == ():
                    # Scalar
                    data = np.frombuffer(data_bytes, dtype=dtype)
                    return [data[0]]  # Return numpy scalar
                else:
                    # Tensor
                    data = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
                    return [data]

        elif pfunc.fn_type == "builtin.rank":
            if len(args) != 0:
                raise ValueError("Rank expects no arguments.")

            # Get the rank from the execution context (this should be set during setup)
            # For now, we'll use a simple approach - the rank should be available in the handler
            rank = getattr(self, "_rank", 0)  # Default to 0 if not set
            return [np.array(rank, dtype=np.uint64)]

        elif pfunc.fn_type == "builtin.prand":
            if len(args) != 0:
                raise ValueError("Prand expects no arguments.")

            shape = pfunc.attrs.get("shape", ())
            # Generate random values with the specified shape
            dtype = np.uint64

            rng = np.random.default_rng()
            info = np.iinfo(dtype)
            data = rng.integers(
                low=info.min,
                high=info.max,
                size=shape,
                dtype=dtype,
                endpoint=True,  # includes the high value in the possible results
            )
            return [data]

        else:
            raise ValueError(f"Unsupported function type: {pfunc.fn_type}")
