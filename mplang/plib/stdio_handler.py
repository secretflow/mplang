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

from mplang.core.base import TensorLike
from mplang.core.pfunc import PFunction, PFunctionHandler


class StdioHandler(PFunctionHandler):
    """Handler for standard I/O operations using numpy as intermediate data format.

    This handler provides read and write operations for tensor data using
    numpy's save/load functionality. It supports reading from and writing to
    .npy files, with automatic conversion between different tensor formats
    (JAX arrays, PyTorch tensors, numpy arrays) and numpy arrays.
    """

    # override
    def setup(self) -> None:
        """Set up the runtime environment."""

    # override
    def teardown(self) -> None:
        """Clean up the runtime environment."""

    def list_fn_names(self) -> list[str]:
        """List function names that this handler can execute."""
        return ["Read", "Write"]

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
        """Execute read or write operations.

        Args:
            pfunc: PFunction containing operation type and attributes
            args: Input arguments - empty for Read, single tensor for Write

        Returns:
            list[TensorLike]: For Read - list containing loaded data;
                             For Write - list containing the original object that was written.

        Raises:
            ValueError: If required attributes are missing or wrong number of args
            RuntimeError: If file I/O operations fail
        """
        if pfunc.fn_type == "Read":
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

        elif pfunc.fn_type == "Write":
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

        else:
            raise ValueError(f"Unsupported function type: {pfunc.fn_type}")
