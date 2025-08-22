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
from mplang.core.pfunc import HybridHandler, PFunction
from mplang.core.table import TableLike, TableType
from mplang.utils import table_utils


class BuiltinHandler(HybridHandler):
    """Handler for builtin operations including identity and standard I/O operations.

    This handler provides:
    1. Identity operation for pass-through functionality
    2. Read and write operations for tensor data using numpy as intermediate data format

    The I/O operations support reading from and writing to .npy files, with automatic
    conversion between different tensor formats (JAX arrays, PyTorch tensors, numpy arrays)
    and numpy arrays.
    """

    # Function name constants
    IDENTITY = "builtin.identity"
    READ = "builtin.read"
    WRITE = "builtin.write"
    CONSTANT = "builtin.constant"
    RANK = "builtin.rank"
    PRAND = "builtin.prand"

    # override
    def setup(self, rank: int) -> None:
        self._my_rank = rank

    # override
    def teardown(self) -> None: ...

    # override
    def list_fn_names(self) -> list[str]:
        """List function names that this handler can execute."""
        return [
            self.IDENTITY,
            self.READ,
            self.WRITE,
            self.CONSTANT,
            self.RANK,
            self.PRAND,
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

    def _identity(
        self, pfunc: PFunction, args: list[TensorLike | TableLike]
    ) -> list[TensorLike | TableLike]:
        """Execute builtin.identity operation."""
        if len(args) != 1:
            raise ValueError("Identity expects exactly one argument.")
        return args

    def _read(
        self, pfunc: PFunction, args: list[TensorLike | TableLike]
    ) -> list[TensorLike | TableLike]:
        """Execute builtin.read operation."""
        from mplang.core.table import TableType
        from mplang.utils import table_utils

        path = pfunc.attrs.get("path")
        if path is None:
            raise ValueError("Read function requires 'path' attribute.")
        if len(args) != 0:
            raise ValueError("Read expects no arguments.")

        output_type = pfunc.outs_info[0]

        try:
            if isinstance(output_type, TableType):
                # Read table data from CSV file
                with open(path, "rb") as f:
                    csv_bytes = f.read()
                df = table_utils.csv_to_dataframe(csv_bytes)
                return [df]
            else:
                # Read tensor data from numpy file
                data = np.load(path)
                return [data]
        except Exception as e:
            raise RuntimeError(f"Failed to read from {path}: {e}") from e

    def _write(
        self, pfunc: PFunction, args: list[TensorLike | TableLike]
    ) -> list[TensorLike | TableLike]:
        """Execute builtin.write operation."""
        from mplang.core.table import TableLike
        from mplang.utils import table_utils

        path = pfunc.attrs.get("path")
        if path is None:
            raise ValueError("Write function requires 'path' attribute.")
        if len(args) != 1:
            raise ValueError("Write expects exactly one argument.")

        obj = args[0]

        try:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            if isinstance(obj, TableLike):
                # Handle table-like objects using CSV serialization
                csv_bytes = table_utils.dataframe_to_csv(obj)
                with open(path, "wb") as f:
                    f.write(csv_bytes)
            else:
                # Handle tensor-like objects using numpy serialization
                np_array = self._convert_to_numpy(obj)  # type: ignore
                np.save(path, np_array)

            return [obj]  # Return the original object
        except Exception as e:
            raise RuntimeError(f"Failed to write to {path}: {e}") from e

    def _constant(
        self,
        pfunc: PFunction,
        args: list[TensorLike | TableLike],
    ) -> list[TensorLike | TableLike]:
        """Execute builtin.constant operation."""
        if len(args) != 0:
            raise ValueError("Constant expects no arguments.")

        data_bytes = pfunc.attrs.get("data_bytes")

        if data_bytes is None:
            raise ValueError("Constant function requires 'data_bytes' attribute.")

        output_type = pfunc.outs_info[0]
        data_format = pfunc.attrs.get("data_format")

        if isinstance(output_type, TableType):
            if data_format != "bytes[csv]":
                raise ValueError(f"Only 'bytes[csv]' is supported, got {data_format}")
            df = table_utils.csv_to_dataframe(data_bytes)
            return [df]
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

    def _rank(
        self, pfunc: PFunction, args: list[TensorLike | TableLike]
    ) -> list[TensorLike | TableLike]:
        """Execute builtin.rank operation."""
        if len(args) != 0:
            raise ValueError("Rank expects no arguments.")

        return [np.array(self._my_rank, dtype=np.uint64)]

    def _prand(
        self, pfunc: PFunction, args: list[TensorLike | TableLike]
    ) -> list[TensorLike | TableLike]:
        """Execute builtin.prand operation."""
        if len(args) != 0:
            raise ValueError("Prand expects no arguments.")

        shape = pfunc.attrs.get("shape", ())
        # Generate random values with the specified shape
        dtype = np.dtype(np.uint64)

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

    # override
    def execute(
        self,
        pfunc: PFunction,
        args: list[TensorLike | TableLike],
    ) -> list[TensorLike | TableLike]:
        """Execute builtin operations.

        Args:
            pfunc: PFunction containing operation type and attributes
            args: Input arguments - varies by operation type

        Returns:
            list[TensorLike | TableLike]: Results based on operation type

        Raises:
            ValueError: If required attributes are missing or wrong number of args
            RuntimeError: If file I/O operations fail
        """
        if pfunc.fn_type == self.IDENTITY:
            return self._identity(pfunc, args)
        elif pfunc.fn_type == self.READ:
            return self._read(pfunc, args)
        elif pfunc.fn_type == self.WRITE:
            return self._write(pfunc, args)
        elif pfunc.fn_type == self.CONSTANT:
            return self._constant(pfunc, args)
        elif pfunc.fn_type == self.RANK:
            return self._rank(pfunc, args)
        elif pfunc.fn_type == self.PRAND:
            return self._prand(pfunc, args)
        else:
            raise ValueError(f"Unsupported function type: {pfunc.fn_type}")
