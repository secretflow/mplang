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


import json
from abc import ABC
from typing import Any, cast

import numpy as np
import pyarrow as pa


class VectorArray(pa.ExtensionArray, ABC):
    """Abstract base class for PyArrow extension arrays storing vector data.

    This is the base class for all vector array implementations in MPLang,
    which extends PyArrow's ExtensionArray to support custom vector types.
    Concrete implementations include DenseVectorArray, SparseVectorArray,
    and OneHotVectorArray.

    Attributes:
        type: The VectorType associated with this array
        storage: The underlying PyArrow array storing the data
    """


class DenseVectorArray(VectorArray):
    def to_numpy(
        self, zero_copy_only: bool = True, writable: bool = False
    ) -> np.ndarray:
        """Convert dense vectors to numpy arrays.

        Args:
            zero_copy_only: Ignored - conversion to 2D array always requires a copy.
            writable: If True, ensures the returned array is writable.

        Returns:
            2D numpy array of shape (n_vectors, size) with dense vector representations.
        """
        _ = zero_copy_only  # Ignored: conversion to 2D array always requires a copy
        _ = writable  # Unused: stacking creates writable arrays by default

        size = self.type.size
        value_type = self.type.value_type

        # Map PyArrow type to numpy dtype
        np_dtype = _pa_type_to_np_dtype(value_type)

        n_vectors = len(self.storage)
        result = np.zeros((n_vectors, size), dtype=np_dtype)

        for i in range(n_vectors):
            list_val = self.storage[i].as_py()
            if list_val is not None:
                result[i] = list_val

        return result


class SparseVectorItem:
    """Scalar value representing a single sparse vector from a SparseVectorArray.

    Similar to PyArrow's ListScalar for FixedSizeListArray access.
    Allows accessing individual vector elements via indexing and converting to Python types.
    """

    def __init__(self, value: pa.StructScalar, size: int):
        self._value = value
        self._size = size

    def __getitem__(self, index: int) -> int | float:
        """Get the value at the given index in the sparse vector.

        Args:
            index: The position in the vector to access

        Returns:
            The value at the given index (Python native type), or 0 if not set

        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of bounds for sparse vector of size {self._size}"
            )

        if not self._value.is_valid:
            return 0

        # Access child ListScalars directly: indices is at position 0, values at 1
        indices = self._value[0]
        values = self._value[1]
        for idx, val in zip(indices, values, strict=True):
            if idx.as_py() == index:
                return cast(int | float, val.as_py())

        return 0

    def as_py(self) -> list:
        """Convert the sparse vector to a dense Python list.

        Returns:
            A Python list representing the dense vector with zeros for unset elements
        """
        result = [0] * self._size
        if self._value.is_valid:
            # Access child ListScalars directly
            indices = self._value[0]
            values = self._value[1]
            for idx, val in zip(indices, values, strict=True):
                idx_py = idx.as_py()
                if 0 <= idx_py < self._size:
                    result[idx_py] = val.as_py()
        return result

    def __repr__(self) -> str:
        return f"SparseVectorItem({self.as_py()})"


class SparseVectorArray(VectorArray):
    def __getitem__(self, key: int | list[int]) -> Any:
        """Access individual sparse vectors or elements.

        Args:
            key: Index to access the vector at position. Can be int for single
                 element or slice for sub-array.

        Returns:
            SparseVectorItem if key is int, otherwise the sliced array

        Raises:
            IndexError: If key is out of bounds
        """
        result = self.storage[key]
        if isinstance(result, pa.StructScalar):
            return SparseVectorItem(result, self.type.size)
        return result

    def to_pylist(self) -> list:
        """Convert the sparse vectors to a Python list of lists (dense representation).

        Returns:
            A Python list where each element is a dense vector represented as a list,
            with zeros for sparse elements that were not set.
        """
        # Access underlying ListArrays directly for efficiency
        indices_list = self.storage.field("indices")
        values_list = self.storage.field("values")

        size = self.type.size
        n_rows = len(self.storage)
        result = []

        for i in range(n_rows):
            struct_val = self.storage[i]
            if struct_val.is_valid:
                # Get sub-array for this vector
                indices_arr = indices_list[i]
                values_arr = values_list[i]
                dense = [0] * size
                for idx, val in zip(indices_arr, values_arr, strict=True):
                    idx_py = idx.as_py()
                    if 0 <= idx_py < size:
                        dense[idx_py] = val.as_py()
                result.append(dense)
            else:
                result.append([0] * size)
        return result

    def to_numpy(
        self, zero_copy_only: bool = True, writable: bool = False
    ) -> np.ndarray:
        """Convert sparse vectors to dense numpy arrays.

        Args:
            zero_copy_only: Ignored - sparse to dense conversion always requires a copy.
            writable: Ignored - np.zeros creates writable arrays by default.

        Returns:
            2D numpy array of shape (n_vectors, size) with dense vector representations.
        """
        _ = zero_copy_only, writable  # Not applicable for sparse-to-dense conversion
        value_type = self.type.value_type
        np_dtype = _pa_type_to_np_dtype(value_type)
        size = self.type.size
        n_rows = len(self.storage)

        result = np.zeros((n_rows, size), dtype=np_dtype)

        # Access underlying ListArrays for batch processing
        indices_list = self.storage.field("indices")
        values_list = self.storage.field("values")

        for i in range(n_rows):
            if self.storage[i].is_valid:
                idx_arr = indices_list[i]
                val_arr = values_list[i]
                for idx, val in zip(idx_arr, val_arr, strict=True):
                    idx_py = idx.as_py()
                    if 0 <= idx_py < size:
                        result[i, idx_py] = val.as_py()

        return result


def _pa_type_to_np_dtype(pa_type: pa.DataType) -> type[np.generic]:
    """Convert PyArrow data type to numpy dtype.

    Args:
        pa_type: PyArrow data type

    Returns:
        Corresponding numpy dtype
    """
    type_id = pa_type.id
    dtype_map = {
        pa.int8().id: np.int8,
        pa.int16().id: np.int16,
        pa.int32().id: np.int32,
        pa.int64().id: np.int64,
        pa.uint8().id: np.uint8,
        pa.uint16().id: np.uint16,
        pa.uint32().id: np.uint32,
        pa.uint64().id: np.uint64,
        pa.float16().id: np.float16,
        pa.float32().id: np.float32,
        pa.float64().id: np.float64,
    }
    if type_id not in dtype_map:
        raise ValueError(f"Unsupported PyArrow type for numpy conversion: {pa_type}")
    return dtype_map[type_id]  # type: ignore[return-value]


class OneHotVectorItem:
    """Scalar value representing a single one-hot vector from a OneHotVectorArray.

    Similar to PyArrow's ListScalar for FixedSizeListArray access.
    The one-hot vector is stored as the index of the active position.
    """

    def __init__(self, value: pa.Scalar, size: int) -> None:
        self._value = value
        self._size = size

    def __getitem__(self, index: int) -> int | float:
        """Get the value at the given index in the one-hot vector.

        Args:
            index: The position in the vector to access

        Returns:
            1 if index is the active position, 0 otherwise

        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of bounds for one-hot vector of size {self._size}"
            )

        active_index: Any = self._value.as_py()
        return 1 if active_index == index else 0

    def as_py(self) -> list:
        """Convert the one-hot vector to a dense Python list.

        Returns:
            A Python list of zeros with 1 at the active position
        """
        result = [0] * self._size
        active_index = self._value.as_py()
        if active_index is not None and 0 <= active_index < self._size:
            result[active_index] = 1
        return result

    def __repr__(self) -> str:
        return f"OneHotVectorItem({self.as_py()})"


class OneHotVectorArray(VectorArray):
    @classmethod
    def from_array(
        cls, array: pa.Array, vector_type: "OneHotVectorType"
    ) -> "OneHotVectorArray":
        """Create a one-hot vector array from an integer array.

        Each integer value represents the active position in the one-hot vector.

        Args:
            array: PyArrow array of integer values representing the active position
            vector_type: OneHotVectorType specifying size, value_type, and index_type

        Returns:
            OneHotVectorArray where each element is a one-hot vector

        Raises:
            TypeError: If the input array is not an integer type, or if array.type
                       is not compatible with vector_type.index_type

        Examples:
            >>> import pyarrow as pa
            >>> from mplang.extend.arrow import OneHotVectorArray, onehot_vector
            >>> ints = pa.array([0, 1, 2, 1])
            >>> vt = onehot_vector(
            ...     size=3, value_type=pa.float32(), index_type=pa.int32()
            ... )
            >>> result = OneHotVectorArray.from_array(ints, vt)
            >>> result.to_pylist()
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
        """
        if not pa.types.is_integer(array.type):
            raise TypeError(f"Input array must be of integer type, got {array.type}")

        storage = pa.array(array.cast(vector_type.storage_type))
        result = pa.ExtensionArray.from_storage(vector_type, storage)
        assert isinstance(result, OneHotVectorArray)
        return result

    def __getitem__(self, key: Any) -> Any:
        """Access individual one-hot vectors.

        Args:
            key: Index to access the vector at position. Can be int for single
                 element or slice for sub-array.

        Returns:
            OneHotVectorItem if key is int, otherwise the sliced array

        Raises:
            IndexError: If key is out of bounds
        """
        result = self.storage[key]
        if isinstance(result, pa.Scalar):
            return OneHotVectorItem(result, self.type.size)
        return result

    def to_numpy(
        self, zero_copy_only: bool = True, writable: bool = False
    ) -> np.ndarray:
        """Convert one-hot vectors to dense numpy arrays.

        Args:
            zero_copy_only: Ignored - one-hot to dense conversion always requires a copy.
            writable: If True, ensures the returned array is writable.

        Returns:
            2D numpy array of shape (n_vectors, size) with dense vector representations.
        """
        _ = zero_copy_only  # Ignored: one-hot vectors cannot be zero-copied to numpy
        _ = writable  # Unused: np.zeros creates writable arrays by default

        size = self.type.size
        value_type = self.type.value_type

        # Map PyArrow type to numpy dtype
        np_dtype = _pa_type_to_np_dtype(value_type)
        one = np.dtype(np_dtype).type(1)

        n_vectors = len(self.storage)
        result = np.zeros((n_vectors, size), dtype=np_dtype)

        for i in range(n_vectors):
            idx = self.storage[i].as_py()
            if idx is not None and 0 <= idx < size:
                result[i, idx] = one

        return result

    def to_pylist(self) -> list:
        """Convert the one-hot vectors to a Python list of lists.

        Returns:
            A Python list where each element is a one-hot vector represented as a list
        """
        result = []
        for scalar in self.storage:
            active_index = scalar.as_py()
            onehot = [0] * self.type.size
            if active_index is not None and 0 <= active_index < self.type.size:
                onehot[active_index] = 1
            result.append(onehot)
        return result


class VectorType(pa.ExtensionType, ABC):
    """Base class for PyArrow extension types for vectors."""

    def __init__(
        self,
        size: int,
        value_type: pa.DataType,
        storage_type: pa.DataType,
        extension_name: str,
    ):
        """Initialize the vector type.

        Args:
            size: The size/dimension of the vectors
            value_type: The data type of vector elements (e.g., pa.float32())
        """
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type)):
            raise TypeError(
                f"value_type must be a numeric type (int8, int16, int32, int64, "
                f"uint8, uint16, uint32, uint64, float16, float32, float64), "
                f"got {value_type}"
            )
        self._size = size
        self._value_type = value_type
        super().__init__(storage_type, extension_name)

    @property
    def size(self) -> int:
        """Get the size/dimension of the vectors."""
        return self._size

    @property
    def value_type(self) -> pa.DataType:
        """Get the data type of vector elements."""
        return self._value_type

    def __eq__(self, other: Any) -> bool:
        """Equality check."""
        if not isinstance(other, VectorType):
            return False
        return (
            self.extension_name == other.extension_name
            and self.size == other.size
            and self.value_type == other.value_type
            and self.storage_type == other.storage_type
        )

    def __ne__(self, other: Any) -> bool:
        """Inequality check."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Hash implementation."""
        return hash((self.extension_name, self.storage_type, self.size))


class DenseVectorType(VectorType):
    """PyArrow extension type for dense vectors."""

    EXTENSION_NAME = "mplang.dense_vector"

    def __init__(self, size: int, value_type: pa.DataType | None = None) -> None:
        value_type = value_type or pa.float32()
        storage_type = pa.list_(value_type, list_size=size)
        super().__init__(size, value_type, storage_type, self.EXTENSION_NAME)

    def __str__(self) -> str:
        return f"dense_vector[{self.value_type}, size={self.size}]"

    def __arrow_ext_serialize__(self) -> bytes:
        """Serialize the type metadata including value_type and size."""
        return b""

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> "DenseVectorType":
        """Deserialize the type metadata including value_type and size."""
        if not isinstance(storage_type, pa.FixedSizeListType):
            raise ValueError(
                f"Expected FixedSizeListType for DenseVector, got {type(storage_type)}"
            )
        return cls(storage_type.list_size, storage_type.value_type)

    def __arrow_ext_class__(self) -> type:
        return DenseVectorArray


class SparseVectorType(VectorType):
    """PyArrow extension type for sparse vectors."""

    EXTENSION_NAME = "mplang.sparse_vector"

    def __init__(
        self,
        size: int,
        value_type: pa.DataType | None = None,
        index_type: pa.DataType | None = None,
    ):
        value_type = value_type or pa.float32()
        index_type = index_type or pa.int32()
        if not pa.types.is_integer(index_type):
            raise TypeError(f"index_type must be an integer type, got {index_type}")
        storage_type = pa.struct(
            [
                pa.field("indices", pa.list_(index_type)),
                pa.field("values", pa.list_(value_type)),
            ]
        )
        super().__init__(size, value_type, storage_type, self.EXTENSION_NAME)

    def __str__(self) -> str:
        return f"sparse_vector[{self.value_type}, size={self.size}]"

    def __arrow_ext_serialize__(self) -> bytes:
        """Serialize the metadata."""
        metadata = {"size": self.size}
        return json.dumps(metadata).encode("utf-8")

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> "SparseVectorType":
        """Deserialize the metadata."""
        if not isinstance(storage_type, pa.StructType):
            raise ValueError(
                f"Expected StructType for SparseVector, got {type(storage_type)}"
            )
        metadata = json.loads(serialized.decode("utf-8"))
        size = metadata["size"]
        value_type = storage_type["values"].type
        index_type = storage_type["indices"].type
        return cls(size, value_type, index_type)

    def __arrow_ext_class__(self) -> type:
        return SparseVectorArray


_TYPE_MAP = {
    pa.float16().id: pa.float16(),
    pa.float32().id: pa.float32(),
    pa.float64().id: pa.float64(),
    pa.int8().id: pa.int8(),
    pa.int16().id: pa.int16(),
    pa.int32().id: pa.int32(),
    pa.int64().id: pa.int64(),
    pa.uint8().id: pa.uint8(),
    pa.uint16().id: pa.uint16(),
    pa.uint32().id: pa.uint32(),
    pa.uint64().id: pa.uint64(),
}


class OneHotVectorType(VectorType):
    """PyArrow extension type for one-hot vectors.

    One-hot vectors are stored as the index of the active position.
    The value_type specifies the type of the non-zero element value,
    and the size parameter specifies the dimension of the vector.
    """

    EXTENSION_NAME = "mplang.onehot_vector"

    def __init__(
        self,
        size: int,
        value_type: pa.DataType | None = None,
        index_type: pa.DataType | None = None,
    ):
        value_type = value_type or pa.float32()
        index_type = index_type or pa.int32()
        if not pa.types.is_integer(index_type):
            raise TypeError(f"index_type must be an integer type, got {index_type}")
        storage_type = index_type
        super().__init__(size, value_type, storage_type, self.EXTENSION_NAME)

    def __str__(self) -> str:
        return f"onehot_vector[{self.value_type}, size={self.size}]"

    def __arrow_ext_serialize__(self) -> bytes:
        """Serialize the metadata."""
        metadata = {"size": self.size, "value_type_id": self.value_type.id}
        return json.dumps(metadata).encode("utf-8")

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> "OneHotVectorType":
        """Deserialize the metadata."""
        metadata = json.loads(serialized.decode("utf-8"))
        size = metadata["size"]
        value_type = _TYPE_MAP[metadata["value_type_id"]]
        index_type = storage_type
        return cls(size, value_type, index_type)

    def __arrow_ext_class__(self) -> type:
        return OneHotVectorArray


# Utility functions
def dense_vector(size: int, value_type: pa.DataType | None = None) -> DenseVectorType:
    """Create a DenseVectorType instance.

    Args:
        size: Fixed size of vectors
        value_type: Data type of vector elements

    Returns:
        DenseVectorType instance
    """
    return DenseVectorType(size, value_type)


def sparse_vector(
    size: int,
    value_type: pa.DataType | None = None,
    index_type: pa.DataType | None = None,
) -> SparseVectorType:
    """Create a SparseVectorType instance.

    Args:
        size: Dimension of vectors
        value_type: Data type of vector elements
        index_type: Data type for storing indices (defaults to pa.int32())

    Returns:
        SparseVectorType instance
    """
    return SparseVectorType(size, value_type, index_type)


def onehot_vector(
    size: int,
    value_type: pa.DataType | None = None,
    index_type: pa.DataType | None = None,
) -> OneHotVectorType:
    """Create a OneHotVectorType instance.

    Args:
        size: Dimension of the one-hot vectors
        value_type: Data type of the non-zero element value
        index_type: Data type for storing the active position index (defaults to pa.int32())

    Returns:
        OneHotVectorType instance
    """
    return OneHotVectorType(size, value_type, index_type)


# Register the extension types with PyArrow
def _register_vector_types() -> None:
    """Register the vector extension types with PyArrow."""

    # Simply attempt registration - PyArrow will handle duplicates gracefully
    try:
        pa.register_extension_type(DenseVectorType(pa.float32(), 1))
    except Exception:
        pass  # Type might already be registered

    try:
        pa.register_extension_type(SparseVectorType(pa.float32(), 1))
    except Exception:
        pass  # Type might already be registered

    try:
        pa.register_extension_type(OneHotVectorType(pa.int32(), 1))
    except Exception:
        pass  # Type might already be registered


# Auto-register on import (must be after all class definitions)
_register_vector_types()


def table_to_numpy(table: pa.Table) -> np.ndarray:
    """Convert a PyArrow Table to a 2D numpy array.

    If a column contains VectorType data (dense/sparse/onehot vectors),
    the column will be expanded into multiple columns (one per dimension).
    Regular numeric columns remain as single columns.

    Args:
        table: PyArrow Table to convert

    Returns:
        2D numpy array of shape (n_rows, n_features), where VectorType
        columns are expanded into multiple feature columns.
    """
    # Fast path: no VectorType columns - use pandas which is more efficient
    has_vector_type = any(isinstance(col.type, VectorType) for col in table.columns)
    if not has_vector_type:
        result = table.to_pandas().to_numpy()
        return cast(np.ndarray, result)

    # Slow path: handle VectorType columns
    result_parts = []

    for col in table.columns:
        if isinstance(col.type, VectorType):
            # VectorArray.to_numpy() returns 2D array (n_rows, vector_size)
            vector_array = col.combine_chunks()
            arr_2d = vector_array.to_numpy()
            result_parts.append(arr_2d)
        else:
            # Regular numeric column - ensure it's 2D
            arr = col.to_numpy()
            if arr.ndim == 1:
                result_parts.append(arr.reshape(-1, 1))
            else:
                result_parts.append(arr)

    # Concatenate all parts column-wise into a 2D array
    return np.concatenate(result_parts, axis=1)
