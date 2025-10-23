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
from typing import Any, final

import numpy as np

try:
    # Check if JAX is available
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

__all__ = [
    "BINARY",
    "BOOL",
    "COMPLEX64",
    "COMPLEX128",
    "DATE",
    "DECIMAL",
    "FLOAT16",
    "FLOAT32",
    "FLOAT64",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "INTERVAL",
    "JSON",
    "STRING",
    "TIME",
    "TIMESTAMP",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "UUID",
    "DType",
    "from_numpy",
    "to_numpy",
]


@final
@dataclass(frozen=True)
class DType:
    """Custom dtype representation that can convert between different libraries."""

    name: str
    bitwidth: int
    is_signed: bool | None = None  # None for non-numeric types
    is_floating: bool = False
    is_complex: bool = False
    is_table_only: bool = False  # True for types only supported in tables

    def __post_init__(self) -> None:
        # Validate the dtype configuration
        if self.is_complex and not self.is_floating:
            raise ValueError("Complex types must be floating point")
        if self.is_floating and self.is_signed is None:
            # Floating point types are always signed
            object.__setattr__(self, "is_signed", True)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"DType('{self.name}')"

    def short_name(self) -> str:
        """Return a short name for the dtype."""
        # Map common types to short names
        name_map = {
            "bool": "bool",
            "int8": "i8",
            "int16": "i16",
            "int32": "i32",
            "int64": "i64",
            "uint8": "u8",
            "uint16": "u16",
            "uint32": "u32",
            "uint64": "u64",
            "float16": "f16",
            "float32": "f32",
            "float64": "f64",
            "complex64": "c64",
            "complex128": "c128",
            # Table-only types
            "string": "str",
            "date": "date",
            "time": "time",
            "timestamp": "timestamp",
            "decimal": "decimal",
            "binary": "binary",
            "json": "json",
            "uuid": "uuid",
            "interval": "interval",
        }
        return name_map.get(self.name, self.name)

    @classmethod
    def from_numpy(cls, np_dtype: Any) -> DType:
        """Convert from NumPy dtype to custom DType."""
        np_dtype = np.dtype(np_dtype)
        name = np_dtype.name

        if np_dtype.kind == "b":  # boolean
            return cls(name, 8, None, False, False)  # bool is typically 8 bits
        elif np_dtype.kind in ("i", "u"):  # integer
            return cls(name, np_dtype.itemsize * 8, np_dtype.kind == "i", False, False)
        elif np_dtype.kind == "f":  # floating
            return cls(name, np_dtype.itemsize * 8, True, True, False)
        elif np_dtype.kind == "c":  # complex
            return cls(name, np_dtype.itemsize * 8, True, True, True)
        elif np_dtype.kind in ("U", "S", "O"):  # unicode, byte string, or object
            # For string types, bitwidth represents the maximum number of bytes per element (i.e., np_dtype.itemsize)
            # Object is often used for strings.
            return STRING
        else:
            raise ValueError(f"Unsupported NumPy dtype kind: {np_dtype.kind}")

    @classmethod
    def from_jax(cls, jax_dtype: Any) -> DType:
        """Convert from JAX dtype to custom DType."""
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is not available")
        # Special handling for PRNG KeyTy: <class jax._src.prng.KeyTy>
        if jnp.issubdtype(jax_dtype, jax.dtypes.prng_key):
            return cls.from_numpy(np.uint32)

        # JAX dtypes are essentially NumPy dtypes
        return cls.from_numpy(jax_dtype)

    @classmethod
    def from_python_type(cls, py_type: type) -> DType:
        """Convert from Python builtin type to custom DType."""
        if py_type is bool:
            return cls("bool", 8, None, False, False)
        elif py_type is int:
            # Use platform-dependent int size (usually 64-bit)
            return cls("int64", 64, True, False, False)
        elif py_type is float:
            return cls("float64", 64, True, True, False)
        elif py_type is complex:
            return cls("complex128", 128, True, True, True)
        else:
            raise ValueError(f"Unsupported Python type: {py_type}")

    @classmethod
    def from_any(cls, dtype_like: Any) -> DType:
        """Convert from any supported dtype representation."""
        if isinstance(dtype_like, cls):
            return dtype_like

        # Try pandas specific dtype conversion first
        try:
            return cls._from_pandas_dtype(dtype_like)
        except (ImportError, TypeError):
            # ImportError if pandas is not installed
            # TypeError if it's not a pandas dtype we can handle
            pass

        try:
            return cls._from_arrow_dtype(dtype_like)
        except (ImportError, TypeError):
            # ImportError if pyarrow is not installed
            # TypeError if it's not a pyarrow dtype we can handle
            pass

        if isinstance(dtype_like, type) and dtype_like in (bool, int, float, complex):
            return cls.from_python_type(dtype_like)
        elif hasattr(dtype_like, "dtype") and not isinstance(dtype_like, type):
            # Objects with dtype attribute (arrays, etc.) but not dtype types themselves
            return cls.from_numpy(dtype_like.dtype)
        else:
            # Try NumPy conversion first (handles dtype types, strings, etc.)
            try:
                return cls.from_numpy(dtype_like)
            except (TypeError, ValueError):
                pass

            # Try JAX conversion if available
            if _JAX_AVAILABLE:
                try:
                    return cls.from_jax(dtype_like)
                except (TypeError, ValueError):
                    pass

            raise ValueError(f"Cannot convert {type(dtype_like)} to DType")

    @classmethod
    def _from_pandas_dtype(cls, dtype_like: Any) -> DType:
        """Convert pandas-specific dtypes to DType."""
        # Check if pandas is available
        try:
            import pandas as pd
            from pandas.api.types import is_any_real_numeric_dtype, is_bool_dtype
        except ImportError:
            raise ImportError("pandas not available") from None

        if not hasattr(dtype_like, "__module__") or "pandas" not in str(
            dtype_like.__module__
        ):
            # If it's not a pandas dtype, don't handle it here
            raise TypeError("Not a pandas dtype")

        if isinstance(dtype_like, pd.StringDtype):
            return STRING
        elif is_bool_dtype(dtype_like):
            # Catches pd.BooleanDtype() and 'bool'
            return BOOL
        elif is_any_real_numeric_dtype(dtype_like):
            # Catches Int64Dtype, Float64Dtype, etc.
            return cls.from_numpy(dtype_like.numpy_dtype)

        raise TypeError(f"Unsupported pandas dtype: {dtype_like}")

    @classmethod
    def _from_arrow_dtype(cls, dtype_like: Any) -> DType:
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError("pyarrow not available") from None

        if not isinstance(dtype_like, pa.DataType):
            raise TypeError("Not a pyarrow dtype")

        ARROW_DTYPE_MAPPING = {
            pa.bool_(): BOOL,
            pa.int8(): INT8,
            pa.int16(): INT16,
            pa.int32(): INT32,
            pa.int64(): INT64,
            pa.uint8(): UINT8,
            pa.uint16(): UINT16,
            pa.uint32(): UINT32,
            pa.uint64(): UINT64,
            pa.float16(): FLOAT16,
            pa.float32(): FLOAT32,
            pa.float64(): FLOAT64,
            pa.string(): STRING,
            pa.large_string(): STRING,
        }
        result = ARROW_DTYPE_MAPPING.get(dtype_like)
        if result is not None:
            return result
        raise TypeError(f"Unsupported arrow dtype: {dtype_like}")

    def to_numpy(self) -> np.dtype:
        """Convert custom DType to NumPy dtype."""
        return np.dtype(self.name)

    def to_jax(self) -> Any:
        """Convert custom DType to JAX dtype."""
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is not available")

        return jnp.dtype(self.name)

    def to_python_type(self) -> type:
        """Convert to Python builtin type if possible."""
        if self.name == "bool":
            return bool
        elif self.name.startswith("int") or self.name.startswith("uint"):
            return int
        elif self.name.startswith("float"):
            return float
        elif self.name.startswith("complex"):
            return complex
        else:
            raise ValueError(f"Cannot convert {self.name} to Python builtin type")

    def numpy_dtype(self) -> np.dtype:
        """Convert DType to NumPy dtype for compatibility with external libraries."""
        return self.to_numpy()


# Common dtype constants for convenience
BOOL = DType("bool", 8, None, False, False)
INT8 = DType("int8", 8, True, False, False)
INT16 = DType("int16", 16, True, False, False)
INT32 = DType("int32", 32, True, False, False)
INT64 = DType("int64", 64, True, False, False)
UINT8 = DType("uint8", 8, False, False, False)
UINT16 = DType("uint16", 16, False, False, False)
UINT32 = DType("uint32", 32, False, False, False)
UINT64 = DType("uint64", 64, False, False, False)
FLOAT16 = DType("float16", 16, True, True, False)
FLOAT32 = DType("float32", 32, True, True, False)
FLOAT64 = DType("float64", 64, True, True, False)
COMPLEX64 = DType("complex64", 64, True, True, True)
COMPLEX128 = DType("complex128", 128, True, True, True)

# Table-only types (marked with is_table_only=True)
STRING = DType("string", 0, None, False, False, True)  # Variable length string
DATE = DType("date", 32, None, False, False, True)  # Date only
TIME = DType("time", 32, None, False, False, True)  # Time only
TIMESTAMP = DType("timestamp", 64, None, False, False, True)  # Timestamp
DECIMAL = DType("decimal", 128, True, False, False, True)  # Arbitrary precision decimal
BINARY = DType("binary", 0, None, False, False, True)  # Binary data
JSON = DType("json", 0, None, False, False, True)  # JSON data
UUID = DType("uuid", 128, None, False, False, True)  # UUID type

# Additional types commonly used in relational databases but keep minimal
INTERVAL = DType("interval", 64, None, False, False, True)  # Time interval


# Helper functions for easy conversion


def from_numpy(np_dtype: Any) -> DType:
    """Convert from NumPy dtype to custom DType."""
    return DType.from_numpy(np_dtype)


def to_numpy(dtype: DType) -> np.dtype:
    """Convert custom DType to NumPy dtype."""
    return dtype.to_numpy()
