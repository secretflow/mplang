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
from typing import final

import numpy as np

try:
    pass

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


@final
@dataclass(frozen=True)
class DType:
    """Custom dtype representation that can convert between different libraries."""

    name: str
    bitwidth: int
    is_signed: bool | None = None  # None for non-numeric types
    is_floating: bool = False
    is_complex: bool = False

    def __post_init__(self):
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
        }
        return name_map.get(self.name, self.name)

    @classmethod
    def from_numpy(cls, np_dtype) -> DType:
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
        else:
            raise ValueError(f"Unsupported NumPy dtype kind: {np_dtype.kind}")

    @classmethod
    def from_jax(cls, jax_dtype) -> DType:
        """Convert from JAX dtype to custom DType."""
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is not available")
        # JAX dtypes are essentially NumPy dtypes
        return cls.from_numpy(jax_dtype)

    @classmethod
    def from_python_type(cls, py_type: type) -> DType:
        """Convert from Python builtin type to custom DType."""
        if py_type == bool:
            return cls("bool", 8, None, False, False)
        elif py_type == int:
            # Use platform-dependent int size (usually 64-bit)
            return cls("int64", 64, True, False, False)
        elif py_type == float:
            return cls("float64", 64, True, True, False)
        elif py_type == complex:
            return cls("complex128", 128, True, True, True)
        else:
            raise ValueError(f"Unsupported Python type: {py_type}")

    @classmethod
    def from_any(cls, dtype_like) -> DType:
        """Convert from any supported dtype representation."""
        if isinstance(dtype_like, cls):
            return dtype_like
        elif isinstance(dtype_like, type) and dtype_like in (bool, int, float, complex):
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

    def to_numpy(self) -> np.dtype:
        """Convert custom DType to NumPy dtype."""
        return np.dtype(self.name)

    def to_jax(self):
        """Convert custom DType to JAX dtype."""
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is not available")
        import jax.numpy as jnp

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


# Helper functions for easy conversion


def from_numpy(np_dtype) -> DType:
    """Convert from NumPy dtype to custom DType."""
    return DType.from_numpy(np_dtype)


def to_numpy(dtype: DType) -> np.dtype:
    """Convert custom DType to NumPy dtype."""
    return dtype.to_numpy()
