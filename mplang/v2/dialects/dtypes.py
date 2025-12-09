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

"""Dtype conversion utilities between MPLang ScalarType and external libraries.

This module provides bidirectional conversion between MPLang's type system
(ScalarType hierarchy) and external library types (NumPy, JAX, PyArrow, Pandas).

Usage:
    from mplang.v2.dialects import dtypes

    # MPLang ScalarType → JAX/NumPy
    jax_dtype = dtypes.to_jax(scalar_types.f32)    # → jnp.float32
    np_dtype = dtypes.to_numpy(scalar_types.i64)   # → np.dtype('int64')

    # JAX/NumPy → MPLang ScalarType
    scalar_type = dtypes.from_dtype(np.float32)    # → scalar_types.f32
    scalar_type = dtypes.from_dtype(jnp.int64)     # → scalar_types.i64

    # PyArrow/Pandas → MPLang ScalarType
    scalar_type = dtypes.from_arrow(pa.int64())    # → scalar_types.i64
    scalar_type = dtypes.from_pandas(df["col"].dtype)  # → scalar_types.f64
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

import mplang.v2.edsl.typing as scalar_types

# ==============================================================================
# MPLang ScalarType → JAX/NumPy conversion
# ==============================================================================

# Mapping from MPLang ScalarType instances to JAX dtypes
_SCALAR_TO_JAX: dict[scalar_types.ScalarType, Any] = {
    # Signed integers
    scalar_types.i8: jnp.int8,
    scalar_types.i16: jnp.int16,
    scalar_types.i32: jnp.int32,
    scalar_types.i64: jnp.int64,
    # Unsigned integers
    scalar_types.u8: jnp.uint8,
    scalar_types.u16: jnp.uint16,
    scalar_types.u32: jnp.uint32,
    scalar_types.u64: jnp.uint64,
    # Floating point
    scalar_types.f16: jnp.float16,
    scalar_types.f32: jnp.float32,
    scalar_types.f64: jnp.float64,
    # Complex
    scalar_types.c64: jnp.complex64,
    scalar_types.c128: jnp.complex128,
    # Boolean (i1)
    scalar_types.bool_: jnp.bool_,
}


def to_jax(dtype: scalar_types.ScalarType) -> Any:
    """Convert MPLang scalar type to JAX dtype.

    Args:
        dtype: MPLang ScalarType (IntegerType, FloatType, or ComplexType)

    Returns:
        Corresponding JAX/NumPy dtype

    Raises:
        TypeError: If dtype is not a ScalarType
        ValueError: If dtype has no JAX equivalent

    Examples:
        >>> dtypes.to_jax(scalar_types.f32)
        <class 'jax.numpy.float32'>
        >>> dtypes.to_jax(scalar_types.i64)
        <class 'jax.numpy.int64'>
    """
    if not isinstance(dtype, scalar_types.ScalarType):
        raise TypeError(f"Expected ScalarType, got {type(dtype).__name__}")

    # Direct lookup
    if dtype in _SCALAR_TO_JAX:
        return _SCALAR_TO_JAX[dtype]

    # Handle dynamically created types (same structure but different instance)
    if isinstance(dtype, scalar_types.IntegerType):
        if dtype.bitwidth == 1:
            return jnp.bool_
        prefix = "int" if dtype.signed else "uint"
        try:
            return getattr(jnp, f"{prefix}{dtype.bitwidth}")
        except AttributeError:
            pass
    elif isinstance(dtype, scalar_types.FloatType):
        try:
            return getattr(jnp, f"float{dtype.bitwidth}")
        except AttributeError:
            pass
    elif isinstance(dtype, scalar_types.ComplexType):
        total_bits = dtype.inner_type.bitwidth * 2
        try:
            return getattr(jnp, f"complex{total_bits}")
        except AttributeError:
            pass

    raise ValueError(f"No JAX dtype equivalent for {dtype}")


def to_numpy(dtype: scalar_types.ScalarType) -> np.dtype:
    """Convert MPLang scalar type to NumPy dtype.

    Args:
        dtype: MPLang ScalarType

    Returns:
        Corresponding NumPy dtype

    Examples:
        >>> dtypes.to_numpy(scalar_types.f32)
        dtype('float32')
    """
    jax_dtype = to_jax(dtype)
    return np.dtype(jax_dtype)  # type: ignore[no-any-return]


# ==============================================================================
# JAX/NumPy → MPLang ScalarType conversion
# ==============================================================================

# Reverse mapping (built dynamically to stay in sync)
_JAX_TO_SCALAR: dict[Any, scalar_types.ScalarType] = {
    v: k for k, v in _SCALAR_TO_JAX.items()
}

# NumPy dtype to MPLang ScalarType mapping
_NUMPY_TO_SCALAR: dict[type, scalar_types.ScalarType] = {
    np.int8: scalar_types.i8,
    np.int16: scalar_types.i16,
    np.int32: scalar_types.i32,
    np.int64: scalar_types.i64,
    np.uint8: scalar_types.u8,
    np.uint16: scalar_types.u16,
    np.uint32: scalar_types.u32,
    np.uint64: scalar_types.u64,
    np.float16: scalar_types.f16,
    np.float32: scalar_types.f32,
    np.float64: scalar_types.f64,
    np.complex64: scalar_types.c64,
    np.complex128: scalar_types.c128,
    np.bool_: scalar_types.bool_,
}


def from_dtype(dtype: Any) -> scalar_types.ScalarType:
    """Convert JAX/NumPy dtype to MPLang scalar type.

    Args:
        dtype: JAX dtype, NumPy dtype, or dtype-like object

    Returns:
        Corresponding MPLang ScalarType

    Raises:
        ValueError: If dtype cannot be converted

    Examples:
        >>> dtypes.from_dtype(jnp.float32)
        f32
        >>> dtypes.from_dtype(np.dtype("int64"))
        i64
    """
    # Direct lookup for JAX types
    if dtype in _JAX_TO_SCALAR:
        return _JAX_TO_SCALAR[dtype]

    # Direct lookup for NumPy scalar types
    if dtype in _NUMPY_TO_SCALAR:
        return _NUMPY_TO_SCALAR[dtype]

    # Handle np.dtype objects
    if isinstance(dtype, np.dtype):
        dtype_type = dtype.type
        if dtype_type in _NUMPY_TO_SCALAR:
            return _NUMPY_TO_SCALAR[dtype_type]

    # Try to normalize to a dtype object
    try:
        normalized = jnp.dtype(dtype)
        if normalized in _JAX_TO_SCALAR:
            return _JAX_TO_SCALAR[normalized]
    except Exception:
        pass

    # Fallback: match by name
    name = getattr(dtype, "name", str(dtype)).lower()

    # Integer types
    if "int8" in name and "uint" not in name:
        return scalar_types.i8
    elif "int16" in name and "uint" not in name:
        return scalar_types.i16
    elif "int32" in name and "uint" not in name:
        return scalar_types.i32
    elif "int64" in name and "uint" not in name:
        return scalar_types.i64
    elif "uint8" in name:
        return scalar_types.u8
    elif "uint16" in name:
        return scalar_types.u16
    elif "uint32" in name:
        return scalar_types.u32
    elif "uint64" in name:
        return scalar_types.u64
    # Float types
    elif "float16" in name:
        return scalar_types.f16
    elif "float32" in name:
        return scalar_types.f32
    elif "float64" in name:
        return scalar_types.f64
    # Complex types
    elif "complex64" in name:
        return scalar_types.c64
    elif "complex128" in name:
        return scalar_types.c128
    # Boolean
    elif "bool" in name:
        return scalar_types.bool_

    raise ValueError(f"Cannot convert dtype '{dtype}' to MPLang ScalarType")


# ==============================================================================
# PyArrow → MPLang ScalarType conversion
# ==============================================================================


def from_arrow(arrow_type: Any) -> scalar_types.BaseType:
    """Convert PyArrow type to MPLang scalar type.

    Args:
        arrow_type: PyArrow DataType (e.g., pa.int64(), pa.float32())

    Returns:
        Corresponding MPLang BaseType (ScalarType or CustomType)

    Raises:
        ValueError: If arrow_type cannot be converted

    Examples:
        >>> import pyarrow as pa
        >>> dtypes.from_arrow(pa.int64())
        i64
        >>> dtypes.from_arrow(pa.float32())
        f32
        >>> dtypes.from_arrow(pa.string())
        Custom[string]
    """
    import pyarrow as pa

    if pa.types.is_boolean(arrow_type):
        return scalar_types.bool_
    elif pa.types.is_int8(arrow_type):
        return scalar_types.i8
    elif pa.types.is_int16(arrow_type):
        return scalar_types.i16
    elif pa.types.is_int32(arrow_type):
        return scalar_types.i32
    elif pa.types.is_int64(arrow_type):
        return scalar_types.i64
    elif pa.types.is_uint8(arrow_type):
        return scalar_types.u8
    elif pa.types.is_uint16(arrow_type):
        return scalar_types.u16
    elif pa.types.is_uint32(arrow_type):
        return scalar_types.u32
    elif pa.types.is_uint64(arrow_type):
        return scalar_types.u64
    elif pa.types.is_float16(arrow_type):
        return scalar_types.f16
    elif pa.types.is_float32(arrow_type):
        return scalar_types.f32
    elif pa.types.is_float64(arrow_type) or pa.types.is_floating(arrow_type):
        return scalar_types.f64
    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return scalar_types.STRING
    elif pa.types.is_date(arrow_type):
        return scalar_types.DATE
    elif pa.types.is_time(arrow_type):
        return scalar_types.TIME
    elif pa.types.is_timestamp(arrow_type):
        return scalar_types.TIMESTAMP
    elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return scalar_types.BINARY
    else:
        raise ValueError(
            f"Cannot convert PyArrow type '{arrow_type}' to MPLang ScalarType"
        )


# ==============================================================================
# Pandas dtype → MPLang ScalarType conversion
# ==============================================================================


def from_pandas(pd_dtype: Any) -> scalar_types.BaseType:
    """Convert Pandas dtype to MPLang scalar type.

    Args:
        pd_dtype: Pandas dtype (e.g., df["col"].dtype)

    Returns:
        Corresponding MPLang BaseType (ScalarType or CustomType)

    Raises:
        ValueError: If pd_dtype cannot be converted

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"x": [1, 2, 3]})
        >>> dtypes.from_pandas(df["x"].dtype)
        i64
    """
    # Get the dtype name as string for matching
    dtype_name = str(pd_dtype)

    if dtype_name == "bool":
        return scalar_types.bool_
    elif dtype_name in ("int8", "Int8"):
        return scalar_types.i8
    elif dtype_name in ("int16", "Int16"):
        return scalar_types.i16
    elif dtype_name in ("int32", "Int32"):
        return scalar_types.i32
    elif dtype_name in ("int64", "Int64"):
        return scalar_types.i64
    elif dtype_name in ("uint8", "UInt8"):
        return scalar_types.u8
    elif dtype_name in ("uint16", "UInt16"):
        return scalar_types.u16
    elif dtype_name in ("uint32", "UInt32"):
        return scalar_types.u32
    elif dtype_name in ("uint64", "UInt64"):
        return scalar_types.u64
    elif dtype_name in ("float16", "Float16"):
        return scalar_types.f16
    elif dtype_name in ("float32", "Float32"):
        return scalar_types.f32
    elif dtype_name in ("float64", "Float64"):
        return scalar_types.f64
    elif dtype_name in ("complex64",):
        return scalar_types.c64
    elif dtype_name in ("complex128",):
        return scalar_types.c128
    elif dtype_name == "object" or dtype_name.startswith("string"):
        return scalar_types.STRING
    elif dtype_name.startswith("datetime"):
        return scalar_types.TIMESTAMP
    elif dtype_name.startswith("timedelta"):
        return scalar_types.INTERVAL
    else:
        raise ValueError(
            f"Cannot convert Pandas dtype '{pd_dtype}' to MPLang ScalarType"
        )
