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

"""Tests for mplang.v2.dialects.dtypes module."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from mplang.v2.dialects import dtypes
from mplang.v2.edsl import typing as t


class TestToJax:
    """Tests for to_jax conversion."""

    @pytest.mark.parametrize(
        "scalar_type,expected",
        [
            (t.i8, jnp.int8),
            (t.i16, jnp.int16),
            (t.i32, jnp.int32),
            (t.i64, jnp.int64),
            (t.u8, jnp.uint8),
            (t.u16, jnp.uint16),
            (t.u32, jnp.uint32),
            (t.u64, jnp.uint64),
            (t.f16, jnp.float16),
            (t.f32, jnp.float32),
            (t.f64, jnp.float64),
            (t.c64, jnp.complex64),
            (t.c128, jnp.complex128),
            (t.bool_, jnp.bool_),
        ],
    )
    def test_scalar_to_jax(self, scalar_type, expected):
        """Test conversion from MPLang scalar types to JAX dtypes."""
        assert dtypes.to_jax(scalar_type) == expected

    def test_invalid_type_raises(self):
        """Test that non-ScalarType raises TypeError."""
        with pytest.raises(TypeError, match="Expected ScalarType"):
            dtypes.to_jax("not a type")

        with pytest.raises(TypeError, match="Expected ScalarType"):
            dtypes.to_jax(t.TensorType(t.f32, (3, 4)))


class TestToNumpy:
    """Tests for to_numpy conversion."""

    @pytest.mark.parametrize(
        "scalar_type,expected_name",
        [
            (t.i32, "int32"),
            (t.i64, "int64"),
            (t.f32, "float32"),
            (t.f64, "float64"),
            (t.bool_, "bool"),
        ],
    )
    def test_scalar_to_numpy(self, scalar_type, expected_name):
        """Test conversion from MPLang scalar types to NumPy dtypes."""
        result = dtypes.to_numpy(scalar_type)
        assert isinstance(result, np.dtype)
        assert result.name == expected_name


class TestFromDtype:
    """Tests for from_dtype conversion (NumPy/JAX → MPLang)."""

    @pytest.mark.parametrize(
        "input_dtype,expected",
        [
            # JAX dtypes
            (jnp.int32, t.i32),
            (jnp.int64, t.i64),
            (jnp.float32, t.f32),
            (jnp.float64, t.f64),
            (jnp.bool_, t.bool_),
            # NumPy dtypes
            (np.int32, t.i32),
            (np.int64, t.i64),
            (np.float32, t.f32),
            (np.float64, t.f64),
            (np.bool_, t.bool_),
            # NumPy dtype objects
            (np.dtype("int64"), t.i64),
            (np.dtype("float32"), t.f32),
        ],
    )
    def test_from_dtype(self, input_dtype, expected):
        """Test conversion from JAX/NumPy dtypes to MPLang scalar types."""
        assert dtypes.from_dtype(input_dtype) == expected

    def test_invalid_dtype_raises(self):
        """Test that unsupported dtype raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert dtype"):
            dtypes.from_dtype("unsupported_type_xyz")


class TestFromArrow:
    """Tests for from_arrow conversion (PyArrow → MPLang)."""

    @pytest.mark.parametrize(
        "arrow_type,expected",
        [
            # Integer types
            (pa.int8(), t.i8),
            (pa.int16(), t.i16),
            (pa.int32(), t.i32),
            (pa.int64(), t.i64),
            (pa.uint8(), t.u8),
            (pa.uint16(), t.u16),
            (pa.uint32(), t.u32),
            (pa.uint64(), t.u64),
            # Float types
            (pa.float16(), t.f16),
            (pa.float32(), t.f32),
            (pa.float64(), t.f64),
            # Boolean
            (pa.bool_(), t.bool_),
            # String types
            (pa.string(), t.STRING),
            (pa.large_string(), t.STRING),
            # Date/Time types
            (pa.date32(), t.DATE),
            (pa.date64(), t.DATE),
            (pa.time32("s"), t.TIME),
            (pa.time64("us"), t.TIME),
            (pa.timestamp("ns"), t.TIMESTAMP),
            # Binary
            (pa.binary(), t.BINARY),
            (pa.large_binary(), t.BINARY),
        ],
    )
    def test_from_arrow(self, arrow_type, expected):
        """Test conversion from PyArrow types to MPLang types."""
        assert dtypes.from_arrow(arrow_type) == expected

    def test_unsupported_arrow_type_raises(self):
        """Test that unsupported PyArrow type raises ValueError."""
        # List type is not supported
        with pytest.raises(ValueError, match="Cannot convert PyArrow type"):
            dtypes.from_arrow(pa.list_(pa.int32()))


class TestFromPandas:
    """Tests for from_pandas conversion (Pandas dtype → MPLang)."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            # Integer types
            ([1, 2, 3], t.i64),  # Default int
            (pd.array([1, 2], dtype="int32"), t.i32),
            (pd.array([1, 2], dtype="Int64"), t.i64),  # Nullable int
            # Float types
            ([1.0, 2.0], t.f64),  # Default float
            (pd.array([1.0], dtype="float32"), t.f32),
            # Boolean
            ([True, False], t.bool_),
            # String
            (["a", "b"], t.STRING),
        ],
    )
    def test_from_pandas(self, data, expected):
        """Test conversion from Pandas dtypes to MPLang types."""
        df = pd.DataFrame({"col": data})
        assert dtypes.from_pandas(df["col"].dtype) == expected

    def test_datetime_conversion(self):
        """Test datetime dtype conversion."""
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        assert dtypes.from_pandas(df["ts"].dtype) == t.TIMESTAMP

    def test_unsupported_pandas_dtype_raises(self):
        """Test that unsupported Pandas dtype raises ValueError."""
        # Category type is not directly supported
        df = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c"])})
        with pytest.raises(ValueError, match="Cannot convert Pandas dtype"):
            dtypes.from_pandas(df["cat"].dtype)


class TestRoundTrip:
    """Test round-trip conversions."""

    @pytest.mark.parametrize(
        "scalar_type",
        [t.i8, t.i16, t.i32, t.i64, t.u8, t.u16, t.u32, t.u64, t.f32, t.f64, t.bool_],
    )
    def test_jax_roundtrip(self, scalar_type):
        """Test MPLang → JAX → MPLang round-trip."""
        jax_dtype = dtypes.to_jax(scalar_type)
        back = dtypes.from_dtype(jax_dtype)
        assert back == scalar_type

    @pytest.mark.parametrize(
        "scalar_type",
        [t.i32, t.i64, t.f32, t.f64, t.bool_],
    )
    def test_numpy_roundtrip(self, scalar_type):
        """Test MPLang → NumPy → MPLang round-trip."""
        np_dtype = dtypes.to_numpy(scalar_type)
        back = dtypes.from_dtype(np_dtype)
        assert back == scalar_type
