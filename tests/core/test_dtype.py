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

import numpy as np
import pytest

from mplang.core.dtype import (
    BOOL,
    COMPLEX64,
    COMPLEX128,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    DType,
)
from mplang.core.mask import Mask
from mplang.core.mpir import dtype_to_proto
from mplang.core.mptype import MPType
from mplang.core.tensor import TensorType

try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class TestDType:
    """Test cases for the DType class."""

    def test_dtype_creation(self):
        """Test basic DType creation and validation."""
        # Test valid boolean dtype
        bool_dtype = DType("bool", 1, True, False, False)
        assert bool_dtype.name == "bool"
        assert bool_dtype.bitwidth == 1
        assert bool_dtype.is_signed
        assert not bool_dtype.is_floating
        assert not bool_dtype.is_complex

        # Test valid integer dtype
        int32_dtype = DType("int32", 32, True, False, False)
        assert int32_dtype.name == "int32"
        assert int32_dtype.bitwidth == 32
        assert int32_dtype.is_signed
        assert not int32_dtype.is_floating
        assert not int32_dtype.is_complex

        # Test valid float dtype
        float64_dtype = DType("float64", 64, True, True, False)
        assert float64_dtype.name == "float64"
        assert float64_dtype.bitwidth == 64
        assert float64_dtype.is_signed
        assert float64_dtype.is_floating
        assert not float64_dtype.is_complex

        # Test valid complex dtype
        complex128_dtype = DType("complex128", 128, True, True, True)
        assert complex128_dtype.name == "complex128"
        assert complex128_dtype.bitwidth == 128
        assert complex128_dtype.is_signed
        assert complex128_dtype.is_floating
        assert complex128_dtype.is_complex

    def test_dtype_constants(self):
        """Test predefined dtype constants."""
        # Test boolean
        assert BOOL.name == "bool"
        assert BOOL.bitwidth == 8  # NumPy bool is 8 bits

        # Test integers
        assert INT8.name == "int8"
        assert INT8.bitwidth == 8
        assert INT64.name == "int64"
        assert INT64.bitwidth == 64

        # Test unsigned integers
        assert UINT32.name == "uint32"
        assert UINT32.bitwidth == 32
        assert not UINT32.is_signed

        # Test floats
        assert FLOAT32.name == "float32"
        assert FLOAT32.bitwidth == 32
        assert FLOAT32.is_floating

        # Test complex
        assert COMPLEX64.name == "complex64"
        assert COMPLEX64.bitwidth == 64
        assert COMPLEX64.is_complex

    def test_from_numpy(self):
        """Test creating DType from NumPy dtypes."""
        # Test various numpy dtypes
        test_cases = [
            (np.bool_, BOOL),
            (np.int8, INT8),
            (np.int16, INT16),
            (np.int32, INT32),
            (np.int64, INT64),
            (np.uint8, UINT8),
            (np.uint16, UINT16),
            (np.uint32, UINT32),
            (np.uint64, UINT64),
            (np.float16, FLOAT16),
            (np.float32, FLOAT32),
            (np.float64, FLOAT64),
            (np.complex64, COMPLEX64),
            (np.complex128, COMPLEX128),
        ]

        for np_dtype, expected_dtype in test_cases:
            result = DType.from_numpy(np_dtype)
            assert result == expected_dtype

        # Test with numpy dtype objects
        result = DType.from_numpy(np.dtype("float32"))
        assert result == FLOAT32

    def test_from_python_type(self):
        """Test creating DType from Python types."""
        assert DType.from_python_type(bool) == BOOL
        assert DType.from_python_type(int) == INT64
        assert DType.from_python_type(float) == FLOAT64
        assert DType.from_python_type(complex) == COMPLEX128

        # Test with unsupported type
        with pytest.raises(ValueError):
            DType.from_python_type(str)

    def test_from_any(self):
        """Test creating DType from various inputs."""
        # From numpy types
        assert DType.from_any(np.float32) == FLOAT32
        assert DType.from_any(np.dtype("int64")) == INT64

        # From python types
        assert DType.from_any(float) == FLOAT64
        assert DType.from_any(int) == INT64

        # From DType (should return same)
        assert DType.from_any(FLOAT32) == FLOAT32

        # From string
        assert DType.from_any("float64") == FLOAT64

        # Test with unsupported type
        with pytest.raises(ValueError):
            DType.from_any("unsupported")

    def test_to_numpy(self):
        """Test converting DType to NumPy dtype."""
        test_cases = [
            (BOOL, np.bool_),
            (INT32, np.int32),
            (UINT64, np.uint64),
            (FLOAT32, np.float32),
            (COMPLEX128, np.complex128),
        ]

        for dtype, expected_np_type in test_cases:
            result = dtype.to_numpy()
            assert result == np.dtype(expected_np_type)

    def test_to_python_type(self):
        """Test converting DType to Python type."""
        assert BOOL.to_python_type() is bool
        assert INT64.to_python_type() is int
        assert FLOAT64.to_python_type() is float
        assert COMPLEX128.to_python_type() is complex

        # Unsigned integers should map to int
        assert UINT32.to_python_type() is int

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_to_jax(self):
        """Test converting DType to JAX dtype."""
        test_cases = [
            (BOOL, jnp.bool_),
            (INT32, jnp.int32),
            (FLOAT32, jnp.float32),
            (COMPLEX64, jnp.complex64),
        ]

        for dtype, expected_jax_type in test_cases:
            result = dtype.to_jax()
            assert result == expected_jax_type

    def test_equality(self):
        """Test DType equality comparison."""
        # Same dtypes should be equal
        assert FLOAT32 == FLOAT32
        assert INT64 == INT64

        # Different dtypes should not be equal
        assert FLOAT32 != FLOAT64
        assert INT32 != UINT32

        # Equal dtypes created separately should be equal
        dtype1 = DType("float32", 32, True, True, False)
        dtype2 = DType("float32", 32, True, True, False)
        assert dtype1 == dtype2

    def test_repr(self):
        """Test DType string representation."""
        assert repr(FLOAT32) == "DType('float32')"
        assert repr(INT64) == "DType('int64')"
        assert repr(BOOL) == "DType('bool')"

    def test_hash(self):
        """Test DType hashing."""
        # Same dtypes should have same hash
        assert hash(FLOAT32) == hash(FLOAT32)

        # Different dtypes should have different hashes (usually)
        assert hash(FLOAT32) != hash(FLOAT64)

        # Can be used in sets and dicts
        dtype_set = {FLOAT32, INT64, BOOL}
        assert len(dtype_set) == 3

    def test_dtype_conversion_chain(self):
        """Test conversion chain between different representations."""
        # Start with DType
        dtype = FLOAT64

        # Convert to numpy and back
        np_dtype = dtype.to_numpy()
        back_to_dtype = DType.from_numpy(np_dtype)
        assert dtype == back_to_dtype

        # Convert to python type and back
        py_type = dtype.to_python_type()
        from_py_dtype = DType.from_python_type(py_type)
        assert dtype == from_py_dtype

    def test_mpir_dtype_to_proto_compatibility(self):
        """Test that mpir.dtype_to_proto works with new DType objects."""
        # Create TensorType with DType
        tensor_info = TensorType(FLOAT32, (2, 2))

        # dtype_to_proto should work with DType objects
        proto_result = dtype_to_proto(tensor_info.dtype)

        # Should be the same as using numpy dtype directly
        proto_result_np = dtype_to_proto(np.float32)
        assert proto_result == proto_result_np


class TestTensorInfo:
    """Test cases for the TensorType class."""

    def test_tensor_info_creation(self):
        """Test TensorType compatibility with custom dtypes."""
        # Test with NumPy array
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        tensor_info = TensorType.from_obj(arr)

        # dtype should now return DType, not numpy dtype
        assert isinstance(tensor_info.dtype, DType)
        assert tensor_info.dtype == FLOAT32
        assert tensor_info.shape == (2, 3)

        # Use to_numpy() for compatibility
        assert tensor_info.to_numpy() == np.dtype("float32")

        # Test with scalar
        scalar_info = TensorType.from_obj(42)
        assert isinstance(scalar_info.dtype, DType)
        assert scalar_info.dtype == INT64
        assert scalar_info.shape == ()
        assert scalar_info.to_numpy() == np.dtype("int64")

        # Test with float scalar
        float_scalar_info = TensorType.from_obj(3.14)
        assert isinstance(float_scalar_info.dtype, DType)
        assert float_scalar_info.dtype == FLOAT64
        assert float_scalar_info.shape == ()
        assert float_scalar_info.to_numpy() == np.dtype("float64")

    def test_tensor_info_constructor_with_dtype(self):
        """Test TensorType constructor with various dtype inputs."""
        # Test constructing TensorType with custom DType
        tensor_info = TensorType(FLOAT32, (2, 3))
        assert isinstance(tensor_info.dtype, DType)
        assert tensor_info.dtype == FLOAT32
        assert tensor_info.to_numpy() == np.dtype("float32")

        # Test constructing with numpy dtype (should be converted)
        tensor_info_np = TensorType(np.dtype("int32"), (5,))
        assert isinstance(tensor_info_np.dtype, DType)
        assert tensor_info_np.dtype == INT32
        assert tensor_info_np.to_numpy() == np.dtype("int32")

    def test_tensor_info_unified_api(self):
        """Test the unified dtype API for TensorType."""
        # Test TensorType
        tensor_info = TensorType(np.float32, (3, 4))

        # dtype should return DType
        assert isinstance(tensor_info.dtype, DType)
        assert tensor_info.dtype.name == "float32"

        # to_numpy() for explicit conversion
        assert tensor_info.to_numpy() == np.dtype("float32")

    def test_dtype_consistency(self):
        """Test that dtype and to_numpy() are consistent."""
        # Create from DType
        tensor_info = TensorType(FLOAT32, (2, 2))

        # The numpy dtype should match what dtype can produce
        assert tensor_info.to_numpy() == tensor_info.dtype.to_numpy()

        # Create from numpy dtype
        tensor_info2 = TensorType(np.int16, (3, 3))
        assert tensor_info2.dtype == INT16

    def test_dtype_enhanced_functionality(self):
        """Test that dtype provides enhanced functionality."""
        tensor_info = TensorType(np.float32, (2, 2))
        dtype = tensor_info.dtype

        # Enhanced properties
        assert dtype.to_numpy() == np.float32
        assert dtype.to_python_type() is float
        if JAX_AVAILABLE:
            assert dtype.to_jax() == jnp.float32

        # Type properties
        assert dtype.is_floating
        assert not dtype.is_complex
        assert dtype.bitwidth == 32

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with unusual numpy dtypes
        tensor_info = TensorType(np.dtype("<f4"), (2, 2))  # Little-endian float32
        assert tensor_info.dtype == FLOAT32

        # Test hash consistency
        dtype_set = {tensor_info.dtype, FLOAT32}
        assert len(dtype_set) == 1  # Should be the same


class TestMPType:
    """Test cases for the MPType class."""

    def test_mp_type_creation(self):
        """Test MPType compatibility with custom dtypes."""
        # Test with tensor
        arr = np.array([1, 2, 3, 4], dtype=np.int64)
        mp_type = MPType.from_tensor(arr, pmask=Mask.from_int(0b111))

        # dtype should now return DType, not numpy dtype
        assert isinstance(mp_type.dtype, DType)
        assert mp_type.dtype == INT64
        assert mp_type.shape == (4,)
        assert mp_type.pmask == Mask.from_int(0b111)

        # Use to_numpy() for compatibility
        assert mp_type.to_numpy() == np.dtype("int64")

        # Test with scalar
        scalar_mp_type = MPType.from_tensor(42.0, pmask=Mask.from_int(0b101))
        assert isinstance(scalar_mp_type.dtype, DType)
        assert scalar_mp_type.dtype == FLOAT64
        assert scalar_mp_type.shape == ()
        assert scalar_mp_type.pmask == Mask.from_int(0b101)
        assert scalar_mp_type.to_numpy() == np.dtype("float64")

        # Test with custom attributes
        attr_mp_type = MPType.from_tensor(
            arr, pmask=Mask.from_int(0b111), encrypted=True
        )
        assert attr_mp_type.attrs == {"encrypted": True}

    def test_mp_type_constructor_with_dtype(self):
        """Test MPType constructor with various dtype inputs."""
        # Test constructing MPType with custom DType
        mp_type = MPType.tensor(FLOAT32, (2, 3), 0b111)
        assert isinstance(mp_type.dtype, DType)
        assert mp_type.dtype == FLOAT32
        assert mp_type.to_numpy() == np.dtype("float32")

        # Test constructing with numpy dtype (should be converted)
        mp_type_np = MPType.tensor(INT32, (5,), 0b101)
        assert isinstance(mp_type_np.dtype, DType)
        assert mp_type_np.dtype == INT32
        assert mp_type_np.to_numpy() == np.dtype("int32")

    def test_mp_type_unified_api(self):
        """Test the unified dtype API for MPType."""
        # Test MPType
        mp_type = MPType.tensor(INT64, (2, 2), 0b111)

        # dtype should return DType
        assert isinstance(mp_type.dtype, DType)
        assert mp_type.dtype.name == "int64"

        # to_numpy() for explicit conversion
        assert mp_type.to_numpy() == np.dtype("int64")

    def test_array_creation_workflow(self):
        """Test typical array creation and manipulation workflow."""
        # Create tensor info from numpy array
        arr = np.random.random((3, 4)).astype(np.float32)
        tensor_info = TensorType.from_obj(arr)

        # Verify dtype information
        assert tensor_info.dtype == FLOAT32
        assert tensor_info.dtype.is_floating
        assert tensor_info.dtype.bitwidth == 32

        # Create compatible numpy array
        new_arr = np.zeros((2, 2), dtype=tensor_info.to_numpy())
        assert new_arr.dtype == np.float32

        # Create MPType with same dtype
        mp_type = MPType.tensor(tensor_info.dtype, (5, 5), 0b111)
        assert mp_type.dtype == tensor_info.dtype
