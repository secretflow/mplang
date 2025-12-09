# Copyright 2025 Ant Group Co., Ltd.
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

import pytest

from mplang.v1.core.dtypes import FLOAT32, INT32
from mplang.v1.core.expr import (
    deduce_mask,
    ensure_scalar,
    ensure_tensorlist_equal,
    type_equal,
)
from mplang.v1.core.mask import Mask


class MockTensor:
    """Mock tensor class for testing."""

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    def __str__(self):
        return f"Tensor(dtype={self.dtype}, shape={self.shape})"


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    return {
        "float_2x3": MockTensor(FLOAT32, (2, 3)),
        "float_2x3_dup": MockTensor(FLOAT32, (2, 3)),
        "int_2x3": MockTensor(INT32, (2, 3)),
        "float_3x2": MockTensor(FLOAT32, (3, 2)),
        "scalar_float": MockTensor(FLOAT32, ()),
        "vector_float": MockTensor(FLOAT32, (3,)),
    }


class TestTypeEqual:
    """Test type_equal utility function."""

    def test_single_tensor_always_equal(self, sample_tensors):
        """Single tensor should always be equal to itself."""
        assert type_equal(sample_tensors["float_2x3"])

    @pytest.mark.parametrize(
        "t1_key,t2_key,expected",
        [
            ("float_2x3", "float_2x3_dup", True),  # Same types
            ("float_2x3", "int_2x3", False),  # Different dtypes
            ("float_2x3", "float_3x2", False),  # Different shapes
        ],
    )
    def test_two_tensors(self, sample_tensors, t1_key, t2_key, expected):
        """Test type equality between two tensors."""
        t1, t2 = sample_tensors[t1_key], sample_tensors[t2_key]
        assert type_equal(t1, t2) == expected

    def test_multiple_tensors(self, sample_tensors):
        """Test type equality with multiple tensors."""
        t1 = sample_tensors["float_2x3"]
        t2 = sample_tensors["float_2x3_dup"]
        t3 = sample_tensors["int_2x3"]

        assert type_equal(t1, t2, t2)  # All same types
        assert not type_equal(t1, t2, t3)  # Mixed types


class TestEnsureScalar:
    """Test ensure_scalar utility function."""

    def test_scalar_tensor_passes(self, sample_tensors):
        """Scalar tensor should not raise exception."""
        ensure_scalar(sample_tensors["scalar_float"])

    def test_non_scalar_tensor_raises(self, sample_tensors):
        """Non-scalar tensor should raise TypeError."""
        with pytest.raises(TypeError):
            ensure_scalar(sample_tensors["vector_float"])


class TestEnsureTensorlistEqual:
    """Test ensure_tensorlist_equal utility function."""

    def test_insufficient_arguments(self, sample_tensors):
        """Should raise ValueError with less than 2 argument lists."""
        with pytest.raises(ValueError):
            ensure_tensorlist_equal([sample_tensors["float_2x3"]])

    def test_length_mismatch(self, sample_tensors):
        """Should raise ValueError when tensor lists have different lengths."""
        t1 = sample_tensors["float_2x3"]
        t2 = sample_tensors["float_2x3_dup"]

        with pytest.raises(ValueError):
            ensure_tensorlist_equal([t1, t2], [t1])

    def test_type_mismatch(self, sample_tensors):
        """Should raise TypeError when corresponding tensors have different types."""
        t1 = sample_tensors["float_2x3"]
        t2 = sample_tensors["float_2x3_dup"]
        t3 = sample_tensors["int_2x3"]

        with pytest.raises(TypeError):
            ensure_tensorlist_equal([t1, t2], [t1, t3])

    def test_valid_case(self, sample_tensors):
        """Should not raise when all corresponding tensors have same types."""
        t1 = sample_tensors["float_2x3"]
        t2 = sample_tensors["float_2x3_dup"]

        ensure_tensorlist_equal([t1, t2], [t1, t2])  # Should not raise


class TestDeduceMask:
    """Test deduce_mask utility function."""

    def test_empty_masks(self):
        """Empty arguments should return None."""
        assert deduce_mask() is None

    @pytest.mark.parametrize(
        "masks,expected",
        [
            ([None], None),
            ([Mask(3), None], None),
            ([Mask(3)], Mask(3)),
            ([Mask(3), Mask(7)], Mask(3)),
            ([Mask(5), Mask(3)], Mask(1)),
        ],
    )
    def test_mask_combinations(self, masks, expected):
        """Test various mask combinations."""
        result = deduce_mask(*masks)
        if expected is None:
            assert result is None
        else:
            assert result == expected
