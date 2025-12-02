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

from mplang.v1.core.dtypes import DATE, FLOAT32, FLOAT64, INT32, JSON, STRING
from mplang.v1.core.mask import Mask
from mplang.v1.core.mptype import MPType


class TestTensor:
    """Test tensor-related functionality."""

    def test_tensor_creation(self):
        """Test tensor creation."""
        tensor_type = MPType.tensor(FLOAT32, (3, 4))

        assert tensor_type.is_tensor
        assert tensor_type.dtype == FLOAT32
        assert tensor_type.shape == (3, 4)

    def test_tensor_attribute_access(self):
        """Test tensor attribute access."""
        tensor_type = MPType.tensor(FLOAT32, (3, 4))

        # Should work
        assert tensor_type.dtype == FLOAT32
        assert tensor_type.shape == (3, 4)

        # Should fail
        with pytest.raises(
            AttributeError, match="schema is only available for table types"
        ):
            _ = tensor_type.schema

    def test_table_only_dtype_validation(self):
        """Test that table-only dtypes cannot be used in tensors."""
        # STRING type should fail in tensor
        with pytest.raises(
            ValueError, match="Data type 'string' is only supported in tables"
        ):
            MPType.tensor(STRING, (10,))

        # DATE type should fail in tensor
        with pytest.raises(
            ValueError, match="Data type 'date' is only supported in tables"
        ):
            MPType.tensor(DATE, (5, 5))

        # JSON type should fail in tensor
        with pytest.raises(
            ValueError, match="Data type 'json' is only supported in tables"
        ):
            MPType.tensor(JSON, (3,))

    def test_tensor_with_pmask_and_attrs(self):
        """Test tensor with pmask and attributes."""
        tensor_type = MPType.tensor(
            INT32, (10,), pmask=Mask.from_int(0b1101), device="GPU", precision="high"
        )

        assert tensor_type.pmask == Mask.from_int(0b1101)
        assert tensor_type.attrs["device"] == "GPU"
        assert tensor_type.attrs["precision"] == "high"

    def test_tensor_equality_and_hashing(self):
        """Test tensor equality and hashing."""
        t1 = MPType.tensor(FLOAT32, (2, 3))
        t2 = MPType.tensor(FLOAT32, (2, 3))
        t3 = MPType.tensor(FLOAT64, (2, 3))

        assert t1 == t2
        assert t1 != t3
        assert hash(t1) == hash(t2)
        assert hash(t1) != hash(t3)

    def test_tensor_string_representation(self):
        """Test tensor string representation."""
        # Test basic tensor representation
        tensor_type = MPType.tensor(FLOAT32, (3, 4))
        tensor_str = str(tensor_type)
        assert "f32[3, 4]" == tensor_str

        # Test with pmask and attributes
        tensor_with_attrs = MPType.tensor(
            INT32, (10,), pmask=Mask.from_int(0b1101), device="GPU"
        )
        tensor_attrs_str = str(tensor_with_attrs)
        assert "i32[10]<D>" in tensor_attrs_str
        assert 'device="GPU"' in tensor_attrs_str

    def test_to_numpy_tensor_only(self):
        """Test to_numpy method only works for tensors."""
        tensor_type = MPType.tensor(FLOAT32, (3, 4))
        numpy_dtype = tensor_type.to_numpy()
        assert str(numpy_dtype) == "float32"

    def test_from_tensor_factory_method(self):
        """Test from_tensor factory method."""
        # Test with scalar
        scalar_type = MPType.from_tensor(42)
        assert scalar_type.is_tensor
        assert scalar_type.shape == ()

        # Test with numpy array
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        array_type = MPType.from_tensor(arr)
        assert array_type.is_tensor
        assert array_type.dtype == FLOAT32
        assert array_type.shape == (2, 2)

        # Test with list converted to numpy array
        list_arr = np.array([1, 2, 3])
        list_type = MPType.from_tensor(list_arr)
        assert list_type.is_tensor
        assert list_type.shape == (3,)
