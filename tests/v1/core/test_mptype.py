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

from mplang.v1.core.dtypes import DATE, FLOAT32, FLOAT64, INT32, INT64, JSON, STRING
from mplang.v1.core.mask import Mask
from mplang.v1.core.mptype import MPType
from mplang.v1.core.table import TableType


class TestMPType:
    """Test MPType functionality with tensor and table support."""

    def test_tensor_creation(self):
        """Test tensor MPType creation."""
        tensor_type = MPType.tensor(FLOAT32, (3, 4))

        assert tensor_type.is_tensor
        assert tensor_type.dtype == FLOAT32
        assert tensor_type.shape == (3, 4)

    def test_table_creation(self):
        """Test table MPType creation."""
        schema = TableType.from_dict({
            "user_id": INT64,
            "score": FLOAT64,
            "rank": INT32,
        })

        table_type = MPType.table(schema)

        assert table_type.is_table
        assert table_type.schema == schema

    def test_table_creation_from_dict(self):
        """Test table MPType creation from dict."""
        schema_dict = {"id": INT64, "value": FLOAT32}
        table_type = MPType.table(schema_dict)

        assert table_type.is_table
        assert table_type.schema.column_names() == ("id", "value")

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

    def test_table_attribute_access(self):
        """Test table attribute access."""
        schema = TableType.from_dict({"id": INT64, "value": FLOAT32})
        table_type = MPType.table(schema)

        # Should work
        assert table_type.schema == schema

        # Should fail
        with pytest.raises(
            AttributeError, match="dtype is only available for tensor types"
        ):
            _ = table_type.dtype

        with pytest.raises(
            AttributeError, match="shape is only available for tensor types"
        ):
            _ = table_type.shape

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

        # But should work in tables
        schema = TableType.from_dict({
            "name": STRING,
            "birth_date": DATE,
            "metadata": JSON,
        })
        table_type = MPType.table(schema)
        assert table_type.is_table

    def test_mptype_with_pmask_and_attrs(self):
        """Test MPType with pmask and attributes."""
        # Test tensor with pmask and attrs
        tensor_type = MPType.tensor(
            INT32, (10,), pmask=Mask.from_int(0b1101), device="GPU", precision="high"
        )

        assert tensor_type.pmask == Mask.from_int(0b1101)
        assert tensor_type.attrs["device"] == "GPU"
        assert tensor_type.attrs["precision"] == "high"

        # Test table with pmask and attrs
        schema = TableType.from_dict({"col1": FLOAT32, "col2": INT64})
        table_type = MPType.table(
            schema, pmask=Mask.from_int(0b11), storage="distributed", format="parquet"
        )

        assert table_type.pmask == Mask.from_int(0b11)
        assert table_type.attrs["storage"] == "distributed"
        assert table_type.attrs["format"] == "parquet"

    def test_equality_and_hashing(self):
        """Test equality and hashing."""
        # Test tensor equality
        t1 = MPType.tensor(FLOAT32, (2, 3))
        t2 = MPType.tensor(FLOAT32, (2, 3))
        t3 = MPType.tensor(FLOAT64, (2, 3))

        assert t1 == t2
        assert t1 != t3
        assert hash(t1) == hash(t2)
        assert hash(t1) != hash(t3)

        # Test table equality
        schema1 = TableType.from_dict({"a": INT32, "b": FLOAT32})
        schema2 = TableType.from_dict({"a": INT32, "b": FLOAT32})
        schema3 = TableType.from_dict({"a": INT64, "b": FLOAT32})

        r1 = MPType.table(schema1)
        r2 = MPType.table(schema2)
        r3 = MPType.table(schema3)

        assert r1 == r2
        assert r1 != r3
        assert hash(r1) == hash(r2)
        assert hash(r1) != hash(r3)

        # Test tensor vs table inequality
        assert t1 != r1

    def test_string_representation(self):
        """Test string representation."""
        # Test tensor representation
        tensor_type = MPType.tensor(FLOAT32, (3, 4))
        tensor_str = str(tensor_type)
        assert "f32[3, 4]" == tensor_str

        # Test table representation
        schema = TableType.from_dict({"id": INT64, "name": STRING})
        table_type = MPType.table(schema)
        table_str = str(table_type)
        assert "Tbl(id:i64, name:str)" == table_str

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

        # Should fail for tables
        schema = TableType.from_dict({"id": INT64})
        table_type = MPType.table(schema)
        with pytest.raises(
            AttributeError, match="to_numpy is only available for tensor types"
        ):
            table_type.to_numpy()

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
