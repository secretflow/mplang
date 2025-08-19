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

from mplang.core.dtype import DATE, FLOAT32, FLOAT64, INT32, INT64, JSON, STRING
from mplang.core.mask import Mask
from mplang.core.mptype import MPType
from mplang.core.relation import RelationSchema


class TestMPType:
    """Test MPType functionality with tensor and relation support."""

    def test_tensor_creation(self):
        """Test tensor MPType creation."""
        tensor_type = MPType.tensor(FLOAT32, (3, 4))

        assert tensor_type.is_tensor
        assert tensor_type.dtype == FLOAT32
        assert tensor_type.shape == (3, 4)

    def test_relation_creation(self):
        """Test relation MPType creation."""
        schema = RelationSchema.from_dict({
            "user_id": INT64,
            "score": FLOAT64,
            "rank": INT32,
        })

        relation_type = MPType.relation(schema)

        assert relation_type.is_relation
        assert relation_type.schema == schema

    def test_relation_creation_from_dict(self):
        """Test relation MPType creation from dict."""
        schema_dict = {"id": INT64, "value": FLOAT32}
        relation_type = MPType.relation(schema_dict)

        assert relation_type.is_relation
        assert relation_type.schema.column_names() == ("id", "value")

    def test_tensor_attribute_access(self):
        """Test tensor attribute access."""
        tensor_type = MPType.tensor(FLOAT32, (3, 4))

        # Should work
        assert tensor_type.dtype == FLOAT32
        assert tensor_type.shape == (3, 4)

        # Should fail
        with pytest.raises(
            AttributeError, match="schema is only available for relation types"
        ):
            _ = tensor_type.schema

    def test_relation_attribute_access(self):
        """Test relation attribute access."""
        schema = RelationSchema.from_dict({"id": INT64, "value": FLOAT32})
        relation_type = MPType.relation(schema)

        # Should work
        assert relation_type.schema == schema

        # Should fail
        with pytest.raises(
            AttributeError, match="dtype is only available for tensor types"
        ):
            _ = relation_type.dtype

        with pytest.raises(
            AttributeError, match="shape is only available for tensor types"
        ):
            _ = relation_type.shape

    def test_relation_only_dtype_validation(self):
        """Test that relation-only dtypes cannot be used in tensors."""
        # STRING type should fail in tensor
        with pytest.raises(
            ValueError, match="Data type 'string' is only supported in relations"
        ):
            MPType.tensor(STRING, (10,))

        # DATE type should fail in tensor
        with pytest.raises(
            ValueError, match="Data type 'date' is only supported in relations"
        ):
            MPType.tensor(DATE, (5, 5))

        # JSON type should fail in tensor
        with pytest.raises(
            ValueError, match="Data type 'json' is only supported in relations"
        ):
            MPType.tensor(JSON, (3,))

        # But should work in relations
        schema = RelationSchema.from_dict({
            "name": STRING,
            "birth_date": DATE,
            "metadata": JSON,
        })
        relation_type = MPType.relation(schema)
        assert relation_type.is_relation

    def test_mptype_with_pmask_and_attrs(self):
        """Test MPType with pmask and attributes."""
        # Test tensor with pmask and attrs
        tensor_type = MPType.tensor(
            INT32, (10,), pmask=Mask.from_int(0b1101), device="GPU", precision="high"
        )

        assert tensor_type.pmask == Mask.from_int(0b1101)
        assert tensor_type.attrs["device"] == "GPU"
        assert tensor_type.attrs["precision"] == "high"

        # Test relation with pmask and attrs
        schema = RelationSchema.from_dict({"col1": FLOAT32, "col2": INT64})
        relation_type = MPType.relation(
            schema, pmask=Mask.from_int(0b11), storage="distributed", format="parquet"
        )

        assert relation_type.pmask == Mask.from_int(0b11)
        assert relation_type.attrs["storage"] == "distributed"
        assert relation_type.attrs["format"] == "parquet"

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

        # Test relation equality
        schema1 = RelationSchema.from_dict({"a": INT32, "b": FLOAT32})
        schema2 = RelationSchema.from_dict({"a": INT32, "b": FLOAT32})
        schema3 = RelationSchema.from_dict({"a": INT64, "b": FLOAT32})

        r1 = MPType.relation(schema1)
        r2 = MPType.relation(schema2)
        r3 = MPType.relation(schema3)

        assert r1 == r2
        assert r1 != r3
        assert hash(r1) == hash(r2)
        assert hash(r1) != hash(r3)

        # Test tensor vs relation inequality
        assert t1 != r1

    def test_string_representation(self):
        """Test string representation."""
        # Test tensor representation
        tensor_type = MPType.tensor(FLOAT32, (3, 4))
        tensor_str = str(tensor_type)
        assert "f32[3, 4]" == tensor_str

        # Test relation representation
        schema = RelationSchema.from_dict({"id": INT64, "name": STRING})
        relation_type = MPType.relation(schema)
        relation_str = str(relation_type)
        assert "Rel(id:i64, name:str)" == relation_str

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

        # Should fail for relations
        schema = RelationSchema.from_dict({"id": INT64})
        relation_type = MPType.relation(schema)
        with pytest.raises(
            AttributeError, match="to_numpy is only available for tensor types"
        ):
            relation_type.to_numpy()

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
