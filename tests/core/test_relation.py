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

import pytest

from mplang.core.dtype import DATE, FLOAT32, INT32, INT64, JSON, STRING
from mplang.core.relation import RelationType


class TestRelationType:
    """Test RelationType functionality."""

    def test_creation_from_dict(self):
        """Test creating RelationType from dictionary."""
        schema = RelationType.from_dict({"id": INT64, "value": FLOAT32, "age": INT32})

        assert len(schema) == 3
        assert schema.num_columns() == 3
        assert schema.column_names() == ("id", "value", "age")
        assert schema.column_types() == (INT64, FLOAT32, INT32)

    def test_creation_from_pairs(self):
        """Test creating RelationType from pairs."""
        pairs = [("id", INT64), ("name", STRING)]
        schema = RelationType.from_pairs(pairs)

        assert len(schema) == 2
        assert schema.column_names() == ("id", "name")

    def test_column_access(self):
        """Test column access methods."""
        schema = RelationType.from_dict({"id": INT64, "value": FLOAT32, "age": INT32})

        # Test has_column
        assert schema.has_column("id") is True
        assert schema.has_column("email") is False

        # Test get_column_type
        assert schema.get_column_type("id") == INT64
        assert schema.get_column_type("value") == FLOAT32

        with pytest.raises(KeyError):
            schema.get_column_type("nonexistent")

    def test_indexing(self):
        """Test indexing functionality."""
        schema = RelationType.from_dict({"id": INT64, "value": FLOAT32})

        # Test integer indexing
        assert schema[0] == ("id", INT64)
        assert schema[1] == ("value", FLOAT32)

        # Test string indexing
        assert schema["id"] == INT64
        assert schema["value"] == FLOAT32

        with pytest.raises(KeyError):
            _ = schema["nonexistent"]

    def test_iteration(self):
        """Test iteration over schema."""
        schema = RelationType.from_dict({"id": INT64, "value": FLOAT32})

        columns = list(schema)
        assert len(columns) == 2
        assert columns[0] == ("id", INT64)
        assert columns[1] == ("value", FLOAT32)

    def test_to_dict(self):
        """Test conversion back to dictionary."""
        original_dict = {"id": INT64, "value": FLOAT32}
        schema = RelationType.from_dict(original_dict)
        result_dict = schema.to_dict()

        assert result_dict == original_dict

    def test_string_representation(self):
        """Test string representation."""
        schema = RelationType.from_dict({"id": INT64, "value": FLOAT32})

        repr_str = repr(schema)
        assert "RelationType<" in repr_str
        assert "id:i64" in repr_str
        assert "value:f32" in repr_str

    def test_relation_only_types(self):
        """Test schemas with relation-only data types."""
        schema = RelationType.from_dict({
            "name": STRING,
            "birth_date": DATE,
            "metadata": JSON,
        })

        assert schema.get_column_type("name") == STRING
        assert schema.get_column_type("birth_date") == DATE
        assert schema.get_column_type("metadata") == JSON

    def test_validation(self):
        """Test schema validation."""
        # Test empty schema
        with pytest.raises(ValueError, match="RelationType cannot be empty"):
            RelationType(())

        # Test duplicate column names
        with pytest.raises(ValueError, match="Column names must be unique"):
            RelationType((("id", INT64), ("id", FLOAT32)))

        # Test invalid column names
        with pytest.raises(ValueError, match="Column names must be non-empty strings"):
            RelationType((("", INT64),))

        # Invalid type should be caught by validation
        with pytest.raises(ValueError, match="Column names must be non-empty strings"):
            # This will be caught by __post_init__ validation
            schema = RelationType.__new__(RelationType)
            object.__setattr__(schema, "columns", ((None, INT64),))
            schema.__post_init__()

    def test_equality(self):
        """Test schema equality."""
        schema1 = RelationType.from_dict({"id": INT64, "value": FLOAT32})
        schema2 = RelationType.from_dict({"id": INT64, "value": FLOAT32})
        schema3 = RelationType.from_dict({"id": INT32, "value": FLOAT32})

        assert schema1 == schema2
        assert schema1 != schema3
        assert hash(schema1) == hash(schema2)
        assert hash(schema1) != hash(schema3)
