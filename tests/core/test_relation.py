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
from mplang.core.relation import RelationLike, RelationType


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


class TestRelationLike:
    """Test RelationLike protocol functionality."""

    def test_mock_dataframe_satisfies_protocol(self):
        """Test that a mock DataFrame object satisfies RelationLike protocol."""

        class MockDataFrame:
            """Mock DataFrame-like object with dtypes and columns."""

            def __init__(self):
                self.dtypes = {"id": "int64", "name": "object", "value": "float32"}
                self.columns = ["id", "name", "value"]

        mock_df = MockDataFrame()

        # Test protocol satisfaction
        assert isinstance(mock_df, RelationLike)

        # Test attribute access
        assert hasattr(mock_df, "dtypes")
        assert hasattr(mock_df, "columns")
        assert mock_df.dtypes == {"id": "int64", "name": "object", "value": "float32"}
        assert mock_df.columns == ["id", "name", "value"]

    def test_pandas_dataframe_protocol_compatibility(self):
        """Test pandas DataFrame compatibility if pandas is available."""

        try:
            import pandas as pd

            # Create a simple DataFrame
            df = pd.DataFrame({
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [1.0, 2.5, 3.7],
            })

            # Test protocol satisfaction
            assert isinstance(df, RelationLike)

            # Test attribute access
            assert hasattr(df, "dtypes")
            assert hasattr(df, "columns")

            # Verify attributes work as expected
            assert len(df.dtypes) == 3
            assert len(df.columns) == 3
            assert "id" in df.columns
            assert "name" in df.columns
            assert "value" in df.columns

        except ImportError:
            pytest.skip("pandas not available")

    def test_protocol_with_custom_relation_object(self):
        """Test protocol with a custom relation-like object."""

        class CustomTable:
            """Custom table implementation."""

            @property
            def dtypes(self):
                return {"col1": "int32", "col2": "string"}

            @property
            def columns(self):
                return ["col1", "col2"]

        table = CustomTable()

        # Test protocol satisfaction
        assert isinstance(table, RelationLike)

        # Test property access
        assert table.dtypes == {"col1": "int32", "col2": "string"}
        assert table.columns == ["col1", "col2"]

    def test_protocol_rejection_of_non_relation_objects(self):
        """Test that non-relation objects are correctly rejected."""

        # Test objects without required attributes
        class NoAttributes:
            """Object with no attributes."""

        class OnlyDtypes:
            """Object with only dtypes attribute."""

            def __init__(self):
                self.dtypes = {}

        class OnlyColumns:
            """Object with only columns attribute."""

            def __init__(self):
                self.columns = []

        # Test regular objects
        assert not isinstance(42, RelationLike)
        assert not isinstance("string", RelationLike)
        assert not isinstance([], RelationLike)
        assert not isinstance({}, RelationLike)

        # Test objects with partial implementation
        assert not isinstance(NoAttributes(), RelationLike)
        assert not isinstance(OnlyDtypes(), RelationLike)
        assert not isinstance(OnlyColumns(), RelationLike)

    def test_protocol_with_minimal_implementation(self):
        """Test protocol with minimal required implementation."""

        class MinimalRelation:
            """Minimal implementation with just required attributes."""

            dtypes = None
            columns = None

        minimal = MinimalRelation()

        # Even with None values, protocol should be satisfied
        assert isinstance(minimal, RelationLike)
        assert hasattr(minimal, "dtypes")
        assert hasattr(minimal, "columns")

    def test_protocol_runtime_checking(self):
        """Test runtime protocol checking behavior."""

        class DynamicTable:
            """Table that gains attributes dynamically."""

        table = DynamicTable()

        # Initially should not satisfy protocol
        assert not isinstance(table, RelationLike)

        # Add required attributes dynamically
        table.dtypes = {"col": "int"}
        table.columns = ["col"]

        # Now should satisfy protocol
        assert isinstance(table, RelationLike)

    def test_protocol_with_property_implementation(self):
        """Test protocol with property-based implementation."""

        class PropertyTable:
            """Table with property-based attributes."""

            def __init__(self):
                self._dtypes = {"a": "float", "b": "int"}
                self._columns = ["a", "b"]

            @property
            def dtypes(self):
                return self._dtypes

            @property
            def columns(self):
                return self._columns

        table = PropertyTable()

        # Test protocol satisfaction
        assert isinstance(table, RelationLike)

        # Test that properties work correctly
        assert table.dtypes == {"a": "float", "b": "int"}
        assert table.columns == ["a", "b"]

    def test_protocol_with_callable_attributes(self):
        """Test protocol behavior when attributes are callable."""

        class CallableAttributeTable:
            """Table where dtypes/columns are callable."""

            def dtypes(self):
                return {"x": "double", "y": "string"}

            def columns(self):
                return ["x", "y"]

        table = CallableAttributeTable()

        # This SHOULD satisfy the protocol since Python protocol checking
        # only looks for attribute existence, not whether they're properties vs methods
        assert isinstance(table, RelationLike)

        # However, accessing them as attributes will return method objects
        assert callable(table.dtypes)
        assert callable(table.columns)
