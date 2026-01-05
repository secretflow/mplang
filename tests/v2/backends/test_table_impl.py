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

"""Tests for Table Runtime Implementation."""

import os

import pyarrow as pa
import pytest

import mplang.v2.edsl.typing as elt
from mplang.v2.backends.table_impl import ParquetReader, TableValue
from mplang.v2.dialects import table


def _get_table(val) -> pa.Table:
    """Extract pa.Table from various wrapper types."""
    if hasattr(val, "runtime_obj"):
        val = val.runtime_obj
    if isinstance(val, TableValue):
        return val.data
    return val


def test_table_ops_e2e():
    """Test basic table operations (constant, run_sql, conversions) end-to-end."""

    path = "test_table_ops_e2e.parquet"

    def workload():
        # Create constant table
        data = {
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
        }
        t1 = table.constant(data)

        # Run SQL
        # Output type must be specified for run_sql
        out_schema = elt.TableType({"a": elt.f64, "b": elt.f64})

        t2 = table.run_sql(
            "SELECT CAST(a AS DOUBLE) AS a, b * 2 AS b FROM t1",
            out_type=out_schema,
            t1=t1,  # type: ignore
        )

        # Table to Tensor
        # t2 has 3 rows.
        tensor_val = table.table2tensor(t2, number_rows=3)

        # Tensor to Table
        t3 = table.tensor2table(tensor_val, column_names=["a", "b"])  # type: ignore

        # write & read
        table.write(t2, path)
        t4 = table.read(path, schema=out_schema)

        return t3, t4

    # Execute
    result = workload()

    expected = pa.table({"a": [1.0, 2.0, 3.0], "b": [8.0, 10.0, 12.0]})
    for item in result:
        res_table = _get_table(item)
        assert isinstance(res_table, pa.Table)
        assert res_table == expected

    os.remove(path)


def test_table_constant_dataframe():
    """Test creating constant table from DataFrame."""
    import pandas as pd

    df_in = pd.DataFrame({"x": [10, 20], "y": ["foo", "bar"]})
    t = table.constant(df_in)  # type: ignore

    res_table = _get_table(t)
    assert isinstance(res_table, pa.Table)
    df_out = res_table.to_pandas()
    pd.testing.assert_frame_equal(df_in, df_out)


def test_parquet_reader():
    """Test ParquetReader implementation for reading parquet files with column selection."""
    import os
    import tempfile

    import pyarrow.parquet as pq

    # Create test data
    data = pa.table({
        "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
        "name": pa.array(["Alice", "Bob", "Charlie", "David", "Eve"], type=pa.string()),
        "score": pa.array([85.5, 90.0, 78.5, 95.5, 88.0], type=pa.float64()),
        "active": pa.array([True, False, True, True, False], type=pa.bool_()),
    })

    # Write to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        pq.write_table(data, tmp_path)

    try:
        # Test 1: Read all columns
        reader_all = ParquetReader(tmp_path, columns=None)

        # Test basic properties
        assert reader_all.num_rows == 5
        # When columns=None, schema is a ParquetSchema object
        assert hasattr(reader_all.schema, "names")
        assert reader_all.schema.names == ["id", "name", "score", "active"]

        # Test that we can access column metadata
        for field_name in ["id", "name", "score", "active"]:
            assert field_name in reader_all.schema.names

        # Test reading batches
        batch1 = reader_all.read_next_batch()
        assert batch1.num_rows == 5  # All data in one batch
        assert batch1.column_names == ["id", "name", "score", "active"]

        # Verify data
        assert batch1.to_pandas()["id"].tolist() == [1, 2, 3, 4, 5]
        assert batch1.to_pandas()["name"].tolist() == [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Eve",
        ]
        assert batch1.to_pandas()["score"].tolist() == [85.5, 90.0, 78.5, 95.5, 88.0]
        assert batch1.to_pandas()["active"].tolist() == [True, False, True, True, False]

        # Test 2: Read selected columns
        reader_selected = ParquetReader(tmp_path, columns=["id", "score"])

        # Test properties with column selection
        assert reader_selected.num_rows == 5
        # Schema is always a pa.Schema object
        assert isinstance(reader_selected.schema, pa.Schema)
        assert len(reader_selected.schema) == 2
        # Extract field names from the schema
        schema_names = reader_selected.schema.names
        assert schema_names == ["id", "score"]

        # Verify the field names and types are correct
        id_field = reader_selected.schema.field("id")
        score_field = reader_selected.schema.field("score")
        assert id_field.name == "id"
        assert id_field.type == pa.int64()
        assert score_field.name == "score"
        assert score_field.type == pa.float64()

        # Test reading batches with selected columns
        batch2 = reader_selected.read_next_batch()
        assert batch2.num_rows == 5
        assert batch2.column_names == ["id", "score"]

        # Verify selected data
        assert batch2.to_pandas()["id"].tolist() == [1, 2, 3, 4, 5]
        assert batch2.to_pandas()["score"].tolist() == [85.5, 90.0, 78.5, 95.5, 88.0]

        # Test 3: Test cast interface (inherited from pa.RecordBatchReader)
        reader_cast = ParquetReader(tmp_path, columns=["id", "score"])

        # Cast id column from int64 to float64 using reader's cast method
        # This should return a new RecordBatchReader that casts batches lazily
        casted_reader = reader_cast.cast(
            pa.schema([("id", pa.float64()), ("score", pa.float64())])
        )

        # Verify the casted reader has the correct schema
        assert casted_reader.schema.field("id").type == pa.float64()
        assert casted_reader.schema.field("score").type == pa.float64()

        # Read from the casted reader and verify data
        casted_batch = casted_reader.read_next_batch()
        assert casted_batch.schema.field("id").type == pa.float64()
        assert casted_batch.schema.field("score").type == pa.float64()
        assert casted_batch.to_pandas()["id"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert casted_batch.to_pandas()["score"].tolist() == [
            85.5,
            90.0,
            78.5,
            95.5,
            88.0,
        ]

        # Test 4: Test iterator protocol
        reader_iter = ParquetReader(tmp_path, columns=["name"])
        batches = list(reader_iter)
        assert len(batches) == 1
        assert batches[0].column_names == ["name"]
        assert batches[0].to_pandas()["name"].tolist() == [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Eve",
        ]

        # Test 5: Test context manager
        with ParquetReader(tmp_path, columns=["active"]) as reader_ctx:
            batch_ctx = reader_ctx.read_next_batch()
            assert batch_ctx.column_names == ["active"]
            assert batch_ctx.to_pandas()["active"].tolist() == [
                True,
                False,
                True,
                True,
                False,
            ]

        # Test 6: Test empty columns list (should read all)
        reader_empty = ParquetReader(tmp_path, columns=[])
        assert reader_empty.num_rows == 5
        # Empty list is treated like None - returns ParquetSchema
        assert hasattr(reader_empty.schema, "names")
        assert len(reader_empty.schema.names) == 4

        # Test 7: Test with non-existent column (should work normally)
        reader_partial = ParquetReader(tmp_path, columns=["id", "nonexistent"])
        # Only existing columns should be in schema
        assert isinstance(reader_partial.schema, pa.Schema)
        assert len(reader_partial.schema) == 1
        assert reader_partial.schema.names[0] == "id"

        # Test 8: Test cast validation - wrong number of columns
        reader_cast_val = ParquetReader(tmp_path, columns=["id", "score"])
        with pytest.raises(ValueError) as exc_info:
            reader_cast_val.cast(pa.schema([("id", pa.float64())]))
        assert "target schema has 1 columns, but current schema has 2 columns" in str(
            exc_info.value
        )

        # Test 9: Test cast validation - wrong field name
        with pytest.raises(ValueError) as exc_info:
            reader_cast_val.cast(
                pa.schema([("id", pa.float64()), ("wrong_name", pa.float64())])
            )
        assert "field name at position 1 differs" in str(exc_info.value)
        assert "Current: 'score', Target: 'wrong_name'" in str(exc_info.value)

        # Test 10: Test cast with no changes - should not set _cast
        # Cast to the same schema
        same_schema = pa.schema([("id", pa.int64()), ("score", pa.float64())])
        reader_no_change = ParquetReader(tmp_path, columns=["id", "score"])
        casted_reader_no_change = reader_no_change.cast(same_schema)

        # _cast should be False when there are no changes
        assert (
            not hasattr(casted_reader_no_change, "_cast")
            or not casted_reader_no_change._cast
        )

        # Reading should work normally
        batch = casted_reader_no_change.read_next_batch()
        assert batch.schema.field("id").type == pa.int64()
        assert batch.schema.field("score").type == pa.float64()

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_file_table_source():
    """Test FileTableSource implementation for different file formats."""
    import os
    import tempfile

    import duckdb
    import pyarrow.parquet as pq

    from mplang.v2.backends.table_impl import FileTableSource, TableReader

    # Create test data
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [85.5, 90.1, 78.5, 95.5, 88.1],
        "active": [True, False, True, True, False],
    }
    table = pa.table(data)

    # Define schema - using pa.Schema matching the table
    schema = pa.schema([
        ("id", pa.int64()),
        ("name", pa.string()),
        ("score", pa.float64()),
        ("active", pa.bool_()),
    ])

    cast_schema = pa.schema([("id", pa.int64()), ("score", pa.string())])
    cast_table = pa.table({"id": data["id"], "score": data["score"]}).cast(cast_schema)

    # Test with different file formats
    formats = {
        "parquet": lambda p, t: pq.write_table(t, p),
        "csv": lambda p, t: t.to_pandas().to_csv(p, index=False),
        "json": lambda p, t: t.to_pandas().to_json(p, orient="records", lines=True),
    }

    for format_name, write_func in formats.items():
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            suffix=f".{format_name}", delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name
            write_func(tmp_path, table)

        try:
            # Test 1: Create FileTableSource
            source = FileTableSource(path=tmp_path, format=format_name, schema=schema)
            assert source.path == tmp_path
            assert source.format == format_name
            assert source.schema == schema

            # Test 2: Open and read data
            with source.open() as reader:
                assert isinstance(reader, TableReader)

                # Read all data
                result = reader.read_all()
                assert result == table

            # Test 3: register and exec sql
            with duckdb.connect() as conn:
                source.register(conn, "table1")
                result = conn.execute("SELECT * FROM table1").fetch_arrow_table()
                assert result == table

            # Test 4: Batch read
            with source.open(batch_size=2) as reader:
                batches = list(reader)
                num_rows = [batch.num_rows for batch in batches]
                # 5 rows with batch_size=2 => 3 batches
                assert len(batches) == 3 and num_rows == [2, 2, 1]
                assert pa.Table.from_batches(batches) == table

            # Test 5: Test casting - data source schema differs from target schema
            source1 = FileTableSource(
                path=tmp_path, format=format_name, schema=cast_schema
            )
            with source1.open() as reader:
                result = reader.read_all()
                assert result == cast_table

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Test unsupported format
    with pytest.raises(ValueError, match="Unsupported format: unsupported"):
        source_bad = FileTableSource(
            path="test.txt", format="unsupported", schema=schema
        )
        with source_bad.open():
            pass  # This should raise ValueError during open


def test_write_basic(tmp_path):
    filename = tmp_path / "test_write_basic"
    data = {"x": [10, 20], "y": ["foo", "bar"]}
    src_table = pa.table(data)
    t1 = table.constant(data)
    formats = ["parquet", "csv", "json"]
    schema = elt.TableType({"x": elt.i64, "y": elt.STRING})
    for fmt in formats:
        path = filename.with_suffix(f".{fmt}")
        table.write(t1, path=str(path), format="auto")
        result = table.read(str(path), schema=schema)
        res_table = _get_table(result)
        assert res_table == src_table


def test_write_multi_tables(tmp_path):
    data1 = {"x": [10, 20], "y": ["foo", "bar"]}
    data2 = {"z": [0.1, 0.2]}

    def workload():
        t1 = table.constant(data1)
        t2 = table.constant(data2)
        path = str(tmp_path / "test_multi_write.parquet")
        table.write([t1, t2], path=path)
        result = table.read(
            path, schema=elt.TableType({"x": elt.i64, "y": elt.STRING, "z": elt.f64})
        )
        return result

    result = workload()
    res_table = _get_table(result)
    assert res_table == pa.table({**data1, **data2})

    # Test row data inconsistency
    data3 = {"a": [1, 2], "b": [3, 4]}  # 2 rows
    data4 = {"c": [5, 6, 7]}  # 3 rows - different count
    path2 = "test_multi_write_inconsistent.parquet"

    def workload_inconsistent():
        t3 = table.constant(data3)
        t4 = table.constant(data4)
        table.write([t3, t4], path=path2)

    with pytest.raises(ValueError, match=r"Batch 1 has \d+ rows, expected \d+"):
        workload_inconsistent()

    # Test duplicate column names
    data5 = {"x": [1, 2], "y": [3, 4]}
    data6 = {"y": [5, 6]}  # 'y' is also in data5
    path3 = "test_multi_write_duplicate.parquet"

    def workload_duplicate():
        t5 = table.constant(data5)
        t6 = table.constant(data6)
        table.write([t5, t6], path=path3)

    with pytest.raises(
        ValueError, match=r"Duplicate column name 'y' found across tables"
    ):
        workload_duplicate()
