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

import numpy as np
import pyarrow as pa
import pytest

import mplang.edsl.typing as elt
from mplang.backends.table_impl import TableValue
from mplang.dialects import table


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


@pytest.mark.parametrize("format_name", ["parquet", "csv", "json"])
def test_read_enforces_declared_schema(tmp_path, format_name):
    """table.read materializes with the declared schema (integral y cast to f64).

    Replaces the removed FileTableSource.register schema-enforcement tests:
    enforcement now happens at read time instead of DuckDB view registration.
    """
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pq

    path = tmp_path / f"labels.{format_name}"
    inferred_table = pa.table({"id": [1, 2], "y": [0, 1]})  # y looks integral
    match format_name:
        case "parquet":
            pq.write_table(inferred_table, path)
        case "csv":
            pa_csv.write_csv(inferred_table, path)
        case "json":
            inferred_table.to_pandas().to_json(path, orient="records", lines=True)

    schema = elt.TableType({"id": elt.i64, "y": elt.f64})
    result = _get_table(table.read(str(path), schema=schema))

    expected = pa.table({
        "id": pa.array([1, 2], type=pa.int64()),
        "y": pa.array([0.0, 1.0], type=pa.float64()),
    })
    assert result == expected
    assert result.schema.field("y").type == pa.float64()


def test_read_rejects_missing_declared_column(tmp_path):
    """A declared column absent from the file fails fast at read time."""
    import pyarrow.csv as pa_csv

    path = tmp_path / "missing.csv"
    pa_csv.write_csv(pa.table({"y": [0, 1]}), path)

    schema = elt.TableType({"y": elt.f64, "missing": elt.f64})
    # pyarrow raises ArrowInvalid (parquet/csv) or KeyError (json) for absent columns.
    with pytest.raises((pa.ArrowInvalid, KeyError)):
        _get_table(table.read(str(path), schema=schema))


def test_csv_schema_survives_sql_table2tensor_compile_evaluate(tmp_path):
    """Regression for catalog f64 labels whose CSV values look integral."""
    import pyarrow.csv as pa_csv

    import mplang as mp
    from mplang.backends.tensor_impl import TensorValue
    from mplang.runtime.interpreter import Interpreter

    path = tmp_path / "labels.csv"
    pa_csv.write_csv(pa.table({"feature": [10, 20], "y": [0, 1]}), path)
    input_type = elt.TableType({"feature": elt.i64, "y": elt.f64})
    output_type = elt.TableType({"y": elt.f64})

    def workload():
        source = table.read(str(path), schema=input_type)
        selected = table.run_sql(
            "SELECT y FROM source", out_type=output_type, source=source
        )
        return table.table2tensor(selected, number_rows=-1)

    interpreter = Interpreter()
    program = mp.compile(workload, context=interpreter)
    result = mp.evaluate(program, context=interpreter)

    assert program.graph.outputs[0].type == elt.TensorType(elt.f64, (-1, 1))
    assert isinstance(result.runtime_obj, TensorValue)
    np.testing.assert_array_equal(
        result.runtime_obj.unwrap(),
        np.array([[0.0], [1.0]], dtype=np.float64),
    )


def test_table2tensor_rejects_runtime_dtype_different_from_declared_type():
    """Backend values cannot silently violate table2tensor's IR output type."""

    def workload():
        source = table.constant({"y": [0, 1]})
        declared_float = table.run_sql(
            "SELECT y FROM source",
            out_type=elt.TableType({"y": elt.f64}),
            source=source,
        )
        return table.table2tensor(declared_float, number_rows=-1)

    with pytest.raises(
        TypeError, match=r"runtime dtype int64.*declared tensor dtype float64"
    ):
        workload()


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
    """Multi-table write combines same-length inputs column-wise and rejects
    row-count / column-name conflicts.

    The happy path reads two Parquet sources with *different* row-group sizes
    (chunks [2,2,1] vs [3,2]); after the eager-read refactor these materialize
    as multi-chunk pa.Tables, so MultiTableReader must rechunk before combining
    them or it raises ``Batch 1 has 3 rows, expected 2``.
    """
    import pyarrow.parquet as pq

    data1 = {"x": [10, 20, 30, 40, 50], "y": ["foo", "bar", "baz", "qux", "quux"]}
    data2 = {"z": [0.1, 0.2, 0.3, 0.4, 0.5]}
    p1 = tmp_path / "src1.parquet"
    p2 = tmp_path / "src2.parquet"
    pq.write_table(pa.table(data1), p1, row_group_size=2)  # row groups [2, 2, 1]
    pq.write_table(pa.table(data2), p2, row_group_size=3)  # row groups [3, 2]
    # regression precondition: differing native chunk sizes
    assert [b.num_rows for b in pq.read_table(p1).to_batches()] == [2, 2, 1]
    assert [b.num_rows for b in pq.read_table(p2).to_batches()] == [3, 2]

    path = tmp_path / "test_multi_write.parquet"

    def workload():
        t1 = table.read(str(p1), schema=elt.TableType({"x": elt.i64, "y": elt.STRING}))
        t2 = table.read(str(p2), schema=elt.TableType({"z": elt.f64}))
        table.write([t1, t2], path=str(path))
        return table.read(
            str(path),
            schema=elt.TableType({"x": elt.i64, "y": elt.STRING, "z": elt.f64}),
        )

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


def test_read_write_with_store(tmp_path, monkeypatch):
    """Test that read/write resolve relative paths via ObjectStore."""
    from mplang.edsl.context import set_root_context
    from mplang.runtime.interpreter import Interpreter
    from mplang.runtime.object_store import FileSystemBackend, ObjectStore

    # Table I/O materialises relative paths into a CWD-relative ".tmp/"
    # staging dir before delegating to the store, and store.download() leaves
    # a symlink in that staging dir pointing into the store root.  If that
    # staging dir lives in the project root it survives across pytest
    # sessions; once the previous session's tmp_path is reaped the symlink
    # dangles and the next run's write fails with FileNotFoundError.  chdir
    # into tmp_path so the staging dir is created under the per-session
    # tmp_path and cleaned up by pytest -- parallel-safe and free of any
    # pre-existing project-root ".tmp/" leftovers.
    monkeypatch.chdir(tmp_path)

    store = ObjectStore(persistent=FileSystemBackend(root_path=str(tmp_path)))
    interp = Interpreter(store=store)
    set_root_context(interp, force=True)

    try:
        data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
        schema = elt.TableType({"a": elt.i64, "b": elt.f64})
        t1 = table.constant(data)

        # Write with a relative path — should resolve under tmp_path
        table.write(t1, "output/result.parquet")
        assert os.path.exists(tmp_path / "output" / "result.parquet")

        # Read with the same relative path
        t2 = table.read("output/result.parquet", schema=schema)
        res = t2 if isinstance(t2, pa.Table) else _get_table(t2)
        assert res == pa.table(data)
    finally:
        from mplang.edsl.context import pop_context

        pop_context()


def test_read_write_absolute_path_with_store(tmp_path):
    """Test that absolute paths bypass ObjectStore path resolution."""
    from mplang.edsl.context import set_root_context
    from mplang.runtime.interpreter import Interpreter
    from mplang.runtime.object_store import FileSystemBackend, ObjectStore

    store_root = tmp_path / "store_root"
    store_root.mkdir()
    store = ObjectStore(persistent=FileSystemBackend(root_path=str(store_root)))
    interp = Interpreter(store=store)
    set_root_context(interp, force=True)

    try:
        data = {"x": [10, 20]}
        schema = elt.TableType({"x": elt.i64})
        t1 = table.constant(data)

        # Write with absolute path — should NOT resolve via store
        abs_path = str(tmp_path / "direct" / "output.parquet")
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        table.write(t1, abs_path)
        assert os.path.exists(abs_path)
        # Verify it did NOT write under store_root
        assert not os.path.exists(store_root / "direct" / "output.parquet")

        # Read with absolute path
        t2 = table.read(abs_path, schema=schema)
        res = _get_table(t2)
        assert res == pa.table(data)
    finally:
        from mplang.edsl.context import pop_context

        pop_context()
