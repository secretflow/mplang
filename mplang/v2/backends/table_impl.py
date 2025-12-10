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

"""Table Runtime Implementation.

Implements execution logic for Table primitives using DuckDB and PyArrow.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

import duckdb
import pandas as pd
import pyarrow as pa

from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import table
from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import Interpreter
from mplang.v2.runtime.value import WrapValue

# =============================================================================
# TableValue Wrapper
# =============================================================================


@serde.register_class
class TableValue(WrapValue[pa.Table]):
    """Runtime value wrapping a PyArrow Table.

    Provides serialization via Arrow IPC format (streaming).
    Future: may extend to support other backends (Polars, DuckDB relations, etc.)
    """

    _serde_kind: ClassVar[str] = "table_impl.TableValue"

    # =========== Wrap/Unwrap ===========

    def _convert(self, data: Any) -> pa.Table:
        """Convert input data to pa.Table."""
        if isinstance(data, TableValue):
            return data.unwrap()
        if isinstance(data, pd.DataFrame):
            return pa.Table.from_pandas(data)
        if not isinstance(data, pa.Table):
            raise TypeError(f"Expected pa.Table or pd.DataFrame, got {type(data)}")
        return data

    # unwrap() is inherited from WrapValue

    # =========== Serialization ===========

    def to_json(self) -> dict[str, Any]:
        # Serialize using Arrow IPC streaming format
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, self._data.schema) as writer:
            writer.write_table(self._data)
        ipc_bytes = sink.getvalue().to_pybytes()
        return {"ipc": base64.b64encode(ipc_bytes).decode("ascii")}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TableValue:
        ipc_bytes = base64.b64decode(data["ipc"])
        reader = pa.ipc.open_stream(ipc_bytes)
        table = reader.read_all()
        return cls(table)


# Module-level helpers for convenience (delegate to class methods)
def _wrap(val: pa.Table | pd.DataFrame | TableValue) -> TableValue:
    """Wrap a table-like value into TableValue."""
    return TableValue.wrap(val)


def _unwrap(val: TableValue | pa.Table | pd.DataFrame) -> pa.Table:
    """Unwrap TableValue to pa.Table, also accepts raw pa.Table/DataFrame."""
    if isinstance(val, TableValue):
        return val.unwrap()
    if isinstance(val, pd.DataFrame):
        return pa.Table.from_pandas(val)
    if isinstance(val, pa.Table):
        return val
    # Handle RecordBatchReader from newer PyArrow versions
    if isinstance(val, pa.lib.RecordBatchReader):
        return val.read_all()
    raise TypeError(
        f"Expected TableValue, pa.Table, pd.DataFrame, or RecordBatchReader, got {type(val)}"
    )


# =============================================================================
# Table Primitive Implementations
# =============================================================================


@table.run_sql_p.def_impl
def run_sql_impl(interpreter: Interpreter, op: Operation, *args: Any) -> TableValue:
    """Execute SQL query on input tables."""
    query = op.attrs["query"]
    dialect = op.attrs.get("dialect", "duckdb")
    table_names = op.attrs["table_names"]

    if dialect != "duckdb":
        raise ValueError(f"Unsupported dialect: {dialect}")

    # Use in-memory DuckDB connection
    conn = duckdb.connect(":memory:")

    for name, arg in zip(table_names, args, strict=True):
        conn.register(name, _unwrap(arg))

    # Execute query and fetch result as Arrow table
    try:
        arrow_result = conn.execute(query).arrow()
        # In newer DuckDB versions, .arrow() returns RecordBatchReader
        res = arrow_result.read_all()
        return _wrap(res)
    except Exception as e:
        raise RuntimeError(f"Failed to execute SQL query: {query}") from e


@table.table2tensor_p.def_impl
def table2tensor_impl(interpreter: Interpreter, op: Operation, table_val: Any) -> Any:
    """Convert table to tensor (numpy array).

    Returns TensorValue if tensor_impl is available, otherwise raw np.ndarray.
    """
    from mplang.v2.backends.tensor_impl import TensorValue

    tbl = _unwrap(table_val)
    df = tbl.to_pandas()
    # Convert to numpy array
    # Note: This assumes the table is homogeneous as enforced by abstract_eval
    arr = df.to_numpy()
    return TensorValue.wrap(arr)


@table.tensor2table_p.def_impl
def tensor2table_impl(
    interpreter: Interpreter, op: Operation, tensor_val: TensorValue
) -> TableValue:
    """Convert tensor (numpy array) to table."""
    column_names = op.attrs["column_names"]

    # Unwrap TensorValue
    arr = tensor_val.unwrap()

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")

    if arr.shape[1] != len(column_names):
        raise ValueError(
            f"Shape mismatch: tensor has {arr.shape[1]} columns, "
            f"but {len(column_names)} names provided"
        )

    # Create dictionary for DataFrame/Table creation
    data = {}
    for i, name in enumerate(column_names):
        data[name] = arr[:, i]

    return _wrap(pa.Table.from_pydict(data))


@table.constant_p.def_impl
def constant_impl(interpreter: Interpreter, op: Operation) -> TableValue:
    """Create constant table."""
    # data is stored in attrs by default bind if not TraceObject
    data = op.attrs["data"]

    # Handle pandas DataFrame if passed directly (though attrs usually store basic types)
    # If data was a DataFrame, it might have been stored as is if the IR supports it.
    # If data was a dict, it's fine.

    if isinstance(data, pd.DataFrame):
        return _wrap(pa.Table.from_pandas(data))

    return _wrap(pa.Table.from_pydict(data))


def _infer_format(path: str, format_hint: str) -> str:
    """Infer file format from path extension or hint."""
    if format_hint != "auto":
        return format_hint

    path_lower = path.lower()
    if path_lower.endswith((".parquet", ".pq")):
        return "parquet"
    elif path_lower.endswith(".csv"):
        return "csv"
    elif path_lower.endswith((".json", ".jsonl")):
        return "json"
    elif path_lower.endswith((".feather", ".arrow")):
        return "feather"
    else:
        # Default to parquet
        return "parquet"


@table.read_p.def_impl
def read_impl(interpreter: Interpreter, op: Operation) -> TableValue:
    """Read table from file.

    Supported formats: parquet, csv, json, feather
    """
    import pyarrow.csv as pv_csv
    import pyarrow.json as pv_json
    import pyarrow.parquet as pq

    path: str = op.attrs["path"]
    format_hint: str = op.attrs.get("format", "auto")

    fmt = _infer_format(path, format_hint)

    if fmt == "parquet":
        return _wrap(pq.read_table(path))
    elif fmt == "csv":
        return _wrap(pv_csv.read_csv(path))
    elif fmt == "json":
        return _wrap(pv_json.read_json(path))
    elif fmt == "feather":
        import pyarrow.feather as feather

        return _wrap(feather.read_table(path))
    else:
        raise ValueError(f"Unsupported format: {fmt}")


@table.write_p.def_impl
def write_impl(interpreter: Interpreter, op: Operation, table_val: Any) -> None:
    """Write table to file.

    Supported formats: parquet, csv, json, feather
    """
    import pyarrow.csv as pv_csv
    import pyarrow.parquet as pq

    path: str = op.attrs["path"]
    format_hint: str = op.attrs.get("format", "parquet")

    fmt = _infer_format(path, format_hint)

    tbl = _unwrap(table_val)

    if fmt == "parquet":
        pq.write_table(tbl, path)
    elif fmt == "csv":
        pv_csv.write_csv(tbl, path)
    elif fmt == "json":
        # PyArrow doesn't have direct JSON write, convert to pandas
        df = tbl.to_pandas()
        df.to_json(path, orient="records", lines=True)
    elif fmt == "feather":
        import pyarrow.feather as feather

        feather.write_feather(tbl, path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
