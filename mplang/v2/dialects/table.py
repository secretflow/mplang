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

"""Table dialect: table operations backed by plaintext/private SQL engines."""

from __future__ import annotations

from typing import Any

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt

run_sql_p: el.Primitive[Any] = el.Primitive("table.run_sql")
table2tensor_p: el.Primitive[el.Object] = el.Primitive("table.table2tensor")
tensor2table_p: el.Primitive[el.Object] = el.Primitive("table.tensor2table")
constant_p: el.Primitive[el.Object] = el.Primitive("table.constant")
read_p: el.Primitive[el.Object] = el.Primitive("table.read")
write_p: el.Primitive[el.Object] = el.Primitive("table.write")


def _current_tracer() -> el.Tracer:
    ctx = el.get_current_context()
    if not isinstance(ctx, el.Tracer):
        raise TypeError(f"Expected Tracer context, got {type(ctx)}")
    return ctx


@run_sql_p.def_trace
def _run_sql_trace(
    query: str,
    *,
    out_type: elt.TableType,
    dialect: str = "duckdb",
    **tables: el.TraceObject,
) -> el.TraceObject:
    tracer = _current_tracer()
    if not isinstance(out_type, elt.TableType):
        raise TypeError("run_sql out_type must be TableType")
    if not tables:
        raise ValueError("run_sql requires at least one table input")

    ordered = list(tables.items())
    inputs = []
    names = []
    for name, table in ordered:
        if not isinstance(table, el.TraceObject):
            raise TypeError(f"Table '{name}' must be TraceObject")
        inputs.append(table._graph_value)
        names.append(name)

    [value] = tracer.graph.add_op(
        opcode="table.run_sql",
        inputs=inputs,
        output_types=[out_type],
        attrs={"query": query, "dialect": dialect, "table_names": names},
    )
    return el.TraceObject(value, tracer)


@table2tensor_p.def_abstract_eval
def _table2tensor_ae(table_t: elt.TableType, *, number_rows: int) -> elt.TensorType:
    """Infer tensor type for table.table2tensor."""

    if not isinstance(number_rows, int):
        raise TypeError("number_rows must be an int")
    if number_rows < 0:
        raise ValueError("number_rows must be >= 0")
    if not table_t.schema:
        raise ValueError("Cannot convert empty table to tensor")
    column_types = list(table_t.schema.values())
    first = column_types[0]

    def _scalar_dtype(col: elt.BaseType) -> elt.BaseType:
        if hasattr(col, "element_type"):
            tensor_col = col  # type: ignore[assignment]
            if tensor_col.shape not in ((), None):  # type: ignore[attr-defined]
                raise TypeError(
                    "table2tensor expects scalar columns (rank-0 TensorType)"
                )
            return tensor_col.element_type  # type: ignore[attr-defined,no-any-return]
        return col

    first_scalar = _scalar_dtype(first)
    for col in column_types[1:]:
        if _scalar_dtype(col) != first_scalar:
            raise TypeError("All table columns must share the same scalar dtype")
    if not isinstance(first_scalar, elt.BaseType):
        raise TypeError("All table columns must share the same dtype for table2tensor")
    return elt.TensorType(first_scalar, (number_rows, len(column_types)))


@tensor2table_p.def_abstract_eval
def _tensor2table_ae(
    tensor_t: elt.TensorType, *, column_names: list[str]
) -> elt.TableType:
    """Infer table type for table.tensor2table."""

    if len(tensor_t.shape) != 2:
        raise TypeError(
            f"tensor2table expects rank-2 tensor (N, F), got rank {len(tensor_t.shape)}"
        )
    n_cols = tensor_t.shape[1]
    if not column_names:
        raise ValueError("column_names must be provided")
    if len(column_names) != n_cols:
        raise ValueError("column_names length must match tensor second dimension")
    seen: set[str] = set()
    schema: dict[str, elt.BaseType] = {}
    for idx, name in enumerate(column_names):
        if not isinstance(name, str):
            raise TypeError(
                f"column_names[{idx}] must be str, got {type(name).__name__}"
            )
        if name.strip() == "":
            raise ValueError("column names must be non-empty/non-whitespace")
        if name in seen:
            raise ValueError(f"duplicate column name: {name!r}")
        seen.add(name)
        schema[name] = tensor_t.element_type
    # Each column shares the tensor's element dtype.
    return elt.TableType(schema)


def run_sql(
    query: str,
    *,
    out_type: elt.TableType,
    dialect: str = "duckdb",
    **tables: el.TraceObject,
) -> el.TraceObject:
    """Trace a SQL query over plaintext/private tables.

    Inserts a `table.run_sql` op with the provided query string and table inputs.
    The `out_type` describes the resulting table schema (columns + types).
    """

    return run_sql_p.bind(  # type: ignore[no-any-return]
        query,
        out_type=out_type,
        dialect=dialect,
        **tables,
    )


def table2tensor(table: el.TraceObject, *, number_rows: int) -> el.Object:
    """Convert a homogeneous table into a dense tensor."""

    return table2tensor_p.bind(table, number_rows=number_rows)


def tensor2table(tensor: el.TraceObject, *, column_names: list[str]) -> el.Object:
    """Convert a rank-2 tensor (N, F) into a table with named columns."""

    return tensor2table_p.bind(tensor, column_names=column_names)


@constant_p.def_abstract_eval
def _constant_ae(*, data: Any) -> elt.TableType:
    """Infer table type for constant data.

    Args:
        data: Dictionary mapping column names to lists of values,
              pandas DataFrame, PyArrow Table, or any data convertible to DataFrame

    Returns:
        TableType inferred from schema

    Raises:
        TypeError: If data cannot be converted to DataFrame
    """
    import pandas as pd
    import pyarrow as pa

    from mplang.v2.dialects import dtypes

    # Handle PyArrow Table directly
    if isinstance(data, pa.Table):
        schema: dict[str, elt.BaseType] = {}
        for field in data.schema:
            schema[field.name] = dtypes.from_arrow(field.type)
        return elt.TableType(schema)

    # Handle pandas DataFrame
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        # Dict or other types - convert to DataFrame
        df = pd.DataFrame(data)

    # Infer schema from pandas dtypes
    schema = {}
    for col_name in df.columns:
        schema[str(col_name)] = dtypes.from_pandas(df[col_name].dtype)

    return elt.TableType(schema)


def constant(data: dict[str, list]) -> el.Object:
    """Create a table constant value.

    This creates a constant table that can be used in table computations.
    The constant value is embedded directly into the computation graph.

    Args:
        data: Dictionary mapping column names to lists of values,
              pandas DataFrame, or any data convertible to DataFrame.
              All columns must have the same length.

    Returns:
        Object representing the constant table (TraceObject in trace mode,
        InterpObject in interp mode)

    Raises:
        TypeError: If data cannot be converted to DataFrame
        ValueError: If columns have different lengths

    Example:
        >>> # From dict
        >>> table = constant({
        ...     "id": [1, 2, 3],
        ...     "name": ["alice", "bob", "charlie"],
        ...     "score": [95.5, 87.2, 92.8],
        ... })
        >>> # From DataFrame
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        >>> table = constant(df)
    """
    return constant_p.bind(data=data)  # type: ignore[no-any-return]


# =============================================================================
# Table I/O: read and write
# =============================================================================


@read_p.def_abstract_eval
def _read_ae(*, path: str, schema: elt.TableType, format: str) -> elt.TableType:
    """Infer output type for table.read.

    Args:
        path: File path to read from
        schema: Expected table schema
        format: File format ("auto", "csv", "parquet")

    Returns:
        The provided schema (since we can't inspect the file at trace time)

    Raises:
        TypeError: If schema is not a TableType
        ValueError: If path is empty or format is invalid
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if not isinstance(schema, elt.TableType):
        raise TypeError(f"schema must be TableType, got {type(schema).__name__}")
    if format not in ("auto", "csv", "parquet"):
        raise ValueError(f"format must be 'auto', 'csv', or 'parquet', got {format!r}")
    return schema


def read(
    path: str,
    *,
    schema: elt.TableType,
    format: str = "auto",
) -> el.Object:
    """Read a table from a file.

    This creates a table.read operation that reads data from the specified path
    at runtime. The schema must be provided since the file cannot be inspected
    at trace/compile time.

    Args:
        path: File path to read from. In distributed scenarios, each party
            interprets this path relative to its own filesystem.
        schema: Expected table schema. Must match the actual file structure.
        format: File format. Options:
            - "auto": Detect from file extension (.csv, .parquet)
            - "csv": Read as CSV
            - "parquet": Read as Parquet

    Returns:
        Table object with the specified schema.

    Example:
        >>> schema = TableType({
        ...     "id": TensorType(i64, ()),
        ...     "value": TensorType(f64, ()),
        ... })
        >>> tbl = table.read("/data/input.csv", schema=schema)
    """
    return read_p.bind(path=path, schema=schema, format=format)  # type: ignore[no-any-return]


@write_p.def_abstract_eval
def _write_ae(table_type: elt.TableType, *, path: str, format: str) -> elt.TableType:
    """Infer output type for table.write.

    Args:
        table_type: Input table's type
        path: File path to write to
        format: Output format ("csv", "parquet")

    Returns:
        The input table type (passthrough)

    Raises:
        TypeError: If input is not a TableType
        ValueError: If path is empty or format is invalid
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if not isinstance(table_type, elt.TableType):
        raise TypeError(f"Expected TableType input, got {type(table_type).__name__}")
    if format not in ("csv", "parquet"):
        raise ValueError(f"format must be 'csv' or 'parquet', got {format!r}")
    return table_type


def write(
    table: el.Object | Any,
    path: str,
    *,
    format: str = "parquet",
) -> el.Object | None:
    """Write a table to a file.

    This creates a table.write operation that persists the table data at runtime.
    The operation returns the input table unchanged, allowing chaining.

    If a runtime value (e.g., PyArrow Table, DataFrame, dict) is passed instead of
    a traced object, it will be wrapped with table.constant() automatically.

    Args:
        table: Table to write. Can be a TraceObject, PyArrow Table, DataFrame, or dict.
        path: Destination file path. In distributed scenarios, each party
            interprets this path relative to its own filesystem.
        format: Output format. Options:
            - "csv": Write as CSV
            - "parquet": Write as Parquet (default, more efficient)

    Returns:
        The input table (passthrough for chaining), or None in interpreter mode.

    Example:
        >>> result = table.run_sql("SELECT ...", out_type=schema, input=tbl)
        >>> table.write(result, "/data/output.parquet")
    """
    # Auto-wrap runtime values
    if not isinstance(table, el.Object):
        table = constant(table)
    return write_p.bind(table, path=path, format=format)  # type: ignore[no-any-return]


__all__ = [
    "constant",
    "constant_p",
    "read",
    "read_p",
    "run_sql",
    "run_sql_p",
    "table2tensor",
    "table2tensor_p",
    "tensor2table",
    "tensor2table_p",
    "write",
    "write_p",
]
