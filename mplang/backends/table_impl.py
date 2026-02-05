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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import duckdb
import pandas as pd
import pyarrow as pa

import mplang.edsl.typing as elt
from mplang.backends.tensor_impl import TensorValue
from mplang.dialects import table
from mplang.edsl import serde
from mplang.edsl.graph import Operation
from mplang.runtime.interpreter import Interpreter
from mplang.runtime.value import WrapValue
from mplang.utils.logging import get_logger

logger = get_logger(__name__)


class BatchReader(ABC):
    @property
    @abstractmethod
    def schema(self) -> pa.Schema: ...

    @abstractmethod
    def read_next_batch(self) -> pa.RecordBatch: ...
    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> pa.RecordBatch:
        return self.read_next_batch()


class TableReader(BatchReader):
    """A reader for streaming table data from PyArrow RecordBatchReader or Table.

    This class provides an efficient way to read large tables in batches,
    with support for custom batch sizes and proper handling of data boundaries.
    It implements the iterator protocol for easy consumption of data.
    """

    def __init__(
        self,
        data: pa.RecordBatchReader | pa.Table,
        num_rows: int = -1,
        batch_size: int = -1,
    ) -> None:
        """Initialize a TableReader.

        Args:
            data: Either a RecordBatchReader or Table to read from
            num_rows: Expected number of rows in the data. -1 indicates unknown
            batch_size: Size of each batch to read. -1 means use default/reader's batch size
        """
        # Store the underlying reader and row count based on input type
        if isinstance(data, pa.RecordBatchReader):
            self._reader = data
            self._num_rows = num_rows
        else:
            # Convert Table to RecordBatchReader for consistent interface
            self._reader = data.to_reader()
            self._num_rows = data.num_rows

        # Configuration for batch reading
        self._batch_size = batch_size

        # Internal state for handling custom batch sizes
        self._remain: pa.RecordBatch | None = (
            None  # Stores partial batch from previous read
        )
        self._eof = False  # Flag to indicate end of data

    @property
    def num_rows(self) -> int:
        """Get the total number of rows in the table.

        Returns:
            Total number of rows, or -1 if unknown
        """
        return self._num_rows

    @property
    def schema(self) -> pa.Schema:
        """Get the schema of the table.

        Returns:
            PyArrow Schema describing the table's columns and types
        """
        return self._reader.schema

    def read_all(self) -> pa.Table:
        """Read all remaining data as a Table.

        This is a convenience method that reads all data from the reader
        and returns it as a single PyArrow Table.

        Returns:
            Complete table containing all remaining data
        """
        return self._reader.read_all()

    def read_next_batch(self) -> pa.RecordBatch:
        """Read the next batch of records.

        This method respects the configured batch size. If the native reader
        returns batches larger than the configured size, this method will split
        them appropriately. Any partial data from previous reads is included
        in the returned batch.

        Returns:
            Next RecordBatch of data

        Raises:
            StopIteration: When no more data is available
        """
        # Check if we've reached end of file
        if self._eof:
            raise StopIteration

        # Get the next batch using internal logic
        batch = self._read_next_batch()

        # Handle end of data
        if batch is None:
            self._eof = True
            raise StopIteration

        return batch

    def _read_next_batch(self) -> pa.RecordBatch | None:
        """Internal method to read and process the next batch.

        This method handles the complex logic of:
        - Using default batch size when none is specified
        - Accumulating data from multiple native batches to reach the target size
        - Splitting oversized batches and saving the remainder
        - Converting between Table and RecordBatch formats as needed

        Returns:
            Next RecordBatch of the configured size, or None if no more data
        """
        # If no batch size specified, just return the reader's native batches
        if self._batch_size <= 0:
            try:
                batch = self._reader.read_next_batch()
                # Convert to RecordBatch if the reader returns a Table
                if isinstance(batch, pa.Table) and batch.num_rows > 0:
                    return batch.to_batches()[0]
                return batch
            except StopIteration:
                return None

        # We have a custom batch size - need to accumulate/split batches
        batches: list[pa.RecordBatch] = []
        num_rows: int = 0

        # First, include any remaining data from the previous read
        if self._remain is not None:
            num_rows = self._remain.num_rows
            batches = [self._remain]
            self._remain = None

        # Keep reading until we have enough rows or run out of data
        while num_rows < self._batch_size:
            try:
                batch = self._reader.read_next_batch()

                # Handle the case where reader returns a Table instead of RecordBatch
                if isinstance(batch, pa.Table):
                    if batch.num_rows > 0:
                        # Convert each batch from the Table
                        for rb in batch.to_batches():
                            num_rows += rb.num_rows
                            if rb.num_rows > 0:  # Skip empty batches
                                batches.append(rb)
                else:
                    # Already a RecordBatch
                    num_rows += batch.num_rows
                    if batch.num_rows > 0:  # Skip empty batches
                        batches.append(batch)
            except StopIteration:
                # Mark EOF but continue processing what we have
                self._eof = True
                break

        # If we didn't get any data, return None
        if num_rows == 0:
            return None

        # Split the last batch if we have more rows than needed
        if num_rows > self._batch_size:
            last = batches[-1]
            remain_size = num_rows - self._batch_size
            last_size = last.num_rows - remain_size

            # Keep only what we need from the last batch
            batches[-1] = last.slice(0, last_size)
            # Save the remainder for the next read
            self._remain = last.slice(last_size, remain_size)

        # Optimized path: if we only have one batch, return it directly
        if len(batches) == 1:
            return batches[0]

        # Otherwise, combine all batches and return as a single RecordBatch
        combined = pa.Table.from_batches(batches)
        return combined.to_batches()[0]

    def close(self) -> None:
        """Close the reader and release all resources.

        This method should be called when the reader is no longer needed.
        It closes the underlying reader and clears internal state.
        """
        # Close the underlying reader
        self._reader.close()
        # Clear internal state
        self._remain = None
        self._eof = False


DEFAULT_BATCH_SIZE = 1_000_000


class TableSource(ABC):
    """Abstract base class for lazy table operations.

    Provides deferred execution for table operations to prevent OOM issues.
    """

    @abstractmethod
    def register(
        self, conn: duckdb.DuckDBPyConnection, name: str, replace: bool = True
    ) -> None: ...

    @abstractmethod
    def open(self, batch_size: int = DEFAULT_BATCH_SIZE) -> TableReader:
        """Read data as a stream of record batches."""
        ...


class ParquetReader(pa.RecordBatchReader):
    """A reader that implements the pa.RecordBatchReader interface for Parquet files."""

    def __init__(self, source: Any, columns: list[str] | None = None):
        import pyarrow.parquet as pq

        file = pq.ParquetFile(source)

        # Use schema_arrow to get the proper pa.Schema
        if columns:
            # Filter the schema to only include selected columns
            fields = [
                file.schema_arrow.field(col)
                for col in columns
                if col in file.schema_arrow.names
            ]
            schema = pa.schema(fields)
        else:
            schema = file.schema_arrow

        self._file = file
        self._schema = schema
        self._cast = False
        self._num_rows = int(file.metadata.num_rows)
        self._iter = file.iter_batches(columns=columns)

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def cast(self, target_schema: pa.Schema) -> ParquetReader:
        # Validate that the number of columns is the same
        if len(target_schema) != len(self._schema):
            raise ValueError(
                f"Cannot cast schema: target schema has {len(target_schema)} columns, "
                f"but current schema has {len(self._schema)} columns"
            )

        # Check if there are any changes in the schema
        schema_changed = False
        for i, (target_field, current_field) in enumerate(
            zip(target_schema, self._schema, strict=True)
        ):
            # Check if field names are the same (allowing type changes)
            if target_field.name != current_field.name:
                raise ValueError(
                    f"Cannot cast schema: field name at position {i} differs. "
                    f"Current: '{current_field.name}', Target: '{target_field.name}'. "
                    f"Field names must match."
                )
            # Check if types are different
            if target_field.type != current_field.type:
                schema_changed = True

        # Only set _cast if there are actual changes
        if schema_changed:
            self._schema = target_schema
            self._cast = True

        return self

    def read_all(self) -> pa.Table:
        batches = []
        try:
            while True:
                batch = self.read_next_batch()
                batches.append(batch)
        except StopIteration:
            pass
        if batches:
            return pa.Table.from_batches(batches)
        return pa.Table.from_batches([], schema=self._schema)

    def read_next_batch(self) -> pa.RecordBatch:
        batch = next(self._iter)
        if self._cast:
            return batch.cast(self._schema)
        else:
            return batch

    def close(self) -> None:
        """Close the Parquet reader and release resources."""
        self._file.close()


_type_mapping = {
    elt.bool_: pa.bool_(),
    elt.i8: pa.int8(),
    elt.i16: pa.int16(),
    elt.i32: pa.int32(),
    elt.i64: pa.int64(),
    elt.u8: pa.uint8(),
    elt.u16: pa.uint16(),
    elt.u32: pa.uint32(),
    elt.u64: pa.uint64(),
    elt.f16: pa.float16(),
    elt.f32: pa.float32(),
    elt.f64: pa.float64(),
    elt.STRING: pa.string(),
    elt.DATE: pa.date64(),
    elt.TIME: pa.time32("ms"),
    elt.TIMESTAMP: pa.timestamp("ms"),
    elt.DECIMAL: pa.decimal128(38, 10),
    elt.BINARY: pa.binary(),
    elt.JSON: pa.json_(),
}


def _pa_schema(s: elt.TableType) -> pa.Schema:
    fields = []
    for k, v in s.schema.items():
        if v not in _type_mapping:
            raise ValueError(f"cannot convert to pyarrow type. type={v}, name={k}")
        fields.append(pa.field(k, _type_mapping[v]))

    return pa.schema(fields)


@dataclass
class FileTableSource(TableSource):
    """Lazy table handle for file-based operations with streaming reads."""

    path: str
    format: str
    schema: pa.Schema | None = None

    def register(
        self, conn: duckdb.DuckDBPyConnection, name: str, replace: bool = True
    ) -> None:
        """Register the file as a view in DuckDB."""
        func_name = ""
        match self.format:
            case "parquet":
                func_name = "read_parquet"
            case "csv":
                func_name = "read_csv_auto"
            case "json":
                func_name = "read_json_auto"
            case _:
                raise ValueError(f"Unsupported format: {self.format}")

        safe_path = self.path.replace("'", "''")
        base_query = f"SELECT * FROM {func_name}('{safe_path}')"
        if replace:
            query = f"CREATE OR REPLACE VIEW {name} AS {base_query}"
        else:
            query = f"CREATE VIEW {name} AS {base_query}"
        conn.execute(query)

    def open(self, batch_size: int = DEFAULT_BATCH_SIZE) -> TableReader:
        """Create a streaming reader for the file."""
        import pyarrow.csv as pa_csv
        import pyarrow.json as pa_json

        columns = self.schema.names if self.schema else None

        reader = None
        num_rows = -1
        match self.format.lower():
            case "parquet":
                reader = ParquetReader(self.path, columns)
                num_rows = reader.num_rows
            case "csv":
                read_options = pa_csv.ReadOptions(use_threads=True)
                convert_options = pa_csv.ConvertOptions(
                    column_types=self.schema,
                    include_columns=columns,
                )
                reader = pa_csv.open_csv(
                    self.path,
                    read_options=read_options,
                    convert_options=convert_options,
                )
            case "json":
                read_options = pa_json.ReadOptions(use_threads=True)
                table = pa_json.read_json(self.path, read_options=read_options)
                if columns:
                    table = table.select(columns)
                reader = table.to_reader()
                num_rows = table.num_rows
            case _:
                raise ValueError(f"Unsupported format: {self.format}")

        if self.schema and self.schema != reader.schema:
            reader = reader.cast(self.schema)

        return TableReader(reader, num_rows=num_rows, batch_size=batch_size)


class DuckDBState:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn
        self.tables: dict[str, Any] = {}


@dataclass(frozen=True)
class QueryTableSource(TableSource):
    """Handle for existing DuckDB relations (kept for compatibility)."""

    relation: duckdb.DuckDBPyRelation
    state: DuckDBState

    def register(
        self, conn: duckdb.DuckDBPyConnection, name: str, replace: bool = True
    ) -> None:
        self.relation.create_view(name, replace)

    def open(self, batch_size: int = DEFAULT_BATCH_SIZE) -> TableReader:
        """Read from the DuckDB relation."""
        if batch_size <= 0:
            batch_size = DEFAULT_BATCH_SIZE
        reader = self.relation.arrow(batch_size)
        return TableReader(reader)


# =============================================================================
# TableValue Wrapper
# =============================================================================


@serde.register_class
class TableValue(WrapValue[pa.Table | TableSource]):
    """Runtime value wrapping a PyArrow Table.

    Provides serialization via Arrow IPC format (streaming).
    Future: may extend to support other backends (Polars, DuckDB relations, etc.)
    """

    _serde_kind: ClassVar[str] = "table_impl.TableValue"

    @property
    def data(self) -> pa.Table:
        """Get the underlying PyArrow Table data.

        For lazy TableSource, this triggers a full read of the data and caches
        the result in self._data. Subsequent calls will return the cached table.

        Returns:
            The PyArrow Table containing all data
        """
        if isinstance(self._data, TableSource):
            source = self._data
            with source.open() as reader:
                self._data = reader.read_all()

        return self._data

    # =========== Wrap/Unwrap ===========

    def _convert(self, data: Any) -> pa.Table | TableSource:
        """Convert input data to pa.Table or TableSource."""
        if isinstance(data, TableValue):
            return data.unwrap()
        if isinstance(data, pd.DataFrame):
            data = pa.Table.from_pandas(data)
        if not isinstance(data, pa.Table | TableSource):
            raise TypeError(f"Expected pa.Table or TableSource, got {type(data)}")
        return data

    # =========== Serialization ===========

    def to_json(self) -> dict[str, Any]:
        # Serialize using Arrow IPC streaming format
        data = self.data
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, data.schema) as writer:
            writer.write_table(data)
        ipc_bytes = sink.getvalue().to_pybytes()
        return {"ipc": base64.b64encode(ipc_bytes).decode("ascii")}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TableValue:
        ipc_bytes = base64.b64decode(data["ipc"])
        reader = pa.ipc.open_stream(ipc_bytes)
        table = reader.read_all()
        return cls(table)


# Module-level helpers for convenience (delegate to class methods)
def _wrap(val: pa.Table | pd.DataFrame | TableSource | TableValue) -> TableValue:
    """Wrap a table-like value into TableValue."""
    return TableValue.wrap(val)


def _unwrap(val: TableValue | pa.Table | pd.DataFrame) -> pa.Table:
    """Unwrap TableValue to pa.Table, also accepts raw pa.Table/DataFrame."""
    if isinstance(val, TableValue):
        return val.data
    if isinstance(val, pd.DataFrame):
        return pa.Table.from_pandas(val)
    if isinstance(val, pa.Table):
        return val
    # Handle RecordBatchReader from newer PyArrow versions
    if isinstance(val, pa.RecordBatchReader):
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

    state: DuckDBState | None = None
    tables: list[TableValue] = []
    for arg in args:
        tbl = _wrap(arg)
        tables.append(tbl)
        data = tbl.unwrap()
        if isinstance(data, QueryTableSource):
            if state is None:
                state = data.state
            elif state != data.state:
                raise ValueError("All tables must belong to the same DuckDB connection")

    if state is None:
        conn = duckdb.connect()
        state = DuckDBState(conn)

    try:
        conn = state.conn
        # register tables or create view
        for name, tbl in zip(table_names, tables, strict=True):
            data = tbl.unwrap()
            if name in state.tables:
                if state.tables[name] is not data:
                    # TODO: rename and rewrite sql??
                    raise ValueError(f"{name} has been registered.")
            else:
                state.tables[name] = data
            if isinstance(data, TableSource):
                data.register(state.conn, name)
            else:
                conn.register(name, data)

        relation = conn.sql(query)
        return _wrap(QueryTableSource(relation, state))
    except Exception as e:
        raise RuntimeError(f"Failed to execute SQL query: {query}") from e


@table.table2tensor_p.def_impl
def table2tensor_impl(interpreter: Interpreter, op: Operation, table_val: Any) -> Any:
    """Convert table to tensor (numpy array).

    Returns TensorValue if tensor_impl is available, otherwise raw np.ndarray.
    """
    from mplang.backends.tensor_impl import TensorValue

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

    if isinstance(data, pa.Table):
        return _wrap(data)
    else:
        return _wrap(pa.table(data))


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
    else:
        # Default to parquet
        return "parquet"


@table.read_p.def_impl
def read_impl(interpreter: Interpreter, op: Operation) -> TableValue:
    """Read table from file.

    Supported formats: parquet, csv, json
    """
    import os

    path: str = op.attrs["path"]
    schema: elt.TableType = op.attrs["schema"]
    format_hint: str = op.attrs.get("format", "auto")
    fmt = _infer_format(path, format_hint)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not exists")

    pa_schema = _pa_schema(schema) if schema else None
    return _wrap(FileTableSource(path=path, format=fmt, schema=pa_schema))


class MultiTableReader(BatchReader):
    def __init__(self, readers: list[TableReader]) -> None:
        fields = {}
        for r in readers:
            for f in r.schema:
                if f.name in fields:
                    raise ValueError(f"Field name conflict. {f.name}")
                fields[f.name] = f

        self._readers = readers
        self._schema = pa.schema(list(fields.values()))

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def read_next_batch(self) -> pa.RecordBatch:
        num_rows = -1
        columns: list[pa.ChunkedArray] = []
        for idx, r in enumerate(self._readers):
            batch = r.read_next_batch()
            if num_rows == -1:
                num_rows = batch.num_rows
            elif num_rows != batch.num_rows:
                raise ValueError(
                    f"Batch {idx} has {batch.num_rows} rows, expected {num_rows}"
                )
            columns.extend(batch.columns)
        return pa.RecordBatch.from_arrays(columns, names=self._schema.names)

    def close(self) -> None:
        for r in self._readers:
            r.close()


@table.write_p.def_impl
def write_impl(interpreter: Interpreter, op: Operation, *tables: TableValue) -> None:
    """Write table to file.

    Supported formats: parquet, csv, json

    For LazyTable, performs streaming writes when supported.
    For regular Tables, performs direct writes.
    """
    import os

    path: str = op.attrs["path"]
    format_hint: str = op.attrs.get("format", "parquet")

    fmt = _infer_format(path, format_hint)

    batch_size = DEFAULT_BATCH_SIZE if len(tables) > 1 else -1
    readers: list[TableReader] = []
    for t in tables:
        data = t.unwrap()
        readers.append(
            data.open(batch_size)
            if isinstance(data, TableSource)
            else TableReader(data)
        )

    reader: BatchReader = readers[0] if len(readers) == 1 else MultiTableReader(readers)

    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pa_pq

    @runtime_checkable
    class BatchWriter(Protocol):
        def write_batch(self, batch: pa.RecordBatch) -> None: ...
        def close(self) -> None: ...

    class JsonWriter(BatchWriter):
        def __init__(self, path: str) -> None:
            self._path = path
            self._batches: list[pa.RecordBatch] = []

        def write_batch(self, batch: pa.RecordBatch) -> None:
            self._batches.append(batch)

        def close(self) -> None:
            # PyArrow doesn't have direct JSON write, convert to pandas
            tbl = pa.Table.from_batches(self._batches)
            df = tbl.to_pandas()
            df.to_json(self._path, orient="records", lines=True)

    def _safe_remove_file(path: str) -> None:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass  # Ignore cleanup errors

    try:
        match fmt:
            case "parquet":
                writer = pa_pq.ParquetWriter(path, reader.schema)
            case "csv":
                writer = pa_csv.CSVWriter(path, reader.schema)
            case "json":
                writer = JsonWriter(path)
            case _:
                raise ValueError(f"Unsupported format: {fmt}")
    except Exception as e:
        reader.close()
        _safe_remove_file(path)
        raise e

    try:
        for batch in reader:
            writer.write_batch(batch)
    except Exception as e:
        _safe_remove_file(path)
        raise e
    finally:
        reader.close()
        writer.close()
