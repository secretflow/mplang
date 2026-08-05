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
from typing import Any, ClassVar, Protocol, Self, cast, runtime_checkable

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

# Target rows per batch when rechunking inputs that must stream in lockstep
# (see write_impl: MultiTableReader zips one batch per reader, so non-aligned
# native chunk boundaries — e.g. Parquet row groups of different sizes — must
# be normalized to a common size before they can be combined column-wise).
DEFAULT_BATCH_SIZE = 1_000_000


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

        # Otherwise, combine all batches and return as a single RecordBatch.
        # ``from_batches`` keeps each input as a separate chunk, so without
        # ``combine_chunks`` the ``to_batches()[0]`` below would return only
        # the first chunk and silently drop the rest — truncating the output
        # the first time multiple native batches accumulate under one batch.
        combined = pa.Table.from_batches(batches).combine_chunks()
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

    @property
    def data(self) -> pa.Table:
        """Get the underlying PyArrow Table data."""
        return self._data

    # =========== Wrap/Unwrap ===========

    def _convert(self, data: Any) -> pa.Table:
        """Convert input data to pa.Table."""
        if isinstance(data, TableValue):
            return data.unwrap()
        if isinstance(data, pd.DataFrame):
            data = pa.Table.from_pandas(data)
        if not isinstance(data, pa.Table):
            raise TypeError(f"Expected pa.Table, got {type(data)}")
        return data

    # =========== Serialization ===========

    def to_json(self) -> dict[str, Any]:
        # Serialize using Arrow IPC streaming format (base64-encoded for JSON transport)
        data = self.data
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, data.schema) as writer:
            writer.write_table(data)
        ipc_bytes = sink.getvalue().to_pybytes()
        return {"ipc": base64.b64encode(ipc_bytes).decode("ascii")}

    def to_binary_json(self, ctx: serde.BinaryContext) -> dict[str, Any]:
        """Binary-aware serialization: stores Arrow IPC bytes as a raw segment.

        The IPC bytes are pushed directly into *ctx* without base64 encoding.
        """
        data = self.data
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, data.schema) as writer:
            writer.write_table(data)
        ipc_bytes = sink.getvalue().to_pybytes()
        seg_idx = ctx.add_segment(ipc_bytes)
        return {"ipc_bref": seg_idx}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TableValue:
        ipc_bytes = base64.b64decode(data["ipc"])
        reader = pa.ipc.open_stream(ipc_bytes)
        table = reader.read_all()
        return cls(table)

    @classmethod
    def from_binary_json(
        cls, data: dict[str, Any], segments: list[memoryview]
    ) -> TableValue:
        """Binary-aware deserialization: reads Arrow IPC bytes from raw segments."""
        if "ipc_bref" in data:
            ipc_bytes = segments[data["ipc_bref"]]
        else:
            # Fallback: base64-encoded "ipc" from to_json()
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
    """Execute SQL query on input tables.

    Inputs are materialized ``pa.Table`` values; they are registered with a
    fresh DuckDB connection and the query result is materialized back into a
    ``pa.Table`` immediately, so the returned ``TableValue`` never holds a live
    DuckDB relation (and is therefore picklable for ``store.save``).
    """
    query = op.attrs["query"]
    dialect = op.attrs.get("dialect", "duckdb")
    table_names = op.attrs["table_names"]

    if dialect != "duckdb":
        raise ValueError(f"Unsupported dialect: {dialect}")

    tables = [_unwrap(arg) for arg in args]

    try:
        with duckdb.connect() as conn:
            for name, tbl in zip(table_names, tables, strict=True):
                conn.register(name, tbl)
            result = conn.execute(query).fetch_arrow_table()
        return _wrap(result)
    except Exception as e:
        raise RuntimeError(f"Failed to execute SQL query: {query}") from e


@table.table2tensor_p.def_impl
def table2tensor_impl(interpreter: Interpreter, op: Operation, table_val: Any) -> Any:
    """Convert table to tensor (numpy array).

    Returns TensorValue if tensor_impl is available, otherwise raw np.ndarray.
    """
    from mplang.backends.tensor_impl import TensorValue
    from mplang.extend.arrow.vector import table_to_numpy

    tbl = _unwrap(table_val)

    # Validate row count before expensive table_to_numpy conversion
    expected_rows = op.attrs["number_rows"]
    if expected_rows >= 0 and tbl.num_rows != expected_rows:
        raise ValueError(
            f"table2tensor: expected {expected_rows} rows, "
            f"got {tbl.num_rows}. Either fix the input data or "
            f"use number_rows=-1 for dynamic row count."
        )

    arr = table_to_numpy(tbl)

    output_type = op.outputs[0].type
    if not isinstance(output_type, elt.TensorType) or not isinstance(
        output_type.element_type, elt.ScalarType
    ):
        raise TypeError(f"table2tensor: expected scalar TensorType, got {output_type}")

    from mplang.dialects import dtypes

    expected_dtype = dtypes.to_numpy(cast(elt.ScalarType, output_type.element_type))
    if arr.dtype != expected_dtype:
        raise TypeError(
            f"table2tensor: runtime dtype {arr.dtype} does not match "
            f"declared tensor dtype {expected_dtype}"
        )

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
    """Read table from file into a materialized ``pa.Table``.

    Supported formats: parquet, csv, json. The declared schema is enforced at
    read time (parse-time column types for CSV, post-read cast otherwise), so
    integral-looking values declared as ``f64`` parse straight to float64.
    """
    import os

    path: str = op.attrs["path"]
    schema: elt.TableType = op.attrs["schema"]
    format_hint: str = op.attrs.get("format", "auto")
    fmt = _infer_format(path, format_hint)

    def _do_read(resolved: str) -> TableValue:
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"{resolved} does not exist")
        pa_schema = _pa_schema(schema) if schema else None
        columns = pa_schema.names if pa_schema else None

        match fmt:
            case "parquet":
                import pyarrow.parquet as pq

                table = pq.read_table(resolved, columns=columns)
            case "csv":
                import pyarrow.csv as pa_csv

                table = pa_csv.read_csv(
                    resolved,
                    convert_options=pa_csv.ConvertOptions(
                        column_types=pa_schema,
                        include_columns=columns,
                    ),
                )
            case "json":
                import pyarrow.json as pa_json

                table = pa_json.read_json(resolved)
                if columns:
                    table = table.select(columns)
            case _:
                raise ValueError(f"Unsupported format: {fmt}")

        # Parquet/JSON enforce types only via post-read cast; CSV already used
        # parse-time column_types. Guard both against no-op casts.
        if pa_schema and table.schema != pa_schema:
            table = table.cast(pa_schema)
        return _wrap(table)

    if interpreter.store:
        local_path = os.path.abspath(os.path.join(".tmp", path))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        interpreter.store.download(path, local_path)
        return _do_read(local_path)
    return _do_read(path)


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

    def _do_write(resolved: str) -> None:
        # MultiTableReader zips one batch per reader, requiring matching row
        # counts per round. Eagerly read pa.Tables keep native chunk boundaries
        # (e.g. Parquet row groups), so two same-length inputs chunked as
        # [2,2,1] vs [3,2] would otherwise raise "Batch 1 has 3 rows, expected
        # 2". Normalizing to a common batch_size makes aligned batches (equal
        # total rows => equal full-batch counts and matching final partial
        # batch) without losing the genuine row-count mismatch error below.
        batch_size = DEFAULT_BATCH_SIZE if len(tables) > 1 else -1
        readers: list[TableReader] = []
        for t in tables:
            data = t.unwrap()  # always pa.Table now
            readers.append(TableReader(data, batch_size=batch_size))

        reader: BatchReader = (
            readers[0] if len(readers) == 1 else MultiTableReader(readers)
        )

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

        def _safe_remove_file(p: str) -> None:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError as e:
                    # Log the error for debugging purposes, but continue cleanup attempt
                    logger.warning("Failed to remove file %s: %s", p, e)

        try:
            match fmt:
                case "parquet":
                    writer = pa_pq.ParquetWriter(resolved, reader.schema)
                case "csv":
                    writer = pa_csv.CSVWriter(resolved, reader.schema)
                case "json":
                    writer = JsonWriter(resolved)
                case _:
                    raise ValueError(f"Unsupported format: {fmt}")
        except Exception as e:
            reader.close()
            _safe_remove_file(resolved)
            raise e

        try:
            for batch in reader:
                writer.write_batch(batch)
        except Exception as e:
            _safe_remove_file(resolved)
            raise e
        finally:
            reader.close()
            writer.close()

    if interpreter.store:
        local_path = os.path.abspath(os.path.join(".tmp", path))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _do_write(local_path)
        interpreter.store.upload(local_path, path)
    else:
        _do_write(path)
