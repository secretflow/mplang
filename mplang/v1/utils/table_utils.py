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

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.orc as pa_orc
import pyarrow.parquet as pa_pq

from mplang.v1.core.table import TableLike

__all__ = ["decode_table", "encode_table", "read_table", "write_table"]


def _parse_kwargs(kwargs: dict[str, Any], keys: list[str]) -> dict[str, Any] | None:
    if not kwargs:
        return None

    return {key: kwargs[key] for key in keys if key in kwargs}


_csv_read_option_keys = [
    "skip_rows",
    "skip_rows_after_names",
    "column_names",
    "autogenerate_column_names",
    "encoding",
]
_csv_parse_option_keys = [
    "delimiter",
    "quote_char",
    "double_quote",
    "escape_char",
    "newlines_in_values",
    "ignore_empty_lines",
]
_csv_convert_option_keys = [
    "check_utf8",
    "column_types",
    "null_values",
    "true_values",
    "false_values",
    "decimal_point",
    "strings_can_be_null",
    "quoted_strings_can_be_null",
    "include_columns",
    "include_missing_columns",
    "auto_dict_encode",
    "auto_dict_max_cardinality",
    "timestamp_parsers",
]


def read_table(
    source: Any,
    format: str = "parquet",
    columns: list[str] | None = None,
    **kwargs: Any,
) -> pa.Table:
    """Read data from a file and return a PyArrow table.

    Args:
        source: The source to read data from (file path, file-like object, etc.)
        format: The format of the data source ("parquet", "csv", or "orc")
        columns: List of column names to read (None means all columns)
        **kwargs: Additional keyword arguments passed to the underlying reader

    Returns:
        A PyArrow Table containing the data from the source

    Raises:
        ValueError: If an unsupported format is specified
    """
    match format:
        case "csv":
            if columns:
                kwargs["include_columns"] = columns
            read_args = _parse_kwargs(kwargs, _csv_read_option_keys)
            parse_args = _parse_kwargs(kwargs, _csv_parse_option_keys)
            convert_args = _parse_kwargs(kwargs, _csv_convert_option_keys)

            read_opts = pa_csv.ReadOptions(**read_args) if read_args else None
            parse_opts = pa_csv.ParseOptions(**parse_args) if parse_args else None
            conv_opts = pa_csv.ConvertOptions(**convert_args) if convert_args else None
            return pa_csv.read_csv(
                source,
                read_options=read_opts,
                parse_options=parse_opts,
                convert_options=conv_opts,
            )
        case "orc":
            return pa_orc.read_table(source, columns=columns, **kwargs)
        case "parquet":
            return pa_pq.read_table(source, columns=columns, **kwargs)
        case _:
            raise ValueError(f"unsupported data format. {format}")


def write_table(
    data: TableLike,
    where: Any,
    format: str = "parquet",
    **kwargs: Any,
) -> None:
    """Write a table-like object to a file in the specified format.

    Args:
        data: The table-like object to write (PyArrow Table or other compatible format)
        where: The destination to write to (file path, file-like object, etc.)
        format: The format to write the data in ("parquet", "csv", or "orc")
        **kwargs: Additional keyword arguments passed to the underlying writer

    Returns:
        None

    Raises:
        ValueError: If the table has no columns or an unsupported format is specified
    """
    # Convert data to PyArrow Table if needed
    table = data if isinstance(data, pa.Table) else pa.table(data)
    if len(table.column_names) == 0:
        raise ValueError("Cannot convert Table with no columns.")

    match format:
        case "csv":
            options = pa_csv.WriteOptions(**kwargs) if kwargs else None
            pa_csv.write_csv(table, where, write_options=options)
        case "orc":
            pa_orc.write_table(table, where, **kwargs)
        case "parquet":
            pa_pq.write_table(table, where, **kwargs)
        case _:
            raise ValueError(f"unsupported data format. {format}")


def decode_table(
    data: bytes,
    format: str = "parquet",
    columns: list[str] | None = None,
    **kwargs: Any,
) -> pa.Table:
    """Decode a bytes object into a PyArrow table.

    Args:
        data: The bytes object containing the encoded table data
        format: The format of the encoded data ("parquet", "csv", or "orc")
        columns: List of column names to decode (None means all columns)
        **kwargs: Additional keyword arguments passed to the underlying reader

    Returns:
        A PyArrow Table decoded from the bytes data
    """
    buffer = io.BytesIO(data)
    return read_table(buffer, format=format, columns=columns, **kwargs)


def encode_table(data: TableLike, format: str = "parquet", **kwargs: Any) -> bytes:
    """Encode a table-like object into bytes.

    Args:
        data: The table-like object to encode (PyArrow Table or other compatible format)
        format: The format to encode the data in ("parquet", "csv", or "orc")
        **kwargs: Additional keyword arguments passed to the underlying writer

    Returns:
        Bytes object containing the encoded table data
    """
    buffer = io.BytesIO()
    write_table(data, buffer, format, **kwargs)
    return buffer.getvalue()
