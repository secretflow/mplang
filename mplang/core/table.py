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

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mplang.core.dtype import DType

__all__ = ["TableLike", "TableType", "dataframe_to_table_constant"]


@runtime_checkable
class TableLike(Protocol):
    """
    Protocol for objects structurally resembling tables from common libraries
    (pandas DataFrame, pyarrow Table, etc.), focusing on dtypes and columns attributes.
    """

    @property
    def dtypes(self) -> Any: ...

    @property
    def columns(self) -> Any: ...


@dataclass(frozen=True)
class TableType:
    """Table schema: ordered list of column name-type pairs.

    Represents table structure in relational algebra, containing column names
    and their corresponding data types.

    Examples:
        >>> schema = TableType.from_dict({
        ...     "id": DType.i64(),
        ...     "name": DType.string(),
        ... })
        >>> schema = TableType((("id", DType.i64()), ("name", DType.string())))
    """

    columns: tuple[tuple[str, DType], ...]
    _column_map: dict[str, DType] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the table schema."""
        if not self.columns:
            raise ValueError("TableType cannot be empty")

        # Validate column name uniqueness
        names = [name for name, _ in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("Column names must be unique")

        # Validate column names are non-empty
        for name, dtype in self.columns:
            if not name or not isinstance(name, str):
                raise ValueError("Column names must be non-empty strings")
            if not isinstance(dtype, DType):
                raise ValueError(f"Column type must be DType, got {type(dtype)}")

        # Create column name to type mapping for O(1) lookups
        object.__setattr__(self, "_column_map", dict(self.columns))

    @classmethod
    def from_dict(cls, schema_dict: dict[str, DType]) -> TableType:
        """Create table schema from dictionary.

        Args:
            schema_dict: Mapping from column names to data types

        Returns:
            TableType instance
        """
        return cls(tuple(schema_dict.items()))

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, DType]]) -> TableType:
        """Create table schema from list of name-type pairs.

        Args:
            pairs: List of tuples containing column name and data type

        Returns:
            TableType instance
        """
        return cls(tuple(pairs))

    def column_names(self) -> tuple[str, ...]:
        """Get all column names."""
        return tuple(name for name, _ in self.columns)

    def column_types(self) -> tuple[DType, ...]:
        """Get all column data types."""
        return tuple(dtype for _, dtype in self.columns)

    def get_column_type(self, name: str) -> DType:
        """Get data type by column name.

        Args:
            name: Column name

        Returns:
            Corresponding data type

        Raises:
            KeyError: If column name does not exist
        """
        try:
            return self._column_map[name]
        except KeyError:
            raise KeyError(f"Column '{name}' not found in schema") from None

    def has_column(self, name: str) -> bool:
        """Check if contains specified column name.

        Args:
            name: Column name

        Returns:
            True if contains the column, False otherwise
        """
        return name in self.column_names()

    def num_columns(self) -> int:
        """Get number of columns."""
        return len(self.columns)

    def to_dict(self) -> dict[str, DType]:
        """Convert to dictionary form."""
        return dict(self.columns)

    def __repr__(self) -> str:
        """String representation."""
        cols = ", ".join(f"{name}:{dtype.short_name()}" for name, dtype in self.columns)
        return f"TableType<{cols}>"

    def __len__(self) -> int:
        """Get number of columns."""
        return len(self.columns)

    def __iter__(self) -> Iterator[tuple[str, DType]]:
        """Support iteration."""
        return iter(self.columns)

    def __getitem__(self, index: int | str) -> tuple[str, DType] | DType:
        """Support index access.

        Args:
            index: Integer index or column name

        Returns:
            If integer index, returns (column name, data type) tuple
            If column name, returns corresponding data type
        """
        if isinstance(index, int):
            return self.columns[index]
        elif isinstance(index, str):
            return self.get_column_type(index)
        else:
            raise TypeError(f"Index must be int or str, got {type(index)}")


def dataframe_to_table_constant(data) -> tuple[TableType, bytes]:
    """Convert pandas DataFrame to TableType and JSON serialized bytes.

    This helper function converts a pandas DataFrame to the format needed for
    table constants in multi-party computations.

    Args:
        data: pandas DataFrame to convert

    Returns:
        tuple: (TableType schema, JSON serialized bytes)

    Raises:
        ImportError: If pandas is not available
        TypeError: If data is not a pandas DataFrame

    Note:
        This function uses JSON serialization which is not designed for large tables.
        Consider using dedicated table loading mechanisms for substantial datasets.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame constant support") from None

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data)}")

    if len(data.columns) == 0:
        raise ValueError("Cannot create a table constant from a DataFrame with no columns.")

    # Convert DataFrame to JSON for serialization
    json_str = data.to_json(orient="records")
    json_bytes = json_str.encode("utf-8")

    # Create table type from DataFrame schema
    from mplang.core.dtype import STRING, DType

    schema_dict = {}
    for col_name in data.columns:
        pandas_dtype = data[col_name].dtype
        # Convert pandas dtype to DType
        if pandas_dtype.kind in ("O", "U", "S"):  # object, unicode, string
            schema_dict[col_name] = (
                DType.from_numpy(pandas_dtype) if pandas_dtype.kind != "O" else STRING
            )
        else:
            schema_dict[col_name] = DType.from_numpy(pandas_dtype)

    table_type = TableType.from_dict(schema_dict)
    return table_type, json_bytes
