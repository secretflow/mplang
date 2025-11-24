"""Table Runtime Implementation.

Implements execution logic for Table primitives using DuckDB and PyArrow.
"""

from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from mplang2.dialects import table
from mplang2.edsl.graph import Operation
from mplang2.edsl.interpreter import Interpreter


@table.run_sql_p.def_impl
def run_sql_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Execute SQL query on input tables."""
    query = op.attrs["query"]
    dialect = op.attrs.get("dialect", "duckdb")
    table_names = op.attrs["table_names"]

    if dialect != "duckdb":
        raise ValueError(f"Unsupported dialect: {dialect}")

    # Use in-memory DuckDB connection
    conn = duckdb.connect(":memory:")

    for name, arg in zip(table_names, args, strict=True):
        if isinstance(arg, (pa.Table, pd.DataFrame)):
            conn.register(name, arg)
        else:
            raise TypeError(
                f"Expected pyarrow.Table or pd.DataFrame for table '{name}', "
                f"got {type(arg)}"
            )

    # Execute query and fetch result as Arrow table
    try:
        res = conn.execute(query).arrow()
        return res
    except Exception as e:
        raise RuntimeError(f"Failed to execute SQL query: {query}") from e


@table.table2tensor_p.def_impl
def table2tensor_impl(interpreter: Interpreter, op: Operation, table_val: Any) -> Any:
    """Convert table to tensor (numpy array)."""
    if not isinstance(table_val, (pa.Table, pd.DataFrame)):
        raise TypeError(
            f"Expected pyarrow.Table or pd.DataFrame, got {type(table_val)}"
        )

    if isinstance(table_val, pa.Table):
        df = table_val.to_pandas()
    else:
        df = table_val

    # Convert to numpy array
    # Note: This assumes the table is homogeneous as enforced by abstract_eval
    return df.to_numpy()


@table.tensor2table_p.def_impl
def tensor2table_impl(interpreter: Interpreter, op: Operation, tensor_val: Any) -> Any:
    """Convert tensor (numpy array) to table."""
    column_names = op.attrs["column_names"]

    if not isinstance(tensor_val, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(tensor_val)}")

    if tensor_val.ndim != 2:
        raise ValueError(f"Expected 2D array, got {tensor_val.ndim}D")

    if tensor_val.shape[1] != len(column_names):
        raise ValueError(
            f"Shape mismatch: tensor has {tensor_val.shape[1]} columns, "
            f"but {len(column_names)} names provided"
        )

    # Create dictionary for DataFrame/Table creation
    data = {}
    for i, name in enumerate(column_names):
        data[name] = tensor_val[:, i]

    return pa.Table.from_pydict(data)


@table.constant_p.def_impl
def constant_impl(interpreter: Interpreter, op: Operation) -> Any:
    """Create constant table."""
    # data is stored in attrs by default bind if not TraceObject
    data = op.attrs["data"]

    # Handle pandas DataFrame if passed directly (though attrs usually store basic types)
    # If data was a DataFrame, it might have been stored as is if the IR supports it.
    # If data was a dict, it's fine.

    if isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data)

    return pa.Table.from_pydict(data)
