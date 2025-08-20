#!/usr/bin/env python3
"""
Example demonstrating TableLike constant support in mplang.

This example shows how to use pandas DataFrames as constants in multi-party computations.
"""

import pandas as pd

from mplang.core.context_mgr import with_ctx
from mplang.core.mask import Mask
from mplang.core.primitive import constant
from mplang.core.trace import TraceContext, trace


def example_dataframe_constant():
    """Example using a pandas DataFrame as a constant."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 28],
        "salary": [50000.0, 60000.0, 70000.0, 55000.0],
    })

    print("Original DataFrame:")
    print(df)
    print(f"DataFrame dtypes:\n{df.dtypes}")

    # Create a trace context for 2-party computation
    mask = Mask(3)  # 0b11 for 2 parties
    trace_context = TraceContext(world_size=2, mask=mask)

    # Define a function that uses the DataFrame as a constant
    def use_dataframe_constant():
        """Function that creates a constant from a DataFrame."""
        # NOTE: constant primitive is not designed for large tables - use dedicated
        # table loading mechanisms for substantial datasets
        table_const = constant(df)
        return table_const

    # Trace the function
    with with_ctx(trace_context):
        traced_fn = trace(trace_context, use_dataframe_constant)

    # Examine the result
    print(f"\nTraced function outputs: {len(traced_fn.out_vars)}")
    result = traced_fn.out_vars[0]
    print(f"Result type: {type(result)}")
    print(f"Result mptype: {result.mptype}")
    print(f"Table schema: {result.mptype._type}")

    # Check the expression
    from mplang.core.table import TableType
    from mplang.expr.ast import ConstExpr

    print(f"Expression type: {type(result.expr)}")
    print(f"Is ConstExpr: {isinstance(result.expr, ConstExpr)}")
    print(f"Expression data type: {type(result.expr.typ)}")
    print(f"Is TableType: {isinstance(result.expr.typ, TableType)}")

    # Verify the JSON serialization
    import json

    json_data = json.loads(result.expr.data_bytes.decode("utf-8"))
    print(f"\nSerialized data (first record): {json_data[0]}")
    print(f"Number of records: {len(json_data)}")

    return traced_fn


def example_empty_dataframe():
    """Example using an empty DataFrame to demonstrate schema handling."""
    # Create an empty DataFrame with defined schema
    df = pd.DataFrame(columns=["user_id", "username", "is_active"])
    df = df.astype({"user_id": "int64", "username": "string", "is_active": "bool"})

    print("Empty DataFrame schema:")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")

    # Create trace context
    mask = Mask(3)
    trace_context = TraceContext(world_size=2, mask=mask)

    def use_empty_dataframe():
        return constant(df)

    with with_ctx(trace_context):
        traced_fn = trace(trace_context, use_empty_dataframe)

    result = traced_fn.out_vars[0]
    print(f"\nEmpty DataFrame table schema: {result.mptype._type}")

    # Verify empty serialization
    import json

    json_data = json.loads(result.expr.data_bytes.decode("utf-8"))
    print(f"Serialized empty data: {json_data}")


if __name__ == "__main__":
    print("=== Example: DataFrame as Constant ===")
    example_dataframe_constant()

    print("\n" + "=" * 50)
    print("=== Example: Empty DataFrame Schema ===")
    example_empty_dataframe()
