"""Table dialect: table operations backed by plaintext/private SQL engines."""

from __future__ import annotations

from typing import Any

import mplang2.edsl as el
import mplang2.edsl.typing as elt

run_sql_p: el.Primitive[Any] = el.Primitive("table.run_sql")
table2tensor_p: el.Primitive[el.Object] = el.Primitive("table.table2tensor")
tensor2table_p: el.Primitive[el.Object] = el.Primitive("table.tensor2table")
constant_p: el.Primitive[el.Object] = el.Primitive("table.constant")


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
        schema[name] = elt.TensorType(tensor_t.element_type, ())
    # Each column shares the tensor dtype; treat scalar leaves per row.
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
def _constant_ae(data: dict[str, list]) -> elt.TableType:
    """Infer table type for constant data.

    Args:
        data: Dictionary mapping column names to lists of values,
              or any data convertible to pandas DataFrame

    Returns:
        TableType inferred from DataFrame schema

    Raises:
        TypeError: If data cannot be converted to DataFrame
    """
    import pandas as pd

    # Unified pandas bridge for all data types
    df = pd.DataFrame(data)

    # Infer schema from pandas dtypes
    schema: dict[str, elt.BaseType] = {}
    for col_name in df.columns:
        pd_dtype = df[col_name].dtype

        # Map pandas dtypes to EDSL scalar types
        if pd_dtype == "bool":
            col_type: elt.TensorType = elt.TensorType(elt.i8, ())
        elif pd_dtype in ("int64", "Int64"):
            col_type = elt.TensorType(elt.i64, ())
        elif pd_dtype in ("int32", "Int32"):
            col_type = elt.TensorType(elt.i32, ())
        elif pd_dtype in ("float64", "Float64"):
            col_type = elt.TensorType(elt.f64, ())
        elif pd_dtype in ("float32", "Float32"):
            col_type = elt.TensorType(elt.f32, ())
        elif pd_dtype == "object" or pd_dtype.name.startswith("string"):
            # String columns - use i64 as placeholder
            col_type = elt.TensorType(elt.i64, ())
        else:
            raise TypeError(
                f"Unsupported pandas dtype for column '{col_name}': {pd_dtype}"
            )

        schema[str(col_name)] = col_type

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


__all__ = [
    "constant",
    "constant_p",
    "run_sql",
    "run_sql_p",
    "table2tensor",
    "table2tensor_p",
    "tensor2table",
    "tensor2table_p",
]
