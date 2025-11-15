"""Table dialect: table operations backed by plaintext/private SQL engines."""

from __future__ import annotations

from mplang.edsl.context import get_current_context
from mplang.edsl.primitive import Primitive
from mplang.edsl.tracer import TraceObject, Tracer
from mplang.edsl.typing import BaseType, TableType, TensorType

run_sql_p = Primitive("table.run_sql")
table2tensor_p = Primitive("table.table2tensor")
tensor2table_p = Primitive("table.tensor2table")


def _current_tracer() -> Tracer:
    ctx = get_current_context()
    if not isinstance(ctx, Tracer):
        raise TypeError(f"Expected Tracer context, got {type(ctx)}")
    return ctx


@run_sql_p.def_trace
def _run_sql_trace(
    query: str,
    *,
    out_type: TableType,
    dialect: str = "duckdb",
    **tables: TraceObject,
) -> TraceObject:
    tracer = _current_tracer()
    if not isinstance(out_type, TableType):
        raise TypeError("run_sql out_type must be TableType")
    if not tables:
        raise ValueError("run_sql requires at least one table input")

    ordered = list(tables.items())
    inputs = []
    names = []
    for name, table in ordered:
        if not isinstance(table, TraceObject):
            raise TypeError(f"Table '{name}' must be TraceObject")
        inputs.append(table._graph_value)
        names.append(name)

    [value] = tracer.graph.add_op(
        opcode="table.run_sql",
        inputs=inputs,
        output_types=[out_type],
        attrs={"query": query, "dialect": dialect, "table_names": names},
    )
    return TraceObject(value, tracer)


@table2tensor_p.def_abstract_eval
def _table2tensor_ae(table_t: TableType, *, number_rows: int) -> TensorType:
    """Infer tensor type for table.table2tensor."""

    if not isinstance(number_rows, int):
        raise TypeError("number_rows must be an int")
    if number_rows < 0:
        raise ValueError("number_rows must be >= 0")
    if not table_t.schema:
        raise ValueError("Cannot convert empty table to tensor")
    column_types = list(table_t.schema.values())
    first = column_types[0]

    def _scalar_dtype(col: BaseType) -> BaseType:
        if hasattr(col, "element_type"):
            tensor_col = col  # type: ignore[assignment]
            if tensor_col.shape not in ((), None):
                raise TypeError(
                    "table2tensor expects scalar columns (rank-0 TensorType)"
                )
            return tensor_col.element_type
        return col

    first_scalar = _scalar_dtype(first)
    for col in column_types[1:]:
        if _scalar_dtype(col) != first_scalar:
            raise TypeError("All table columns must share the same scalar dtype")
    if not isinstance(first_scalar, BaseType):
        raise TypeError("All table columns must share the same dtype for table2tensor")
    return TensorType(first_scalar, (number_rows, len(column_types)))


@tensor2table_p.def_abstract_eval
def _tensor2table_ae(tensor_t: TensorType, *, column_names: list[str]) -> TableType:
    """Infer table type for table.tensor2table."""

    if len(tensor_t.shape) != 2:
        raise TypeError("tensor2table expects rank-2 tensor (N, F)")
    n_cols = tensor_t.shape[1]
    if not column_names:
        raise ValueError("column_names must be provided")
    if len(column_names) != n_cols:
        raise ValueError("column_names length must match tensor second dimension")
    seen: set[str] = set()
    schema: dict[str, BaseType] = {}
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
        schema[name] = TensorType(tensor_t.element_type, ())
    # Each column shares the tensor dtype; treat scalar leaves per row.
    return TableType(schema)


def run_sql(
    query: str,
    *,
    out_type: TableType,
    dialect: str = "duckdb",
    **tables: TraceObject,
) -> TraceObject:
    """Trace a SQL query over plaintext/private tables.

    Inserts a `table.run_sql` op with the provided query string and table inputs.
    The `out_type` describes the resulting table schema (columns + types).
    """

    return run_sql_p.bind(
        query,
        out_type=out_type,
        dialect=dialect,
        **tables,
    )


def table2tensor(table: TraceObject, *, number_rows: int) -> TraceObject:
    """Convert a homogeneous table into a dense tensor."""

    return table2tensor_p.bind(table, number_rows=number_rows)


def tensor2table(tensor: TraceObject, *, column_names: list[str]) -> TraceObject:
    """Convert a rank-2 tensor (N, F) into a table with named columns."""

    return tensor2table_p.bind(tensor, column_names=column_names)


__all__ = [
    "run_sql",
    "run_sql_p",
    "table2tensor",
    "table2tensor_p",
    "tensor2table",
    "tensor2table_p",
]
