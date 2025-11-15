"""Table dialect: table operations backed by plaintext/private SQL engines."""

from __future__ import annotations

from mplang.edsl.context import get_current_context
from mplang.edsl.primitive import Primitive
from mplang.edsl.tracer import TraceObject, Tracer
from mplang.edsl.typing import TableType, TensorType

run_sql_p = Primitive("table.run_sql")
table_to_tensor_p = Primitive("table.to_tensor")
tensor_to_table_p = Primitive("table.from_tensor")


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


@table_to_tensor_p.def_trace
def _table_to_tensor_trace(table: TraceObject, *, column: str, out_type: TensorType):
    tracer = _current_tracer()
    if not isinstance(table, TraceObject):
        raise TypeError("table_to_tensor input must be TraceObject")
    [value] = tracer.graph.add_op(
        opcode="table.to_tensor",
        inputs=[table._graph_value],
        output_types=[out_type],
        attrs={"column": column},
    )
    return TraceObject(value, tracer)


@tensor_to_table_p.def_trace
def _tensor_to_table_trace(
    tensor: TraceObject, *, column: str, out_type: TableType
) -> TraceObject:
    tracer = _current_tracer()
    if not isinstance(tensor, TraceObject):
        raise TypeError("tensor_to_table input must be TraceObject")
    [value] = tracer.graph.add_op(
        opcode="table.from_tensor",
        inputs=[tensor._graph_value],
        output_types=[out_type],
        attrs={"column": column},
    )
    return TraceObject(value, tracer)


def run_sql(
    query: str,
    *,
    out_type: TableType,
    dialect: str = "duckdb",
    **tables: TraceObject,
) -> TraceObject:
    """Trace a SQL query over plaintext/private tables."""

    return run_sql_p.bind(
        query,
        out_type=out_type,
        dialect=dialect,
        **tables,
    )


def table_to_tensor(
    table: TraceObject, *, column: str, out_type: TensorType
) -> TraceObject:
    """Project a single column from a table into a tensor."""

    return table_to_tensor_p.bind(table, column=column, out_type=out_type)


def tensor_to_table(
    tensor: TraceObject, *, column: str, out_type: TableType
) -> TraceObject:
    """Wrap a tensor column into a table."""

    return tensor_to_table_p.bind(tensor, column=column, out_type=out_type)


__all__ = [
    "run_sql",
    "run_sql_p",
    "table_to_tensor",
    "table_to_tensor_p",
    "tensor_to_table",
    "tensor_to_table_p",
]
