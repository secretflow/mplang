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

from typing import Any

import sqlglot as sg
from jax.tree_util import PyTreeDef, tree_flatten
from sqlglot import exp as sge
from sqlglot.optimizer import annotate_types as opt_annot
from sqlglot.optimizer import qualify as opt_qualify

from mplang.v1.core import MPObject, PFunction, TableType
from mplang.v1.core.dtypes import (
    BINARY,
    BOOL,
    DATE,
    DECIMAL,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    INTERVAL,
    JSON,
    STRING,
    TIME,
    TIMESTAMP,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    UUID,
    DType,
)
from mplang.v1.ops.base import stateless_mod

_SQL_MOD = stateless_mod("sql")


# Static dtype mappings (MPLang <-> SQL)
MP_TO_SQL_TYPE: dict[DType, str] = {
    # Floats
    FLOAT64: "DOUBLE",
    FLOAT32: "FLOAT",
    # Signed ints
    INT8: "TINYINT",
    INT16: "SMALLINT",
    INT32: "INT",
    INT64: "BIGINT",
    # Unsigned ints (portable approximations)
    UINT8: "SMALLINT",
    UINT16: "INT",
    UINT32: "BIGINT",
    UINT64: "DECIMAL(38)",
    # Booleans & strings
    BOOL: "BOOLEAN",
    STRING: "VARCHAR",
    # Dates / times
    DATE: "DATE",
    TIME: "TIME",
    TIMESTAMP: "TIMESTAMP",
    # Other table types
    DECIMAL: "DECIMAL",
    JSON: "JSON",
    BINARY: "BLOB",
    UUID: "UUID",
    INTERVAL: "INTERVAL",
}

SQL_TYPE_TO_MP: dict[str, DType] = {
    # Floats
    "double": FLOAT64,
    "double precision": FLOAT64,
    "float": FLOAT32,
    "real": FLOAT32,
    # Signed ints
    "bigint": INT64,
    "long": INT64,
    "int": INT32,
    "integer": INT32,
    "int4": INT32,
    "smallint": INT16,
    "int2": INT16,
    "tinyint": INT8,
    "int1": INT8,
    # Unsigned (rare in SQL)
    "uint8": UINT8,
    "ubyte": UINT8,
    "uint16": UINT16,
    "uint32": UINT32,
    "uint64": UINT64,
    # Booleans / strings
    "bool": BOOL,
    "boolean": BOOL,
    "char": STRING,
    "varchar": STRING,
    "text": STRING,
    "string": STRING,
    # Dates / times
    "date": DATE,
    "time": TIME,
    "timestamp": TIMESTAMP,
    # Decimal / numeric
    "decimal": DECIMAL,
    "numeric": DECIMAL,
    # Others
    "json": JSON,
    "binary": BINARY,
    "varbinary": BINARY,
    "blob": BINARY,
    "uuid": UUID,
    "interval": INTERVAL,
}


def _deduce_out_schema(
    parsed: sge.Expression,
    dialect: str,
    in_schemas: dict[str, TableType],
) -> TableType:
    """Deduce output schema using sqlglot's qualify + annotate_types.

    This implementation leverages sqlglot's optimizer to resolve table/column
    references (including star expansion) and annotate expression types. It then
    maps sqlglot DataType to mplang DType and returns a TableType.
    """

    # 1) Build sqlglot schema from MPObject/TableType inputs
    def _dtype_to_sql(dt: DType) -> str:
        return MP_TO_SQL_TYPE.get(dt, "VARCHAR")

    sqlglot_schema: dict[str, dict[str, str]] = {
        tname: {col: _dtype_to_sql(dt) for col, dt in schema.columns}
        for tname, schema in in_schemas.items()
    }

    # 2) Parse with read dialect; 3) Qualify (resolve names, expand star); 4) Annotate types
    qualified = opt_qualify.qualify(parsed, schema=sqlglot_schema, dialect=dialect)
    typed = opt_annot.annotate_types(qualified, schema=sqlglot_schema)

    # 5) Extract projection names and types
    select = typed if isinstance(typed, sge.Select) else typed.find(sge.Select)
    if select is None:
        raise NotImplementedError(
            "Only SELECT queries are supported for schema deduction"
        )

    def _sqlglot_type_to_dtype(tobj: Any) -> DType:
        ts = str(tobj).lower().replace(" with time zone", "").strip()
        base = ts.split("(", 1)[0].strip()
        return SQL_TYPE_TO_MP.get(base, STRING)

    pairs: list[tuple[str, DType]] = []
    idx = 0
    used: set[str] = set()
    for proj in select.expressions:
        name = getattr(proj, "alias_or_name", None) or getattr(proj, "name", None)
        if not name:
            name = f"expr_{idx}"
            idx += 1
        t = getattr(proj, "type", None)
        if t is None:
            raise NotImplementedError(
                "Cannot infer type for projection; please provide out_type explicitly"
            )
        dtype = _sqlglot_type_to_dtype(t)
        if name in used:
            raise ValueError(
                f"Duplicate output column name '{name}' after qualification"
            )
        used.add(name)
        pairs.append((name, dtype))

    return TableType.from_pairs(pairs)


@_SQL_MOD.op_def()
def run_sql(
    query: str,
    *,
    out_type: TableType | None = None,
    dialect: str = "duckdb",
    **in_tables: Any,
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Build a sql.run PFunction from a SQL query with optional schema deduction.

    API: run_sql(query: str, *, out_type: TableType | None = None, dialect: str = "duckdb", **in_tables) -> (PFunction, [MPObject], PyTreeDef)

    Semantics:
    - Parses the SQL and binds only the tables that are actually referenced in the query by name.
    - If ``out_type`` is not provided, attempts to deduce the output table schema using sqlglot (qualify + annotate types).
    - Returns a triad consisting of the constructed PFunction (``fn_type='sql.run'``), the ordered list of input MPObjects, and the output PyTreeDef.

    Difference vs ``run_sql_raw``: this op can infer ``out_type`` and will parse the SQL to filter inputs; ``run_sql_raw`` requires an explicit ``out_type`` and does not parse/filter inputs.
    """
    # Extract required table names from SQL (order by first appearance)
    parsed = sg.parse_one(query, read=dialect)
    required_names: list[str] = []
    for t in parsed.find_all(sge.Table):
        # Prefer .name; fallback to str(this) if needed
        tname = getattr(t, "name", None) or str(t.this)
        if tname not in required_names:
            required_names.append(tname)

    # Disallow extras not referenced by the query to avoid surprises
    extra = set(in_tables.keys()) - set(required_names)
    if extra:
        raise ValueError(
            f"Unexpected tables provided that are not referenced in SQL: {sorted(extra)}"
        )

    # Validate required tables and require MPObject for runtime registration
    in_names: list[str] = []
    ins_info: list[TableType] = []
    in_vars: list[MPObject] = []
    for name in required_names:
        if name not in in_tables:
            raise KeyError(f"Missing required table '{name}' for SQL query")
        obj = in_tables[name]
        if not isinstance(obj, MPObject):
            raise TypeError(
                f"Table '{name}' must be an MPObject (for runtime registration), got {type(obj).__name__}"
            )
        assert obj.schema is not None, f"Input table '{name}' missing schema"
        in_vars.append(obj)
        ins_info.append(obj.schema)
        in_names.append(name)

    if out_type is None:
        in_schemas: dict[str, TableType] = {
            n: in_tables[n].schema for n in required_names
        }
        out_type = _deduce_out_schema(parsed, dialect, in_schemas)

    pfn = PFunction(
        fn_type="sql.run",
        ins_info=tuple(ins_info),
        outs_info=(out_type,),
        fn_name="",
        fn_text=query,
        in_names=tuple(in_names),
        dialect=dialect,
    )
    _, treedef = tree_flatten(out_type)
    return pfn, in_vars, treedef


@_SQL_MOD.op_def()
def run_sql_raw(
    query: str,
    out_type: TableType,
    *,
    dialect: str = "duckdb",
    in_tables: dict[str, MPObject] | None = None,
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Build a sql.run PFunction from a SQL query with an explicit output schema.

    API: run_sql_raw(query: str, out_type: TableType, *, dialect: str = "duckdb", in_tables: dict[str, MPObject] | None = None) -> (PFunction, [MPObject], PyTreeDef)

    Semantics:
    - Does not parse the SQL; carries all tables provided via ``in_tables`` in the mapping's iteration order.
    - Requires an explicit ``out_type``; no schema deduction is attempted.
    - Returns a triad consisting of the constructed PFunction (``fn_type='sql.run'``), the ordered list of input MPObjects, and the output PyTreeDef.

    Difference vs ``run_sql``: this op requires ``out_type`` and does not parse/filter inputs; ``run_sql`` can infer ``out_type`` and selects only tables referenced by the query.
    """

    # Collect inputs strictly as provided by caller
    in_names: list[str] = []
    ins_info: list[TableType] = []
    in_vars: list[MPObject] = []
    if in_tables:
        for name, tbl in in_tables.items():
            if not isinstance(tbl, MPObject):
                raise TypeError(f"Input table '{name}' is not an MPObject {type(tbl)}")
            assert tbl.schema is not None, f"Input table '{name}' is missing a schema"
            in_names.append(name)
            ins_info.append(tbl.schema)
            in_vars.append(tbl)

    pfn = PFunction(
        fn_type="sql.run",
        fn_name="",
        fn_text=query,
        ins_info=tuple(ins_info),
        outs_info=(out_type,),
        in_names=tuple(in_names),
        dialect=dialect,
    )
    _, treedef = tree_flatten(out_type)
    return pfn, in_vars, treedef
