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


import inspect
from collections.abc import Callable
from typing import Any

import ibis
from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core import dtype
from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction
from mplang.core.table import TableType
from mplang.frontend.base import FeOperation, stateless_mod
from mplang.utils.func_utils import normalize_fn


def ibis2sql(
    expr: ibis.Table,
    in_schemas: list[ibis.Schema],
    in_names: list[str],
    fn_name: str = "",
) -> PFunction:
    """
    Compile a ibis expr to sql and return the PFunction.

    Args:
        expr: ibis expr.
        in_schemas: the input table schemas
        in_names: the input table names, If there is only one table, it is usually defaulted to "table"
    Return:
        PFunction: The compiled PFunction
    """
    assert len(in_schemas) == len(in_names), (
        f"length of input table names and schemas mismatch. {len(in_schemas)}!={len(in_names)}"
    )

    def _convert(s: ibis.Schema) -> TableType:
        return TableType.from_pairs([
            (name, dtype.from_numpy(dt.to_numpy())) for name, dt in s.fields.items()
        ])

    ins_info = [_convert(s) for s in in_schemas]
    outs_info = [_convert(expr.schema())]

    sql = ibis.to_sql(expr, dialect="duckdb")
    pfn = PFunction(
        fn_type="sql[duckdb]",
        fn_name=fn_name,
        fn_text=sql,
        ins_info=tuple(ins_info),
        outs_info=tuple(outs_info),
        in_names=tuple(in_names),
    )
    return pfn


def is_ibis_function(func: Callable) -> bool:
    """
    Verify whether a function is an ibis function.
    The func signature should like def foo(t0:ibis.Table, t1:ibis.Table)->ibis.Table
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return False

    ret_anno = sig.return_annotation
    if ret_anno is ibis.Table:
        return True

    for param in sig.parameters.values():
        par_anno = param.annotation
        if par_anno is ibis.Table:
            return True

    return False


_IBIS_MOD = stateless_mod("ibis")


class IbisCompiler(FeOperation):
    """Ibis compiler frontend operation."""

    def trace(
        self, func: Callable, *args: Any, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Compile an Ibis function to SQL format.

        Args:
            func: The Ibis function to compile
            *args: Positional arguments to the function
            **kwargs: Keyword arguments to the function

        Returns:
            tuple[PFunction, list[MPObject], Any]: The compiled PFunction, input variables, and output tree
        """

        def is_variable(arg: Any) -> bool:
            return isinstance(arg, MPObject)

        normalized_fn, in_vars = normalize_fn(func, args, kwargs, is_variable)

        in_args, in_schemas, in_names = [], [], []
        idx = 0
        for arg in in_vars:
            columns = [(p[0], p[1].to_numpy()) for p in arg.schema.columns]
            schema = ibis.schema(columns)
            name = f"table{idx}"
            table = ibis.table(schema=schema, name=name)
            in_args.append(table)
            in_schemas.append(schema)
            in_names.append(name)
            idx += 1

        result = normalized_fn(in_args)
        assert isinstance(result, ibis.Table)
        pfunc = ibis2sql(result, in_schemas, in_names, func.__name__)
        _, treedef = tree_flatten(result)
        return pfunc, in_vars, treedef


ibis_compile = IbisCompiler(_IBIS_MOD, "compile")
