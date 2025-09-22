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

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction
from mplang.core.table import TableType
from mplang.frontend.base import FeOperation, stateless_mod

_SQL_MOD = stateless_mod("sql")


class SqlFE(FeOperation):
    def __init__(self, dialect: str = "duckdb"):
        # Bind to sql module with a stable op name for registry/dispatch
        super().__init__(_SQL_MOD, "run")
        self._dialect = dialect

    def trace(
        self,
        sql: str,
        out_type: TableType,
        in_tables: dict[str, MPObject] | None = None,
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        in_names: list[str] = []
        ins_info: list[TableType] = []
        in_vars: list[MPObject] = []
        if in_tables:
            for name, tbl in in_tables.items():
                assert isinstance(tbl, MPObject)
                assert tbl.schema is not None
                in_names.append(name)
                ins_info.append(tbl.schema)
                in_vars.append(tbl)

        pfn = PFunction(
            fn_type=f"sql[{self._dialect}]",
            fn_name="",
            fn_text=sql,
            ins_info=tuple(ins_info),
            outs_info=(out_type,),
            in_names=tuple(in_names),
        )
        _, treedef = tree_flatten(out_type)
        return pfn, in_vars, treedef


sql_run = SqlFE("duckdb")
