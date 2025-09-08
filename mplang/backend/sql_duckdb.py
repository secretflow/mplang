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

from mplang.core.pfunc import PFunction, TableHandler
from mplang.core.table import TableLike


class DuckDBHandler(TableHandler):
    def __init__(self) -> None:
        super().__init__()

    def setup(self, rank: int) -> None: ...
    def teardown(self) -> None: ...
    def list_fn_names(self) -> list[str]:
        return ["sql[duckdb]"]

    def execute(self, pfunc: PFunction, args: list[TableLike]) -> list[TableLike]:
        if pfunc.fn_type == "sql[duckdb]":
            return self.do_run(pfunc, args)
        else:
            raise ValueError(f"unsupported fn_type, {pfunc.fn_type}")

    def do_run(
        self,
        pfunc: PFunction,
        args: list[TableLike],
    ) -> list[TableLike]:
        import duckdb
        import pandas as pd

        conn = duckdb.connect(":memory:")

        if len(args) > 0:
            assert "in_names" in pfunc.attrs, (
                f"cannot find in_names in attrs while having {len(args)} inputs"
            )
            in_names = pfunc.attrs["in_names"]
            # register input tables
            for arg, name in zip(args, in_names, strict=True):
                # assert isinstance(arg, pd.DataFrame)
                if isinstance(arg, pd.DataFrame):
                    df = arg
                elif isinstance(arg, list):
                    # const df, only for test
                    df = pd.DataFrame.from_records(arg)
                else:
                    raise ValueError(f"unsupport type, {type(arg)}")
                conn.register(name, df)

        res_df = conn.execute(pfunc.fn_text).fetchdf()
        return [res_df]
