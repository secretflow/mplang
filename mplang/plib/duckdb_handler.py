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

import json

from mplang.core.base import TensorLike
from mplang.core.pfunc import PFunction, PFunctionHandler, PFuncTypes


class DuckDBHandler(PFunctionHandler):
    def __init__(self) -> None:
        super().__init__()

    def setup(self) -> None: ...
    def teardown(self) -> None: ...
    def list_fn_names(self) -> list[str]:
        return [PFuncTypes.IBIS_SQL]

    def execute(self, pfunc: PFunction, args: list[TensorLike]) -> list[TensorLike]:
        if pfunc.fn_type == PFuncTypes.IBIS_SQL:
            return self.do_run(pfunc, args)
        else:
            raise ValueError(f"unsupported fn_type, {pfunc.fn_type}")

    def do_run(
        self,
        pfunc: PFunction,
        args: list[TensorLike],
    ) -> list[TensorLike]:
        import duckdb
        import numpy as np
        import pandas as pd

        if len(args) != 1:
            raise ValueError(f"len of args mismatch. len(args)=={len(args)}")
        if "in_schema" not in pfunc.attrs or "out_schema" not in pfunc.attrs:
            raise ValueError(
                f"cannot find in_schema or out_schema in attrs{list(pfunc.attrs.keys())}."
            )

        in_schema: list[tuple[str, str]] = json.loads(pfunc.attrs["in_schema"])
        out_schema: list[tuple[str, str]] = json.loads(pfunc.attrs["out_schema"])
        in_columns: list[str] = [pair[0] for pair in in_schema]

        arg0 = args[0]
        assert isinstance(arg0, np.ndarray)
        df = pd.DataFrame(arg0, columns=in_columns, copy=False)
        df = df.astype(dict(in_schema), copy=False)
        conn = duckdb.connect(":memory:")
        conn.register("table", df)
        result_df = conn.execute(pfunc.fn_text).fetchdf()
        if len(result_df.columns) != len(out_schema):
            raise ValueError(
                f"invalid output schema, {result_df.columns}, {out_schema}"
            )

        result_np = result_df.to_numpy(copy=False)
        return [result_np]
