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

from __future__ import annotations

from typing import Any

from mplang.backend.base import kernel_def
from mplang.core.pfunc import PFunction


@kernel_def("sql[duckdb]")
def _duckdb_sql(pfunc: PFunction, *args: Any) -> Any:
    import duckdb
    import pandas as pd

    conn = duckdb.connect(":memory:")
    if args:
        in_names = pfunc.attrs.get("in_names")
        if in_names is None:
            raise ValueError("duckdb sql missing in_names attr")
        for arg, name in zip(args, in_names, strict=True):
            if isinstance(arg, pd.DataFrame):
                df = arg
            elif isinstance(arg, list):  # const list-of-dict for tests
                df = pd.DataFrame.from_records(arg)
            else:
                raise ValueError(f"unsupported duckdb input type {type(arg)}")
            conn.register(name, df)
    res_df = conn.execute(pfunc.fn_text).fetchdf()
    return res_df
