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

from mplang.v1.core import PFunction
from mplang.v1.kernels.base import kernel_def
from mplang.v1.kernels.value import TableValue


@kernel_def("duckdb.run_sql")
def _duckdb_sql(pfunc: PFunction, *args: TableValue) -> TableValue:
    import duckdb

    # TODO: maybe we could translate the sql to duckdb dialect
    # instead of raising an exception
    if pfunc.attrs.get("dialect") != "duckdb":
        raise ValueError("duckdb.run_sql must have dialect=duckdb attr")

    conn = duckdb.connect(":memory:")
    if args:
        in_names = pfunc.attrs.get("in_names")
        if in_names is None:
            raise ValueError("duckdb sql missing in_names attr")
        for arg, name in zip(args, in_names, strict=True):
            # Use Arrow directly for zero-copy data transfer
            arrow_table = arg.to_arrow()
            conn.register(name, arrow_table)
    # Fetch result as Arrow table for consistency
    if pfunc.fn_text is None:
        raise ValueError("SQL function text is None")
    res_arrow = conn.execute(pfunc.fn_text).fetch_arrow_table()
    return TableValue(res_arrow)
