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

import numpy.testing as npt
import pandas as pd

import mplang.v1 as mp
import mplang.v1.core as mp_core
from mplang.v1.kernels.context import RuntimeContext


class TestDuckDBKernel:
    def test_duckdb_run(self):
        runtime = RuntimeContext(rank=0, world_size=1)
        dialect = "duckdb"
        table_name = "_table_"
        in_schema = {"a": mp_core.INT32, "b": mp_core.INT32, "c": mp_core.FLOAT32}
        out_schema = in_schema.copy()
        out_schema["d"] = mp_core.INT32

        pfn = mp_core.PFunction(
            fn_type="sql.run",
            fn_text=f"SELECT a, b, c, a+b as d FROM {table_name}",
            ins_info=[mp.TableType.from_dict(in_schema)],
            outs_info=[mp.TableType.from_dict(out_schema)],
            dialect=dialect,
            in_names=[table_name],
        )

        in_df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7.1, 8.1, 9.1],
        })
        expected = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7.1, 8.1, 9.1],
            "d": [5, 7, 9],
        })
        from mplang.v1.kernels.value import TableValue

        (out_val,) = runtime.run_kernel(pfn, [TableValue(in_df)])
        assert isinstance(out_val, TableValue)
        out_df = out_val.to_pandas()
        npt.assert_allclose(out_df, expected, rtol=1e-7, atol=1e-8)
