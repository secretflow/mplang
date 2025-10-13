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

import ibis
import numpy.testing as npt
import pandas as pd
import pytest

import mplang
import mplang as mp
from mplang.core.pfunc import PFunction
from mplang.core.table import TableType
from mplang.kernels.context import RuntimeContext
from mplang.ops import ibis_cc


class TestDuckDBKernel:
    def test_duckdb_run(self):
        runtime = RuntimeContext(rank=0, world_size=1)
        tbl_name = "table"
        schema = {"a": "int", "b": "int", "c": "float"}
        in_tbl = ibis.table(schema=schema, name=tbl_name)
        # Use explicit add to keep static analyzers happy (ibis Column supports + at runtime)
        result_expr = in_tbl["a"].add(in_tbl["b"])  # type: ignore[attr-defined]
        new_table = in_tbl.mutate(d=result_expr)
        pfn = ibis_cc.ibis2sql(new_table, [in_tbl.schema()], [tbl_name])
        # Build PFunction for generic sql run op (fn_type sql.run)
        assert isinstance(pfn, PFunction) and pfn.fn_type == "sql.run"
        # outs_info produced from ibis schema; sanity
        assert len(pfn.outs_info) == 1
        assert isinstance(pfn.outs_info[0], TableType)

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
        from mplang.kernels.value import TableValue

        (out_val,) = runtime.run_kernel(pfn, [TableValue(in_df)])
        assert isinstance(out_val, TableValue)
        out_df = out_val.to_pandas()
        npt.assert_allclose(out_df, expected, rtol=1e-7, atol=1e-8)

    @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    @pytest.mark.parametrize(
        "features,new_feature_name",
        [
            (["a", "b"], "r"),
            (["a", "b"], "a"),
        ],
    )
    def test_binary_op(self, op: str, features, new_feature_name):
        def _binary_op(t: ibis.Table, op: str) -> ibis.Table:
            match op:
                case "+":
                    result_expr = t[features[0]] + t[features[1]]
                case "-":
                    result_expr = t[features[0]] - t[features[1]]
                case "*":
                    result_expr = t[features[0]] * t[features[1]]
                case "/":
                    result_expr = t[features[0]] / t[features[1]]
                case _:
                    raise ValueError(f"Unsupported operation: {op}")

            res_t = t.mutate(**{new_feature_name: result_expr})
            return res_t

        def example():
            data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [4.1, 5.1, 6.1]})
            in_tbl = mp.constant(data)
            out_tbl = mp.run_ibis_at(0, _binary_op, in_tbl, op)

            return out_tbl

        sim2 = mplang.Simulator.simple(2)
        res = mplang.evaluate(sim2, example)
        print(f"output table: {op}, {new_feature_name}\n", mplang.fetch(sim2, res))

    def test_union(self):
        """
        Multiple inputs single output
        """

        def _union(t1: ibis.Table, t2: ibis.Table) -> ibis.Table:
            return t1.union(t2, distinct=False)

        def example():
            import pandas as pd

            df1 = pd.DataFrame({"f0": [0.1, 0.2], "f1": [1, 2]})
            df2 = pd.DataFrame({"f0": [0.3, 0.4], "f1": [3, 4]})
            t1 = mp.constant(df1)
            t2 = mp.constant(df2)
            res = mp.run_ibis_at(0, _union, t1, t2)
            return res

        sim2 = mplang.Simulator.simple(2)
        res = mplang.evaluate(sim2, example)
        out_tbl = mplang.fetch(sim2, res)
        print(f"output table: \n {out_tbl[0]}")
