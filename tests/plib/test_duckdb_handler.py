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
import numpy as np
import numpy.testing as npt
import pytest

import mplang
from mplang import plibs, simp
from mplang.plib import ibis_fe
from mplang.plib.duckdb_handler import DuckDBHandler


class TestDuckDBHandler:
    def test_duckdb_run(self):
        # ibis_fe
        in_tbl = ibis.table(schema={"a": "int", "b": "int", "c": "float"}, name="table")
        result_expr = in_tbl["a"] + in_tbl["b"]
        new_table = in_tbl.mutate(d=result_expr)
        pfn = ibis_fe.compile(new_table, in_tbl.schema())

        # duckdb run
        dh = DuckDBHandler()

        in_tbl = np.array([
            [1, 4, 7.1],
            [2, 5, 8.1],
            [3, 6, 9.1],
        ])
        expected_tbl = np.array([
            [1, 4, 7.1, 5],
            [2, 5, 8.1, 7],
            [3, 6, 9.1, 9],
        ])
        ot_tbls = dh.execute(pfn, [in_tbl])
        assert len(ot_tbls) == 1
        ot_tbl = ot_tbls[0]

        npt.assert_allclose(ot_tbl, expected_tbl, rtol=1e-7, atol=1e-8)

    @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    @pytest.mark.parametrize(
        "features,new_feature_name",
        [
            (["a", "b"], "r"),
            (["a", "b"], "a"),
        ],
    )
    def test_binary_op(self, op: str, features, new_feature_name):
        def example():
            t = ibis.table(schema={"a": "int", "b": "int", "c": "float"}, name="table")

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

            new_table = t.mutate(**{new_feature_name: result_expr})
            in_tbl_data = np.array([[1, 2, 3], [4, 5, 6], [4.0, 5.0, 6.0]])
            in_tbl = simp.constant(in_tbl_data)
            in_tbl.attrs["schema"] = ibis.schema({"a": "int", "b": "int", "c": "float"})

            out_tbl = plibs.prun_ibis(new_table, in_tbl)
            return out_tbl

        sim2 = mplang.Simulator(2)
        res = mplang.evaluate(sim2, example)
        print(f"output table: {op}, {new_feature_name}\n", mplang.fetch(sim2, res))
