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

import pandas as pd

import mplang
import mplang.v1 as mp
from mplang.v1.core.dtypes import INT32
from mplang.v1.core.table import TableType
from mplang.v1.ops import sql_cc


def test_sqlrun():
    # test without input
    sql = "select a from 'a.csv'"
    out_type = TableType.from_pairs([("a", INT32)])
    pfn, input_args, _output_tree = sql_cc.run_sql_raw(sql, out_type)
    assert pfn.fn_text == sql
    assert pfn.outs_info[0] == out_type
    assert len(input_args) == 0

    # test with inputs
    sim2 = mplang.Simulator.simple(2)
    mplang.set_ctx(sim2)

    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [4.1, 5.1, 6.1]})
    in_tbl = mp.constant(data)

    sql = "select a from table"
    out_type = TableType.from_pairs([("a", INT32)])
    in_tables = {"table": in_tbl}
    pfn, input_args, _output_tree = sql_cc.run_sql_raw(
        sql, out_type, in_tables=in_tables
    )
    assert pfn.fn_text == sql
    assert pfn.outs_info[0] == out_type
    assert len(input_args) == 1
