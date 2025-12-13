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

import numpy as np

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects.table import run_sql, table2tensor, tensor2table
from mplang.v2.runtime.interpreter import InterpObject


def _sample_table() -> InterpObject:
    ttype = elt.Table[{"value": elt.Tensor[elt.f32, ()]}]
    data = np.array([(1.0,)], dtype=[("value", np.float64)])
    return InterpObject(data, ttype)


def test_table_run_sql_op_emitted():
    table = _sample_table()

    def wrapper(tbl):
        return run_sql(
            "SELECT value FROM input_table",
            out_type=tbl.type,
            input_table=tbl,
        )

    traced = el.trace(wrapper, table)
    graph = traced.graph
    assert len(graph.operations) == 1
    op = graph.operations[0]
    assert op.opcode == "table.run_sql"
    assert op.attrs["table_names"] == ["input_table"]


def test_table_to_tensor_and_back():
    table = _sample_table()

    def wrapper(tbl):
        tensor = table2tensor(tbl, number_rows=1)
        return tensor2table(tensor, column_names=["value"])

    traced = el.trace(wrapper, table)
    graph = traced.graph
    assert len(graph.operations) == 2
    assert [op.opcode for op in graph.operations] == [
        "table.table2tensor",
        "table.tensor2table",
    ]
