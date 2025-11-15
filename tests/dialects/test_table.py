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

from mplang.dialects.table import run_sql, table_to_tensor, tensor_to_table
from mplang.edsl.interpreter import InterpObject
from mplang.edsl.tracer import trace
from mplang.edsl.typing import Table, Tensor, f32


def _sample_table() -> InterpObject:
    ttype = Table[{"value": Tensor[f32, ()]}]
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

    traced = trace(wrapper, table)
    graph = traced.graph
    assert len(graph.operations) == 1
    op = graph.operations[0]
    assert op.opcode == "table.run_sql"
    assert op.attrs["table_names"] == ["input_table"]


def test_table_to_tensor_and_back():
    table = _sample_table()
    tensor_type = Tensor[f32, ()]
    table_type = table.type

    def wrapper(tbl):
        tensor = table_to_tensor(tbl, column="value", out_type=tensor_type)
        return tensor_to_table(tensor, column="value", out_type=table_type)

    traced = trace(wrapper, table)
    graph = traced.graph
    assert len(graph.operations) == 2
    assert [op.opcode for op in graph.operations] == [
        "table.to_tensor",
        "table.from_tensor",
    ]
