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

from mplang.frontend import ibis_cc


def test_ibis2sql():
    table_name = "table"
    in_tbl = ibis.table(schema={"a": "int", "b": "int", "c": "float"}, name=table_name)
    result_expr = in_tbl["a"] + in_tbl["b"]
    new_table = in_tbl.mutate(d=result_expr)
    pfn = ibis_cc.ibis2sql(new_table, [in_tbl.schema()], [table_name])
    assert pfn.fn_text is not None
    assert "in_names" in pfn.attrs
    assert len(pfn.ins_info) == 1 and len(pfn.outs_info) == 1
    assert len(pfn.ins_info[0].columns) == 3
    assert len(pfn.outs_info[0].columns) == 4
