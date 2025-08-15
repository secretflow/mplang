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

import mplang.plib.ibis_fe as ibis_fe
from mplang.core import dtype


def test_ibis_compile():
    in_tbl = ibis.table(schema={"a": "int", "b": "int", "c": "float"}, name="table")
    result_expr = in_tbl["a"] + in_tbl["b"]
    new_table = in_tbl.mutate(d=result_expr)
    pfn = ibis_fe.compile(new_table, in_tbl.schema())
    assert pfn.fn_text is not None
    assert "in_schema" in pfn.attrs
    assert "out_schema" in pfn.attrs
    assert pfn.ins_info[0].dtype == dtype.FLOAT64
    assert pfn.outs_info[0].dtype == dtype.FLOAT64
