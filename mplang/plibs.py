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

import ibis

from mplang.core import primitive as prim
from mplang.core.base import Mask, MPObject
from mplang.plib import ibis_fe


def prun_ibis(
    out_tbl_expr: ibis.Table, in_tbl: MPObject, pmask: Mask | None = None
) -> MPObject:
    assert "schema" in in_tbl.attrs
    in_schema: ibis.Schema = in_tbl.attrs["schema"]
    pfn = ibis_fe.compile(out_tbl_expr, in_schema)
    res = prim.peval(pfn, [in_tbl], pmask)
    assert len(res) == 1
    out = res[0]
    out.attrs["schema"] = out_tbl_expr.schema()
    return out
