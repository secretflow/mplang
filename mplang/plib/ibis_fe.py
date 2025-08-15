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


import json

import ibis
import numpy as np

from mplang.core import dtype
from mplang.core.base import TensorInfo
from mplang.core.pfunc import PFunction, PFuncTypes


def compile(expr: ibis.Table, in_schema: ibis.Schema) -> PFunction:
    out_schema = expr.schema()
    np_in_schema = in_schema.to_numpy()
    np_ot_schema = out_schema.to_numpy()
    np_in_types = [p[1] for p in np_in_schema]
    np_ot_types = [p[1] for p in np_ot_schema]
    ts_in_dt = dtype.from_numpy(np.result_type(*np_in_types))
    ts_ot_dt = dtype.from_numpy(np.result_type(*np_ot_types))

    ins_info = [TensorInfo(ts_in_dt, (-1, len(in_schema.fields)))]
    ots_info = [TensorInfo(ts_ot_dt, (-1, len(out_schema.fields)))]

    in_schema_attr = json.dumps([(p[0], p[1].name) for p in np_in_schema])
    ot_schema_attr = json.dumps([(p[0], p[1].name) for p in np_ot_schema])

    sql = ibis.to_sql(expr)
    pfn = PFunction(
        fn_type=PFuncTypes.IBIS_SQL,
        fn_name="run_sql",
        fn_text=sql,
        fn_body=None,
        ins_info=tuple(ins_info),
        outs_info=tuple(ots_info),
        attrs={"in_schema": in_schema_attr, "out_schema": ot_schema_attr},
    )
    return pfn
