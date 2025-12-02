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
import pytest

import mplang.v1 as mp
from mplang.v1.core.dtypes import FLOAT32, INT32
from mplang.v1.core.mpobject import MPContext, MPObject
from mplang.v1.core.mptype import MPType
from mplang.v1.ops.sql_cc import run_sql
from mplang.v1.runtime.simulation import Simulator


@pytest.fixture
def sim():
    return Simulator.simple(1)


def test_run_sql2_happy_path_with_out_type(sim):
    # Prepare input table as MPObject using simp API constant, within a function context
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7.1, 8.1, 9.1]})
    out_schema = {"a": INT32, "b": INT32, "c": FLOAT32, "d": INT32}

    @mp.function
    def fn():
        tbl = mp.constant(df)
        return mp.run_at(
            0,
            run_sql,
            "SELECT a, b, c, a+b as d FROM t",
            out_type=mp.TableType.from_dict(out_schema),
            t=tbl,
        )

    res = mp.evaluate(sim, fn)
    out_val = mp.fetch(sim, res)
    # fetch returns per-party list; single-party index 0
    out_df = out_val[0].to_pandas() if hasattr(out_val[0], "to_pandas") else out_val[0]

    expected = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7.1, 8.1, 9.1],
        "d": [5, 7, 9],
    })
    pd.testing.assert_frame_equal(out_df.reset_index(drop=True), expected)


def test_run_sql2_schema_deduction(sim):
    # Build a lightweight dummy MPObject carrying only table schema
    class DummyTable(MPObject):
        def __init__(self, schema: dict[str, mp.DType]):
            self._mptype = MPType.table(mp.TableType.from_dict(schema))

        @property
        def mptype(self) -> MPType:
            return self._mptype

        @property
        def ctx(self) -> MPContext:  # not used in this test path
            return None  # type: ignore[return-value]

    schema = {"a": INT32, "b": INT32, "c": FLOAT32}
    tbl = DummyTable(schema)

    # No out_type provided; should deduce: a, b, c, d=int32 (a+b)
    pfunc, _inputs, _ = run_sql("SELECT a, b, c, a+b as d FROM t", t=tbl)
    assert pfunc.fn_type == "sql.run"
    assert pfunc.attrs.get("dialect") == "duckdb"
    assert pfunc.attrs.get("in_names") == ("t",)

    out = pfunc.outs_info[0]
    assert isinstance(out, mp.TableType)
    cols = dict(out.columns)
    assert list(cols.keys()) == ["a", "b", "c", "d"]
    # c is FLOAT32 inferred from input; d should be INT32 by simple numeric promotion
    assert cols["a"] == INT32
    assert cols["b"] == INT32
    assert cols["c"] == FLOAT32
    assert cols["d"] == INT32


def test_run_sql2_function_type_inference_via_optimizer(sim):
    class DummyTable(MPObject):
        def __init__(self, schema: dict[str, mp.DType]):
            self._mptype = MPType.table(mp.TableType.from_dict(schema))

        @property
        def mptype(self) -> MPType:
            return self._mptype

        @property
        def ctx(self) -> MPContext:  # not used in this test path
            return None  # type: ignore[return-value]

    tbl = DummyTable({"a": INT32})

    # Function call should be supported via optimizer-based type inference
    pfunc, _inputs, _ = run_sql("SELECT ABS(a) FROM t", t=tbl)
    out = pfunc.outs_info[0]
    assert isinstance(out, mp.TableType)
    cols = dict(out.columns)
    # Column name may be auto-generated (expr_0); verify single int32 column
    assert list(cols.values()) == [INT32]
