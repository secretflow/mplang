# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Table Runtime Implementation."""

import numpy as np
import pyarrow as pa

import mplang.v2.backends.table_impl  # noqa: F401
import mplang.v2.edsl.typing as elt
from mplang.v2.backends.table_impl import TableValue
from mplang.v2.dialects import table


def _get_table(val) -> pa.Table:
    """Extract pa.Table from various wrapper types."""
    if hasattr(val, "runtime_obj"):
        val = val.runtime_obj
    if isinstance(val, TableValue):
        return val.unwrap()
    return val


def test_table_ops_e2e():
    """Test basic table operations (constant, run_sql, conversions) end-to-end."""

    def workload():
        # Create constant table
        data = {
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
        }
        t1 = table.constant(data)

        # Run SQL
        # Output type must be specified for run_sql
        # table2tensor requires homogeneous types, so we cast 'a' to double
        out_schema = elt.TableType({
            "a": elt.TensorType(elt.f64, ()),
            "b": elt.TensorType(elt.f64, ()),
        })

        t2 = table.run_sql(
            "SELECT CAST(a AS DOUBLE) AS a, b * 2 AS b FROM t1",
            out_type=out_schema,
            t1=t1,
        )

        # Table to Tensor
        # t2 has 3 rows.
        tensor_val = table.table2tensor(t2, number_rows=3)

        # Tensor to Table
        t3 = table.tensor2table(tensor_val, column_names=["a", "b"])

        return t3

    # Execute
    result = workload()

    # Verify
    # result should be an object wrapping the runtime value (TableValue -> pyarrow Table)
    res_table = _get_table(result)
    assert isinstance(res_table, pa.Table)

    df = res_table.to_pandas()
    np.testing.assert_array_equal(df["a"], [1, 2, 3])
    np.testing.assert_array_equal(df["b"], [8.0, 10.0, 12.0])


def test_table_constant_dataframe():
    """Test creating constant table from DataFrame."""
    import pandas as pd

    df_in = pd.DataFrame({"x": [10, 20], "y": ["foo", "bar"]})
    t = table.constant(df_in)

    res_table = _get_table(t)
    assert isinstance(res_table, pa.Table)
    df_out = res_table.to_pandas()
    pd.testing.assert_frame_equal(df_in, df_out)
