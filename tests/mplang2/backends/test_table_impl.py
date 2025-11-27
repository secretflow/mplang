"""Tests for Table Runtime Implementation."""

import numpy as np
import pyarrow as pa

import mplang2.backends.table_impl  # noqa: F401
import mplang2.edsl.typing as elt
from mplang2.dialects import table


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
    # result should be an object wrapping the runtime value (pyarrow Table)
    res_table = result.runtime_obj
    assert isinstance(res_table, pa.Table)

    df = res_table.to_pandas()
    np.testing.assert_array_equal(df["a"], [1, 2, 3])
    np.testing.assert_array_equal(df["b"], [8.0, 10.0, 12.0])


def test_table_constant_dataframe():
    """Test creating constant table from DataFrame."""
    import pandas as pd

    df_in = pd.DataFrame({"x": [10, 20], "y": ["foo", "bar"]})
    t = table.constant(df_in)

    assert isinstance(t.runtime_obj, pa.Table)
    df_out = t.runtime_obj.to_pandas()
    pd.testing.assert_frame_equal(df_in, df_out)
