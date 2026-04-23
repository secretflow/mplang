# Copyright 2026 Ant Group Co., Ltd.
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

"""Tests for TableType row dimension fixes (table_row_dimension.md).

Layer 1: Runtime row count validation in table2tensor_impl.
Layer 2: Support number_rows=-1 for dynamic row dimensions.
Layer 3: TableType.nrows field and serde.
"""

import numpy as np
import pytest

import mplang.edsl as el
import mplang.edsl.typing as elt
from mplang.dialects.table import table2tensor, tensor2table
from mplang.edsl.typing import TableType
from mplang.runtime.interpreter import InterpObject


def _sample_table(schema: dict[str, elt.BaseType] | None = None) -> InterpObject:
    if schema is None:
        schema = {"a": elt.f64}
    ttype = elt.TableType(schema)
    data = np.array([(1.0,)], dtype=[("a", np.float64)])
    return InterpObject(data, ttype)  # type: ignore


# =============================================================================
# Layer 1: Runtime row count validation
# =============================================================================


class TestTable2TensorRuntimeValidation:
    """table2tensor_impl rejects mismatched row counts."""

    def test_row_count_match(self):
        """Runtime accepts matching row count (existing behavior preserved)."""
        import pyarrow as pa

        import mplang.backends.table_impl  # noqa: F401 - register impl

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        from mplang.dialects import table

        def workload():
            tbl = table.constant(data)
            return table.table2tensor(tbl, number_rows=3)

        result = workload()
        arr = result.runtime_obj.unwrap() if hasattr(result, "runtime_obj") else result
        assert arr.shape[0] == 3

    def test_row_count_mismatch(self):
        """Runtime rejects mismatched row count."""
        import mplang.backends.table_impl  # noqa: F401 - register impl

        from mplang.dialects import table

        data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]}

        def workload():
            tbl = table.constant(data)
            return table.table2tensor(tbl, number_rows=3)

        with pytest.raises(ValueError, match="expected 3 rows, got 5"):
            workload()


# =============================================================================
# Layer 2: Dynamic row dimension (number_rows=-1)
# =============================================================================


class TestTable2TensorDynamicRows:
    """Support number_rows=-1 for dynamic row dimensions."""

    def test_dynamic_rows_abstract_eval(self):
        """number_rows=-1 produces TensorType with dynamic first dim."""
        from mplang.dialects.table import _table2tensor_ae

        result = _table2tensor_ae(elt.TableType({"a": elt.f64}), number_rows=-1)
        assert result == elt.TensorType(elt.f64, (-1, 1))

    def test_dynamic_rows_trace(self):
        """number_rows=-1 traces correctly into IR."""
        table = _sample_table()

        def wrapper(tbl):
            return table2tensor(tbl, number_rows=-1)

        traced = el.trace(wrapper, table)
        graph = traced.graph
        assert len(graph.operations) == 1
        op = graph.operations[0]
        assert op.opcode == "table.table2tensor"
        assert op.attrs["number_rows"] == -1
        # Output type should have dynamic first dim
        out_type = op.outputs[0].type
        assert isinstance(out_type, elt.TensorType)
        assert out_type.shape[0] == -1

    def test_dynamic_rows_impl(self):
        """Dynamic mode accepts any row count at runtime."""
        import mplang.backends.table_impl  # noqa: F401 - register impl

        from mplang.dialects import table

        data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}

        def workload():
            tbl = table.constant(data)
            return table.table2tensor(tbl, number_rows=-1)

        result = workload()
        arr = result.runtime_obj.unwrap() if hasattr(result, "runtime_obj") else result
        assert arr.shape[0] == 7

    def test_rejects_negative_below_minus_one(self):
        """number_rows < -1 raises ValueError."""
        from mplang.dialects.table import _table2tensor_ae

        with pytest.raises(ValueError, match="number_rows must be >= -1"):
            _table2tensor_ae(elt.TableType({"a": elt.f64}), number_rows=-2)

    def test_rejects_zero(self):
        """number_rows == 0 raises ValueError."""
        from mplang.dialects.table import _table2tensor_ae

        with pytest.raises(ValueError, match="number_rows must be != 0"):
            _table2tensor_ae(elt.TableType({"a": elt.f64}), number_rows=0)

    def test_static_rows_still_works(self):
        """Positive number_rows still works as before."""
        from mplang.dialects.table import _table2tensor_ae

        result = _table2tensor_ae(
            elt.TableType({"a": elt.f64, "b": elt.f64}), number_rows=10
        )
        assert result == elt.TensorType(elt.f64, (10, 2))


# =============================================================================
# Layer 3: TableType.nrows
# =============================================================================


class TestTableTypeNrows:
    """TableType carries optional nrows metadata."""

    def test_nrows_default(self):
        t = TableType({"a": elt.i32})
        assert t.nrows == -1
        assert t.is_dynamic
        assert not t.is_static

    def test_nrows_static(self):
        t = TableType({"a": elt.i32}, nrows=100)
        assert t.nrows == 100
        assert t.is_static
        assert not t.is_dynamic

    def test_equality_with_nrows(self):
        """Different nrows means different types."""
        assert TableType({"a": elt.i32}, nrows=10) != TableType({"a": elt.i32})
        assert TableType({"a": elt.i32}, nrows=10) == TableType(
            {"a": elt.i32}, nrows=10
        )

    def test_equality_without_nrows(self):
        """Default nrows types are equal."""
        assert TableType({"a": elt.i32}) == TableType({"a": elt.i32})

    def test_hash_with_nrows(self):
        """Different nrows produces different hash."""
        s1 = {TableType({"a": elt.i32}, nrows=10)}
        s2 = {TableType({"a": elt.i32})}
        assert s1 != s2

    def test_str_with_nrows(self):
        t = TableType({"f1": elt.f64, "f2": elt.f64}, nrows=100)
        s = str(t)
        assert "nrows=100" in s

    def test_str_without_nrows(self):
        t = TableType({"f1": elt.f64})
        s = str(t)
        assert "nrows" not in s

    def test_serde_roundtrip(self):
        """Serde roundtrip preserves nrows."""
        t = TableType({"a": elt.f64, "b": elt.i32}, nrows=50)
        j = t.to_json()
        t2 = TableType.from_json(j)
        assert t == t2
        assert t2.nrows == 50

    def test_serde_roundtrip_default_nrows(self):
        """Serde roundtrip works with default nrows."""
        t = TableType({"a": elt.f64})
        j = t.to_json()
        t2 = TableType.from_json(j)
        assert t == t2
        assert t2.nrows == -1

    def test_serde_compat_old_json(self):
        """Old JSON without nrows deserializes with nrows=-1."""
        from mplang.edsl import serde

        j = {"schema": {"a": serde.to_json(elt.f64)}}
        t = TableType.from_json(j)
        assert t.nrows == -1

    def test_serde_json_omits_default_nrows(self):
        """to_json omits nrows when it's the default -1."""
        t = TableType({"a": elt.f64})
        j = t.to_json()
        assert "nrows" not in j

    def test_serde_json_includes_static_nrows(self):
        """to_json includes nrows when it's set."""
        t = TableType({"a": elt.f64}, nrows=42)
        j = t.to_json()
        assert j["nrows"] == 42


class TestTable2TensorAutoNrows:
    """table2tensor infers number_rows from TableType.nrows when not provided."""

    def test_auto_nrows_from_table_type(self):
        """When number_rows is None and TableType has static nrows, use it."""
        schema = {"a": elt.f64}
        ttype = elt.TableType(schema, nrows=5)
        data = np.array([(1.0,)], dtype=[("a", np.float64)])
        table_obj = InterpObject(data, ttype)  # type: ignore

        def wrapper(tbl):
            return table2tensor(tbl)

        traced = el.trace(wrapper, table_obj)
        graph = traced.graph
        op = graph.operations[0]
        assert op.attrs["number_rows"] == 5

    def test_auto_nrows_dynamic_fallback(self):
        """When number_rows is None and TableType has no nrows, fallback to -1."""
        table_obj = _sample_table()

        def wrapper(tbl):
            return table2tensor(tbl)

        traced = el.trace(wrapper, table_obj)
        graph = traced.graph
        op = graph.operations[0]
        assert op.attrs["number_rows"] == -1

    def test_explicit_nrows_overrides(self):
        """Explicit number_rows overrides TableType.nrows."""
        schema = {"a": elt.f64}
        ttype = elt.TableType(schema, nrows=5)
        data = np.array([(1.0,)], dtype=[("a", np.float64)])
        table_obj = InterpObject(data, ttype)  # type: ignore

        def wrapper(tbl):
            return table2tensor(tbl, number_rows=10)

        traced = el.trace(wrapper, table_obj)
        graph = traced.graph
        op = graph.operations[0]
        assert op.attrs["number_rows"] == 10
