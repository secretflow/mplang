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
import pandas as pd
import pytest

from mplang.v1.core.cluster import (
    ClusterSpec,
    Device,
    Node,
    RuntimeInfo,
)
from mplang.v1.core.dtypes import DType
from mplang.v1.core.mpobject import MPContext, MPObject
from mplang.v1.core.mptype import MPType
from mplang.v1.core.table import TableType
from mplang.v1.core.tensor import TensorType
from mplang.v1.ops.basic import table_to_tensor, tensor_to_table


class DummyContext(MPContext):
    def __init__(self):
        # Build minimal single-node, single-device spec
        runtime = RuntimeInfo(version="dev", platform="local", op_bindings={})
        node = Node(name="p0", rank=0, endpoint="local", runtime_info=runtime)
        device = Device(name="p0_local", kind="ppu", members=[node])
        spec = ClusterSpec(nodes={node.name: node}, devices={device.name: device})
        super().__init__(spec)


class TableMP(MPObject):
    def __init__(self, df: pd.DataFrame):
        schema = TableType.from_pairs([
            (c, DType.from_any(str(df[c].dtype))) for c in df.columns
        ])
        self._mptype = MPType.table(schema)
        self._ctx = DummyContext()
        self._df = df

    @property
    def mptype(self) -> MPType:
        return self._mptype

    @property
    def ctx(self) -> MPContext:
        return self._ctx

    # runtime payload (not part of public MPObject interface, but backend expects DataFrame)
    def runtime_obj(self):
        return self._df


class TensorMP(MPObject):
    def __init__(self, arr: np.ndarray, dtype: DType):
        self._mptype = MPType.tensor(dtype, tuple(arr.shape))
        self._ctx = DummyContext()
        self._arr = arr

    @property
    def mptype(self) -> MPType:
        return self._mptype

    @property
    def ctx(self) -> MPContext:
        return self._ctx

    def runtime_obj(self):
        return self._arr


def test_table_to_tensor_happy_path_all_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    table_obj = TableMP(df)
    pfunc, _inputs, _treedef = table_to_tensor(
        table_obj,
        number_rows=len(df),
    )
    assert pfunc.fn_type == "basic.table_to_tensor"
    out_type = pfunc.outs_info[0]
    assert isinstance(out_type, TensorType)
    assert out_type.shape == (3, 2)


def test_table_to_tensor_hetero_dtype_failure():
    df = pd.DataFrame({"x": [1, 2], "y": [1.1, 2.2]})
    table_obj = TableMP(df)
    with pytest.raises(TypeError):
        table_to_tensor(
            table_obj,
            number_rows=len(df),
        )


def test_table_to_tensor_homogeneous_dtype_infers():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    table_obj = TableMP(df)
    pfunc, _, _ = table_to_tensor(
        table_obj,
        number_rows=len(df),
    )
    out_type = pfunc.outs_info[0]
    assert isinstance(out_type, TensorType)
    assert out_type.dtype == DType.from_any("int64")


def test_table_to_tensor_string_numeric_hetero_failure():
    df = pd.DataFrame({"s": ["1", "2"], "x": [1, 2]})
    table_obj = TableMP(df)
    with pytest.raises(TypeError):
        table_to_tensor(
            table_obj,
            number_rows=len(df),
        )


def test_table_to_tensor_no_subset_projection():
    # Ensure op packs all columns (cannot subset); create 3-column table
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    table_obj = TableMP(df)
    pfunc, _, _ = table_to_tensor(table_obj, number_rows=len(df))
    out_type = pfunc.outs_info[0]
    assert isinstance(out_type, TensorType)
    assert out_type.shape == (2, 3)


def test_table_to_tensor_hetero_numeric_failure():
    df = pd.DataFrame({
        "f64": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
        "i32": pd.Series([1, 2, 3], dtype="int32"),
    })
    table_obj = TableMP(df)
    with pytest.raises(TypeError):
        table_to_tensor(table_obj, number_rows=len(df))


def test_tensor_to_table_round_trip_schema():
    arr = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int64)
    tensor_obj = TensorMP(arr, DType.from_any("int64"))
    pfunc_tbl, _inputs, _treedef = tensor_to_table(
        tensor_obj,
        column_names=["a", "b"],
    )
    assert pfunc_tbl.fn_type == "basic.tensor_to_table"
    out_type = pfunc_tbl.outs_info[0]
    assert isinstance(out_type, TableType)
    assert out_type.has_column("a") and out_type.has_column("b")


def test_tensor_to_table_rank_error():
    arr = np.array([1, 2, 3], dtype=np.int64)  # rank 1
    tensor_obj = TensorMP(arr, DType.from_any("int64"))
    with pytest.raises(TypeError):
        tensor_to_table(tensor_obj, column_names=["c"])  # rank mismatch


def test_tensor_to_table_missing_column_names_param():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor_obj = TensorMP(arr, DType.from_any("int64"))
    # Python will raise TypeError because column_names is now required positional argument
    with pytest.raises(TypeError):  # signature enforcement
        tensor_to_table(tensor_obj)  # type: ignore[call-arg]


def test_table_to_tensor_negative_rows():
    df = pd.DataFrame({"a": [1]})
    table_obj = TableMP(df)
    with pytest.raises(ValueError):
        table_to_tensor(table_obj, number_rows=-1)


def test_tensor_to_table_duplicate_names_failure():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor_obj = TensorMP(arr, DType.from_any("int64"))
    with pytest.raises(ValueError):
        tensor_to_table(tensor_obj, column_names=["a", "a"])  # duplicate


def test_tensor_to_table_empty_name_failure():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor_obj = TensorMP(arr, DType.from_any("int64"))
    with pytest.raises(ValueError):
        tensor_to_table(tensor_obj, column_names=["", "b"])  # empty name


def test_tensor_to_table_whitespace_name_failure():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor_obj = TensorMP(arr, DType.from_any("int64"))
    with pytest.raises(ValueError):
        tensor_to_table(tensor_obj, column_names=["  ", "b"])  # whitespace only
