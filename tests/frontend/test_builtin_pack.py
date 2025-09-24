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

import numpy as np
import pandas as pd
import pytest

import mplang
import mplang.simp as simp
from mplang.core.dtype import UINT8, DType
from mplang.core.table import TableType
from mplang.core.tensor import TensorType
from mplang.frontend import builtin
from mplang.utils import table_utils


@pytest.mark.integration
def test_builtin_pack_unpack_tensor_runtime() -> None:
    sim = mplang.Simulator.simple(1)
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    tensor_ty = TensorType.from_obj(arr)

    @mplang.function
    def fn():
        x = simp.runAt(0, lambda: np.arange(6, dtype=np.int32).reshape(2, 3))()
        packed = simp.runAt(0, builtin.pack)(x)
        unpacked = simp.runAt(0, builtin.unpack)(packed, out_ty=tensor_ty)
        return x, packed, unpacked

    x, packed, unpacked = mplang.evaluate(sim, fn)
    x_v, packed_v, unpacked_v = mplang.fetch(sim, (x, packed, unpacked))
    np.testing.assert_array_equal(x_v[0], unpacked_v[0])
    assert packed_v[0].dtype == np.uint8
    assert packed_v[0].ndim == 1
    assert packed_v[0].size == arr.size * arr.dtype.itemsize
    assert packed.mptype._type.dtype == UINT8
    assert packed.mptype._type.shape == (-1,)
    assert unpacked.mptype._type == tensor_ty


@pytest.mark.integration
def test_builtin_pack_unpack_table_runtime() -> None:
    sim = mplang.Simulator.simple(1)
    data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    table_schema = TableType.from_pairs([
        ("a", DType.from_any("int64")),
        ("b", DType.from_any("float64")),
    ])
    expected_bytes = table_utils.dataframe_to_csv(pd.DataFrame(data))

    @mplang.function
    def fn():
        table = simp.constant(pd.DataFrame(data))
        packed = simp.runAt(0, builtin.pack)(table)
        unpacked = simp.runAt(0, builtin.unpack)(packed, out_ty=table_schema)
        return packed, unpacked

    packed, unpacked = mplang.evaluate(sim, fn)
    packed_v, unpacked_v = mplang.fetch(sim, (packed, unpacked))
    pd.testing.assert_frame_equal(unpacked_v[0], pd.DataFrame(data))
    assert packed_v[0].dtype == np.uint8
    assert packed_v[0].ndim == 1
    assert packed_v[0].tobytes() == expected_bytes
    assert packed.mptype._type.dtype == UINT8
    assert packed.mptype._type.shape == (-1,)
    assert unpacked.mptype._type == table_schema
