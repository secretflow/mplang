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

import os
from typing import Any

import numpy as np
import pandas as pd

from mplang.backend.base import cur_kctx, kernel_def
from mplang.core.pfunc import PFunction
from mplang.core.table import TableType
from mplang.utils import table_utils


def _to_numpy(obj: Any) -> np.ndarray:  # minimal helper to avoid duplicating logic
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "numpy"):
        try:
            return np.asarray(obj.numpy())  # type: ignore
        except Exception:
            pass
    return np.asarray(obj)


@kernel_def("builtin.identity")
def _identity(pfunc: PFunction, value: Any) -> Any:
    # Runtime guarantees exactly one argument; no extra arity checks here.
    return value


@kernel_def("builtin.read")
def _read(pfunc: PFunction) -> Any:
    path = pfunc.attrs.get("path")
    if path is None:
        raise ValueError("missing path attr for builtin.read")
    out_t = pfunc.outs_info[0]
    try:
        if isinstance(out_t, TableType):
            with open(path, "rb") as f:
                csv_bytes = f.read()
            df = table_utils.csv_to_dataframe(csv_bytes)
            return df
        else:
            data = np.load(path)
            return data
    except Exception as e:  # pragma: no cover - filesystem errors
        raise RuntimeError(f"builtin.read failed: {e}") from e


@kernel_def("builtin.write")
def _write(pfunc: PFunction, obj: Any) -> Any:
    path = pfunc.attrs.get("path")
    if path is None:
        raise ValueError("missing path attr for builtin.write")
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if hasattr(obj, "__dataframe__") or isinstance(obj, pd.DataFrame):
            csv_bytes = table_utils.dataframe_to_csv(obj)  # type: ignore
            with open(path, "wb") as f:
                f.write(csv_bytes)
        else:
            np.save(path, _to_numpy(obj))
        return obj
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"builtin.write failed: {e}") from e


@kernel_def("builtin.constant")
def _constant(pfunc: PFunction) -> Any:
    data_bytes = pfunc.attrs.get("data_bytes")
    if data_bytes is None:
        raise ValueError("missing data_bytes attr for builtin.constant")
    out_t = pfunc.outs_info[0]
    fmt = pfunc.attrs.get("data_format")
    if isinstance(out_t, TableType):
        if fmt != "bytes[csv]":
            raise ValueError(f"unsupported table constant format {fmt}")
        df = table_utils.csv_to_dataframe(data_bytes)
        return df
    # tensor path
    shape = out_t.shape  # type: ignore[attr-defined,union-attr]
    dtype = out_t.dtype.numpy_dtype()  # type: ignore[attr-defined,union-attr]
    arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return arr


@kernel_def("builtin.rank")
def _rank(pfunc: PFunction) -> Any:
    ctx = cur_kctx()
    return np.array(ctx.rank, dtype=np.uint64)


@kernel_def("builtin.prand")
def _prand(pfunc: PFunction) -> Any:
    shape = pfunc.attrs.get("shape", ())
    rng = np.random.default_rng()
    info = np.iinfo(np.uint64)
    data = rng.integers(
        low=info.min, high=info.max, size=shape, dtype=np.uint64, endpoint=True
    )
    return data


@kernel_def("builtin.table_to_tensor")
def _table_to_tensor(pfunc: PFunction, table: Any) -> Any:
    if not isinstance(table, pd.DataFrame):
        raise TypeError("expected pandas DataFrame")
    if table.shape[1] == 0:
        raise ValueError("cannot pack empty table")
    mat = np.column_stack([table[col].to_numpy() for col in table.columns])
    return mat


@kernel_def("builtin.tensor_to_table")
def _tensor_to_table(pfunc: PFunction, tensor: Any) -> Any:
    arr = _to_numpy(tensor)
    if arr.ndim != 2:
        raise ValueError("tensor_to_table expects rank-2 array")
    col_names = pfunc.attrs.get("column_names")
    if col_names is None:
        raise ValueError("missing column_names attr")
    df = pd.DataFrame(arr, columns=list(col_names))
    return df


def _summ(v: Any) -> str:
    try:
        if isinstance(v, pd.DataFrame):
            return str(v.head(8).to_string(index=False))
        arr = _to_numpy(v)
        return str(
            np.array2string(
                arr, threshold=64, edgeitems=3, precision=6, suppress_small=True
            )
        )
    except Exception as e:  # pragma: no cover
        return f"<unprintable {type(v).__name__}: {e}>"


@kernel_def("builtin.debug_print")
def _debug_print(pfunc: PFunction, val: Any) -> Any:
    prefix = pfunc.attrs.get("prefix", "")
    ctx = cur_kctx()
    print(f"[debug_print][rank={ctx.rank}] {prefix}{_summ(val)}")
    return val
