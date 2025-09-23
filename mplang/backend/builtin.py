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

"""Flat builtin backend kernels.

Each kernel adheres to signature: fn(pfunc: PFunction, args: tuple) -> tuple.
Registration performed via @backend_kernel with explicit fn_type.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from mplang.backend.base import backend_kernel, cur_kctx
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


@backend_kernel("builtin.identity")
def _identity(pfunc: PFunction, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(args) != 1:
        raise ValueError("builtin.identity expects 1 arg")
    return (args[0],)


@backend_kernel("builtin.read")
def _read(pfunc: PFunction, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if args:
        raise ValueError("builtin.read expects 0 args")
    path = pfunc.attrs.get("path")
    if path is None:
        raise ValueError("missing path attr for builtin.read")
    out_t = pfunc.outs_info[0]
    try:
        if isinstance(out_t, TableType):
            with open(path, "rb") as f:
                csv_bytes = f.read()
            df = table_utils.csv_to_dataframe(csv_bytes)
            return (df,)
        else:
            data = np.load(path)
            return (data,)
    except Exception as e:  # pragma: no cover - filesystem errors
        raise RuntimeError(f"builtin.read failed: {e}") from e


@backend_kernel("builtin.write")
def _write(pfunc: PFunction, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(args) != 1:
        raise ValueError("builtin.write expects 1 arg")
    path = pfunc.attrs.get("path")
    if path is None:
        raise ValueError("missing path attr for builtin.write")
    obj = args[0]
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
        return (obj,)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"builtin.write failed: {e}") from e


@backend_kernel("builtin.constant")
def _constant(pfunc: PFunction, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if args:
        raise ValueError("builtin.constant expects 0 args")
    data_bytes = pfunc.attrs.get("data_bytes")
    if data_bytes is None:
        raise ValueError("missing data_bytes attr for builtin.constant")
    out_t = pfunc.outs_info[0]
    fmt = pfunc.attrs.get("data_format")
    if isinstance(out_t, TableType):
        if fmt != "bytes[csv]":
            raise ValueError(f"unsupported table constant format {fmt}")
        df = table_utils.csv_to_dataframe(data_bytes)
        return (df,)
    # tensor path
    shape = out_t.shape  # type: ignore[attr-defined]
    dtype = out_t.dtype.numpy_dtype()  # type: ignore[attr-defined]
    arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return (arr,)


@backend_kernel("builtin.rank")
def _rank(pfunc: PFunction, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if args:
        raise ValueError("builtin.rank expects 0 args")
    ctx = cur_kctx()
    return (np.array(ctx.rank, dtype=np.uint64),)


@backend_kernel("builtin.prand")
def _prand(pfunc: PFunction, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if args:
        raise ValueError("builtin.prand expects 0 args")
    shape = pfunc.attrs.get("shape", ())
    rng = np.random.default_rng()
    info = np.iinfo(np.uint64)
    data = rng.integers(
        low=info.min, high=info.max, size=shape, dtype=np.uint64, endpoint=True
    )
    return (data,)


@backend_kernel("builtin.table_to_tensor")
def _table_to_tensor(pfunc: PFunction, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(args) != 1:
        raise ValueError("builtin.table_to_tensor expects 1 arg")
    table = args[0]
    if not isinstance(table, pd.DataFrame):
        raise TypeError("expected pandas DataFrame")
    if table.shape[1] == 0:
        raise ValueError("cannot pack empty table")
    mat = np.column_stack([table[col].to_numpy() for col in table.columns])
    return (mat,)


@backend_kernel("builtin.tensor_to_table")
def _tensor_to_table(pfunc: PFunction, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(args) != 1:
        raise ValueError("builtin.tensor_to_table expects 1 arg")
    tensor = args[0]
    arr = _to_numpy(tensor)
    if arr.ndim != 2:
        raise ValueError("tensor_to_table expects rank-2 array")
    col_names = pfunc.attrs.get("column_names")
    if col_names is None:
        raise ValueError("missing column_names attr")
    df = pd.DataFrame(arr, columns=list(col_names))
    return (df,)


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


@backend_kernel("builtin.debug_print")
def _debug_print(pfunc: PFunction, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(args) != 1:
        raise ValueError("builtin.debug_print expects 1 arg")
    val = args[0]
    prefix = pfunc.attrs.get("prefix", "")
    ctx = cur_kctx()
    print(f"[debug_print][rank={ctx.rank}] {prefix}{_summ(val)}")
    return (val,)
