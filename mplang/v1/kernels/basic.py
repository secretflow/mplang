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

from mplang.v1.core import PFunction, TableType, TensorType
from mplang.v1.kernels.base import cur_kctx, kernel_def
from mplang.v1.kernels.value import TableValue, TensorValue, Value
from mplang.v1.runtime.data_providers import get_provider, resolve_uri
from mplang.v1.utils import table_utils


@kernel_def("basic.identity")
def _identity(pfunc: PFunction, value: Value) -> Value:
    # Runtime guarantees exactly one argument; no extra arity checks here.
    return value


@kernel_def("basic.read")
def _read(pfunc: PFunction) -> Value:
    path = pfunc.attrs.get("path")
    if path is None:
        raise ValueError("missing path attr for basic.read")
    out_t = pfunc.outs_info[0]
    uri = resolve_uri(str(path))
    prov = get_provider(uri.scheme)
    if prov is None:
        raise NotImplementedError(f"no resource provider for scheme: {uri.scheme}")
    ctx = cur_kctx()
    try:
        data = prov.read(uri, out_t, ctx=ctx)
    except Exception as e:  # pragma: no cover - provider errors
        raise RuntimeError(f"basic.read failed: {e}") from e

    if isinstance(data, Value):
        return data

    if isinstance(out_t, TableType):
        return TableValue(data)
    elif isinstance(out_t, TensorType):
        return TensorValue(np.asarray(data))
    else:
        raise TypeError(
            f"basic.read only supports TableType/TensorType outputs, got {type(out_t).__name__}"
        )


@kernel_def("basic.write")
def _write(pfunc: PFunction, obj: Value) -> Value:
    path = pfunc.attrs.get("path")
    if path is None:
        raise ValueError("missing path attr for basic.write")
    uri = resolve_uri(str(path))
    prov = get_provider(uri.scheme)
    if prov is None:
        raise NotImplementedError(f"no resource provider for scheme: {uri.scheme}")
    # Pass Value object directly to provider - let provider decide how to handle it
    ctx = cur_kctx()
    try:
        prov.write(uri, obj, ctx=ctx)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"basic.write failed: {e}") from e
    return obj


@kernel_def("basic.constant")
def _constant(pfunc: PFunction) -> Value:
    """Return constants as Value types (TensorValue or TableValue)."""
    data_bytes = pfunc.attrs.get("data_bytes")
    if data_bytes is None:
        raise ValueError("missing data_bytes attr for basic.constant")
    out_t = pfunc.outs_info[0]
    fmt = pfunc.attrs.get("data_format")
    if isinstance(out_t, TableType):
        if fmt != "bytes[parquet]":
            raise ValueError(f"unsupported table constant format {fmt}")
        df = table_utils.decode_table(data_bytes, format="parquet")
        return TableValue(df)
    # tensor path
    shape = out_t.shape  # type: ignore[attr-defined,union-attr]
    dtype = out_t.dtype.numpy_dtype()  # type: ignore[attr-defined,union-attr]
    arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return TensorValue(arr)


@kernel_def("basic.rank")
def _rank(pfunc: PFunction) -> TensorValue:
    """Return rank as TensorValue."""
    ctx = cur_kctx()
    arr = np.array(ctx.rank, dtype=np.uint64)
    return TensorValue(arr)


@kernel_def("basic.prand")
def _prand(pfunc: PFunction) -> TensorValue:
    """Return random data as TensorValue."""
    shape = pfunc.attrs.get("shape", ())
    rng = np.random.default_rng()
    info = np.iinfo(np.uint64)
    data = rng.integers(
        low=info.min, high=info.max, size=shape, dtype=np.uint64, endpoint=True
    )
    return TensorValue(data)


@kernel_def("basic.table_to_tensor")
def _table_to_tensor(pfunc: PFunction, table: TableValue) -> TensorValue:
    """Convert table to tensor, return as TensorValue."""
    arrow_table = table.to_arrow()
    if arrow_table.num_columns == 0:
        raise ValueError("cannot pack empty table")
    # Convert Arrow columns to numpy arrays and stack
    mat = np.column_stack([
        arrow_table.column(i).to_numpy() for i in range(arrow_table.num_columns)
    ])
    return TensorValue(mat)


@kernel_def("basic.tensor_to_table")
def _tensor_to_table(pfunc: PFunction, tensor: TensorValue) -> TableValue:
    """Convert tensor to table, return as TableValue."""
    import pyarrow as pa  # type: ignore

    arr = tensor.to_numpy()
    if arr.ndim != 2:
        raise ValueError("tensor_to_table expects rank-2 array")
    col_names = pfunc.attrs.get("column_names")
    if col_names is None:
        raise ValueError("missing column_names attr")
    # Create Arrow table directly from numpy array columns
    arrays = [pa.array(arr[:, i]) for i in range(arr.shape[1])]
    arrow_table = pa.table(dict(zip(col_names, arrays, strict=True)))
    return TableValue(arrow_table)


def _summ(v: Value) -> str:
    try:
        if isinstance(v, TableValue):
            # Use Arrow's native string representation (more efficient)
            arrow_table = v.to_arrow()
            # Show first 8 rows
            preview = arrow_table.slice(0, min(8, arrow_table.num_rows))
            return str(preview)
        if isinstance(v, TensorValue):
            arr = v.to_numpy()
            return str(
                np.array2string(
                    arr, threshold=64, edgeitems=3, precision=6, suppress_small=True
                )
            )
        return repr(v)
    except Exception as e:  # pragma: no cover
        return f"<unprintable {type(v).__name__}: {e}>"


@kernel_def("basic.debug_print")
def _debug_print(pfunc: PFunction, val: Value) -> Value:
    prefix = pfunc.attrs.get("prefix", "")
    ctx = cur_kctx()
    print(f"[debug_print][rank={ctx.rank}] {prefix}{_summ(val)}")
    return val


@kernel_def("basic.pack")
def _pack(pfunc: PFunction, value: Value) -> TensorValue:
    outs_info = pfunc.outs_info
    if len(outs_info) != 1:
        raise ValueError("basic.pack expects single output type")
    out_ty = outs_info[0]
    if not isinstance(out_ty, TensorType):
        raise TypeError("basic.pack must return TensorType")
    if out_ty.dtype.numpy_dtype() != np.uint8:
        raise TypeError("basic.pack output dtype must be uint8")

    if isinstance(value, TableValue):
        # Serialize Arrow table using IPC stream for consistency with Value serde
        import pyarrow as pa  # type: ignore
        import pyarrow.ipc as pa_ipc  # type: ignore

        arrow_table = value.to_arrow()
        sink = pa.BufferOutputStream()
        with pa_ipc.new_stream(sink, arrow_table.schema) as writer:  # type: ignore[arg-type]
            writer.write_table(arrow_table)  # type: ignore[arg-type]
        ipc_bytes = sink.getvalue().to_pybytes()
        return TensorValue(np.frombuffer(ipc_bytes, dtype=np.uint8))

    if isinstance(value, TensorValue):
        arr = value.to_numpy()
        return TensorValue(np.frombuffer(arr.tobytes(order="C"), dtype=np.uint8))

    raise TypeError(f"basic.pack does not support Value type {type(value).__name__}")


@kernel_def("basic.unpack")
def _unpack(pfunc: PFunction, packed: TensorValue) -> Value:
    outs_info = pfunc.outs_info
    if len(outs_info) != 1:
        raise ValueError("basic.unpack expects single output type")
    out_ty = outs_info[0]

    b = packed.to_numpy().astype(np.uint8, copy=False).reshape(-1)

    if isinstance(out_ty, TensorType):
        np_dtype = out_ty.dtype.numpy_dtype()
        shape = tuple(out_ty.shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("basic.unpack does not support dynamic tensor shapes")
        elem_count = int(np.prod(shape))
        expected = elem_count * np.dtype(np_dtype).itemsize
        if b.size != expected:
            raise ValueError(
                f"unpack size mismatch: got {b.size} bytes, expect {expected} for {np_dtype} {shape}"
            )
        arr = np.frombuffer(b.tobytes(), dtype=np_dtype)
        return TensorValue(arr.reshape(shape))

    if isinstance(out_ty, TableType):
        # Deserialize Arrow IPC stream back to TableValue
        import pyarrow as pa  # type: ignore
        import pyarrow.ipc as pa_ipc  # type: ignore

        buf = pa.py_buffer(b.tobytes())
        reader = pa_ipc.open_stream(buf)
        table = reader.read_all()
        return TableValue(table)

    raise TypeError("basic.unpack output type must be TensorType or TableType")
