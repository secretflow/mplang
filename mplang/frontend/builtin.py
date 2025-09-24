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


from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.dtype import UINT8, UINT64
from mplang.core.mpobject import MPObject  # Needed for constant() triad return typing
from mplang.core.pfunc import PFunction
from mplang.core.table import TableLike, TableType
from mplang.core.tensor import ScalarType, Shape, TensorLike, TensorType
from mplang.frontend.base import stateless_mod
from mplang.utils import table_utils

_BUILTIN_MOD = stateless_mod("builtin")


@_BUILTIN_MOD.simple_op()
def identity(x: TensorType) -> TensorType:
    """Return the input type unchanged.

    Args:
        x: The input tensor type. If called with an MPObject, the value is
            captured positionally; the kernel sees only the type.

    Returns:
        The same type as ``x``.
    """
    return x


@_BUILTIN_MOD.simple_op()
def read(*, path: str, ty: TensorType) -> TensorType:
    """Declare reading a value of type ``ty`` from ``path`` (type-only).

    Args:
        path: Non-empty path or URI to read from (stored as an attribute).
        ty: The expected output type/schema.

    Returns:
        Exactly ``ty``.

    Raises:
        ValueError: If ``path`` is empty.
        TypeError: If ``ty`` is not a TensorType or TableType.
    """
    if not isinstance(path, str) or path == "":
        raise ValueError("path must be a non-empty string")
    if not isinstance(ty, (TensorType, TableType)):
        raise TypeError("ty must be a TensorType or TableType")
    # typed_op will attach 'path' as an attribute and build the PFunction
    return ty


@_BUILTIN_MOD.simple_op()
def write(x: TensorType, *, path: str) -> TensorType:
    """Declare writing the input value to ``path`` and return the same type.

    Args:
        x: The value's type to be written; values are captured positionally.
        path: Destination path or URI (attribute).

    Returns:
        The same type as ``x``.
    """
    return x


@_BUILTIN_MOD.op_def()
def constant(
    data: TensorLike | ScalarType | TableLike,
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Embed a literal tensor/table and return the full triad.

    Args:
        data: Constant payload. Supports scalars, array-like tensors, or
            table-like dataframes.

    Returns:
        Tuple[PFunction, list[MPObject], PyTreeDef]:
        - PFunction: ``fn_type='builtin.constant'`` with one output whose type
            matches ``data``; payload serialized via ``data_bytes`` with
            ``data_format`` ('bytes[numpy]' or 'bytes[csv]').
        - list[MPObject]: Empty (no inputs captured).
        - PyTreeDef: Output tree (single leaf).
    """
    import numpy as np

    data_bytes: bytes
    out_type: TableType | TensorType

    if isinstance(data, TableLike):
        data_bytes = table_utils.dataframe_to_csv(data)
        data_format = "bytes[csv]"
        out_type = TableType.from_tablelike(data)
    elif isinstance(data, ScalarType):
        out_type = TensorType.from_obj(data)
        np_data = np.array(data)
        data_bytes = np_data.tobytes()
        data_format = "bytes[numpy]"
    else:
        if hasattr(data, "tobytes"):
            out_type = TensorType.from_obj(data)
            data_bytes = data.tobytes()  # type: ignore[attr-defined]
        else:
            np_data = np.array(data)
            out_type = TensorType.from_obj(np_data)
            data_bytes = np_data.tobytes()
        data_format = "bytes[numpy]"

    pfunc = PFunction(
        fn_type="builtin.constant",
        ins_info=(),
        outs_info=(out_type,),
        data_bytes=data_bytes,
        data_format=data_format,
    )
    _, treedef = tree_flatten(out_type)
    return pfunc, [], treedef


@_BUILTIN_MOD.simple_op()
def rank() -> TensorType:
    """Return the scalar UINT64 tensor type for the current party rank.

    Returns:
        A scalar ``UINT64`` tensor type (shape ``()``).
    """
    return TensorType(UINT64, ())


@_BUILTIN_MOD.simple_op()
def prand(*, shape: Shape = ()) -> TensorType:
    """Declare a private random UINT64 tensor with the given shape.

    Args:
        shape: Output tensor shape. Defaults to ``()``.

    Returns:
        A ``UINT64`` tensor type with the specified shape.
    """
    return TensorType(UINT64, shape)


@_BUILTIN_MOD.simple_op()
def debug_print(
    x: TensorType | TableType, *, prefix: str = ""
) -> TableType | TensorType:
    """Print a value at runtime and return the same type.

    Args:
        x: The value to print (captured positionally; kernel sees only type).
        prefix: Optional text prefix for the printed output.

    Returns:
        The same type as ``x``.
    """
    return x


@_BUILTIN_MOD.simple_op()
def pack(x: TensorType | TableType) -> TensorType:
    """Serialize a tensor/table into a byte vector (type-only).

    Args:
        x: Input type to pack.

    Returns:
        A ``UINT8`` tensor type with shape ``(-1,)`` (length decided at runtime).

    Raises:
        TypeError: If ``x`` is not a TensorType or TableType.
    """

    if not isinstance(x, (TensorType, TableType)):
        raise TypeError("pack expects TensorType or TableType input")

    return TensorType(UINT8, (-1,))


@_BUILTIN_MOD.simple_op()
def unpack(b: TensorType, *, out_ty: TensorType | TableType) -> TensorType | TableType:
    """Deserialize a byte vector into the explicit output type.

    Args:
        b: Byte vector type. Must be ``UINT8`` with shape ``(N,)`` (``N`` may be
            ``-1``).
        out_ty: Resulting type/schema after unpacking.

    Returns:
        Exactly ``out_ty``.

    Raises:
        TypeError: If ``out_ty`` is not a TensorType/TableType, or if ``b`` is
            not a 1-D UINT8 tensor.
    """

    if not isinstance(out_ty, (TensorType, TableType)):
        raise TypeError("out_ty must be TensorType or TableType")

    if b.dtype != UINT8 or len(b.shape) != 1:
        raise TypeError("unpack expects a 1-D UINT8 tensor")

    return out_ty


@_BUILTIN_MOD.simple_op()
def table_to_tensor(table: TableType, *, number_rows: int) -> TensorType:
    """Convert a homogeneous-typed table to a dense 2D tensor.

    Args:
        table: Input table whose columns all share the same dtype.
        number_rows: Number of rows in the resulting tensor. Must be ``>= 0``.

    Returns:
        A rank-2 tensor with dtype equal to the table column dtype and shape
        ``(number_rows, table.num_columns())``.

    Raises:
        ValueError: If the table is empty or ``number_rows < 0``.
        TypeError: If the table has heterogeneous column dtypes or ``number_rows``
            is not an int.
    """
    if table.num_columns() == 0:
        raise ValueError("Cannot pack empty table")
    col_dtypes = list(table.column_types())
    first = col_dtypes[0]
    if not all(dt == first for dt in col_dtypes[1:]):
        raise TypeError(
            "Heterogeneous dtypes; perform casting upstream before table_to_tensor"
        )
    if not isinstance(number_rows, int):
        raise TypeError("number_rows must be an int")
    if number_rows < 0:
        raise ValueError("number_rows must be >= 0")
    shape = (number_rows, table.num_columns())
    return TensorType(first, shape)  # type: ignore[arg-type]


@_BUILTIN_MOD.simple_op()
def tensor_to_table(tensor: TensorType, *, column_names: list[str]) -> TableType:
    """Convert a rank-2 tensor into a table with named columns.

    Args:
        tensor: Rank-2 tensor with shape ``(N, F)``.
        column_names: List of unique, non-whitespace column names of length ``F``.

    Returns:
        A table with ``F`` columns named as provided, each with dtype
        ``tensor.dtype``.

    Raises:
        TypeError: If ``tensor`` is not rank-2, or if any column name is not a
            string.
        ValueError: If names are empty/whitespace, duplicated, or length != ``F``.
    """
    if len(tensor.shape) != 2:
        raise TypeError("tensor_to_table expects a rank-2 tensor (N,F)")
    n_cols = tensor.shape[1]
    if not column_names:
        raise ValueError("column_names required (non-empty)")
    if len(column_names) != n_cols:
        raise ValueError("column_names length must match tensor second dim")
    for i, name in enumerate(column_names):
        if not isinstance(name, str):
            raise TypeError(f"column_names[{i}] must be str, got {type(name).__name__}")
        if name == "" or name.strip() == "":
            raise ValueError("column names must be non-empty and not whitespace-only")
    seen: set[str] = set()
    for name in column_names:
        if name in seen:
            raise ValueError(f"duplicate column name: {name!r}")
        seen.add(name)
    col_types = [tensor.dtype] * n_cols
    return TableType.from_pairs(list(zip(column_names, col_types, strict=True)))
