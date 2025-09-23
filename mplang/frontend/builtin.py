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

from mplang.core.dtype import UINT64
from mplang.core.mpobject import MPObject  # Needed for constant() triad return typing
from mplang.core.pfunc import PFunction
from mplang.core.table import TableLike, TableType
from mplang.core.tensor import ScalarType, Shape, TensorLike, TensorType
from mplang.frontend.base import femod
from mplang.utils import table_utils

_BUILTIN_MOD = femod("builtin")


@_BUILTIN_MOD.typed_op()
def identity(x: TensorType) -> TensorType:
    """Identity on type: captures the underlying MPObject (if any) but kernel sees only the type.

    Under strict typed_op semantics positional MPObject is converted to its TensorType before entering.
    """
    return x


@_BUILTIN_MOD.typed_op()
def read(*, path: str, ty: TensorType) -> TensorType:
    """Type-only kernel for reading a tensor/table from a path.

    Attributes:
      - path: str destination to read from (carried as PFunction attr)
      - ty:   TensorType/TableType describing the expected output type

    Returns: ty (shape/dtype/schema), no inputs captured.
    """
    if not isinstance(path, str) or path == "":
        raise ValueError("path must be a non-empty string")
    if not isinstance(ty, (TensorType, TableType)):
        raise TypeError("ty must be a TensorType or TableType")
    # typed_op will attach 'path' as an attribute and build the PFunction
    return ty


@_BUILTIN_MOD.typed_op()
def write(x: TensorType, *, path: str) -> TensorType:
    """Write op: returns same type it consumes; runtime handles side effect.

    Positional MPObject (tensor/table) will be captured; kernel sees only its type.
    """
    return x


@_BUILTIN_MOD.feop()
def constant(
    data: TensorLike | ScalarType | TableLike,
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
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


@_BUILTIN_MOD.typed_op()
def rank() -> TensorType:
    """Type-only kernel: returns the UINT64 scalar tensor type for current rank.

    Runtime provides the concrete rank value per party during execution; here we
    only declare the output type with no inputs captured and no attributes.
    """
    return TensorType(UINT64, ())


@_BUILTIN_MOD.typed_op()
def prand(*, shape: Shape = ()) -> TensorType:
    """Type-only kernel: private random UINT64 tensor of given shape.

    Shape is attached as a PFunction attribute via typed_op; no inputs.
    """
    return TensorType(UINT64, shape)


@_BUILTIN_MOD.typed_op()
def debug_print(
    x: TensorType | TableType, *, prefix: str = ""
) -> TableType | TensorType:
    """Debug-print pass-through: type identity with side-effect attribute.

    Accepts tensor/table type; MPObject positional (if provided) is captured automatically.
    """
    return x


@_BUILTIN_MOD.typed_op()
def table_to_tensor(table: TableType, *, number_rows: int) -> TensorType:
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


@_BUILTIN_MOD.typed_op()
def tensor_to_table(tensor: TensorType, *, column_names: list[str]) -> TableType:
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
