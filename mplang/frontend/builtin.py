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

from typing import Any

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.dtype import UINT64
from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction
from mplang.core.table import TableLike, TableType
from mplang.core.tensor import ScalarType, Shape, TensorLike, TensorType
from mplang.frontend.base import FEOp
from mplang.utils import table_utils


class Identity(FEOp):
    """Identity function class."""

    def __call__(self, obj: MPObject) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Create an identity operation for an MPObject.

        Args:
            obj: The MPObject to create identity operation for

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for identity, args list with obj, output tree definition
        """
        obj_ty = TensorType.from_obj(obj)
        pfunc = PFunction(
            fn_type="builtin.identity",
            ins_info=(obj_ty,),
            outs_info=(obj_ty,),
        )
        _, treedef = tree_flatten(obj_ty)
        return pfunc, [obj], treedef


identity = Identity()


class Read(FEOp):
    """Read function class."""

    def __call__(
        self, path: str, ty: TensorType, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """
        Read data from a file.

        Args:
            path: The file path to read from
            ty: The tensor type information for the data
            **kwargs: Additional attributes for reading

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for reading, empty args list, output tree definition
        """
        pfunc = PFunction(
            fn_type="builtin.read",
            ins_info=(),
            outs_info=(ty,),
            path=path,
            **kwargs,
        )
        _, treedef = tree_flatten(ty)
        return pfunc, [], treedef


read = Read()


class Write(FEOp):
    """Write function class."""

    def __call__(
        self, obj: MPObject, path: str
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """
        Write data to a file.

        Args:
            obj: The MPObject to write
            path: The file path to write to

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for writing, args list with obj, output tree definition

        Raises:
            ValueError: If obj is not provided
        """
        if obj is None:
            raise ValueError("builtin.write requires an object to write")

        obj_ty = TensorType.from_obj(obj)
        pfunc = PFunction(
            fn_type="builtin.write",
            ins_info=(obj_ty,),
            outs_info=(obj_ty,),
            path=path,
        )
        _, treedef = tree_flatten(obj_ty)
        return pfunc, [obj], treedef


write = Write()


class Constant(FEOp):
    """Constant function class."""

    def __call__(
        self,
        data: TensorLike | ScalarType | TableLike,
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Create a constant tensor or table from data.

        Args:
            data: The constant data to embed. Can be:
                  - A scalar value (int, float, bool)
                  - A numpy array or other tensor-like object
                  - A pandas DataFrame or other table-like object
                  - Any object that can be converted to tensor

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for constant, empty args list, output tree definition
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
            # For scalars, convert to numpy array then to bytes
            np_data = np.array(data)
            data_bytes = np_data.tobytes()
            data_format = "bytes[numpy]"
        else:
            if hasattr(data, "tobytes"):
                # For numpy arrays and other TensorLike objects with tobytes method
                out_type = TensorType.from_obj(data)
                data_bytes = data.tobytes()  # type: ignore[attr-defined]
            else:
                # For other TensorLike objects, convert to numpy first
                np_data = np.array(data)
                out_type = TensorType.from_obj(np_data)
                data_bytes = np_data.tobytes()
            data_format = "bytes[numpy]"

        # Store tensor metadata as simple attributes
        pfunc = PFunction(
            fn_type="builtin.constant",
            ins_info=(),
            outs_info=(out_type,),
            data_bytes=data_bytes,
            data_format=data_format,
        )
        _, treedef = tree_flatten(out_type)
        return pfunc, [], treedef


# Public instance for convenience (mirrors other FEOp usages)
constant = Constant()


class Rank(FEOp):
    """Produce the local party's rank as a UINT64 scalar tensor.

    Frontend wrapper emitting a PFunction with fn_type="builtin.rank" and no inputs.
    Runtime semantics: each participating party obtains its own rank in [0, world_size).
    """

    def __call__(self) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Multi-party get the rank (party identifier) of each party.

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for rank, empty args list, output tree definition
        """
        # Return a scalar UINT64 tensor for rank
        tensor_type = TensorType(UINT64, ())

        pfunc = PFunction(
            fn_type="builtin.rank",
            ins_info=(),
            outs_info=(tensor_type,),
        )
        _, treedef = tree_flatten(tensor_type)
        return pfunc, [], treedef


rank = Rank()


class PRand(FEOp):
    """Produce a private random UINT64 tensor of the given shape.

    Each party generates its own random contents independently; shape is structural only.
    """

    def __call__(
        self, shape: Shape = ()
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Multi-party generate a private random (uint64) tensor with the given shape.

        Args:
            shape: The shape of the random tensor to generate.
                   Must be a tuple of positive integers. Defaults to () for scalar.

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for prand, empty args list, output tree definition
        """
        # For now, only UINT64 is supported
        tensor_type = TensorType(UINT64, shape)

        pfunc = PFunction(
            fn_type="builtin.prand",
            ins_info=(),
            outs_info=(tensor_type,),
            shape=shape,
        )
        _, treedef = tree_flatten(tensor_type)
        return pfunc, [], treedef


prand = PRand()


class TableToTensor(FEOp):
    """Pack all columns of a homogeneous table into a 2-D tensor.

    WHY this exists:
        * Provide a purely structural bridge Table -> Tensor without any semantic transforms.
        * Keep feature dimension ordering identical to the table schema to avoid hidden reordering.
        * Enforce homogeneity early so downstream tensor backends see a single element dtype.

    Non-goals (handled upstream in preprocessing layers such as ibis / pandas):
        * Column selection / projection
        * Per-column casting, encoding, null filling
        * Reordering or feature engineering

    Resulting tensor shape: (N, F) where N is provided explicitly (no implicit row counting) and
    F is the number of columns in the table schema.
    """

    def __call__(
        self,
        table: MPObject,
        number_rows: int,
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Symbolically pack table columns into a rank-2 tensor MPObject.

        Args:
            table: Table-typed MPObject whose *entire* schema is to be packed.
            number_rows: Mandatory total row count (N). Caller supplies authoritative value (catalog / prior compute). Must be >= 0.

        Returns:
            A tuple (pfunc, inputs, treedef):
              pfunc: PFunction describing the structural conversion (fn_type="builtin.table_to_tensor").
              inputs: Single-element list containing the original table MPObject (captured for graph wiring).
              treedef: PyTreeDef corresponding to the resulting TensorType so outer tracing can rebuild structure.

        Raises:
            TypeError: If input object is not a table MPObject; or if table columns are heterogeneous.
            ValueError: If ``number_rows`` < 0.
        """
        # Basic validations on table type
        if not isinstance(table.mptype._type, TableType):  # type: ignore[attr-defined]
            raise TypeError("table_to_tensor expects a Table MPObject")
        schema: TableType = table.mptype._type  # type: ignore[assignment]
        if schema.num_columns() == 0:
            raise ValueError("Cannot pack empty table")

        col_dtypes = list(schema.column_types())
        first = col_dtypes[0]
        if not all(dt == first for dt in col_dtypes[1:]):
            raise TypeError(
                "Heterogeneous dtypes; perform casting upstream before table_to_tensor"
            )
        final_dtype = first

        # Row dimension is now explicitly provided by caller.
        if not isinstance(number_rows, int):  # type: ignore[unreachable]
            raise TypeError("number_rows must be an int")
        if number_rows < 0:
            raise ValueError("number_rows must be >= 0")
        shape = (number_rows, schema.num_columns())
        tensor_type = TensorType(final_dtype, shape)  # type: ignore[arg-type]

        pfunc = PFunction(
            fn_type="builtin.table_to_tensor",
            ins_info=(schema,),
            outs_info=(tensor_type,),
        )
        _, treedef = tree_flatten(tensor_type)
        return pfunc, [table], treedef


table_to_tensor = TableToTensor()


class TensorToTable(FEOp):
    """Unpack a rank-2 tensor (N, F) into a homogeneous table schema.

    WHY this exists:
        * Provide the inverse structural mapping of TableToTensor with explicit, stable column names.
        * Avoid implicit naming or synthetic dtype heterogenization which could hide mismatches.

    Constraints / design rationale:
        * Caller must supply column_names explicitly to make schema intent unambiguous.
        * All columns inherit the single tensor element dtype (no per-column overrides to prevent silent casts).
        * No data materialization or copying; only IR-level structural metadata is emitted.
    """

    def __call__(
        self,
        tensor: MPObject,
        column_names: list[str],
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Symbolically convert a rank-2 tensor into a table schema.

        Args:
            tensor: Tensor-typed MPObject of shape (N, F)
            column_names: Required list of names; length must equal F.

        Returns:
            pfunc, inputs, treedef triple as usual.

        Raises:
            TypeError: If not tensor or rank != 2.
            ValueError: If column_names length mismatches F or empty.
        """
        if not isinstance(tensor.mptype._type, TensorType):  # type: ignore[attr-defined]
            raise TypeError("tensor_to_table expects a Tensor MPObject")

        t_ty: TensorType = tensor.mptype._type  # type: ignore[assignment]
        if len(t_ty.shape) != 2:
            raise TypeError("tensor_to_table expects a rank-2 tensor (N,F)")
        n_cols = t_ty.shape[1]
        if not column_names:
            raise ValueError("column_names required (non-empty)")
        if len(column_names) != n_cols:
            raise ValueError("column_names length must match tensor second dim")
        # Validate column names: non-empty, not purely whitespace, unique.
        for i, name in enumerate(column_names):
            if not isinstance(name, str):
                raise TypeError(
                    f"column_names[{i}] must be str, got {type(name).__name__}"
                )
            if name == "" or name.strip() == "":
                raise ValueError(
                    "column names must be non-empty and not whitespace-only"
                )
        # Uniqueness check (preserve order; fail fast on first duplicate)
        seen: set[str] = set()
        for name in column_names:
            if name in seen:
                raise ValueError(f"duplicate column name: {name!r}")
            seen.add(name)

        col_types = [t_ty.dtype] * n_cols
        schema = TableType.from_pairs(list(zip(column_names, col_types, strict=True)))

        pfunc = PFunction(
            fn_type="builtin.tensor_to_table",
            ins_info=(t_ty,),
            outs_info=(schema,),
            column_names=tuple(column_names),  # duplicated with outs_info
        )
        _, treedef = tree_flatten(schema)
        return pfunc, [tensor], treedef


tensor_to_table = TensorToTable()
