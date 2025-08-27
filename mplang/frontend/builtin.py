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
                data_bytes = data.tobytes()  # type: ignore
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


constant = Constant()


class Rank(FEOp):
    """Rank function class."""

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
    """PRand function class."""

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
