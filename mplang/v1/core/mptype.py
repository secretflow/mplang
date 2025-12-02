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

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mplang.v1.core.mpobject import MPObject

from mplang.v1.core.dtypes import STRING, DType
from mplang.v1.core.mask import Mask
from mplang.v1.core.table import TableLike, TableType
from mplang.v1.core.tensor import ScalarType, Shape, TensorLike, TensorType

# basic type aliases
Rank = int


class MPType:
    """A type that describes the type information of an MPObject."""

    _type: TensorType | TableType
    _pmask: Mask | None
    _attrs: dict[str, Any]

    def __init__(
        self,
        type_info: TensorType | TableType,
        pmask: Mask | None = None,
        attrs: dict[str, Any] | None = None,
    ):
        """Initialize MPType.

        Args:
            type_info: The type information (TensorType for tensors, TableType for tables).
            pmask: The party mask, used for compile/trace time determine which party holds the object.
            attrs: Attributes are key-value pairs that can be used to store additional information about the object.
        """
        self._type = type_info
        self._pmask = pmask
        # Ensure attrs is a copy
        self._attrs = copy.copy(attrs) if attrs is not None else {}

    @classmethod
    def tensor(
        cls,
        dtype: DType | Any,
        shape: Shape,
        pmask: int | Mask | None = None,
        **attrs: Any,
    ) -> MPType:
        """Create a tensor type.

        Args:
            dtype: The data type of the tensor.
            shape: The shape of the tensor.
            pmask: The party mask.
            **attrs: Additional attributes.

        Returns:
            MPType instance for tensor.

        Raises:
            ValueError: If dtype is table-only.
        """
        # Convert dtype to DType if needed and validate
        if not isinstance(dtype, DType):
            dtype = DType.from_any(dtype)

        # Ensure tensor types don't use table-only dtypes
        if dtype.is_table_only:
            raise ValueError(
                f"Data type '{dtype.name}' is only supported in tables, "
                f"not in tensors. Use table types for string, date, and other "
                f"non-numeric data types."
            )

        if isinstance(pmask, int):
            pmask = Mask.from_int(pmask)

        tensor_info = TensorType(dtype, shape)
        return cls(tensor_info, pmask, attrs)

    @classmethod
    def table(
        cls,
        schema: TableType | dict[str, DType],
        pmask: int | Mask | None = None,
        **attrs: Any,
    ) -> MPType:
        """Create a table type.

        Args:
            schema: The table schema or dict mapping column names to types.
            pmask: The party mask.
            **attrs: Additional attributes.

        Returns:
            MPType instance for table.
        """
        if isinstance(schema, dict):
            schema = TableType.from_dict(schema)

        if isinstance(pmask, int):
            pmask = Mask.from_int(pmask)

        return cls(schema, pmask, attrs)

    @property
    def is_tensor(self) -> bool:
        """Check if this is a tensor type."""
        return isinstance(self._type, TensorType)

    @property
    def is_table(self) -> bool:
        """Check if this is a table type."""
        return isinstance(self._type, TableType)

    @property
    def dtype(self) -> DType:
        """The data type of the object.

        This property is readonly (mandatory) and will be used for JAX compilation
        to determine the appropriate data type during trace and compilation phases.

        Only available for tensor types.
        """
        if not isinstance(self._type, TensorType):
            raise AttributeError("dtype is only available for tensor types")
        return self._type.dtype

    @property
    def shape(self) -> Shape:
        """The shape of the object, represented as a tuple of integers.

        For example, a 2D tensor with shape (3, 4) would be represented as (3, 4).
        The shape can be empty, which indicates a scalar.

        This property is readonly (mandatory) and will be used for JAX compilation
        to determine tensor shapes during trace and compilation phases.

        Only available for tensor types.
        """
        if not isinstance(self._type, TensorType):
            raise AttributeError("shape is only available for tensor types")
        return self._type.shape

    @property
    def schema(self) -> TableType:
        """The table schema.

        Only available for table types.
        """
        if not isinstance(self._type, TableType):
            raise AttributeError("schema is only available for table types")
        return self._type

    @property
    def pmask(self) -> Mask | None:
        """The party mask indicating which parties hold the data.

        Value interpretation:
        - When not None: A bitmask where the i'th bit is 1 if the i'th party holds
          the data, and 0 otherwise. For example, 0b1101 means parties 0, 2, and 3
          hold the data, while party 1 does not.
        - When None: Party ownership is unknown at compile/trace time and will be
          completely determined at runtime.

        Semantic meaning:
        This mask can be either manually set or deduced by primitive functions during
        compilation/tracing. When None, it does NOT imply either a full mask (all
        parties) or zero mask (no parties) - the actual ownership pattern is entirely
        runtime-dependent.
        """
        return self._pmask

    @property
    def attrs(self) -> dict[str, Any]:
        """Attributes are key-value pairs that can be used to store additional
        information about the object."""
        return self._attrs

    def raw_type(self) -> TensorType | TableType:
        """Get the raw type information (TensorType or TableType)."""
        return self._type

    def set_attr(self, key: str, value: Any) -> None:
        """Set an attribute for this type."""
        self._attrs[key] = value

    def get_attr(self, key: str, default: Any = None) -> Any:
        """Get an attribute for this type."""
        return self._attrs.get(key, default)

    def __repr__(self) -> str:
        """String representation of MPType.

        Schema:
        - For tensor: dtype[shape]<pmask>{other_attrs}
        - For table: Tbl(col1:type1, col2:type2)<pmask>{other_attrs}

        Examples:
        - u64                        # scalar uint64
        - f32[3, 2]                 # 3x2 float32 tensor
        - f16[3]<3>                 # float16 vector with pmask=3
        - u32[5, 5]<F>{device="P0"} # uint32 matrix with pmask=15 and device attr
        - Tbl(id:i64, name:str)      # table with id and name columns
        """
        if isinstance(self._type, TensorType):
            # Start with short dtype name
            ret = self._type.dtype.short_name()

            # Add shape if not scalar
            if self._type.shape:
                shape_str = ", ".join(str(d) for d in self._type.shape)
                ret += f"[{shape_str}]"
        else:  # TableType
            cols = ", ".join(
                f"{name}:{dtype.short_name()}" for name, dtype in self._type.columns
            )
            ret = f"Tbl({cols})"

        # Add pmask in angle brackets if present
        if self._pmask is not None:
            ret += f"<{self._pmask:X}>"

        # Add other attributes in curly braces if any
        if self._attrs:
            attrs_list = []
            for key, value in self._attrs.items():
                if isinstance(value, str):
                    attrs_list.append(f'{key}="{value}"')
                else:
                    attrs_list.append(f"{key}={value}")
            ret += "{" + ", ".join(attrs_list) + "}"

        return ret

    def __eq__(self, other: object) -> bool:
        """Check if two MPType objects are equal."""
        if not isinstance(other, MPType):
            return False
        return (
            self._type == other._type and self._pmask == other._pmask
            # and self._attrs == other._attrs # TODO(jint): attrs should be optional
        )

    def __hash__(self) -> int:
        """Compute hash for MPType objects."""
        # Make attrs hashable by converting to frozenset of items
        attrs_hash = hash(frozenset(self._attrs.items())) if self._attrs else 0
        return hash((
            self._type,
            self._pmask,
            attrs_hash,
        ))

    def isInstance(self, obj: MPObject) -> bool:
        """Check if the given object is an instance of this MPType."""
        # Import here to avoid circular import
        from mplang.v1.core.mpobject import MPObject

        if not isinstance(obj, MPObject):
            return False

        # Check if the object's type matches this type
        obj_type = obj.mptype
        if type(self._type) is not type(obj_type._type):
            return False

        if self._type != obj_type._type:
            return False

        # Check attributes
        if self._attrs:
            if not isinstance(obj.attrs, dict):
                return False
            for k, v in self._attrs.items():
                if k not in obj.attrs or obj.attrs[k] != v:
                    return False
        return True

    def to_numpy(self) -> np.dtype:
        """Convert to NumPy dtype for compatibility.

        Only available for tensor types.
        """
        if not isinstance(self._type, TensorType):
            raise AttributeError("to_numpy is only available for tensor types")
        return self._type.to_numpy()

    @staticmethod
    def _create_tensor_info(obj: TensorLike | ScalarType) -> TensorType:
        """Helper method to create TensorType from tensor-like objects."""
        if isinstance(obj, ScalarType):
            return TensorType(DType.from_python_type(type(obj)), ())
        elif isinstance(obj, TensorLike):
            return TensorType(DType.from_any(obj.dtype), obj.shape)
        elif isinstance(obj, list | tuple):
            # Convert lists/tuples to numpy arrays for compatibility
            arr = np.array(obj)
            return TensorType(DType.from_any(arr.dtype), arr.shape)
        else:
            raise TypeError(f"Unsupported type: {type(obj)}.")

    @classmethod
    def from_tensor(
        cls,
        obj: TensorLike | ScalarType,
        pmask: Mask | None = None,
        **kwargs: Any,
    ) -> MPType:
        """Create MPType from tensor-like object.

        Args:
            obj: Tensor-like object or scalar.
            pmask: The party mask.
            **kwargs: Additional attributes.

        Returns:
            MPType instance for tensor.
        """
        attrs = copy.copy(kwargs)
        tensor_info = cls._create_tensor_info(obj)
        return cls(tensor_info, pmask, attrs)

    @classmethod
    def from_mpobj(cls, obj: MPObject) -> MPType:
        """Create MPType from MPObject.

        Args:
            obj: MPObject instance.

        Returns:
            MPType instance with same type as the object.
        """
        # assume obj is MPObject-like
        obj_type = obj.mptype
        return cls(obj_type._type, obj.pmask, copy.copy(obj.attrs))

    @classmethod
    def from_obj(cls, obj: Any, pmask: Mask | None = None, **attrs: Any) -> MPType:
        """Create MPType from any object, automatically inferring the type.

        Args:
            obj: Object to create type from.
            pmask: The party mask.
            **attrs: Additional attributes.

        Returns:
            MPType instance.

        Raises:
            TypeError: If object type cannot be inferred.
            NotImplementedError: For table objects (not yet implemented).
        """
        # Check if it's a table-like object using the TableLike protocol
        if isinstance(obj, TableLike):
            # For TableLike objects, try to extract schema information
            try:
                import pandas as pd

                if isinstance(obj, pd.DataFrame):
                    from mplang.v1.core.dtypes import DType

                    schema_dict = {}
                    for col_name in obj.columns:
                        pandas_dtype = obj[col_name].dtype
                        # Convert pandas dtype to DType
                        if pandas_dtype.kind in (
                            "O",
                            "U",
                            "S",
                        ):  # object, unicode, string
                            schema_dict[col_name] = (
                                DType.from_numpy(pandas_dtype)
                                if pandas_dtype.kind != "O"
                                else STRING
                            )
                        else:
                            schema_dict[col_name] = DType.from_numpy(pandas_dtype)
                    schema = TableType.from_dict(schema_dict)
                    return cls(schema, pmask, attrs)
            except ImportError:
                pass
            # For other table-like objects without pandas
            raise NotImplementedError(
                "Table object detection for non-pandas objects not fully implemented yet"
            )

        # Otherwise treat as tensor-like
        return cls.from_tensor(obj, pmask, **attrs)
