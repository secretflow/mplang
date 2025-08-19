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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mplang.core.dtype import DType
from mplang.core.mask import Mask
from mplang.core.relation import RelationSchema

# basic type aliases
Rank = int
Shape = tuple[int, ...]
ScalarType = int | float | bool | complex


@runtime_checkable
class TensorLike(Protocol):
    """
    Protocol for objects structurally resembling tensors from common libraries
    (NumPy, PyTorch, JAX), focusing on dtype and shape attributes.
    """

    @property
    def dtype(self) -> Any: ...

    @property
    def shape(self) -> Shape: ...


@dataclass(frozen=True)
class TensorInfo:
    """A data class that describes the type information of a tensor."""

    dtype: DType
    shape: Shape

    def __init__(self, dtype: DType | Any, shape: Shape):
        # Convert dtype to DType if needed
        if not isinstance(dtype, DType):
            dtype = DType.from_any(dtype)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)

    @classmethod
    def from_obj(cls, obj: TensorLike | ScalarType) -> TensorInfo:
        if isinstance(obj, ScalarType):
            return cls(DType.from_python_type(type(obj)), ())
        elif isinstance(obj, TensorLike):
            return cls(DType.from_any(obj.dtype), obj.shape)
        else:
            raise TypeError(f"Unsupported type: {type(obj)}.")

    def to_numpy(self) -> np.dtype:
        """Convert to NumPy dtype for compatibility."""
        return self.dtype.to_numpy()

    def __repr__(self) -> str:
        shape_str = "x".join(str(d) for d in self.shape)
        dtype_name = str(self.dtype)
        return f"Tensor<{shape_str}x{dtype_name}>" if shape_str else f"{dtype_name}"


class MPKind(Enum):
    """Enumeration for different kinds of MPType."""

    TENSOR = auto()
    RELATION = auto()


class MPType:
    """A type that describes the type information of an MPObject."""

    def __init__(
        self,
        *,
        kind: MPKind,
        dtype: DType | None = None,
        shape: Shape | None = None,
        schema: RelationSchema | None = None,
        pmask: Mask | None = None,
        attrs: dict[str, Any] | None = None,
    ):
        """Initialize MPType.

        Args:
            kind: The kind of MP type (TENSOR or RELATION).
            dtype: The data type of the tensor (required for TENSOR kind).
            shape: The shape of the tensor (required for TENSOR kind).
            schema: The relational schema (required for RELATION kind).
            pmask: The party mask, used for compile/trace time determine which party holds the object.
            attrs: Attributes are key-value pairs that can be used to store additional information about the object.
        """
        self._kind = kind
        self._pmask = pmask
        # Ensure attrs is a copy
        self._attrs = copy.copy(attrs) if attrs is not None else {}

        if kind == MPKind.TENSOR:
            if dtype is None or shape is None:
                raise ValueError("Tensor type requires dtype and shape")
            # Convert dtype to DType if needed
            if not isinstance(dtype, DType):
                dtype = DType.from_any(dtype)
            self._dtype = dtype
            self._shape = shape
            self._schema = None
        elif kind == MPKind.RELATION:
            if schema is None:
                raise ValueError("Relation type requires schema")
            self._schema = schema
            self._dtype = None
            self._shape = None
        else:
            raise ValueError(f"Unsupported MPKind: {kind}")

    @classmethod
    def tensor(
        cls,
        dtype: DType | Any,
        shape: Shape,
        pmask: Mask | None = None,
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
            ValueError: If dtype is relation-only.
        """
        # Convert dtype to DType if needed and validate
        if not isinstance(dtype, DType):
            dtype = DType.from_any(dtype)

        # Ensure tensor types don't use relation-only dtypes
        if dtype.is_relation_only:
            raise ValueError(
                f"Data type '{dtype.name}' is only supported in relations, "
                f"not in tensors. Use relation types for string, date, and other "
                f"non-numeric data types."
            )

        return cls(
            kind=MPKind.TENSOR, dtype=dtype, shape=shape, pmask=pmask, attrs=attrs
        )

    @classmethod
    def relation(
        cls,
        schema: RelationSchema | dict[str, DType],
        pmask: Mask | None = None,
        **attrs: Any,
    ) -> MPType:
        """Create a relation type.

        Args:
            schema: The relational schema or dict mapping column names to types.
            pmask: The party mask.
            **attrs: Additional attributes.

        Returns:
            MPType instance for relation.
        """
        if isinstance(schema, dict):
            schema = RelationSchema.from_dict(schema)
        return cls(kind=MPKind.RELATION, schema=schema, pmask=pmask, attrs=attrs)

    @property
    def kind(self) -> MPKind:
        """The kind of this MPType (TENSOR or RELATION)."""
        return self._kind

    @property
    def dtype(self) -> DType:
        """The data type of the object.

        This property is readonly (mandatory) and will be used for JAX compilation
        to determine the appropriate data type during trace and compilation phases.

        Only available for tensor types.
        """
        if self._kind != MPKind.TENSOR:
            raise AttributeError("dtype is only available for tensor types")
        assert self._dtype is not None  # Type guard
        return self._dtype

    @property
    def shape(self) -> Shape:
        """The shape of the object, represented as a tuple of integers.

        For example, a 2D tensor with shape (3, 4) would be represented as (3, 4).
        The shape can be empty, which indicates a scalar.

        This property is readonly (mandatory) and will be used for JAX compilation
        to determine tensor shapes during trace and compilation phases.

        Only available for tensor types.
        """
        if self._kind != MPKind.TENSOR:
            raise AttributeError("shape is only available for tensor types")
        assert self._shape is not None  # Type guard
        return self._shape

    @property
    def schema(self) -> RelationSchema:
        """The relational schema.

        Only available for relation types.
        """
        if self._kind != MPKind.RELATION:
            raise AttributeError("schema is only available for relation types")
        assert self._schema is not None  # Type guard
        return self._schema

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

    @pmask.setter
    def pmask(self, value: Mask | None) -> None:
        """Set the party mask."""
        self._pmask = value

    @property
    def attrs(self) -> dict[str, Any]:
        """Attributes are key-value pairs that can be used to store additional
        information about the object."""
        return self._attrs

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
        - For relation: Relation<col1:type1, col2:type2><pmask>{other_attrs}

        Examples:
        - u64                        # scalar uint64
        - f32[3, 2]                 # 3x2 float32 tensor
        - f16[3]<3>                 # float16 vector with pmask=3
        - u32[5, 5]<F>{device="P0"} # uint32 matrix with pmask=15 and device attr
        - Relation<id:i64, name:str> # relation with id and name columns
        """
        if self._kind == MPKind.TENSOR:
            # Start with short dtype name
            assert self._dtype is not None
            ret = self._dtype.short_name()

            # Add shape if not scalar
            if self._shape:
                shape_str = ", ".join(str(d) for d in self._shape)
                ret += f"[{shape_str}]"
        else:  # RELATION
            assert self._schema is not None
            cols = ", ".join(
                f"{name}:{dtype.short_name()}" for name, dtype in self._schema.columns
            )
            ret = f"Relation<{cols}>"

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
            self._kind == other._kind
            and self._dtype == other._dtype
            and self._shape == other._shape
            and self._schema == other._schema
            and self._pmask == other._pmask
            and self._attrs == other._attrs
        )

    def __hash__(self) -> int:
        """Compute hash for MPType objects."""
        # Make attrs hashable by converting to frozenset of items
        attrs_hash = hash(frozenset(self._attrs.items())) if self._attrs else 0
        return hash((
            self._kind,
            self._dtype,
            self._shape,
            self._schema,
            self._pmask,
            attrs_hash,
        ))

    def isInstance(self, obj: MPObject) -> bool:
        """Check if the given object is an instance of this MPType."""
        if not isinstance(obj, MPObject):
            return False

        # Check if the object's type matches this type
        obj_type = obj.mptype
        if self._kind != obj_type._kind:
            return False

        if self._kind == MPKind.TENSOR:
            if self._dtype != obj_type._dtype or self._shape != obj_type._shape:
                return False
        elif self._kind == MPKind.RELATION:
            if self._schema != obj_type._schema:
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
        if self._kind != MPKind.TENSOR:
            raise AttributeError("to_numpy is only available for tensor types")
        assert self._dtype is not None
        return self._dtype.to_numpy()

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
        if isinstance(obj, ScalarType):
            return cls.tensor(DType.from_python_type(type(obj)), (), pmask, **attrs)
        elif isinstance(obj, TensorLike):
            return cls.tensor(DType.from_any(obj.dtype), obj.shape, pmask, **attrs)
        elif isinstance(obj, list | tuple):
            # Convert lists/tuples to numpy arrays for compatibility
            arr = np.array(obj)
            return cls.tensor(DType.from_any(arr.dtype), arr.shape, pmask, **attrs)
        else:
            raise TypeError(f"Unsupported type: {type(obj)}.")

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
        if obj_type.kind == MPKind.TENSOR:
            return cls.tensor(obj.dtype, obj.shape, obj.pmask, **obj.attrs)
        else:  # RELATION
            return cls.relation(obj_type.schema, obj.pmask, **obj.attrs)

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
            NotImplementedError: For relation objects (not yet implemented).
        """
        # Check if it's a relation-like object (e.g., pandas DataFrame, pyarrow Table)
        if hasattr(obj, "schema") and hasattr(obj, "columns"):
            # This would need specific implementation for different relation types
            raise NotImplementedError("Relation object detection not implemented yet")

        # Otherwise treat as tensor-like
        return cls.from_tensor(obj, pmask, **attrs)


class MPContext(ABC):
    """The context of an MPObject."""

    @abstractmethod
    def psize(self) -> int:
        """Return the world size."""

    @abstractmethod
    def attrs(self) -> dict[str, Any]:
        """Return the attributes of the context."""

    def attr(self, key: str) -> Any:
        """Return the attribute of the context by key."""
        return self.attrs()[key]


class MPObject(ABC):
    """The base class for all objects in mp-system."""

    @property
    @abstractmethod
    def mptype(self) -> MPType:
        """The type information of the object.

        This property is readonly (mandatory) and will be used for JAX compilation
        to determine the appropriate data type during trace and compilation phases.
        MPType can be passed between different MPObjects as a value.
        """

    @property
    def dtype(self) -> DType:
        return self.mptype.dtype

    @property
    def shape(self) -> Shape:
        return self.mptype.shape

    @property
    def schema(self) -> RelationSchema:
        """The relational schema of the object.

        Only available for relation types.
        """
        return self.mptype.schema

    @property
    def pmask(self) -> Mask | None:
        return self.mptype.pmask

    @property
    def attrs(self) -> dict[str, Any]:
        return self.mptype.attrs

    @property
    @abstractmethod
    def ctx(self) -> MPContext:
        """Return the context of the object."""


# Forward docstrings from MPType to MPObject
MPObject.dtype.__doc__ = MPType.dtype.__doc__
MPObject.shape.__doc__ = MPType.shape.__doc__
MPObject.schema.__doc__ = MPType.schema.__doc__
MPObject.pmask.__doc__ = MPType.pmask.__doc__
MPObject.attrs.__doc__ = MPType.attrs.__doc__
