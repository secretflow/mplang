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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mplang.core.dtype import DType

# basic type aliases
Rank = int
Shape = tuple[int, ...]
ScalarType = int | float | bool | complex


class Mask:
    """A class to represent and manipulate party masks for multi-party computation.
    
    A mask is a bitmask where the i'th bit is 1 if the i'th party holds the data,
    and 0 otherwise. For example, 0b1101 means parties 0, 2, and 3 hold the data.
    """
    
    def __init__(self, value: int | Mask = 0):
        """Initialize a Mask from an integer or another Mask.
        
        Args:
            value: The integer value of the mask, must be non-negative.
        """
        if isinstance(value, Mask):
            self._value = value._value
        elif isinstance(value, int):
            if value < 0:
                raise ValueError(f"Mask value must be non-negative, got {value}")
            self._value = value
        else:
            raise TypeError(f"Mask value must be int or Mask, got {type(value)}")
    
    @property 
    def value(self) -> int:
        """Get the integer value of the mask."""
        return self._value
    
    def bit_count(self) -> int:
        """Return the number of set bits in the mask."""
        return self._value.bit_count()
    
    def enum_mask(self) -> Iterator[int]:
        """Iterate over the indices of set bits in the mask."""
        value = self._value
        i = 0
        while value != 0:
            if value & 1:
                yield i
            value >>= 1
            i += 1
    
    def is_rank_in(self, rank: int) -> bool:
        """Check if a rank is in the party mask."""
        return (1 << rank) & self._value != 0
    
    def ensure_rank_in(self, rank: int) -> None:
        """Ensure a rank is in the party mask, raise ValueError if not."""
        if not self.is_rank_in(rank):
            raise ValueError(f"Rank {rank} is not in the party mask {self._value}")
    
    def is_subset(self, other: Mask | int) -> bool:
        """Check if this mask is a subset of another mask."""
        other_value = other._value if isinstance(other, Mask) else other
        return (self._value & other_value) == self._value
    
    def ensure_subset(self, other: Mask | int) -> None:
        """Ensure this mask is a subset of another mask, raise ValueError if not."""
        if not self.is_subset(other):
            other_value = other._value if isinstance(other, Mask) else other
            raise ValueError(
                f"Expect subset mask {self._value} to be a subset of superset mask {other_value}."
            )
    
    def global_to_relative_rank(self, global_rank: int) -> int:
        """Convert a global rank to a relative rank within this mask."""
        if not (global_rank >= 0 and (self._value & (1 << global_rank))):
            raise ValueError(
                f"Invalid global_rank ({global_rank}) or bit not set in mask (0b{self._value:b})."
            )
        sub_mask = self._value & ((1 << (global_rank + 1)) - 1)
        return sub_mask.bit_count() - 1
    
    def relative_to_global_rank(self, relative_rank: int) -> int:
        """Convert a relative rank within this mask to a global rank."""
        if not (0 <= relative_rank < self._value.bit_count()):
            raise ValueError(
                f"Invalid relative_rank ({relative_rank}) for mask (0b{self._value:b}) "
                f"with {self._value.bit_count()} set bits."
            )
        temp_mask = self._value
        for _ in range(relative_rank):
            temp_mask &= temp_mask - 1
        return (temp_mask & -temp_mask).bit_length() - 1
    
    # Operator overloads for bitwise operations
    def __and__(self, other: Mask | int) -> Mask:
        """Bitwise AND operation (intersection of masks)."""
        other_value = other._value if isinstance(other, Mask) else other
        return Mask(self._value & other_value)
    
    def __or__(self, other: Mask | int) -> Mask:
        """Bitwise OR operation (union of masks)."""
        other_value = other._value if isinstance(other, Mask) else other
        return Mask(self._value | other_value)
    
    def __xor__(self, other: Mask | int) -> Mask:
        """Bitwise XOR operation."""
        other_value = other._value if isinstance(other, Mask) else other
        return Mask(self._value ^ other_value)
    
    def __invert__(self) -> Mask:
        """Bitwise NOT operation (limited to reasonable mask size)."""
        # For masks, we typically want to invert only up to the highest set bit
        # or a reasonable default like 32 bits
        if self._value == 0:
            return Mask(0)
        highest_bit = self._value.bit_length()
        mask_limit = (1 << highest_bit) - 1
        return Mask(mask_limit ^ self._value)
    
    def __lshift__(self, other: int) -> Mask:
        """Left shift operation."""
        return Mask(self._value << other)
    
    def __rshift__(self, other: int) -> Mask:
        """Right shift operation."""
        return Mask(self._value >> other)
    
    # Comparison operations
    def __eq__(self, other: object) -> bool:
        """Check equality with another Mask or int."""
        if isinstance(other, Mask):
            return self._value == other._value
        elif isinstance(other, int):
            return self._value == other
        return False
    
    def __ne__(self, other: object) -> bool:
        """Check inequality with another Mask or int."""
        return not self.__eq__(other)
    
    def __hash__(self) -> int:
        """Hash based on the integer value."""
        return hash(self._value)
    
    def __bool__(self) -> bool:
        """Truth value (True if any bits are set)."""
        return self._value != 0
    
    def __int__(self) -> int:
        """Convert to integer."""
        return self._value
    
    def __repr__(self) -> str:
        """String representation showing both decimal and binary."""
        return f"Mask({self._value}=0b{self._value:b})"
    
    def __str__(self) -> str:
        """String representation."""
        return str(self._value)
    
    @classmethod
    def union(cls, *masks: Mask | int) -> Mask:
        """Return the union of multiple masks."""
        if not masks:
            raise ValueError("At least one mask is required for union.")
        result = masks[0] if isinstance(masks[0], Mask) else Mask(masks[0])
        for mask in masks[1:]:
            result = result | mask
        return result
    
    @classmethod
    def intersection(cls, *masks: Mask | int) -> Mask:
        """Return the intersection of multiple masks.""" 
        if not masks:
            return Mask(0)
        result = masks[0] if isinstance(masks[0], Mask) else Mask(masks[0])
        for mask in masks[1:]:
            result = result & mask
        return result
    
    @classmethod
    def is_disjoint(cls, *masks: Mask | int) -> bool:
        """Check if all masks are disjoint (no common bits)."""
        joint_mask = 0
        for mask in masks:
            mask_value = mask._value if isinstance(mask, Mask) else mask
            if isinstance(mask, int) and mask < 0:
                raise ValueError(f"Not a valid mask, got {mask}")
            if joint_mask & mask_value:
                return False
            joint_mask |= mask_value
        return True


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


class MPType:
    """A type that describes the type information of an MPObject."""

    def __init__(
        self,
        dtype: DType,
        shape: Shape,
        pmask: Mask | int | None = None,
        attrs: dict[str, Any] | None = None,
    ):
        """Initialize MPType.

        Args:
            dtype: The data type of the object.
            shape: The shape of the object, represented as a tuple of integers.
            pmask: The party mask, used for compile/trace time determine which party holds the object.
            attrs: Attributes are key-value pairs that can be used to store additional information about the object.
        """
        # Convert dtype to DType if needed
        if not isinstance(dtype, DType):
            dtype = DType.from_any(dtype)

        self._dtype = dtype
        self._shape = shape
        # Convert pmask to Mask if it's an int
        if isinstance(pmask, int):
            self._pmask = Mask(pmask)
        else:
            self._pmask = pmask
        # Ensure attrs is a copy
        self._attrs = copy.copy(attrs) if attrs is not None else {}

    @property
    def dtype(self) -> DType:
        """The data type of the object.

        This property is readonly (mandatory) and will be used for JAX compilation
        to determine the appropriate data type during trace and compilation phases.
        """
        return self._dtype

    @property
    def shape(self) -> Shape:
        """The shape of the object, represented as a tuple of integers.

        For example, a 2D tensor with shape (3, 4) would be represented as (3, 4).
        The shape can be empty, which indicates a scalar.

        This property is readonly (mandatory) and will be used for JAX compilation
        to determine tensor shapes during trace and compilation phases.
        """
        return self._shape

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
    def pmask(self, value: Mask | int | None) -> None:
        """Set the party mask."""
        if isinstance(value, int):
            self._pmask = Mask(value)
        else:
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

        Schema: dtype[shape]<pmask>{other_attrs}
        Examples:
        - u64                        # scalar uint64
        - f32[3, 2]                 # 3x2 float32 tensor
        - f16[3]<3>                 # float16 vector with pmask=3
        - u32[5, 5]<F>{device="P0"} # uint32 matrix with pmask=15 and device attr
        """
        # Start with short dtype name
        ret = self._dtype.short_name()

        # Add shape if not scalar
        if self._shape:
            shape_str = ", ".join(str(d) for d in self._shape)
            ret += f"[{shape_str}]"

        # Add pmask in angle brackets if present
        if self._pmask is not None:
            ret += f"<{self._pmask.value:X}>"

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
            self._dtype == other._dtype
            and self._shape == other._shape
            and self._pmask == other._pmask
            and self._attrs == other._attrs
        )

    def __hash__(self) -> int:
        """Compute hash for MPType objects."""
        # Make attrs hashable by converting to frozenset of items
        attrs_hash = hash(frozenset(self._attrs.items())) if self._attrs else 0
        return hash((self._dtype, self._shape, self._pmask, attrs_hash))

    def isInstance(self, obj: MPObject) -> bool:
        """Check if the given object is an instance of this MPType."""
        if not isinstance(obj, MPObject):
            return False
        if obj.dtype != self._dtype:
            return False
        if obj.shape != self._shape:
            return False
        if self._attrs:
            if not isinstance(obj.attrs, dict):
                return False
            for k, v in self._attrs.items():
                if k not in obj.attrs or obj.attrs[k] != v:
                    return False
        return True

    def to_numpy(self) -> np.dtype:
        """Convert to NumPy dtype for compatibility."""
        return self._dtype.to_numpy()

    @classmethod
    def from_tensor(
        cls,
        obj: TensorLike | ScalarType,
        pmask: Mask | int | None = None,
        **kwargs: Any,
    ) -> MPType:
        attrs = copy.copy(kwargs)
        if isinstance(obj, ScalarType):
            return cls(DType.from_python_type(type(obj)), (), pmask, attrs)
        elif isinstance(obj, TensorLike):
            return cls(DType.from_any(obj.dtype), obj.shape, pmask, attrs)
        elif isinstance(obj, list | tuple):
            # Convert lists/tuples to numpy arrays for compatibility
            arr = np.array(obj)
            return cls(DType.from_any(arr.dtype), arr.shape, pmask, attrs)
        else:
            raise TypeError(f"Unsupported type: {type(obj)}.")

    @classmethod
    def from_mpobj(cls, obj: MPObject) -> MPType:
        # assume obj is MPObject-like
        return cls(obj.dtype, obj.shape, obj.pmask, obj.attrs)


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
    def pmask(self) -> Mask | None:
        return self.mptype.pmask

    @property
    def attrs(self) -> dict[str, Any]:
        return self.mptype.attrs

    @property
    @abstractmethod
    def ctx(self) -> MPContext:
        """Return the context of the object."""


# Forword docstrings from MPType to MPObject
MPObject.dtype.__doc__ = MPType.dtype.__doc__
MPObject.shape.__doc__ = MPType.shape.__doc__
MPObject.pmask.__doc__ = MPType.pmask.__doc__
MPObject.attrs.__doc__ = MPType.attrs.__doc__
