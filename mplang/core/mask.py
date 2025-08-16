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

"""
Mask class for representing party masks in multi-party computation.

This class encapsulates mask data and operations, replacing the previous
int-based mask representation with a proper type-safe abstraction.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Literal


class Mask:
    """
    A mask representing a set of parties in multi-party computation.

    The mask uses bit positions to represent party ranks:
    - Bit 0 represents party 0
    - Bit 1 represents party 1
    - And so on...

    Examples:
        >>> mask = Mask.from_ranks([0, 1])  # Parties 0 and 1
        >>> mask = Mask.from_int(0b101)  # Parties 0 and 2
        >>> mask = Mask.all(3)  # All parties 0, 1, 2
    """

    _value: int

    def __init__(self, value: Mask | int) -> None:
        """
        Create a mask from an integer value.

        Args:
            value: Integer where each bit represents a party

        Raises:
            ValueError: If value is negative
        """
        if isinstance(value, Mask):
            self._value = value._value
        else:
            if value < 0:
                raise ValueError("Mask value must be non-negative")
            self._value = int(value)

    @classmethod
    def from_int(cls, value: int) -> Mask:
        """Create a mask from an integer."""
        return cls(value)

    @classmethod
    def from_ranks(cls, ranks: int | Iterable[int]) -> Mask:
        """
        Create a mask from one or more ranks.

        Args:
            ranks: Either a single integer rank or an iterable of integer ranks

        Returns:
            Mask with the specified ranks set

        Examples:
            >>> Mask.from_ranks(0)  # Single party 0
            >>> Mask.from_ranks([0, 1, 2])  # Multiple parties
            >>> Mask.from_ranks((1, 3))  # Tuple of parties
        """
        if isinstance(ranks, int):
            if ranks < 0:
                raise ValueError("Rank must be non-negative")
            return cls(1 << ranks)

        mask_value = 0
        for rank in ranks:
            if rank < 0:
                raise ValueError("All ranks must be non-negative")
            mask_value |= 1 << rank
        return cls(mask_value)

    @classmethod
    def all(cls, num_parties: int) -> Mask:
        """Create a mask with all parties up to num_parties-1."""
        if num_parties < 0:
            raise ValueError("Number of parties must be non-negative")
        if num_parties == 0:
            return cls(0)
        return cls((1 << num_parties) - 1)

    @classmethod
    def none(cls) -> Mask:
        """Create an empty mask."""
        return cls(0)

    @staticmethod
    def _ensure_mask_value(value: Mask | int) -> int:
        """
        Ensure a value is converted to its underlying integer mask.

        Args:
            value: Either a Mask instance or an integer

        Returns:
            The underlying integer value of the mask
        """
        if isinstance(value, Mask):
            return value._value
        else:
            return int(value)

    @property
    def value(self) -> int:
        """Get the underlying integer value."""
        return self._value

    def __int__(self) -> int:
        """Allow implicit conversion to int."""
        return self._value

    def __eq__(self, other: object) -> bool:
        """Check equality with another mask or int."""
        if isinstance(other, Mask):
            return self._value == other._value
        elif isinstance(other, int):
            return self._value == other
        else:
            raise TypeError("Invalid type for equal comparison")

    def __hash__(self) -> int:
        """Make Mask hashable."""
        return hash(self._value)

    def __repr__(self) -> str:
        """String representation of the mask."""
        return f"Mask({bin(self._value)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        ranks = list(self.ranks())
        if not ranks:
            return "Mask()"
        return f"Mask({ranks})"

    def __format__(self, format_spec: str) -> str:
        """Support formatting for hexadecimal display."""
        return format(self._value, format_spec)

    def num_parties(self) -> int:
        """Count the number of parties in this mask."""
        return self._value.bit_count()

    def ranks(self) -> Iterator[int]:
        """Iterate over the ranks in this mask."""
        value = self._value
        rank = 0
        while value > 0:
            if value & 1:
                yield rank
            value >>= 1
            rank += 1

    def __iter__(self) -> Iterator[int]:
        """Allow iteration over ranks."""
        return self.ranks()

    def __contains__(self, rank: int) -> bool:
        """Check if a rank is in this mask."""
        if rank < 0:
            return False
        return (self._value & (1 << rank)) != 0

    def is_disjoint(self, other: Mask | int) -> bool:
        """Check if this mask is disjoint with another."""
        other_mask_value = self._ensure_mask_value(other)
        return (self._value & other_mask_value) == 0

    def is_subset(self, other: Mask | int) -> bool:
        """Check if this mask is a subset of another."""
        other_mask_value = self._ensure_mask_value(other)
        return (self._value & other_mask_value) == self._value

    def is_superset(self, other: Mask | int) -> bool:
        """Check if this mask is a superset of another."""
        other_mask_value = self._ensure_mask_value(other)
        return (other_mask_value & self._value) == other_mask_value

    def union(self, other: Mask | int) -> Mask:
        """Return the union of this mask with another."""
        other_mask_value = self._ensure_mask_value(other)
        return Mask(self._value | other_mask_value)

    def intersection(self, other: Mask | int) -> Mask:
        """Return the intersection of this mask with another."""
        other_mask_value = self._ensure_mask_value(other)
        return Mask(self._value & other_mask_value)

    def difference(self, other: Mask | int) -> Mask:
        """Return the difference of this mask with another."""
        other_mask_value = self._ensure_mask_value(other)
        return Mask(self._value & Mask._invert_mask_value(other_mask_value))

    def __or__(self, other: Mask | int) -> Mask:
        """Union operator (|)."""
        return self.union(other)

    def __and__(self, other: Mask | int) -> Mask:
        """Intersection operator (&)."""
        return self.intersection(other)

    def __xor__(self, other: Mask | int) -> Mask:
        """Symmetric difference operator (^)."""
        other_mask_value = self._ensure_mask_value(other)
        return Mask(self._value ^ other_mask_value)

    def __sub__(self, other: Mask | int) -> Mask:
        """Difference operator (-)."""
        return self.difference(other)

    @staticmethod
    def _invert_mask_value(value: int) -> int:
        # Invert the bits of the mask value
        # Use with caution - typically you want to limit to a specific number of parties
        # For now, we limit to 64 bits to avoid negative values
        return ~value & ((1 << 64) - 1)

    def __invert__(self) -> Mask:
        """Bitwise NOT operator (~)."""
        # Note: This creates a mask with potentially infinite bits set
        return Mask(Mask._invert_mask_value(self._value))

    def global_to_relative_rank(self, global_rank: int) -> int:
        """Convert a global rank to relative rank within this mask."""
        if global_rank not in self:
            raise ValueError(f"Global rank {global_rank} not in mask")

        # Count set bits up to global_rank
        mask_up_to_rank = self._value & ((1 << (global_rank + 1)) - 1)
        return bin(mask_up_to_rank).count("1") - 1

    def relative_to_global_rank(self, relative_rank: int) -> int:
        """Convert a relative rank to global rank within this mask."""
        if relative_rank < 0 or relative_rank >= self.num_parties():
            raise ValueError(f"Relative rank {relative_rank} out of range")

        count = 0
        global_rank = 0
        value = self._value

        while value > 0 and count <= relative_rank:
            if value & 1:
                if count == relative_rank:
                    return global_rank
                count += 1
            value >>= 1
            global_rank += 1

        raise ValueError(f"Relative rank {relative_rank} not found in mask")

    def copy(self) -> Mask:
        """Return a copy of this mask."""
        return Mask(self._value)

    def to_bytes(
        self, length: int = 8, byteorder: Literal["little", "big"] = "big"
    ) -> bytes:
        """Convert mask to bytes for serialization."""
        return self._value.to_bytes(length, byteorder=byteorder)

    @property
    def is_empty(self) -> bool:
        """Check if this mask is empty."""
        return self._value == 0

    @property
    def is_single(self) -> bool:
        """Check if this mask contains exactly one party."""
        return (self._value & (self._value - 1)) == 0 and self._value != 0

    def to_json(self) -> int:
        """Serialize to JSON-compatible format."""
        return self._value

    @classmethod
    def from_json(cls, value: int) -> Mask:
        """Deserialize from JSON-compatible format."""
        return cls(value)

    @classmethod
    def from_bytes(
        cls, data: bytes, byteorder: Literal["little", "big"] = "big"
    ) -> Mask:
        """
        Create a mask from bytes for deserialization.

        Args:
            data: Bytes to convert to mask
            byteorder: Byte order ('little' or 'big')

        Returns:
            Mask created from the bytes

        Examples:
            >>> mask = Mask.from_bytes(b"\x05", byteorder="big")
            >>> mask.value == 5
            True
            >>> mask = Mask.from_bytes(b"\x05\x00", byteorder="little")
            >>> mask.value == 5
            True
        """
        value = int.from_bytes(data, byteorder=byteorder)
        return cls(value)
