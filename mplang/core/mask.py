"""
Mask class for representing party masks in multi-party computation.

This class encapsulates mask data and operations, replacing the previous
int-based mask representation with a proper type-safe abstraction.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


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

    def __init__(self, value: int) -> None:
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
    def from_rank(cls, rank: int) -> Mask:
        """Create a mask with a single party."""
        if rank < 0:
            raise ValueError("Rank must be non-negative")
        return cls(1 << rank)

    @classmethod
    def from_ranks(cls, ranks: Iterable[int]) -> Mask:
        """Create a mask from multiple ranks."""
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
        return NotImplemented

    def __hash__(self) -> int:
        """Make Mask hashable."""
        return hash(self._value)

    def __repr__(self) -> str:
        """String representation of the mask."""
        return f"Mask({self._value})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        ranks = list(self.ranks())
        if not ranks:
            return "Mask()"
        return f"Mask({ranks})"

    def __format__(self, format_spec: str) -> str:
        """Support formatting for hexadecimal display."""
        return format(self._value, format_spec)

    def bit_count(self) -> int:
        """Count the number of parties in this mask."""
        return bin(self._value).count("1")

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
        if isinstance(other, int):
            other_mask = other
        else:
            other_mask = other._value
        return (self._value & other_mask) == 0

    def is_subset(self, other: Mask | int) -> bool:
        """Check if this mask is a subset of another."""
        if isinstance(other, int):
            other_mask = other
        else:
            other_mask = other._value
        return (self._value & other_mask) == self._value

    def is_superset(self, other: Mask | int) -> bool:
        """Check if this mask is a superset of another."""
        if isinstance(other, int):
            other_mask = other
        else:
            other_mask = other._value
        return (other_mask & self._value) == other_mask

    def union(self, other: Mask | int) -> Mask:
        """Return the union of this mask with another."""
        if isinstance(other, int):
            other_mask = other
        else:
            other_mask = other._value
        return Mask(self._value | other_mask)

    def intersection(self, other: Mask | int) -> Mask:
        """Return the intersection of this mask with another."""
        if isinstance(other, int):
            other_mask = other
        else:
            other_mask = other._value
        return Mask(self._value & other_mask)

    def difference(self, other: Mask | int) -> Mask:
        """Return the difference of this mask with another."""
        if isinstance(other, int):
            other_mask = other
        else:
            other_mask = other._value
        return Mask(self._value & ~other_mask)

    def __or__(self, other: Mask | int) -> Mask:
        """Union operator (|)."""
        return self.union(other)

    def __and__(self, other: Mask | int) -> Mask:
        """Intersection operator (&)."""
        return self.intersection(other)

    def __xor__(self, other: Mask | int) -> Mask:
        """Symmetric difference operator (^)."""
        if isinstance(other, int):
            other_mask = other
        else:
            other_mask = other._value
        return Mask(self._value ^ other_mask)

    def __sub__(self, other: Mask | int) -> Mask:
        """Difference operator (-)."""
        return self.difference(other)

    def __invert__(self) -> Mask:
        """Bitwise NOT operator (~)."""
        # Note: This creates a mask with potentially infinite bits set
        # Use with caution - typically you want to limit to a specific number of parties
        # For now, we limit to 64 bits to avoid negative values
        return Mask(~self._value & ((1 << 64) - 1))

    def global_to_relative_rank(self, global_rank: int) -> int:
        """Convert a global rank to relative rank within this mask."""
        if global_rank not in self:
            raise ValueError(f"Global rank {global_rank} not in mask")

        # Count set bits up to global_rank
        mask_up_to_rank = self._value & ((1 << (global_rank + 1)) - 1)
        return bin(mask_up_to_rank).count("1") - 1

    def relative_to_global_rank(self, relative_rank: int) -> int:
        """Convert a relative rank to global rank within this mask."""
        if relative_rank < 0 or relative_rank >= self.bit_count():
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

    def to_bytes(self, length: int = 8, byteorder: str = "big") -> bytes:
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
