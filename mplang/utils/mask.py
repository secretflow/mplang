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

"""Mask class and related utilities for multi-party computations."""

from __future__ import annotations

from typing import Iterator, Union

# Type aliases to avoid circular imports
Rank = int


class Mask:
    """A class to encapsulate mask data and operations for multi-party computations.
    
    A mask represents which parties participate in an operation, using a bitmask
    where the i'th bit is 1 if the i'th party is included, and 0 otherwise.
    """
    
    def __init__(self, value: Union[int, "Mask"]) -> None:
        """Initialize a Mask from an integer or another Mask.
        
        Args:
            value: The mask value as an integer or another Mask instance.
            
        Raises:
            ValueError: If the value is negative.
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
        """Get the underlying integer value of the mask."""
        return self._value
    
    def bit_count(self) -> int:
        """Return the number of set bits in the mask."""
        return self._value.bit_count()
    
    def enum(self) -> Iterator[int]:
        """Enumerate the indices of set bits in the mask.
        
        Yields:
            The indices of bits that are set to 1.
        """
        mask = self._value
        i = 0
        while mask != 0:
            if mask & 1:
                yield i
            mask >>= 1
            i += 1
    
    def is_rank_in(self, rank: Rank) -> bool:
        """Check if a rank is in the party mask.
        
        Args:
            rank: The rank to check.
            
        Returns:
            True if the rank is in the mask, False otherwise.
        """
        return (1 << rank) & self._value != 0
    
    def ensure_rank_in(self, rank: Rank) -> None:
        """Ensure a rank is in the party mask, raising an error if not.
        
        Args:
            rank: The rank to check.
            
        Raises:
            ValueError: If the rank is not in the mask.
        """
        if not self.is_rank_in(rank):
            raise ValueError(f"Rank {rank} is not in the party mask {self._value}")
    
    def is_subset(self, superset: "Mask") -> bool:
        """Check if this mask is a subset of another mask.
        
        Args:
            superset: The mask to check against.
            
        Returns:
            True if this mask is a subset of the superset.
        """
        superset_mask = Mask(superset)
        return (self._value & superset_mask._value) == self._value
    
    def ensure_subset(self, superset: "Mask") -> None:
        """Ensure this mask is a subset of another mask.
        
        Args:
            superset: The mask to check against.
            
        Raises:
            ValueError: If this mask is not a subset of the superset.
        """
        superset_mask = Mask(superset)
        if not self.is_subset(superset_mask):
            raise ValueError(
                f"Expect subset mask {self._value} to be a subset of superset mask {superset_mask._value}."
            )
    
    def global_to_relative_rank(self, global_rank: Rank) -> Rank:
        """Convert a global rank to a relative rank within this mask.
        
        Args:
            global_rank: The global rank to convert.
            
        Returns:
            The relative rank within the mask.
            
        Raises:
            ValueError: If the global rank is invalid or not in the mask.
        """
        if not (global_rank >= 0 and (self._value & (1 << global_rank))):
            raise ValueError(
                f"Invalid global_rank ({global_rank}) or bit not set in mask (0b{self._value:b})."
            )
        sub_mask = self._value & ((1 << (global_rank + 1)) - 1)
        return sub_mask.bit_count() - 1
    
    def relative_to_global_rank(self, relative_rank: Rank) -> Rank:
        """Convert a relative rank within this mask to a global rank.
        
        Args:
            relative_rank: The relative rank to convert.
            
        Returns:
            The global rank.
            
        Raises:
            ValueError: If the relative rank is invalid.
        """
        if not (0 <= relative_rank < self._value.bit_count()):
            raise ValueError(
                f"Invalid relative_rank ({relative_rank}) for mask (0b{self._value:b}) "
                f"with {self._value.bit_count()} set bits."
            )
        temp_mask = self._value
        for _ in range(relative_rank):
            temp_mask &= temp_mask - 1
        
        return (temp_mask & -temp_mask).bit_length() - 1
    
    # Arithmetic operations that return new Mask objects
    def __and__(self, other: Union[int, "Mask"]) -> "Mask":
        """Bitwise AND operation."""
        other_mask = Mask(other)
        return Mask(self._value & other_mask._value)
    
    def __or__(self, other: Union[int, "Mask"]) -> "Mask":
        """Bitwise OR operation."""
        other_mask = Mask(other)
        return Mask(self._value | other_mask._value)
    
    def __xor__(self, other: Union[int, "Mask"]) -> "Mask":
        """Bitwise XOR operation."""
        other_mask = Mask(other)
        return Mask(self._value ^ other_mask._value)
    
    def __lshift__(self, shift: int) -> "Mask":
        """Left shift operation."""
        return Mask(self._value << shift)
    
    def __rshift__(self, shift: int) -> "Mask":
        """Right shift operation."""
        return Mask(self._value >> shift)
    
    # Comparison operations
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if isinstance(other, Mask):
            return self._value == other._value
        elif isinstance(other, int):
            return self._value == other
        return False
    
    def __ne__(self, other: object) -> bool:
        """Inequality comparison."""
        return not self.__eq__(other)
    
    def __lt__(self, other: Union[int, "Mask"]) -> bool:
        """Less than comparison."""
        other_mask = Mask(other)
        return self._value < other_mask._value
    
    def __le__(self, other: Union[int, "Mask"]) -> bool:
        """Less than or equal comparison."""
        other_mask = Mask(other)
        return self._value <= other_mask._value
    
    def __gt__(self, other: Union[int, "Mask"]) -> bool:
        """Greater than comparison."""
        other_mask = Mask(other)
        return self._value > other_mask._value
    
    def __ge__(self, other: Union[int, "Mask"]) -> bool:
        """Greater than or equal comparison."""
        other_mask = Mask(other)
        return self._value >= other_mask._value
    
    # Hash and representation
    def __hash__(self) -> int:
        """Hash function for use in sets and dictionaries."""
        return hash(self._value)
    
    def __repr__(self) -> str:
        """String representation of the mask."""
        return f"Mask({self._value})"
    
    def __str__(self) -> str:
        """String representation of the mask."""
        return str(self._value)
    
    def __int__(self) -> int:
        """Convert to integer."""
        return self._value
    
    def __index__(self) -> int:
        """Support for use as an index."""
        return self._value
    
    # Support for 'in' operator with reversed operands
    def __contains__(self, rank: Rank) -> bool:
        """Check if a rank is in the mask (supports 'rank in mask' syntax)."""
        return self.is_rank_in(rank)


# Utility functions that work with multiple masks
def is_disjoint(*masks: Union[int, Mask]) -> bool:
    """Check if all masks are disjoint.
    
    Args:
        *masks: Variable number of masks to check.
        
    Returns:
        True if all masks are disjoint, False otherwise.
        
    Raises:
        ValueError: If any mask is invalid.
    """
    joint_mask = 0
    for mask in masks:
        mask_obj = Mask(mask)
        if joint_mask & mask_obj.value:
            return False
        joint_mask |= mask_obj.value
    return True


def join_masks(*masks: Union[int, Mask]) -> Mask:
    """Return the intersection (AND) of multiple masks.
    
    Args:
        *masks: Variable number of masks to join.
        
    Returns:
        A new Mask representing the intersection.
    """
    if not masks:
        return Mask(0)
    result_mask = Mask(masks[0])
    for mask in masks[1:]:
        result_mask = result_mask & mask
    return result_mask


def union_masks(*masks: Union[int, Mask]) -> Mask:
    """Return the union (OR) of multiple masks.
    
    Args:
        *masks: Variable number of masks to unite.
        
    Returns:
        A new Mask representing the union.
        
    Raises:
        ValueError: If no masks are provided.
    """
    if not masks:
        raise ValueError("At least one mask is required for union.")
    result_mask = Mask(masks[0])
    for mask in masks[1:]:
        result_mask = result_mask | mask
    return result_mask


# Backward compatibility functions for legacy code
def bit_count(mask: Union[int, Mask]) -> int:
    """Return the number of set bits in the mask.
    
    Args:
        mask: The mask to count bits in.
        
    Returns:
        The number of set bits.
    """
    return Mask(mask).bit_count()


def enum_mask(mask: Union[int, Mask]) -> Iterator[int]:
    """Enumerate the indices of set bits in the mask.
    
    Args:
        mask: The mask to enumerate.
        
    Yields:
        The indices of bits that are set to 1.
    """
    return Mask(mask).enum()


def global_to_relative_rank(global_rank: Rank, mask: Union[int, Mask]) -> Rank:
    """Convert a global rank to a relative rank within the mask.
    
    Args:
        global_rank: The global rank to convert.
        mask: The mask to use for conversion.
        
    Returns:
        The relative rank within the mask.
    """
    return Mask(mask).global_to_relative_rank(global_rank)


def relative_to_global_rank(relative_rank: Rank, mask: Union[int, Mask]) -> Rank:
    """Convert a relative rank within the mask to a global rank.
    
    Args:
        relative_rank: The relative rank to convert.
        mask: The mask to use for conversion.
        
    Returns:
        The global rank.
    """
    return Mask(mask).relative_to_global_rank(relative_rank)


def is_rank_in(rank: Rank, mask: Union[int, Mask]) -> bool:
    """Check if a rank is in the party mask.
    
    Args:
        rank: The rank to check.
        mask: The mask to check in.
        
    Returns:
        True if the rank is in the mask, False otherwise.
    """
    return Mask(mask).is_rank_in(rank)


def ensure_rank_in(rank: Rank, mask: Union[int, Mask]) -> None:
    """Ensure a rank is in the party mask.
    
    Args:
        rank: The rank to check.
        mask: The mask to check in.
        
    Raises:
        ValueError: If the rank is not in the mask.
    """
    Mask(mask).ensure_rank_in(rank)


def is_subset(subset_mask: Union[int, Mask], superset_mask: Union[int, Mask]) -> bool:
    """Check if the subset mask is a subset of the superset mask.
    
    Args:
        subset_mask: The mask to check as subset.
        superset_mask: The mask to check as superset.
        
    Returns:
        True if subset_mask is a subset of superset_mask.
    """
    return Mask(subset_mask).is_subset(superset_mask)


def ensure_subset(subset_mask: Union[int, Mask], superset_mask: Union[int, Mask]) -> None:
    """Ensure the subset mask is a subset of the superset mask.
    
    Args:
        subset_mask: The mask to check as subset.
        superset_mask: The mask to check as superset.
        
    Raises:
        ValueError: If subset_mask is not a subset of superset_mask.
    """
    Mask(subset_mask).ensure_subset(superset_mask)