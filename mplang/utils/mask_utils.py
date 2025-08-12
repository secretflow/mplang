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

from collections.abc import Iterator

from mplang.core.base import Mask, Rank


def _is_valid_mask(mask: Mask) -> bool:
    """Check if the mask is a valid non-negative integer."""
    return isinstance(mask, int) and mask >= 0


def bit_count(mask: Mask) -> int:
    assert mask >= 0, f"Expect non-negative mask, got {mask}"
    return mask.bit_count()


def enum_mask(mask: Mask) -> Iterator[int]:
    assert isinstance(mask, Mask) and mask >= 0, f"Expect non-negative mask, got {mask}"
    i = 0
    while mask != 0:
        if mask & 1:
            yield i
        mask >>= 1
        i += 1


def is_disjoint(*masks: Mask) -> bool:
    """Check if all masks are disjoint."""
    joint_mask = 0
    for mask in masks:
        if not _is_valid_mask(mask):
            raise ValueError(f"Not a valid mask, got {mask}")
        if joint_mask & mask:
            return False
        joint_mask |= mask
    return True


def join_masks(*masks: Mask) -> Mask:
    """Return the intersection of multiple masks."""
    if not masks:
        return 0
    result_mask = masks[0]
    for mask in masks[1:]:
        if not _is_valid_mask(mask):
            raise ValueError(f"Not a valid mask, got {mask}")
        result_mask &= mask
    return result_mask


def union_masks(*masks: Mask) -> Mask:
    """Return the union of multiple masks."""
    if not masks:
        raise ValueError("At least one mask is required for union.")
    result_mask = masks[0]
    for mask in masks[1:]:
        if not _is_valid_mask(mask):
            raise ValueError(f"Not a valid mask, got {mask}")
        result_mask |= mask
    return result_mask


def global_to_relative_rank(global_rank: int, mask: int) -> int:
    if not (global_rank >= 0 and (mask & (1 << global_rank))):
        raise ValueError(
            f"Invalid global_rank ({global_rank}) or bit not set in mask (0b{mask:b})."
        )
    sub_mask = mask & ((1 << (global_rank + 1)) - 1)
    return sub_mask.bit_count() - 1


def relative_to_global_rank(relative_rank: int, mask: int) -> int:
    if not (0 <= relative_rank < mask.bit_count()):
        raise ValueError(
            f"Invalid relative_rank ({relative_rank}) for mask (0b{mask:b}) "
            f"with {mask.bit_count()} set bits."
        )
    temp_mask = mask
    for _ in range(relative_rank):
        temp_mask &= temp_mask - 1

    return (temp_mask & -temp_mask).bit_length() - 1


def is_rank_in(rank: Rank, mask: Mask) -> bool:
    """Check if a rank is in the party mask."""
    return (1 << rank) & mask != 0


def ensure_rank_in(rank: Rank, mask: Mask) -> None:
    if (1 << rank) & mask == 0:
        raise ValueError(f"Rank {rank} is not in the party mask {mask}")


def is_subset(subset_mask: Mask, superset_mask: Mask) -> bool:
    return (subset_mask & superset_mask) == subset_mask


def ensure_subset(subset_mask: Mask, superset_mask: Mask) -> None:
    if not is_subset(subset_mask, superset_mask):
        raise ValueError(
            f"Expect subset mask {subset_mask} to be a subset of superset mask {superset_mask}."
        )
