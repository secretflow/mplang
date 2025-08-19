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
Utility functions for expression system.
"""

from collections.abc import Sequence

from mplang.core.mask import Mask
from mplang.core.mptype import TensorLike


def type_equal(*args: TensorLike) -> bool:
    """Check if tensors have identical type properties (dtype, shape).

    Args:
        *args: Variable number of TensorLike objects to compare

    Returns:
        bool: True if all tensors have identical types, False otherwise
    """
    if len(args) <= 1:
        return True
    for i in range(1, len(args)):
        if args[0].dtype != args[i].dtype or args[0].shape != args[i].shape:
            return False
    return True


def ensure_scalar(obj: TensorLike) -> None:
    """Ensure that a tensor is a scalar."""
    if len(obj.shape) != 0:
        raise TypeError(f"Expected a scalar, got {obj}.")


def ensure_tensorlist_equal(*args: Sequence[TensorLike]) -> None:
    """Ensure that multiple tensor lists have the same structure and types."""
    if len(args) < 2:
        raise ValueError(f"expect at least 2 args, got {len(args)}")
    for i in range(1, len(args)):
        if len(args[i]) != len(args[0]):
            raise ValueError(f"Length mismatch: {len(args[i])} vs {len(args[0])}")
        for j in range(len(args[0])):
            if not type_equal(args[0][j], args[i][j]):
                raise TypeError(f"Type mismatch: {args[0][j]} vs {args[i][j]}")


def deduce_mask(*pmasks: Mask | None) -> Mask | None:
    """Deduce the joint mask from multiple participant masks."""
    if len(pmasks) == 0:
        return None

    if any(pmask is None for pmask in pmasks):
        # If any pmask is None, we cannot deduce a specific mask.
        return None

    # return the joint mask of all provided pmasks.
    # We already checked above, but add it here to make mypy happy
    if pmasks[0] is None:
        return None
    result = Mask(pmasks[0])
    for pmask in pmasks[1:]:
        assert pmask is not None  # We already checked above
        result = result.intersection(Mask(pmask))

    return result
