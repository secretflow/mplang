# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Homomorphic Aggregation library.

This module implements efficient aggregation algorithms using BFV rotation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mplang.v2.dialects import bfv, tensor


def _safe_rotate(
    ciphertext: Any, step: int, galois_keys: Any, max_step: int = 1024
) -> Any:
    """Rotate ciphertext by step, decomposing large steps if needed.

    SEAL's rotate_rows requires step to be in range (-slot_count/2, slot_count/2).
    For poly_modulus_degree=4096, slot_count=4096, max valid step is 2047.
    We use a conservative max_step=1024 by default for safety.

    For large steps, we decompose into multiple rotations:
    - rotate(x, 3000) = rotate(rotate(rotate(x, 1024), 1024), 952)
    """
    if step == 0:
        return ciphertext
    if abs(step) <= max_step:
        return bfv.rotate(ciphertext, step, galois_keys)

    # Decompose large step into multiple rotations
    current = ciphertext
    remaining = abs(step)
    sign = 1 if step > 0 else -1

    while remaining > 0:
        rot = min(remaining, max_step)
        current = bfv.rotate(current, sign * rot, galois_keys)
        remaining -= rot

    return current


def _rotate_and_sum_row(
    ciphertext: Any, k: int, galois_keys: Any, max_step: int = 1024
) -> Any:
    """Sum first k elements within a single row (k <= row_size).

    Uses the recursive doubling algorithm with O(log k) rotations.
    """
    if k <= 1:
        return ciphertext

    num_steps = math.ceil(math.log2(k))
    current = ciphertext

    for i in range(num_steps):
        step = 1 << i
        if step >= k:
            break
        rotated = _safe_rotate(current, step, galois_keys, max_step)
        current = bfv.add(current, rotated)

    return current


def rotate_and_sum(
    ciphertext: Any, k: int, galois_keys: Any, slot_count: int = 4096
) -> Any:
    """Aggregate the first k elements of a ciphertext using O(log k) rotations.

    The result is placed in the 0-th slot.
    This assumes the input ciphertext has relevant data in slots 0..k-1
    and zeros (or irrelevant data) elsewhere, OR that the caller will mask the result.

    Args:
        ciphertext: The BFV ciphertext.
        k: The number of elements to sum.
        galois_keys: Keys required for rotation.
        slot_count: Total number of slots (default 4096 for poly_degree=4096).

    Returns:
        Ciphertext where slot 0 contains sum(ciphertext[0..k-1]).

    Note:
        SEAL batching arranges slots as 2 rows of slot_count/2 each.
        - rotate_rows rotates within each row (circular)
        - rotate_columns swaps the two rows

        For k <= row_size (2048), only row rotations are needed.
        For k > row_size, we use rotate_columns to aggregate across rows.
    """
    row_size = slot_count // 2

    if k <= row_size:
        # Simple case: all elements in row 0
        return _rotate_and_sum_row(ciphertext, k, galois_keys)

    # k > row_size: data spans both rows
    # Strategy:
    # 1. Sum row 0 completely (row_size elements)
    # 2. rotate_columns to bring row 1 to row 0 position
    # 3. Sum the first (k - row_size) elements of what was row 1
    # 4. Add the two partial sums

    # Sum row 0 completely
    row0_sum = _rotate_and_sum_row(ciphertext, row_size, galois_keys)

    # Rotate columns: swap row 0 <-> row 1
    # Now row 1's data is in row 0 position
    col_rotated = bfv.rotate_columns(ciphertext, galois_keys)

    # Sum the first (k - row_size) elements (originally in row 1)
    row1_count = k - row_size
    row1_sum = _rotate_and_sum_row(col_rotated, row1_count, galois_keys)

    # Both row0_sum and row1_sum have their results in slot 0
    # Add them together
    return bfv.add(row0_sum, row1_sum)


def aggregate_sparse(
    ciphertext: Any,
    aggregations: list[tuple[int, list[int]]],
    galois_keys: Any,
    encoder: Any,
    vector_size: int,
) -> Any:
    """Perform sparse aggregation.

    Args:
        ciphertext: Input ciphertext.
        aggregations: List of (target_slot, [source_slots]).
                      e.g. [(0, [0, 3, 8]), (1, [1, 5])]
        galois_keys: Rotation keys.
        encoder: BFV encoder for encoding masks.
        vector_size: Total size of the vector (slots).

    Returns:
        Ciphertext with aggregated results in target slots.
    """
    # Naive approach: For each target, sum sources.
    # Optimized approach:
    # 1. Decompose into rotations.
    #    For target t, source s: need rotation by (t-s).
    #    Group by rotation amount.
    # 2. Apply rotations and accumulate.

    # Map: rotation_amount -> mask
    # We want to compute: Result = Sum( Rotate(Input, r) * Mask_r )
    # where Mask_r has 1 at slot t if (t - r) is a source for t.

    # Example: t=0, s={0, 3, 8}.
    #   s=0: rot=0. Mask[0]=1.
    #   s=3: rot=-3. Mask[0]=1.
    #   s=8: rot=-8. Mask[0]=1.
    # Example: t=1, s={1, 5}.
    #   s=1: rot=0. Mask[1]=1.
    #   s=5: rot=-4. Mask[1]=1.

    # Combined:
    # Rot 0: Mask[0]=1, Mask[1]=1. -> Mask = [1, 1, 0...]
    # Rot -3: Mask[0]=1. -> Mask = [1, 0...]
    # Rot -8: Mask[0]=1. -> Mask = [1, 0...]
    # Rot -4: Mask[1]=1. -> Mask = [0, 1, 0...]

    rotations = {}  # shift -> mask_list

    for target, sources in aggregations:
        for src in sources:
            shift = src - target
            if shift not in rotations:
                rotations[shift] = [0] * vector_size
            rotations[shift][target] = 1

    final_result = None

    for shift, mask_list in rotations.items():
        # Optimization: Skip if mask is all zeros (no contribution from this rotation)
        if not any(mask_list):
            continue

        # Create mask plaintext
        # In a real implementation, we encode this list to a Plaintext
        mask_tensor = tensor.constant(np.array(mask_list, dtype=np.int64))
        mask_pt = bfv.encode(mask_tensor, encoder)

        # Rotate
        if shift == 0:
            rotated_c = ciphertext
        else:
            rotated_c = bfv.rotate(ciphertext, shift, galois_keys)

        # Mask
        masked_c = bfv.mul(rotated_c, mask_pt)

        # Accumulate
        if final_result is None:
            final_result = masked_c
        else:
            final_result = bfv.add(final_result, masked_c)

    return final_result


def masked_aggregate(ciphertexts: list[Any], masks: list[Any]) -> Any:
    """Aggregate multiple partial results using masks.

    Args:
        ciphertexts: List of ciphertexts.
        masks: List of plaintexts (masks).

    Returns:
        Sum(ct * mask)
    """
    if not ciphertexts or not masks:
        raise ValueError("Empty input lists")
    if len(ciphertexts) != len(masks):
        raise ValueError("Mismatch in ciphertexts and masks length")

    total = None

    for ct, mask in zip(ciphertexts, masks, strict=True):
        # ct * mask
        masked = bfv.mul(ct, mask)

        if total is None:
            total = masked
        else:
            total = bfv.add(total, masked)

    return total


# ==============================================================================
# SIMD Bucket Packing for Histogram Computation
# ==============================================================================


def strided_rotate_and_sum(
    ciphertext: Any,
    stride: int,
    n_elements: int,
    galois_keys: Any,
    max_step: int = 1024,
) -> Any:
    """Aggregate elements at positions [0, stride, 2*stride, ...] into slot 0.

    This is used for SIMD bucket packing where each bucket's values are
    placed at strided positions.

    Args:
        ciphertext: The BFV ciphertext with values at strided positions.
        stride: Distance between consecutive elements to sum.
        n_elements: Number of elements to aggregate (at positions 0, stride, ..., (n-1)*stride).
        galois_keys: Rotation keys.
        max_step: Maximum rotation step for safe_rotate.

    Returns:
        Ciphertext where slot 0 contains sum of strided elements.

    Example:
        stride=64, n_elements=47 (bucket has 47 samples)
        Values at slots: 0, 64, 128, 192, ...
        Result: slot[0] = sum of all these values
    """
    if n_elements <= 1:
        return ciphertext

    # Use recursive doubling with strided rotations
    # Step 1: rotate by stride, add -> pairs summed at even positions
    # Step 2: rotate by 2*stride, add -> quads summed at positions 0, 4*stride, ...
    # ...
    num_steps = math.ceil(math.log2(n_elements))
    current = ciphertext

    for i in range(num_steps):
        step = stride * (1 << i)
        if step >= n_elements * stride:
            break
        rotated = _safe_rotate(current, step, galois_keys, max_step)
        current = bfv.add(current, rotated)

    return current


def batch_bucket_aggregate(
    ciphertext: Any,
    n_buckets: int,
    samples_per_bucket: int,
    galois_keys: Any,
    slot_count: int = 4096,
) -> Any:
    """Aggregate samples within each bucket region in a packed ciphertext.

    Assumes the ciphertext has the following layout:
    - slot_count is divided into n_buckets regions of size `stride = slot_count // n_buckets`
    - Each bucket b occupies slots [b*stride, b*stride + samples_per_bucket)
    - Samples are placed at consecutive positions within their bucket region

    After aggregation, slot[b * stride] contains sum of bucket b.

    Args:
        ciphertext: Packed ciphertext with samples in bucket regions.
        n_buckets: Number of buckets.
        samples_per_bucket: Max samples per bucket (for rotation count).
        galois_keys: Rotation keys.
        slot_count: Total BFV slots.

    Returns:
        Ciphertext where slot[b * stride] = sum of bucket b's values.
    """
    if samples_per_bucket <= 1:
        return ciphertext

    # Use recursive doubling within each bucket region
    # Since all buckets use the same relative positions, one set of rotations
    # aggregates ALL buckets simultaneously!
    num_steps = math.ceil(math.log2(samples_per_bucket))
    current = ciphertext

    for i in range(num_steps):
        step = 1 << i
        if step >= samples_per_bucket:
            break
        # Rotating by `step` shifts values within each bucket region
        # Add original + rotated to sum pairs/quads/etc.
        rotated = _safe_rotate(current, step, galois_keys)
        current = bfv.add(current, rotated)

    return current


def extract_bucket_results(
    vector: Any,
    n_buckets: int,
    slot_count: int = 4096,
) -> Any:
    """Extract bucket sums from a packed result vector.

    After batch_bucket_aggregate, each bucket's sum is at slot[b * stride].
    This function extracts those values.

    Args:
        vector: Decoded vector from packed ciphertext.
        n_buckets: Number of buckets.
        slot_count: Total slots.

    Returns:
        (n_buckets,) array of bucket sums.
    """
    import jax.numpy as jnp

    stride = slot_count // n_buckets
    indices = jnp.arange(n_buckets) * stride
    return vector[indices]
