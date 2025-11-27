"""Homomorphic Aggregation library.

This module implements efficient aggregation algorithms using BFV rotation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mplang2.dialects import bfv, tensor


def rotate_and_sum(ciphertext: Any, k: int, galois_keys: Any) -> Any:
    """Aggregate the first k elements of a ciphertext using O(log k) rotations.

    The result is placed in the 0-th slot.
    This assumes the input ciphertext has relevant data in slots 0..k-1
    and zeros (or irrelevant data) elsewhere, OR that the caller will mask the result.

    Args:
        ciphertext: The BFV ciphertext.
        k: The number of elements to sum.
        galois_keys: Keys required for rotation.

    Returns:
        Ciphertext where slot 0 contains sum(ciphertext[0..k-1]).
    """
    # We use the recursive doubling algorithm.
    # To sum k elements, we need ceil(log2(k)) steps.
    # If k is not a power of 2, we can treat it as next_power_of_2
    # provided we don't add garbage.
    # For safety, this function implements the standard log-step reduction
    # which sums windows of size 2^p.

    num_steps = math.ceil(math.log2(k))
    current = ciphertext

    for i in range(num_steps):
        step = 1 << i
        # Rotate left by step
        # We need to ensure we have the key for this step.
        # In a real library, we'd check or generate keys.
        rotated = bfv.rotate(current, step, galois_keys)
        current = bfv.add(current, rotated)

    return current


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
