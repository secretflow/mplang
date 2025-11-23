"""Homomorphic Aggregation library.

This module implements efficient aggregation algorithms using BFV rotation.
"""

from __future__ import annotations

import math
from typing import Any

from mplang2.dialects import bfv


def rotate_and_sum(ciphertext: Any, k: int, galois_keys: Any) -> Any:
    """Aggregate the first k elements of a ciphertext using O(log k) rotations.

    The result is placed in the 0-th slot (and other slots as side effects).

    Algorithm:
        Recursive doubling.
        To sum k elements:
        1. Rotate by -1, add. (Sums pairs)
        2. Rotate by -2, add. (Sums quads)
        ...

    Wait, the standard algorithm is:
    To sum N elements (where N is power of 2):
    for i in 0..logN-1:
        step = 2^i
        temp = rotate(current, -step)
        current = add(current, temp)

    Example N=4 [v0, v1, v2, v3]
    i=0, step=1:
        rot(-1) -> [v1, v2, v3, v0]
        add -> [v0+v1, v1+v2, v2+v3, v3+v0]
    i=1, step=2:
        rot(-2) -> [v2+v3, v3+v0, v0+v1, v1+v2]
        add -> [(v0+v1)+(v2+v3), ...] -> [sum, ...]

    If k is not a power of 2, we can just run for next_power_of_2(k).
    The result at slot 0 will be the sum of the first `next_power_of_2(k)` elements.
    Wait, if we want sum of exactly k elements, and k < N (slots),
    and the input has zeros elsewhere?

    The user prompt says: "Aggregate key1 corresponding values, distributed at slots 0, 3, 8".
    This implies we first need to align them?

    The user prompt example:
    "1. Aggregate key1 to slot 0:
       - Align v_k1_b (at slot 3) to slot 0 -> Rotate(-3)
       - Add."

    This is general sparse aggregation.

    But then the user asks about "Log-Step Rotation Sum" for "group of k values".
    "Assume we need to sum k values in a group... Correct way is recursive doubling".
    This implies the values are contiguous or we are summing a dense vector?

    If the values are contiguous [v0, v1, ..., vk-1], then the log-step algo works.
    If they are sparse, we might need to rotate each one to 0 (O(k) rotations).

    However, usually in "Sort and Aggregate" (which seems to be the context),
    after sorting, the values for the same key ARE contiguous.
    So we can assume we are summing a contiguous block.

    But we might have multiple blocks for different keys in the same ciphertext.
    e.g. [A, A, A, B, B, C, C, C, C]

    If we run the log-step algorithm on the whole vector:
    It sums everything in windows.

    Let's implement the generic `rotate_and_sum` which sums contiguous blocks of size `k`.
    Or rather, just sums the whole vector if we treat it as one block?

    The function signature `rotate_and_sum(ciphertext, k, ...)` suggests summing first k elements?

    Let's implement the "Power of 2" summation as described in the prompt.
    "For k elements, we need log2(k) rotations."

    Args:
        ciphertext: The BFV ciphertext.
        k: The number of elements to sum (assumed contiguous starting at 0, or we sum windows of size k?).
           Actually, the algorithm `sum_slots` usually sums the whole vector (N slots) into slot 0.
           If we only want to sum k elements, we can mask before or after?

           If we use the log-step algo for `steps = 1, 2, 4, ..., <k`,
           it effectively computes partial sums.

           Let's assume we want to sum a vector of size `k` (where `k` <= N).
           And we want the result in slot 0.

    """
    # Calculate number of steps needed
    # We need to cover distance k.
    # Actually, if we want to sum k elements [0..k-1] into slot 0.
    # We need to bring index 1 to 0, index 2 to 0, ..., index k-1 to 0.

    # The "Recursive Doubling" algorithm sums [0..N-1] into slot 0.
    # It works for N=2^p.
    # If k is not power of 2, we can round up to next power of 2.
    # But that requires the input to be padded with zeros if we don't want garbage.
    # Or we can be careful.

    # Let's implement the standard log-step reduction which sums everything into slot 0
    # assuming the relevant data is in the first k slots.
    # Note: This will also sum garbage from slots >= k if they are non-zero.
    # The caller is responsible for masking if needed.

    if k <= 1:
        return ciphertext

    num_steps = math.ceil(math.log2(k))
    current = ciphertext

    for i in range(num_steps):
        step = 1 << i
        # Rotate left by step (bring element at i+step to i)
        # rotate(ct, steps) -> positive is left?
        # bfv.rotate doc says: "Positive = left, Negative = right."
        # Wait, usually "rotate left" means shift elements to lower indices.
        # [v0, v1, v2] --rot(1)--> [v1, v2, v0].
        # Yes, that brings v1 to index 0.

        rotated = bfv.rotate(current, step, galois_keys)
        current = bfv.add(current, rotated)

    return current


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
