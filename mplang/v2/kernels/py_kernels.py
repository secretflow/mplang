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

"""Pure Python implementations of performance-critical kernels.

These implementations provide fallback functionality when native C++ kernels
(libmplang_kernels.so) are not available. They are functionally correct but
significantly slower than the optimized C++ versions.
"""

from __future__ import annotations

import numpy as np

from mplang.v2.libs.mpc.common.constants import (
    GOLDEN_RATIO_64,
    SPLITMIX64_GAMMA_2,
    SPLITMIX64_GAMMA_3,
    SPLITMIX64_GAMMA_4,
)

# =============================================================================
# GF(2^128) Arithmetic
# =============================================================================

# Irreducible polynomial: P(x) = x^128 + x^7 + x^2 + x + 1
# In polynomial basis, this means x^128 = x^7 + x^2 + x + 1 (mod P)
_GF128_POLYNOMIAL = 0x87  # x^7 + x^2 + x + 1 = 0b10000111 = 135


def _gf128_clmul64(a: int, b: int) -> tuple[int, int]:
    """Carryless multiplication of two 64-bit integers.

    Returns (lo, hi) where result = hi * 2^64 + lo.
    """
    result_lo = 0
    result_hi = 0

    for i in range(64):
        if (b >> i) & 1:
            # Add a shifted by i positions
            shifted_lo = (a << i) & ((1 << 64) - 1)
            shifted_hi = a >> (64 - i) if i > 0 else 0
            result_lo ^= shifted_lo
            result_hi ^= shifted_hi

    return result_lo, result_hi


def _gf128_clmul128(
    a_lo: int, a_hi: int, b_lo: int, b_hi: int
) -> tuple[int, int, int, int]:
    """Carryless multiplication of two 128-bit values.

    Returns (r0, r1, r2, r3) where result = r3 * 2^192 + r2 * 2^128 + r1 * 2^64 + r0.
    """
    # a_lo * b_lo -> [0:128]
    t0_lo, t0_hi = _gf128_clmul64(a_lo, b_lo)

    # a_hi * b_hi -> [128:256]
    t1_lo, t1_hi = _gf128_clmul64(a_hi, b_hi)

    # a_lo * b_hi -> [64:192]
    t2_lo, t2_hi = _gf128_clmul64(a_lo, b_hi)

    # a_hi * b_lo -> [64:192]
    t3_lo, t3_hi = _gf128_clmul64(a_hi, b_lo)

    # Combine cross terms
    mid_lo = t2_lo ^ t3_lo
    mid_hi = t2_hi ^ t3_hi

    # Result accumulation
    r0 = t0_lo
    r1 = t0_hi ^ mid_lo
    r2 = t1_lo ^ mid_hi
    r3 = t1_hi

    # Handle carry from r1 to r2 (carryless, just XOR overflow)
    # In carryless arithmetic, there's no carry propagation

    return r0, r1, r2, r3


def _gf128_reduce(r0: int, r1: int, r2: int, r3: int) -> tuple[int, int]:
    """Reduce 256-bit polynomial modulo P(x) = x^128 + x^7 + x^2 + x + 1.

    Returns (lo, hi) representing the 128-bit result.
    """
    # Reduction: x^128 = x^7 + x^2 + x + 1 (mod P)
    # So we need to reduce r2 and r3 into r0 and r1

    # r3 contributes at positions [192:256], which after reduction affects [64:128] and [0:64]
    # r2 contributes at positions [128:192], which after reduction affects [0:64]

    # First, reduce r3 (bits 192-255)
    # x^192 = x^64 * x^128 = x^64 * (x^7 + x^2 + x + 1)
    #       = x^71 + x^66 + x^65 + x^64
    # x^256 is beyond our range, but r3 represents bits [192:256]

    # For each bit position p in [192:255] that is set:
    # x^p = x^(p-128) * x^128 = x^(p-128) * 0x87
    # This means bit at position p reduces to XOR with 0x87 shifted by (p-128)

    # Simpler approach: reduce in two stages

    # Stage 1: Reduce r3 (affects r1 and r0 after multiple reductions)
    # r3 * x^192 mod P = r3 * x^64 * (x^7 + x^2 + x + 1)
    q3_lo, q3_hi = _gf128_clmul64(r3, _GF128_POLYNOMIAL)
    # This gives us bits at [64+0:64+128] = [64:192]
    # So it affects r1 and r2

    r1 ^= q3_lo
    r2 ^= q3_hi

    # Stage 2: Reduce r2 (affects r0 and r1)
    # r2 * x^128 mod P = r2 * 0x87
    q2_lo, q2_hi = _gf128_clmul64(r2, _GF128_POLYNOMIAL)
    # This gives bits at [0:128]

    r0 ^= q2_lo
    r1 ^= q2_hi

    return r0, r1


def gf128_mul_single(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two GF(2^128) elements.

    Args:
        a: Shape (2,) uint64 array representing a 128-bit element [lo, hi]
        b: Shape (2,) uint64 array representing a 128-bit element [lo, hi]

    Returns:
        Shape (2,) uint64 array representing the product
    """
    a_lo, a_hi = int(a[0]), int(a[1])
    b_lo, b_hi = int(b[0]), int(b[1])

    r0, r1, r2, r3 = _gf128_clmul128(a_lo, a_hi, b_lo, b_hi)
    res_lo, res_hi = _gf128_reduce(r0, r1, r2, r3)

    return np.array(
        [res_lo & ((1 << 64) - 1), res_hi & ((1 << 64) - 1)], dtype=np.uint64
    )


def gf128_mul_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch multiply GF(2^128) elements.

    Args:
        a: Shape (..., 2) uint64 array
        b: Shape (..., 2) uint64 array

    Returns:
        Shape (..., 2) uint64 array of products
    """
    original_shape = a.shape
    a_flat = a.reshape(-1, 2)
    b_flat = b.reshape(-1, 2)
    n = a_flat.shape[0]

    result = np.zeros_like(a_flat)
    for i in range(n):
        result[i] = gf128_mul_single(a_flat[i], b_flat[i])

    return result.reshape(original_shape)


# =============================================================================
# OKVS (Oblivious Key-Value Store) - 3-Hash Garbled Cuckoo Table
# =============================================================================


def _hash_key_py(key: int, m: int, seed: tuple[int, int]) -> tuple[int, int, int]:
    """Hash a key to 3 distinct indices using simple polynomial hashing.

    This is a pure Python approximation of the AES-based hash in C++.
    For compatibility, we use a deterministic hash based on the key.
    """
    # Simple polynomial hash (not as secure as AES, but deterministic)
    s0, s1 = seed

    # Mix key with seed
    h1 = ((key * GOLDEN_RATIO_64) ^ s0) & ((1 << 64) - 1)
    h2 = ((key * SPLITMIX64_GAMMA_2) ^ s1) & ((1 << 64) - 1)

    # Additional mixing
    h1 = ((h1 ^ (h1 >> 33)) * SPLITMIX64_GAMMA_3) & ((1 << 64) - 1)
    h2 = ((h2 ^ (h2 >> 33)) * SPLITMIX64_GAMMA_4) & ((1 << 64) - 1)

    idx1 = h1 % m
    idx2 = h2 % m
    idx3 = (h1 ^ h2) % m

    # Enforce distinct indices
    if idx2 == idx1:
        idx2 = (idx2 + 1) % m
    if idx3 == idx1 or idx3 == idx2:
        idx3 = (idx3 + 1) % m
        if idx3 == idx1 or idx3 == idx2:
            idx3 = (idx3 + 1) % m

    return int(idx1), int(idx2), int(idx3)


def okvs_solve(
    keys: np.ndarray,
    values: np.ndarray,
    m: int,
    seed: tuple[int, int] = (0xDEADBEEF, 0xCAFEBABE),
) -> np.ndarray:
    """Solve the OKVS system using peeling algorithm.

    Args:
        keys: Shape (n,) uint64 array of keys
        values: Shape (n, 2) uint64 array of values (128-bit each)
        m: Size of output storage

    Returns:
        Shape (m, 2) uint64 array representing the OKVS storage
    """
    n = len(keys)

    # Build graph: for each row, compute its 3 column indices
    rows = []
    col_to_rows: dict[int, list[int]] = {j: [] for j in range(m)}

    for i in range(n):
        h1, h2, h3 = _hash_key_py(int(keys[i]), m, seed)
        rows.append((h1, h2, h3))
        col_to_rows[h1].append(i)
        col_to_rows[h2].append(i)
        col_to_rows[h3].append(i)

    # Compute column degrees
    col_degree = [len(col_to_rows[j]) for j in range(m)]

    # Initialize peel queue with degree-1 columns
    peel_queue = [j for j in range(m) if col_degree[j] == 1]

    row_removed = [False] * n
    col_removed = [False] * m
    assignment_stack: list[tuple[int, int]] = []  # (col, row)

    head = 0
    while head < len(peel_queue):
        j = peel_queue[head]
        head += 1

        if col_removed[j]:
            continue

        # Find the single active row for this column
        owner_row = -1
        for r_idx in col_to_rows[j]:
            if not row_removed[r_idx]:
                owner_row = r_idx
                break

        if owner_row == -1:
            col_removed[j] = True
            continue

        # Peel this (column, row) pair
        assignment_stack.append((j, owner_row))
        col_removed[j] = True
        row_removed[owner_row] = True

        # Update neighbor column degrees
        h1, h2, h3 = rows[owner_row]
        for neighbor in (h1, h2, h3):
            if neighbor == j or col_removed[neighbor]:
                continue
            col_degree[neighbor] -= 1
            if col_degree[neighbor] == 1:
                peel_queue.append(neighbor)

    if len(assignment_stack) != n:
        raise RuntimeError(
            f"OKVS core detected. Failed to peel all rows. "
            f"n={n}, m={m}, solved={len(assignment_stack)}"
        )

    # Back substitution (solve in reverse order)
    output = np.zeros((m, 2), dtype=np.uint64)

    for col, row in reversed(assignment_stack):
        h1, h2, h3 = rows[row]
        # Current sum of columns in this row
        current_sum = output[h1] ^ output[h2] ^ output[h3]
        # Compute value needed for col to make sum equal target
        target = values[row]
        diff = target ^ current_sum
        output[col] = diff

    return output


def okvs_decode(
    keys: np.ndarray,
    storage: np.ndarray,
    m: int,
    seed: tuple[int, int] = (0xDEADBEEF, 0xCAFEBABE),
) -> np.ndarray:
    """Decode values from OKVS storage.

    Args:
        keys: Shape (n,) uint64 array of keys to query
        storage: Shape (m, 2) uint64 array (the solved OKVS)
        m: Size of storage

    Returns:
        Shape (n, 2) uint64 array of decoded values
    """
    n = len(keys)
    output = np.zeros((n, 2), dtype=np.uint64)

    for i in range(n):
        h1, h2, h3 = _hash_key_py(int(keys[i]), m, seed)
        output[i] = storage[h1] ^ storage[h2] ^ storage[h3]

    return output


# =============================================================================
# AES-128 Expansion (PRG Fallback)
# =============================================================================


def aes_expand(seeds: np.ndarray, length: int) -> np.ndarray:
    """Expand seeds to pseudorandom sequence.

    This is a fallback using NumPy's PRNG instead of AES-NI.

    Args:
        seeds: Shape (num_seeds, 2) uint64 array of 128-bit seeds
        length: Number of 128-bit blocks to generate per seed

    Returns:
        Shape (num_seeds, length, 2) uint64 array
    """
    num_seeds = seeds.shape[0]
    output = np.zeros((num_seeds, length, 2), dtype=np.uint64)

    for i in range(num_seeds):
        seed_val = [int(seeds[i, 0]), int(seeds[i, 1])]
        rng = np.random.default_rng(seed_val)
        output[i] = rng.integers(
            0, 0xFFFFFFFFFFFFFFFF, size=(length, 2), dtype=np.uint64
        )
    return output


# =============================================================================
# LDPC Encoding (Sparse)
# =============================================================================


def ldpc_encode(
    message: np.ndarray, h_indices: np.ndarray, h_indptr: np.ndarray, m: int
) -> np.ndarray:
    """Compute syndrome S = H @ message using sparse CSR representation.

    This is the fallback when C++ kernel is not available.

    Args:
        message: (N, 2) uint64 message vector
        h_indices: CSR indices array for H
        h_indptr: CSR indptr array for H (length m+1)
        m: Number of rows in H (syndrome length)

    Returns:
        (m, 2) uint64 syndrome vector
    """
    syndrome = np.zeros((m, 2), dtype=np.uint64)

    for i in range(m):
        # Get column indices for row i
        start, end = int(h_indptr[i]), int(h_indptr[i + 1])
        cols = h_indices[start:end]

        # XOR all selected message elements
        for j in cols:
            syndrome[i] ^= message[int(j)]

    return syndrome
