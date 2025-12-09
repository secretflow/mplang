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

"""LDPC (Low-Density Parity-Check) Code Implementation for Silver VOLE.

This module provides LDPC matrix generation and encoding functions used in
the Silver protocol for efficient silent VOLE generation.

Silver uses a specific LDPC structure optimized for:
1. Fast encoding (quasi-cyclic structure)
2. Efficient syndrome computation
3. Low-density for minimal communication

Reference: "Silver: Silent VOLE and Oblivious Transfer from Hardness of Decoding"
           CRYPTO 2021
"""

import numpy as np
import scipy.sparse as sp
from typing import Any, cast

# ============================================================================
# Constants
# ============================================================================

# Default Silver parameters (from paper)
SILVER_WEIGHT = 5  # Row weight (number of 1s per row)
SILVER_GAP = 16  # Gap parameter for quasi-cyclic structure


# ============================================================================
# LDPC Matrix Generation
# ============================================================================


def generate_silver_ldpc(n: int, m: int, seed: int = 42) -> sp.csr_matrix:
    """Generate Silver-style LDPC parity check matrix.

    Creates a quasi-cyclic LDPC matrix suitable for Silver protocol.
    The matrix has:
    - Dimensions: m x n (m < n for compression)
    - Row weight: SILVER_WEIGHT (sparse)
    - Quasi-cyclic structure for fast encoding

    Args:
        n: Number of columns (message length)
        m: Number of rows (syndrome length, typically n/10 to n/5)
        seed: Random seed for reproducibility

    Returns:
        Sparse CSR matrix H of shape (m, n)
    """
    rng = np.random.RandomState(seed)

    # Use a regular LDPC structure with fixed row weight
    row_weight = min(SILVER_WEIGHT, n)

    # Build sparse matrix in COO format for efficiency
    rows = []
    cols = []

    for i in range(m):
        # Select random column indices for this row
        # Use consistent spacing with some randomness for quasi-cyclic property
        base_positions = np.linspace(0, n - 1, row_weight, dtype=int)
        offsets = rng.randint(-SILVER_GAP, SILVER_GAP + 1, size=row_weight)
        positions = (base_positions + offsets) % n
        positions = np.unique(positions)  # Remove duplicates

        # Ensure we have at least some entries
        while len(positions) < min(3, row_weight):
            extra = rng.randint(0, n, size=row_weight - len(positions))
            positions = np.unique(np.concatenate([positions, extra]))

        for j in positions:
            rows.append(i)
            cols.append(j)

    data = np.ones(len(rows), dtype=np.uint8)
    H = sp.coo_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)

    return H.tocsr()


def generate_silver_ldpc_systematic(
    n: int, k: int, seed: int = 42
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Generate systematic LDPC matrix for Silver.

    Returns both the parity check matrix H and generator matrix G.
    H is (n-k) x n, G is k x n.

    For Silver, we primarily need H for syndrome computation.

    Args:
        n: Codeword length
        k: Message length (k < n)
        seed: Random seed

    Returns:
        Tuple of (H, G) as sparse matrices
    """
    m = n - k  # Number of parity bits
    H = generate_silver_ldpc(n, m, seed)

    # For Silver, G is not strictly needed as we use syndrome encoding
    # Return None for G to save computation
    return H, None


# ============================================================================
# LDPC Encoding (Syndrome Computation)
# ============================================================================


def ldpc_encode_numpy(message: np.ndarray, H: sp.csr_matrix) -> np.ndarray:
    """Compute syndrome s = H · m over GF(2).

    This is the core encoding operation for Silver VOLE.

    Args:
        message: Binary message vector of shape (n,) or (n, 2) for 128-bit
        H: LDPC parity check matrix of shape (m, n)

    Returns:
        Syndrome vector of shape (m,) or (m, 2)
    """
    if message.ndim == 1:
        # Simple binary case
        syndrome = H.dot(message) % 2
        return cast(np.ndarray, syndrome.astype(np.uint8))
    else:
        # 128-bit case: encode each component separately using XOR
        # For GF(2^128), we do bitwise operations
        m, _n = H.shape
        result = np.zeros((m, message.shape[1]), dtype=message.dtype)

        # Use sparse matrix structure for efficiency
        for i in range(m):
            row_start = H.indptr[i]
            row_end = H.indptr[i + 1]
            col_indices = H.indices[row_start:row_end]

            # XOR all selected message elements
            for j in col_indices:
                result[i] ^= message[j]

        return result


def ldpc_decode_syndrome(
    syndrome: np.ndarray, H: sp.csr_matrix, noise_weight: int
) -> np.ndarray:
    """Decode syndrome to recover sparse error vector.

    Uses belief propagation or simple peeling for low-weight errors.

    Args:
        syndrome: Syndrome vector of shape (m,) or (m, 2)
        H: LDPC parity check matrix
        noise_weight: Expected weight of error vector

    Returns:
        Estimated error vector of shape (n,) or (n, 2)
    """
    m, n = H.shape

    # For Silver with low noise, simple syndrome inversion works
    # This is a placeholder - full BP decoder can be added later

    if syndrome.ndim == 1:
        error = np.zeros(n, dtype=np.uint8)
    else:
        error = np.zeros((n, syndrome.shape[1]), dtype=syndrome.dtype)

    # Simple greedy decoder for sparse errors
    # Find columns that match syndrome bits
    remaining_syndrome = syndrome.copy()

    for _ in range(noise_weight):
        # Find column that reduces syndrome the most
        best_col = -1
        best_reduction = 0

        for j in range(n):
            col = H.getcol(j).toarray().flatten()
            if syndrome.ndim == 1:
                reduction = np.sum(col & (remaining_syndrome != 0))
            else:
                reduction = np.sum(col.reshape(-1, 1) & (remaining_syndrome != 0))

            if reduction > best_reduction:
                best_reduction = reduction
                best_col = j

        if best_col == -1 or best_reduction == 0:
            break

        # Flip this bit
        error[best_col] = (
            1
            if syndrome.ndim == 1
            else np.ones(syndrome.shape[1], dtype=syndrome.dtype)
        )

        # Update syndrome
        col = H.getcol(best_col).toarray().flatten()
        if syndrome.ndim == 1:
            remaining_syndrome = (remaining_syndrome + col) % 2
        else:
            for i in range(m):
                if col[i]:
                    remaining_syndrome[i] ^= error[best_col]

    return error


# ============================================================================
# Silver-specific Parameters
# ============================================================================


def get_silver_params(n: int) -> tuple[int, int, int]:
    """Get recommended Silver parameters for given output length.

    Args:
        n: Desired number of VOLE correlations

    Returns:
        Tuple of (code_length, syndrome_length, noise_weight)
    """
    # Silver uses approximately 10:1 compression
    code_length = n
    syndrome_length = max(n // 10, 128)  # At least 128 for security
    noise_weight = 64  # Low noise for efficient decoding

    return code_length, syndrome_length, noise_weight


# ============================================================================
# Utility Functions
# ============================================================================


def matrix_to_sparse_repr(H: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Convert sparse matrix to compact representation for C++ kernel.

    Returns:
        Tuple of (indptr, indices) arrays
    """
    return H.indptr.astype(np.uint64), H.indices.astype(np.uint64)


def verify_ldpc_structure(H: sp.csr_matrix) -> bool:
    """Verify LDPC matrix has correct structure.

    Checks:
    - Sparsity (low density)
    - No all-zero rows
    - Reasonable row weights
    """
    m, n = H.shape

    # Check sparsity
    density = H.nnz / (m * n)
    if density > 0.1:
        print(f"Warning: LDPC density {density:.3f} is high")
        return False

    # Check row weights
    row_weights = np.diff(H.indptr)
    if np.any(row_weights == 0):
        print("Warning: LDPC has zero-weight rows")
        return False

    avg_weight = np.mean(row_weights)
    if avg_weight < 2 or avg_weight > 20:
        print(f"Warning: LDPC average row weight {avg_weight:.1f} unusual")
        return False

    return True
