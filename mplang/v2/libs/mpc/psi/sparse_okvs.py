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

"""Sparse OKVS (Oblivious Key-Value Store) Data Structure.

This module implements Sparse OKVS encoding/decoding for efficient PSI.

Key Properties:
- Communication: O(n) instead of O(N) - sublinear in server set size
- Uses 3-hash Garbled Cuckoo Table for position hints
- Compatible with existing VOLE infrastructure
"""

from typing import Any, cast

import jax.numpy as jnp
import numpy as np

import mplang.v2.dialects.tensor as tensor
import mplang.v2.edsl as el

# ============================================================================
# Constants
# ============================================================================

# Number of hash functions for Cuckoo hashing
NUM_HASHES = 3

# Maximum iterations for Cuckoo insertion before failure
MAX_CUCKOO_ITERATIONS = 500
MASK64 = 0xFFFFFFFFFFFFFFFF


def get_okvs_expansion(n: int) -> float:
    """Get optimal OKVS expansion factor based on dataset size.

    The 3-hash Garbled Cuckoo Table algorithm requires table size M > N for
    the peeling algorithm to successfully solve the system. The minimum safe
    expansion factor ε (where M = (1+ε)*N) depends on N:

    - For N → ∞: Theoretical minimum is ε ≈ 0.23 (M = 1.23N)
    - For finite N: Larger ε needed due to variance in random hash collisions

    Empirical safe thresholds (failure probability < 0.1%):
    - N < 1,000:    ε = 0.6  (M = 1.6N)  - small sets need wide margin
    - N < 10,000:   ε = 0.4  (M = 1.4N)
    - N < 100,000:  ε = 0.3  (M = 1.3N)
    - N ≥ 100,000:  ε = 0.25 (M = 1.25N) - large sets converge to theory

    Using dynamic factor instead of fixed 1.6x saves ~22% communication
    for large-scale PSI (N ≥ 100K).

    Args:
        n: Number of key-value pairs to encode

    Returns:
        Expansion factor ε such that M = (1+ε)*N is safe for peeling
    """
    if n < 1000:
        return 3.0  # Small scale: need very wide safety margin for stability
    elif n < 10000:
        return 1.4  # Medium scale
    elif n < 100000:
        return 1.3  # Large scale
    else:
        return 1.25  # Very large scale: near theoretical minimum


# Legacy constant for backwards compatibility (uses conservative value)
CUCKOO_EXPANSION = 1.6


# ============================================================================
# Sparse OKVS Encoding (Client-side)
# ============================================================================


def sparse_encode_numpy(
    keys: np.ndarray,
    values: np.ndarray,
    server_table_size: int,
    seed: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode keys/values into sparse hints for the server.

    This is a pure-Python/NumPy implementation for the encoding logic.

    Args:
        keys: (n,) uint64 client keys
        values: (n, 2) uint64 values (128-bit field elements)
        server_table_size: Size of server's OKVS table (M = 1.35 * N)
        seed: (2,) uint64 seed for hash function randomization.

    Returns:
        positions: (n * NUM_HASHES,) uint64 - positions to query in server table
        coefficients: (n * NUM_HASHES,) uint64 - XOR mask coefficients (0 or 1)
        client_values: (n, 2) uint64 - client's masked values

    The sparse encoding works as follows:
    1. For each client key k, compute 3 hash positions: h1(k), h2(k), h3(k)
    2. The value at k is retrieved by XOR-ing server table entries at these positions
    3. Client sends positions and gets back the XOR of server entries
    """
    n = len(keys)

    # Compute hash positions for each key (3 positions per key)
    # Using simple hash functions: h_i(k) = hash(k || i) mod table_size
    all_positions = np.zeros((n, NUM_HASHES), dtype=np.uint64)

    # Ensure keys are uint64 for bitwise operations
    keys_u64 = keys.astype(np.uint64)

    # Mix seed into key
    seed0, seed1 = np.uint64(seed[0]), np.uint64(seed[1])
    mixed_key = keys_u64 ^ seed0

    for i in range(NUM_HASHES):
        # Hash: combine key with hash index
        # Using FNV-1a style mixing
        # Rotate seed1 for diversity across hashes
        # Mask calculation to avoid large ints
        offset = (i * 0x9E3779B97F4A7C15) & MASK64
        current_seed = seed1 + np.uint64(offset)

        # Inner mixing
        magic = (0x14650FB0739D0383 + i * 0x27D4EB2F165667C5) & MASK64
        mixed = mixed_key ^ (np.uint64(magic) ^ current_seed)
        
        mixed = mixed * np.uint64(0xBF58476D1CE4E5B9)
        mixed = mixed ^ (mixed >> np.uint64(27))
        mixed = mixed * np.uint64(0x94D049BB133111EB)
        mixed = mixed ^ (mixed >> np.uint64(31))

        # Reduce to table size
        all_positions[:, i] = mixed % np.uint64(server_table_size)

    # Flatten positions
    positions = all_positions.flatten()  # (n * 3,)

    # Coefficients are always 1 for 3-hash OKVS
    # (In more complex schemes, coefficients can vary)
    coefficients = np.ones(n * NUM_HASHES, dtype=np.uint64)

    # Client values are returned as-is (will be masked with server response)
    client_values = values.copy()

    return positions, coefficients, client_values


def compute_hash_positions(
    key: np.ndarray, table_size: int, seed: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Compute 3 hash positions for a single key.

    Args:
        key: scalar uint64 key
        table_size: Size of the hash table
        seed: (2,) uint64 seed

    Returns:
        (3,) uint64 positions
    """
    positions = np.zeros(NUM_HASHES, dtype=np.uint64)
    key = np.uint64(key)  # Ensure uint64
    seed0, seed1 = np.uint64(seed[0]), np.uint64(seed[1])
    mixed_key = key ^ seed0

    for i in range(NUM_HASHES):
        offset = (i * 0x9E3779B97F4A7C15) & MASK64
        current_seed = seed1 + np.uint64(offset)
        
        magic = (0x14650FB0739D0383 + i * 0x27D4EB2F165667C5) & MASK64
        mixed = mixed_key ^ (np.uint64(magic) ^ current_seed)
        
        mixed = mixed * np.uint64(0xBF58476D1CE4E5B9)
        mixed = mixed ^ (mixed >> np.uint64(27))
        mixed = mixed * np.uint64(0x94D049BB133111EB)
        mixed = mixed ^ (mixed >> np.uint64(31))
        positions[i] = mixed % np.uint64(table_size)

    return positions


# ============================================================================
# Sparse OKVS Decoding (Server-side)
# ============================================================================


def sparse_decode_numpy(
    positions: np.ndarray,
    table: np.ndarray,
) -> np.ndarray:
    """Decode values from server table at given positions.

    Args:
        positions: (n * 3,) uint64 - positions to look up
        table: (M, 2) uint64 - server's OKVS table

    Returns:
        (n,) uint64 - XOR of table entries at each key's 3 positions
    """
    n = len(positions) // NUM_HASHES
    positions = positions.reshape(n, NUM_HASHES)

    # Look up and XOR
    result = np.zeros((n, 2), dtype=np.uint64)
    for i in range(NUM_HASHES):
        pos = positions[:, i].astype(np.int64)
        result ^= table[pos]

    return result


# ============================================================================
# EDSL Wrappers
# ============================================================================


def sparse_encode(
    keys: el.Object,
    values: el.Object,
    server_table_size: int,
    seed: el.Object,  # (2,) uint64
) -> tuple[el.Object, el.Object, el.Object]:
    """EDSL wrapper for sparse encoding.

    Args:
        keys: (n,) uint64 tensor of client keys
        values: (n, 2) uint64 tensor of values
        server_table_size: Size of server's table
        seed: (2,) uint64 tensor (random seed)

    Returns:
        Tuple of (positions, coefficients, client_values)
    """

    def _encode_jax(k: Any, v: Any, s: Any, M: int) -> tuple[Any, Any, Any]:
        # Compute positions using vectorized hash
        n = k.shape[0]
        # s: (2,) uint64
        seed0 = s[0]
        seed1 = s[1]

        mixed_key = k ^ seed0

        all_positions = jnp.zeros((n, NUM_HASHES), dtype=jnp.uint64)

        for i in range(NUM_HASHES):
            # Mask constants to 64-bit to prevent JAX coercion errors
            offset = (i * 0x9E3779B97F4A7C15) & MASK64
            current_seed = seed1 + jnp.uint64(offset)
            
            magic = (0x14650FB0739D0383 + i * 0x27D4EB2F165667C5) & MASK64
            mixed = mixed_key ^ (jnp.uint64(magic) ^ current_seed)
            
            mixed = mixed * jnp.uint64(0xBF58476D1CE4E5B9)
            mixed = mixed ^ (mixed >> 27)
            mixed = mixed * jnp.uint64(0x94D049BB133111EB)
            mixed = mixed ^ (mixed >> 31)
            all_positions = all_positions.at[:, i].set(mixed % jnp.uint64(M))

        positions = all_positions.flatten()
        coeffs = jnp.ones(n * NUM_HASHES, dtype=jnp.uint64)

        return positions, coeffs, v

    pos, coef, vals = tensor.run_jax(
        lambda k, v, s: _encode_jax(k, v, s, server_table_size), keys, values, seed
    )
    return pos, coef, vals


def sparse_lookup(
    positions: el.Object,
    table: el.Object,
) -> el.Object:
    """Look up values at sparse positions from table.

    Args:
        positions: (n * 3,) uint64 positions
        table: (M, 2) uint64 server table

    Returns:
        (n, 2) uint64 - XOR of table entries at each key's positions
    """

    def _lookup_jax(pos: Any, tbl: Any) -> Any:
        n = pos.shape[0] // NUM_HASHES
        pos_2d = pos.reshape(n, NUM_HASHES)

        result = jnp.zeros((n, 2), dtype=jnp.uint64)
        for i in range(NUM_HASHES):
            p = pos_2d[:, i].astype(jnp.int32)
            result = result ^ tbl[p]

        return result

    return cast(el.Object, tensor.run_jax(_lookup_jax, positions, table))


def compute_positions(
    keys: el.Object,
    server_table_size: int,
    seed: el.Object,
) -> el.Object:
    """Compute sparse OKVS positions for keys (EDSL wrapper).

    Args:
        keys: (n,) uint64 client keys
        server_table_size: Size of server's table
        seed: (2,) uint64 tensor (random seed)

    Returns:
        (n * NUM_HASHES,) uint64 positions
    """

    def _hash_jax(k: Any, s: Any, M: int) -> Any:
        n = k.shape[0]
        # s: (2,) uint64
        seed0 = s[0]
        seed1 = s[1]

        mixed_key = k ^ seed0

        all_positions = jnp.zeros((n, NUM_HASHES), dtype=jnp.uint64)

        for i in range(NUM_HASHES):
            # Mask constants
            offset = (i * 0x9E3779B97F4A7C15) & MASK64
            current_seed = seed1 + jnp.uint64(offset)
            
            magic = (0x14650FB0739D0383 + i * 0x27D4EB2F165667C5) & MASK64
            mixed = mixed_key ^ (jnp.uint64(magic) ^ current_seed)
            
            mixed = mixed * jnp.uint64(0xBF58476D1CE4E5B9)
            mixed = mixed ^ (mixed >> 27)
            mixed = mixed * jnp.uint64(0x94D049BB133111EB)
            mixed = mixed ^ (mixed >> 31)
            all_positions = all_positions.at[:, i].set(mixed % jnp.uint64(M))

        return all_positions.flatten()

    return cast(
        el.Object,
        tensor.run_jax(lambda k, s: _hash_jax(k, s, server_table_size), keys, seed),
    )
