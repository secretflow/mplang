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

"""Cuckoo Hashing for OPRF-PSI.

Implements JAX-compatible Cuckoo hashing for mapping items to table positions.
Each item hashes to K candidate positions; during lookup, check all K positions.

Reference: KKRT OPRF-PSI uses Cuckoo hashing for row mapping.
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.edsl as el
from mplang.v2.dialects import tensor
from mplang.v2.libs.mpc.common.constants import (
    E_FRAC_1,
    GOLDEN_RATIO_64,
    PI_FRAC_1,
    PI_FRAC_2,
    SPLITMIX64_GAMMA_1,
    SPLITMIX64_GAMMA_2,
)

# =============================================================================
# Cuckoo Hash Parameters
# =============================================================================

NUM_HASH_FUNCTIONS = 3  # Standard: 3 hash functions
STASH_SIZE = 0  # Simple version: no stash (higher failure rate)
MASK64 = 0xFFFFFFFFFFFFFFFF


def hash_to_positions(items: Any, table_size: int, seed: tuple[int, int]) -> Any:
    """Compute K candidate positions for each item.

    Uses polynomial hash family with seeded coefficients:
        h_i(x) = (a_i * x + b_i) mod table_size

    Security: Both coefficients a and b are seeded to prevent
    structural analysis attacks on the hash family.

    Args:
        items: (N, 16) uint8 array - items to hash
        table_size: Size of Cuckoo hash table
        seed: (2,) tuple of uint64 - random seed

    Returns:
        (N, K) int32 array - K candidate positions for each item
    """
    N = items.shape[0]
    K = NUM_HASH_FUNCTIONS

    # Convert items to 64-bit keys (first 8 bytes)
    keys = items[:, :8].view(jnp.uint64).reshape(N)

    # Mix seed into keys
    seed0 = jnp.uint64(seed[0])
    seed1 = jnp.uint64(seed[1])
    keys = keys ^ seed0

    # Base hash coefficients (deterministic starting point)
    a_base = jnp.array(
        [GOLDEN_RATIO_64, SPLITMIX64_GAMMA_1, SPLITMIX64_GAMMA_2], dtype=jnp.uint64
    )
    b_base = jnp.array([PI_FRAC_1, PI_FRAC_2, E_FRAC_1], dtype=jnp.uint64)

    # Security Fix: Seed BOTH coefficients a and b
    # This prevents structural analysis attacks on the hash family
    a = a_base ^ seed0  # Mix seed0 into multiplicative coefficient
    b = b_base ^ seed1  # Mix seed1 into additive coefficient

    # Compute hash positions: (N, K)
    positions = jnp.zeros((N, K), dtype=jnp.int32)
    for i in range(K):
        h = (keys * a[i] + b[i]) % table_size
        positions = positions.at[:, i].set(h.astype(jnp.int32))

    return positions


def cuckoo_insert_batch(
    items: Any,
    table_size: int,
    seed: tuple[int, int],
    max_iters: int = 100,
) -> tuple[Any, Any, Any]:
    """Batch Cuckoo insertion using vectorized logic (JAX-compatible).

    Uses multi-choice parallel insertion:
    1. All items try 1st choice. Collisions resolved by last-write-wins.
    2. Failed items try 2nd choice.
    3. Failed items try 3rd choice.

    Args:
        items: (N, 16) uint8 array - items to insert
        table_size: Size of Cuckoo hash table (should be ~1.3-1.5N)
        max_iters: Ignored in this vectorized version (uses K=3 fixed passes)
        seed: (2,) uint64 seed

    Returns:
        Tuple of:
        - table: (table_size, 16) uint8 - Cuckoo hash table
        - item_to_pos: (N,) int32 - position of each item in table
        - success: (N,) bool - whether each item was successfully inserted
    """
    N = items.shape[0]
    K = NUM_HASH_FUNCTIONS

    positions = hash_to_positions(items, table_size, seed)
    item_to_pos = jnp.full(N, -1, dtype=jnp.int32)
    active_mask = jnp.ones(N, dtype=jnp.bool_)

    # We track which item "owns" each table slot
    table_slots = jnp.full(table_size, -1, dtype=jnp.int32)

    # Track occupied status to forbid overwriting previous successes
    table_occupied = jnp.zeros(table_size, dtype=jnp.bool_)

    item_indices = jnp.arange(N, dtype=jnp.int32)

    for k in range(K):
        # 1. Propose positions for active items
        # Inactive items get -1 proposal
        cand_pos = jnp.where(active_mask, positions[:, k], -1)

        # 2. Filter out already occupied slots
        # Map -1 to safe index 0 for lookup (result discarded via mask)
        safe_lookup = jnp.maximum(cand_pos, 0)
        is_occupied = table_occupied[safe_lookup]
        # Valid proposal: not -1 AND not occupied
        cand_pos_valid = jnp.where((cand_pos >= 0) & (~is_occupied), cand_pos, -1)

        # 3. Attempt write to table_slots using Scatter
        # Extend table to handle -1 dump index (at index table_size)
        ext_slots = jnp.pad(table_slots, (0, 1), constant_values=-1)

        # Map -1 to dump index
        write_pos = jnp.where(cand_pos_valid >= 0, cand_pos_valid, table_size)

        # Write active item indices
        # We write ALL items, but inactive ones write to dump.
        # This is safe because active ones write to valid slots (or dump if collision/occupied).
        ext_slots_updated = ext_slots.at[write_pos].set(item_indices)

        # 4. Verify winners
        winner_indices = ext_slots_updated[write_pos]

        # Success if:
        # a) We had a valid proposal (cand_pos_valid != -1)
        # b) Our index matches the winner
        success_round = (cand_pos_valid >= 0) & (winner_indices == item_indices)

        # 5. Commit state
        # Update global state based on success
        item_to_pos = jnp.where(success_round, cand_pos_valid, item_to_pos)
        active_mask = active_mask & (~success_round)

        # Update table slots (truncate dump)
        table_slots = ext_slots_updated[:table_size]
        table_occupied = table_slots >= 0

    # Construct final table
    safe_indices = jnp.maximum(table_slots, 0)
    final_table = items[safe_indices]
    final_table = jnp.where(table_slots[:, None] >= 0, final_table, 0)

    success_total = item_to_pos >= 0
    return final_table, item_to_pos, success_total


def cuckoo_lookup_positions(items: Any, table_size: int, seed: tuple[int, int]) -> Any:
    """Get Cuckoo lookup positions for each item.

    Returns the K candidate positions where each item could be located
    in a Cuckoo hash table.

    Args:
        items: (M, 16) uint8 array - items to lookup
        table_size: Size of Cuckoo hash table
        seed: (2,) uint64 seed

    Returns:
        (M, K) int32 array - K positions to check for each item
    """
    return hash_to_positions(items, table_size, seed)


# =============================================================================
# EDSL Wrappers
# =============================================================================


def compute_positions(
    items: el.Object,
    table_size: int,
    seed: el.Object,  # (2,) uint64
) -> el.Object:
    """Compute Cuckoo hash positions for items (EDSL wrapper).

    Args:
        items: (N, 16) byte tensor of items
        table_size: Size of Cuckoo hash table
        seed: (2,) uint64 seed

    Returns:
        (N, K) int32 tensor of candidate positions
    """

    def _hash(x: Any, s: Any) -> Any:
        return hash_to_positions(x, table_size, tuple(s))

    return cast(el.Object, tensor.run_jax(_hash, items, seed))
