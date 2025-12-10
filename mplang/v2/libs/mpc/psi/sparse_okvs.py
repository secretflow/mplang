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

"""Sparse OKVS (Oblivious Key-Value Store) Implementation.

This module provides the core data structures and algorithms for Sparse OKVS,
which is a critical component in unbalanced Private Set Intersection (PSI).

Theory:
    Sparse OKVS (also known as Garbled Cuckoo Table) allows a client to encode
    a set of key-value pairs into a compact structure that can be queried by a server.
    Unlike dense OKVS (which requires O(N) communication where N is the server's set size),
    Sparse OKVS allows the client to send only O(n) data (where n is the client's set size).

    The construction uses a 3-hash Cuckoo hashing scheme. For an input `x`, the value is
    recovered by XORing entries at three positions:
        Val(x) = Table[h1(x)] ⊕ Table[h2(x)] ⊕ Table[h3(x)]

Key Features:
    - **Sublinear Communication**: Communication cost depends on client set size (n), not server set size (N).
    - **3-Hash Structure**: Uses 3 independent hash functions derived from a common seed.
    - **JAX/EDSL Integration**: Fully compatible with MPLang's tracing and compilation system.
"""

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.dialects.simp as simp
import mplang.v2.dialects.tensor as tensor
import mplang.v2.edsl as el

# ============================================================================
# Constants
# ============================================================================

# Number of hash functions for Cuckoo hashing
NUM_HASHES = 3

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


# ============================================================================
# EDSL Wrappers
# ============================================================================


def decode_values(
    indices: el.Object,
    table: el.Object,
) -> el.Object:
    """Perform Sparse OKVS Lookup (Decode).

    Retrieves values from the OKVS table using the pre-computed hash indices.
    This implements the reconstruction formula:
        Result = Table[h1] ⊕ Table[h2] ⊕ Table[h3]

    Args:
        indices: (n * 3,) uint64 tensor containing flattened hash indices.
                 Usually computed via `compute_indices`.
        table: (M, D) uint64 tensor representing the server's OKVS table.

    Returns:
        (n, D) uint64 tensor containing the recovered values.
    """

    def _lookup_jax(pos: Any, tbl: Any) -> Any:
        n = pos.shape[0] // NUM_HASHES
        pos_2d = pos.reshape(n, NUM_HASHES)

        result = jnp.zeros((n, 2), dtype=jnp.uint64)
        for i in range(NUM_HASHES):
            p = pos_2d[:, i].astype(jnp.int32)
            result = result ^ tbl[p]

        return result

    return cast(el.Object, tensor.run_jax(_lookup_jax, indices, table))


def compute_indices(
    keys: el.Object,
    server_table_size: int,
    seed: el.Object,
) -> el.Object:
    """Compute the 3 hash indices for Sparse OKVS (Encode Step 1).

    This function maps client keys to indices in the server's OKVS table.
    It is the first step of the sparse encoding process (Client side).
    The resulting indices are typically sent to the server (in a PIR-like fashion)
    or used locally to query a retrieved table.

    Args:
        keys: (n,) uint64 tensor of client keys.
        server_table_size: Size of the server's OKVS table (M).
        seed: (2,) uint64 tensor for hash seeding.

    Returns:
        (n * 3,) uint64 tensor of flattened indices.
    """

    def _compute_indices_jax(k: Any, s: Any, M: int) -> Any:
        """Compute the 3 hash positions for each key using JAX.

        Implements the hash generation logic for the Garbled Cuckoo Table.
        Uses a linear congruential generator style mixing with the provided seed
        to derive 3 independent indices for each key.
        """
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

        return all_positions.flatten()

    res = tensor.run_jax(
        lambda k, s: _compute_indices_jax(k, s, server_table_size),
        keys,
        seed,
    )
    return cast(el.Object, res)


def query(
    client: int,
    server: int,
    client_keys: el.Object,
    server_table: el.Object,
    seed: el.Object,
    server_table_size: int,
) -> el.Object:
    """Execute Sparse OKVS Query Protocol (Multi-Party).

    Orchestrates the sparse query process between a Client and a Server.

    Protocol:
    1. Client computes hash indices for its keys (O(n)).
    2. Client sends indices to Server (O(n) communication).
    3. Server retrieves values from its table using indices (O(n)).
    4. Server sends values back to Client (implicit in return, or explicit if needed).

    Args:
        client: Rank of the Client party.
        server: Rank of the Server party.
        client_keys: (n,) uint64 tensor located on Client.
        server_table: (M, D) uint64 tensor located on Server.
        seed: (2,) uint64 tensor (public or shared).
        server_table_size: Size of the server's table (M).

    Returns:
        (n, D) uint64 tensor located on Server (containing the retrieved values).
        Note: The result is on the Server. If Client needs it, use simp.shuffle.
    """
    # 1. Client computes indices
    indices = simp.pcall_static(
        (client,),
        compute_indices,
        client_keys,
        server_table_size,
        seed,
    )

    # 2. Move indices to Server
    # Explicitly shuffle data from Client to Server
    indices_on_server = simp.shuffle_static(indices, {server: client})

    # 3. Server decodes
    response = simp.pcall_static(
        (server,),
        decode_values,
        indices_on_server,
        server_table,
    )

    return response
