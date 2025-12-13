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

"""Unbalanced PSI Protocol.

This module implements unbalanced PSI for scenarios where client set size n << server set size N.
Uses Seeded OKVS (via derived keys) to prevent pre-computation attacks.

Security Model:
- Session-specific random seed generated at RUNTIME on the Server.
- Both Key and Value derivations use the seed for consistent security.
- WARNING: Online dictionary attacks by active clients remain possible without OPRF.

Protocol:
1. Server generates random Seed at runtime.
2. Server computes K' = H(ServerItems, Seed) and V = H(ServerItems, Seed).
3. Server solves OKVS: Table = Solve(K', V).
4. Server sends Seed + Table to Client.
5. Client computes k' = H(ClientItems, Seed) and v = H(ClientItems, Seed).
6. Client decodes V' = Decode(k', Table).
7. Client checks V' == v.
"""

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects import crypto, field, simp, tensor
from mplang.v2.libs.mpc.psi.okvs_gct import get_okvs_expansion


def psi_unbalanced(
    server: int,
    client: int,
    server_n: int,
    client_n: int,
    server_items: el.Object,
    client_items: el.Object,
) -> el.Object:
    """Unbalanced PSI with O(client_n) communication.

    This protocol is optimized for scenarios where client_n << server_n.

    Security:
    - Uses a cryptographically random Session Seed (128-bit) generated at RUNTIME.
    - Both Key and Value derivations include the Seed.
    - Prevents offline pre-computation (Rainbow Table) attacks.
    - WARNING: Online dictionary attacks by active clients remain possible.

    > [!WARNING]
    > **Security Notice**: This protocol sends the Session Seed to the Client to allow
    > them to compute the OKVS lookups. A malicious Client can perform an online
    > dictionary attack (brute-force hashing) to enumerate Server items.
    > For strict set privacy against malicious clients, use OPRF-PSI (`oprf.py` based)
    > instead of this unbalanced protocol.

    Args:
        server: Rank of server (holds large set N)
        client: Rank of client (holds small set n)
        server_n: Size of server's set
        client_n: Size of client's set
        server_items: (server_n,) uint64 on server
        client_items: (client_n,) uint64 on client

    Returns:
        Intersection indicators on client: (client_n,) uint8
    """
    if server == client:
        raise ValueError("Server and Client must be different parties.")

    if client_n <= 0 or server_n <= 0:
        raise ValueError("Set sizes must be positive.")

    # =========================================================================
    # 1. Server Setup: Generate Runtime Random Seed
    # =========================================================================

    # Generate 16 bytes (128-bit) of cryptographically secure random data
    # AT RUNTIME on the Server party (not during trace!)
    def _gen_runtime_seed() -> Any:
        # Use new API: directly generate (2,) u64 tensor
        return crypto.random_tensor((2,), elt.u64)

    server_seed = simp.pcall_static((server,), _gen_runtime_seed)

    # =========================================================================
    # Hashing Helpers (Both Key and Value use Seed)
    # =========================================================================

    def _compute_hashes(items: Any, seed: Any) -> tuple[Any, Any]:
        """Compute Derived Key K' and Validation Value V for items.

        Both Key and Value are derived using the session Seed to prevent
        pre-computation attacks.

        Key:   K' = AES_Expand(H_key(Item, Seed))[:64bit]
        Value: V  = AES_Expand(H_val(Item, Seed))[:128bit]
        """

        # Domain separator for Key derivation
        KEY_DOMAIN = jnp.uint64(0xA5A5A5A5A5A5A5A5)
        # Domain separator for Value derivation
        VAL_DOMAIN = jnp.uint64(0x5A5A5A5A5A5A5A5A)

        def _prepare_key_seed(x: Any, s: Any) -> Any:
            # x: (N,) u64, s: (2,) u64
            # Mix with KEY domain separator
            k_lo = (x + s[0]) ^ KEY_DOMAIN
            k_hi = (x ^ s[1]) + KEY_DOMAIN
            return jnp.stack([k_lo, k_hi], axis=1)

        def _prepare_val_seed(x: Any, s: Any) -> Any:
            # x: (N,) u64, s: (2,) u64
            # Mix with VAL domain separator (different from key)
            v_lo = (x + s[0]) ^ VAL_DOMAIN
            v_hi = (x ^ s[1]) + VAL_DOMAIN
            return jnp.stack([v_lo, v_hi], axis=1)

        # Derive Keys
        key_seeds = tensor.run_jax(_prepare_key_seed, items, seed)
        h_keys_raw = field.aes_expand(key_seeds, 1)  # (N, 1, 2)

        def _extract_key(h: Any) -> Any:
            return h[:, 0, 0]

        keys = tensor.run_jax(_extract_key, h_keys_raw)

        # Derive Values (ALSO using seed - fixes Value Oracle Attack)
        val_seeds = tensor.run_jax(_prepare_val_seed, items, seed)
        h_vals_raw = field.aes_expand(val_seeds, 1)  # (N, 1, 2)

        def _flatten(h: Any) -> Any:
            return h.reshape(h.shape[0], 2)

        vals = tensor.run_jax(_flatten, h_vals_raw)

        return keys, vals

    # Server computes K' and V
    server_derived_keys, server_values = simp.pcall_static(
        (server,), _compute_hashes, server_items, server_seed
    )

    # Server Solves OKVS
    expansion = get_okvs_expansion(server_n)
    M = int(server_n * expansion)

    def _solve(k: Any, v: Any, s: Any) -> Any:
        return field.solve_okvs(k, v, M, s)

    okvs_table = simp.pcall_static(
        (server,), _solve, server_derived_keys, server_values, server_seed
    )

    # Send to Client
    okvs_table_client = simp.shuffle_static(okvs_table, {client: server})
    client_seed = simp.shuffle_static(server_seed, {client: server})

    # =========================================================================
    # 2. Client Operations
    # =========================================================================

    # Client computes k' and expected V using the SAME hash functions
    client_derived_keys, client_expected_values = simp.pcall_static(
        (client,), _compute_hashes, client_items, client_seed
    )

    # Client Decodes OKVS and Compares
    def _decode_and_compare(keys: Any, table: Any, expected: Any, s: Any) -> Any:
        decoded = field.decode_okvs(keys, table, s)

        def _compare_jax(dec: Any, exp: Any) -> Any:
            match = (dec[:, 0] == exp[:, 0]) & (dec[:, 1] == exp[:, 1])
            return match.astype(jnp.uint8)

        return tensor.run_jax(_compare_jax, decoded, expected)

    intersection_mask = simp.pcall_static(
        (client,),
        _decode_and_compare,
        client_derived_keys,
        okvs_table_client,
        client_expected_values,
        client_seed,
    )

    return cast(el.Object, intersection_mask)
