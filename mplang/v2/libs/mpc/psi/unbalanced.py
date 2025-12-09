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
Uses Sparse OKVS for sublinear communication.
"""

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.dialects.field as field
import mplang.v2.dialects.simp as simp
import mplang.v2.dialects.tensor as tensor
import mplang.v2.edsl as el
from mplang.v2.libs.mpc.psi.sparse_okvs import get_okvs_expansion


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

    Unlike standard VOLE-masked PSI, this version:
    1. Server encodes (key, H(key)) pairs into OKVS
    2. Server sends OKVS to client (O(N) one-time setup)
    3. Client decodes OKVS at their keys and compares (O(n) per query)

    For repeated queries on the same server set, amortized communication is O(n).

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

    # 1. Compute OKVS table size using dynamic expansion factor
    # See get_okvs_expansion() for theoretical basis
    expansion = get_okvs_expansion(server_n)
    M = int(server_n * expansion)

    # 2. Server builds OKVS: encode(key, H(key)) for all server items
    def _server_build_okvs(items: Any) -> Any:
        # Hash items to get values
        def _hash_items(x: Any) -> Any:
            lo = x
            hi = jnp.zeros_like(x)
            seeds = jnp.stack([lo, hi], axis=1)  # (N, 2)
            return seeds

        seeds = tensor.run_jax(_hash_items, items)
        h_items = field.aes_expand(seeds, 1)  # (N, 1, 2)

        def _flatten_hash(h: Any) -> Any:
            return h.reshape(h.shape[0], 2)

        h_flat = tensor.run_jax(_flatten_hash, h_items)  # (N, 2)

        # Build OKVS: Encode(keys=items, values=H(items))
        okvs_table = field.solve_okvs(items, h_flat, m=M)  # (M, 2)
        return okvs_table

    okvs_on_server = simp.pcall_static((server,), _server_build_okvs, server_items)

    # 3. Transfer OKVS to client (O(M) = O(N) communication - one time cost)
    okvs_on_client = simp.shuffle_static(okvs_on_server, {client: server})

    # 4. Client decodes OKVS at each of their keys
    def _client_decode_and_compare(items: Any, okvs_table: Any) -> Any:
        # Get decode values from OKVS
        decoded_values = field.decode_okvs(items, okvs_table)  # (n, 2)

        # Compute expected H(items) locally
        def _hash_items(x: Any) -> Any:
            lo = x
            hi = jnp.zeros_like(x)
            seeds = jnp.stack([lo, hi], axis=1)
            return seeds

        seeds = tensor.run_jax(_hash_items, items)
        h_items = field.aes_expand(seeds, 1)  # (n, 1, 2)

        def _flatten_hash(h: Any) -> Any:
            return h.reshape(h.shape[0], 2)

        expected = tensor.run_jax(_flatten_hash, h_items)  # (n, 2)

        # Compare: if decoded == expected, item is in intersection
        def _compare(dec: Any, exp: Any) -> Any:
            match = (dec[:, 0] == exp[:, 0]) & (dec[:, 1] == exp[:, 1])
            return match.astype(jnp.uint8)

        return tensor.run_jax(_compare, decoded_values, expected)

    intersection_mask = simp.pcall_static(
        (client,), _client_decode_and_compare, client_items, okvs_on_client
    )

    return cast(el.Object, intersection_mask)
