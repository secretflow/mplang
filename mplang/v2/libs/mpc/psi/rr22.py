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

"""Private Set Intersection using VOLE and OKVS (RR22-Style).

This module implements a high-performance PSI protocol based on the "Blazing Fast PSI"
(RR22) paper. The protocol relies on Vector Oblivious Linear Evaluation (VOLE) and
Oblivious Key-Value Stores (OKVS) to achieve efficient set intersection with linear
communication O(N) and computation complexity.

Protocol Overview:
The core idea is to mask a "Polynomial" (encoded via OKVS) with VOLE-correlated randomness,
such that the mask can only be removed (and the polynomial verified) if the parties share
the same element.

Phases:
1.  **Correlated Randomness (VOLE)**:
    Sender and Receiver establish a shared correlation:
    W = V + U * Delta
    - Sender holds U, V.
    - Receiver holds W, Delta.
    - U is random. Delta is a fixed secret scalar (Receiver's key).

2.  **Encoding (OKVS)**:
    Receiver encodes their input set Y into a structure P using OKVS, such that:
    Decode(P, y) = H(y) for all y in Y.
    Here H(y) is a Random Oracle (implemented via Davies-Meyer/AES).

3.  **Masking & Exchange**:
    Receiver masks the structure P with their VOLE share W:
    Q = P ^ W
    Receiver sends Q to Sender.

4.  **Decoding & Verification**:
    Sender attempts to decode Q for each of their items x in X.
    Since OKVS is linear:
    Decode(Q, x) = Decode(P, x) ^ Decode(W, x)

    Sender reconstructs the potential "Target" value T:
    T = Decode(Q, x) ^ Decode(V, x) ^ H(x)

    If x in Y (Intersection):
       Decode(P, x) = H(x)
       Decode(W, x) = Decode(V, x) ^ Decode(U, x) * Delta
    Substitute into T:
       T = H(x) ^ (Decode(V, x) ^ Decode(U, x) * Delta) ^ Decode(V, x) ^ H(x)
       T = Decode(U, x) * Delta

    Thus, verification becomes checking if T == U* * Delta, where U* = Decode(U, x).
    This check is performed securely using hashes to prevent leakage.
"""
from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.edsl as el
import mplang.v2.libs.mpc.ot.silent as silent_ot
from mplang.v2.dialects import field, simp, tensor


def psi_intersect(
    sender: int,
    receiver: int,
    n: int,
    sender_items: el.Object,
    receiver_items: el.Object,
) -> el.Object:
    """Execute OKVS-based PSI Protocol (Original RR22 Logic).

    This implementation follows the RR22 paper's role assignment where:
    - PSI Sender holds Delta (and W).
    - PSI Receiver holds U and V.
    
    This enables the "One Decode" optimization on the Sender side and prevents
    offline brute-force attacks by the Receiver (though Sender could brute-force).

    Args:
        sender: Rank of Sender.
        receiver: Rank of Receiver.
        n: Number of items.
        sender_items: Object located at Sender.
        receiver_items: Object located at Receiver.

    Returns:
        Intersection mask (0/1) located at Receiver.
    """

    # Validation
    if sender == receiver:
        raise ValueError("Sender and Receiver must be different.")
    if n <= 0:
        raise ValueError(f"Input size n must be positive, got {n}.")

    # =========================================================================
    # Phase 1. Parameter Setup
    # =========================================================================
    import mplang.v2.libs.mpc.psi.okvs_gct as okvs_gct

    expansion = okvs_gct.get_okvs_expansion(n)
    M = int(n * expansion)
    if M % 128 != 0:
        M = ((M // 128) + 1) * 128

    # =========================================================================
    # Phase 2. Correlated Randomness (VOLE) - SWAPPED ROLES
    # =========================================================================
    # In the original paper logic (Fig 4), the PSI Sender holds Delta.
    # Therefore, we swap the roles in the OT call.
    #
    # silent_vole_random_u(A, B) gives:
    #   A (OT Sender):   U, V
    #   B (OT Receiver): W, Delta
    #
    # We want PSI Sender to be OT Receiver.
    res_tuple = silent_ot.silent_vole_random_u(receiver, sender, M, base_k=1024)
    
    # PSI Receiver gets U, V
    v_recv, w_sender, u_recv, delta_sender = res_tuple[:4]

    # =========================================================================
    # Phase 3. Receiver Encoding & Masking
    # =========================================================================
    # Receiver computes P such that P(y) = H(y).
    # Receiver masks P with U (Paper's A vector).
    # Q = P ^ U

    from mplang.v2.dialects import crypto
    from mplang.v2.edsl import typing as elt

    def _gen_seed() -> Any:
        return crypto.random_tensor((2,), elt.u64)

    okvs_seed = simp.pcall_static((receiver,), _gen_seed)
    okvs_seed_sender = simp.shuffle_static(okvs_seed, {sender: receiver})

    okvs = okvs_gct.SparseOKVS(M)

    def _recv_ops(y: Any, u: Any, seed: Any) -> Any:
        # 1. Compute H(y)
        def _reshape_seeds(items: Any) -> Any:
            lo = items
            hi = jnp.zeros_like(items)
            return jnp.stack([lo, hi], axis=1)

        seeds = tensor.run_jax(_reshape_seeds, y)
        res_exp = field.aes_expand(seeds, 1)

        def _davies_meyer(enc: Any, s: Any) -> Any:
            enc_flat = enc.reshape(enc.shape[0], 2)
            return jnp.bitwise_xor(enc_flat, s)

        h_y = tensor.run_jax(_davies_meyer, res_exp, seeds)

        # 2. Encode P
        p_storage = okvs.encode(y, h_y, seed)

        # 3. Mask with U (instead of W)
        # Q = P ^ U
        q_storage = field.add(p_storage, u)

        return q_storage

    # Receiver uses U to mask
    q_shared = simp.pcall_static(
        (receiver,), _recv_ops, receiver_items, u_recv, okvs_seed
    )

    q_sender_view = simp.shuffle_static(q_shared, {sender: receiver})

    # =========================================================================
    # Phase 4. Sender "One Decode" & Tag Generation
    # =========================================================================
    # Sender holds W, Delta. Receives Q.
    # W = V + U * Delta
    #
    # Derivation:
    # K = Q * Delta + W
    #   = (P + U) * Delta + (V + U * Delta)
    #   = P * Delta + U * Delta + V + U * Delta
    #   = P * Delta + V
    #
    # Sender computes Tag = Decode(K, x) - H(x) * Delta
    # If x in Intersection: P(x) = H(x)
    # Tag = (P(x) * Delta + V(x)) - P(x) * Delta
    # Tag = V(x)

    def _sender_ops(x: Any, q: Any, w: Any, delta: Any, seed: Any) -> Any:
        # q, w: (M, 2)
        # delta: (2,)

        # 1. Expand Delta for global multiplication (M, 2)
        def _tile_m(d: Any) -> Any:
            return jnp.tile(d, (M // 2, 1)).reshape(M, 2) # Adjust reshape based on lib
        
        # Safe tiling assuming M is aligned
        def _tile_m_simple(d: Any) -> Any:
            return jnp.tile(d, (M, 1))

        delta_expanded_m = tensor.run_jax(_tile_m_simple, delta)

        # 2. Compute Global K = Q * Delta + W
        # This is the O(M) multiplication mentioned in the paper
        q_times_delta = field.mul(q, delta_expanded_m)
        k_storage = field.add(q_times_delta, w)

        # 3. One Decode
        # decoded_val = P(x)*Delta + V(x)
        decoded_k = okvs.decode(x, k_storage, seed)

        # 4. Remove H(x)*Delta
        def _reshape_seeds(items: Any) -> Any:
            lo = items
            hi = jnp.zeros_like(items)
            return jnp.stack([lo, hi], axis=1)

        seeds_x = tensor.run_jax(_reshape_seeds, x)
        res_exp_x = field.aes_expand(seeds_x, 1)

        def _davies_meyer(enc: Any, s: Any) -> Any:
            enc_flat = enc.reshape(enc.shape[0], 2)
            return jnp.bitwise_xor(enc_flat, s)

        h_x = tensor.run_jax(_davies_meyer, res_exp_x, seeds_x)

        # Expand delta for batch N
        def _tile_n(d: Any) -> Any:
            return jnp.tile(d, (n, 1))
        
        delta_expanded_n = tensor.run_jax(_tile_n, delta)
        
        h_x_times_delta = field.mul(h_x, delta_expanded_n)

        # Final Tag = (P*Delta + V) - H*Delta = V(x)
        tag = field.add(decoded_k, h_x_times_delta)

        return tag

    # Execute on Sender
    sender_tags = simp.pcall_static(
        (sender,), 
        _sender_ops, 
        sender_items, 
        q_sender_view, 
        w_sender, 
        delta_sender, 
        okvs_seed_sender
    )
    # =========================================================================
    # Phase 5. Verification (Receiver Side)
    # =========================================================================
    # Sender sends Tags (which should be V(x)) to Receiver. To reduce
    # communication we hash and truncate on the sender side and only send
    # the truncated hash (first 16 bytes).

    # 5.0 Compute hashed & truncated tags on Sender
    from mplang.v2.libs.mpc.ot import extension as ot_extension

    def _hash_and_trunc(tags: Any) -> Any:
        # Compute batched hash on sender and truncate to 16 bytes
        full_h = ot_extension.vec_hash(tags, domain_sep=0x1111, num_rows=n)
        # Use tensor.slice_tensor to slice TraceObjects (start=(0,0), end=(n,16))
        return tensor.slice_tensor(full_h, (0, 0), (n, 16))

    h_sender_trunc = simp.pcall_static((sender,), _hash_and_trunc, sender_tags)

    # 5.1 Send truncated hashes to Receiver (much smaller payload)
    tags_at_recv = simp.shuffle_static(h_sender_trunc, {receiver: sender})

    # 5.2 Receiver computes local V(y) and compares
    def _recv_verify(y: Any, v: Any, seed: Any, remote_tags: Any) -> Any:
        # 1. Decode V locally: target = V(y)
        local_v_y = okvs.decode(y, v, seed)

        # 2. Hash local V(y) and compare with received truncated sender hashes
        # Note: `remote_tags` here is already the truncated hash (16 bytes)
        # sent from the Sender.
        h_local = ot_extension.vec_hash(local_v_y, domain_sep=0x1111, num_rows=n)

        def _core(h_r16: Any, h_l_full: Any) -> Any:
            # h_r16: (n_sender, 16) truncated bytes from sender
            # h_l_full: (n_receiver, k) full hash bytes; truncate to 16
            h_l16 = h_l_full[:, :16]

            eq_matrix = jnp.all(h_r16[:, None, :] == h_l16[None, :, :], axis=2)
            membership = jnp.any(eq_matrix, axis=1)
            return membership.astype(jnp.uint8)

        return tensor.run_jax(_core, remote_tags, h_local)

    intersection_mask = simp.pcall_static(
        (receiver,), _recv_verify, receiver_items, v_recv, okvs_seed, tags_at_recv
    )

    return cast(el.Object, intersection_mask)
