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
    """Execute OKVS-based PSI Protocol.

    Args:
        sender: Rank of Sender.
        receiver: Rank of Receiver.
        n: Number of items (must be same for now).
        sender_items: Object located at Sender containing (N,) u64 items.
        receiver_items: Object located at Receiver containing (N,) u64 items.

    Returns:
        Intersection verification tuple (T, U*, Delta).
    """

    # Validation
    if sender == receiver:
        raise ValueError(
            f"Sender ({sender}) and Receiver ({receiver}) must be different."
        )

    if n <= 0:
        raise ValueError(f"Input size n must be positive, got {n}.")

    # =========================================================================
    # Phase 1. Parameter Setup & Topology
    # =========================================================================
    # OKVS Size M = expansion * N.
    # The expansion factor is critical for the success probability of the "Peeling"
    # algorithm used in OKVS encoding (Garbled Cuckoo Table).
    # Larger N allows smaller expansion (closer to theoretical 1.23) while maintaining safety.
    import mplang.v2.libs.mpc.psi.okvs_gct as okvs_gct

    expansion = okvs_gct.get_okvs_expansion(n)
    M = int(n * expansion)

    # Align M to 128 boundary for efficient batch processing in Silent VOLE (LPN)
    if M % 128 != 0:
        M = ((M // 128) + 1) * 128

    # =========================================================================
    # Phase 2. Correlated Randomness Generation (VOLE)
    # =========================================================================
    # Parties run Silent VOLE (based on LPN assumption) to generate:
    #   Sender:   U, V  (Vectors of size M)
    #   Receiver: W, Delta
    # Correlation: W = V + U * Delta
    #
    # Note: U is uniformly random. It acts as a "One-Time Pad" key for the protocol.

    # silent_vole_random_u returns (v, w, u, delta)
    res_tuple = silent_ot.silent_vole_random_u(sender, receiver, M, base_k=1024)
    v_sender, w_receiver, u_sender, delta_receiver = res_tuple[:4]

    # =========================================================================
    # Phase 3. Receiver Encoding & Masking (OKVS)
    # =========================================================================
    # The Receiver encodes their input set Y into the OKVS structure P.
    # Goal: Decode(P, y) = H(y)  forall y in Y.
    #
    # Then, Receiver masks P with the VOLE output W to get Q:
    # Q = P ^ W
    # This Q is sent to the Sender.

    # 3.1 Generate OKVS Seed (Public/Session Randomness)
    # Used for OKVS hashing distribution. Can be public, but generated at runtime for safety.
    from mplang.v2.dialects import crypto
    from mplang.v2.edsl import typing as elt

    def _gen_seed() -> Any:
        return crypto.random_tensor((2,), elt.u64)

    okvs_seed = simp.pcall_static((receiver,), _gen_seed)
    okvs_seed_sender = simp.shuffle_static(okvs_seed, {sender: receiver})

    # Instantiate OKVS Data Structure
    okvs = okvs_gct.SparseOKVS(M)

    def _recv_ops(y: Any, w: Any, delta: Any, seed: Any) -> Any:
        # y: (N,) Inputs
        # w: (M, 2) VOLE share

        # 3.2 Compute H(y) - The Random Oracle Target
        # We use Davies-Meyer construction: H(x) = E_x(0) ^ x
        # This is a standard, efficient, and robust way to instantiate a RO from AES.

        def _reshape_seeds(items: Any) -> Any:
            # Prepare items as AES keys (128-bit)
            lo = items
            hi = jnp.zeros_like(items)
            return jnp.stack([lo, hi], axis=1)  # (N, 2)

        seeds = tensor.run_jax(_reshape_seeds, y)
        res_exp = field.aes_expand(seeds, 1)  # (N, 1, 2)

        def _davies_meyer(enc: Any, s: Any) -> Any:
            enc_flat = enc.reshape(enc.shape[0], 2)
            return jnp.bitwise_xor(enc_flat, s)

        h_y = tensor.run_jax(_davies_meyer, res_exp, seeds)

        # 3.3 Solve System of Linear Equations (OKVS Encode)
        # We find P such that: P * M_okvs(y) = h_y
        p_storage = okvs.encode(y, h_y, seed)

        # 3.4 Mask with Vole Share
        # Q = P ^ W
        q_storage = field.add(p_storage, w)

        return q_storage

    # Execute on Receiver
    q_shared = simp.pcall_static(
        (receiver,), _recv_ops, receiver_items, w_receiver, delta_receiver, okvs_seed
    )

    # 3.5 Send Q to Sender
    q_sender_view = simp.shuffle_static(q_shared, {sender: receiver})

    # =========================================================================
    # Phase 4. Sender Decoding & Reconstruction
    # =========================================================================
    # Sender uses Q and their local shares (U, V) to reconstruct T.
    #
    # Derivation:
    # 1. S_decoded = Decode(Q, x) = Decode(P ^ W, x) = P(x) ^ W(x)
    # 2. Recall W(x) = V(x) ^ U(x)*Delta (VOLE property)
    # 3. So S_decoded = P(x) ^ V(x) ^ U(x)*Delta
    #
    # 4. Sender computes T = S_decoded ^ V(x) ^ H(x)
    #    T = P(x) ^ V(x) ^ U(x)*Delta ^ V(x) ^ H(x)
    #    T = P(x) ^ H(x) ^ U(x)*Delta
    #
    # 5. If x is in Intersection (Meanings x == y for some y):
    #    Then P(x) == H(x) (by OKVS property)
    #    So T = H(x) ^ H(x) ^ U(x)*Delta
    #    T = U(x)*Delta
    #
    # This relation T == U* * Delta is what we verify in Phase 5.

    def _sender_ops(x: Any, q: Any, u: Any, v: Any, seed: Any) -> tuple[Any, Any]:
        # x: (N,) Sender Items
        # q: (M, 2) Received OKVS

        # 4.1 Decode Q and V at x
        # OKVS Decode is a linear combination of storage positions.
        s_decoded = okvs.decode(x, q, seed)
        v_decoded = okvs.decode(x, v, seed)

        # 4.2 Compute H(x)
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

        # 4.3 Compute T candidate
        # T = S ^ V ^ H(x)
        # Note: s_decoded is (S^V^U*Delta) effectively
        t_val = field.add(s_decoded, v_decoded)
        t_val = field.add(t_val, h_x)

        # 4.4 Compute U* = Decode(U, x)
        # This is the sender's share of the randomness for item x.
        s_u = field.decode_okvs(x, u, seed)

        return t_val, s_u

    t_val_sender, u_star_sender = simp.pcall_static(
        (sender,),
        _sender_ops,
        sender_items,
        q_sender_view,
        u_sender,
        v_sender,
        okvs_seed_sender,
    )

    # =========================================================================
    # Phase 5. Secure Verification
    # =========================================================================
    # The Protocol invariant is T == U* * Delta for intersection items.
    #
    # Security Risk:
    # We must NOT reveal T or Delta to the other party.
    # - If Receiver learns T, they can compute Diff = T - U*Delta = H(x) + ... and attack x.
    # - If Sender learns Delta, VOLE security collapses.
    #
    # Secure Verification Method:
    # 1. Sender sends U* (Random Mask share) to Receiver.
    #    - U* is derived from U (random VOLE inputs) so it reveals nothing about X.
    #
    # 2. Receiver computes Target = U* * Delta.
    #    - This allows Receiver to construct the expected value of T without knowing T's components.
    #
    # 3. Receiver Hashes the Target and sends H(Target) to Sender.
    #    - Hashing prevents Sender from learning Delta algebraically.
    #    - Hash function acts as a commitment.
    #
    # 4. Sender compares H(T) =? H(Target).
    #    - Equality implies x is in Intersection.

    # 5.1 Sender -> Receiver: U*
    u_star_recv = simp.shuffle_static(u_star_sender, {receiver: sender})

    # 5.2 Receiver: Compute Expected Target (U* * Delta)
    def _recv_verify_ops(u_s: Any, delta: Any) -> Any:
        # u_s: (N, 2), delta: (2,)

        # Use tensor.run_jax to isolate JAX operations (tile is not an EDSL primitive)
        def _tile(d: Any) -> Any:
            return jnp.tile(d, (n, 1))

        delta_expanded = tensor.run_jax(_tile, delta)

        # Compute U* * Delta in GF(2^128)
        target = field.mul(u_s, delta_expanded)
        return target

    target_val = simp.pcall_static(
        (receiver,), _recv_verify_ops, u_star_recv, delta_receiver
    )

    # 5.3 Hash Exchange
    # Use robust hashing to prevent algebraic attacks or leakage
    from mplang.v2.libs.mpc.ot import extension as ot_extension

    def _hash_shares(share: el.Object, party: int) -> el.Object:
        """Hash the shares using domain separator for security."""
        return ot_extension.vec_hash(share, domain_sep=0xFEED, num_rows=n)

    # Hash(Target) on Receiver
    h_target_recv = simp.pcall_static(
        (receiver,), lambda x: _hash_shares(x, receiver), target_val
    )

    # Hash(T) on Sender
    h_t_sender = simp.pcall_static(
        (sender,), lambda x: _hash_shares(x, sender), t_val_sender
    )

    # Send Hash to Sender for comparison
    h_target_at_sender = simp.shuffle_static(h_target_recv, {sender: receiver})

    # 5.4 Final Comparison on Sender
    def _compare(h_t: Any, h_target: Any) -> Any:
        # Compare 32-byte hashes (N, 32) row-by-row

        def _core(a: Any, b: Any) -> Any:
            eq = jnp.all(a == b, axis=1)
            return eq.astype(jnp.uint8)  # (N,) 0 or 1

        return tensor.run_jax(_core, h_t, h_target)

    intersection_mask = simp.pcall_static(
        (sender,), _compare, h_t_sender, h_target_at_sender
    )

    return cast(el.Object, intersection_mask)
