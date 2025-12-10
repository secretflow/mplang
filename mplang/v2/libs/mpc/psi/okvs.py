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

This module implements a high-performance PSI protocol.
1. VOLE: Establishes correlated randomness W = V + U * Delta.
2. OKVS: Receiver encodes inputs Y into P such that Decode(P, y) = H(y).
3. Masking: Receiver sends Q = P ^ W.
4. Check: Sender verifies Decode(Q, x) against local V, H(x) to derive U*Delta correlation.
"""

from typing import Any

import jax.numpy as jnp

import mplang.v2.dialects.field as field
import mplang.v2.dialects.simp as simp
import mplang.v2.dialects.tensor as tensor
import mplang.v2.edsl as el
import mplang.v2.libs.mpc.ot.silent as silent_ot


def psi_intersect(
    sender: int,
    receiver: int,
    n: int,
    sender_items: el.Object,
    receiver_items: el.Object,
) -> tuple[el.Object, el.Object, el.Object]:
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

    # 1. Parameter Setup
    # OKVS Size M = expansion * N, where expansion is dynamically chosen
    # based on N (larger N allows smaller expansion, saving communication)
    from mplang.v2.libs.mpc.psi.sparse_okvs import get_okvs_expansion

    expansion = get_okvs_expansion(n)
    M = int(n * expansion)
    # Align to 128 for VOLE batching if needed
    if M % 128 != 0:
        M = ((M // 128) + 1) * 128

    # 2. Run VOLE (Random)
    # Sender gets U, V (Size M). Receiver gets W = V + U*Delta (Size M).
    # We need providers for U and Delta.

    # Run Silent VOLE (Random U)
    # v_sender: (M, 2), w_receiver: (M, 2)
    # Uses Block-wise Linear Expansion to save bandwidth.
    res_tuple = silent_ot.silent_vole_random_u(sender, receiver, M, base_k=1024)
    v_sender, w_receiver, u_sender, delta_receiver = res_tuple[:4]

    # 3. Receiver OKVS Encode & Mask
    def _recv_ops(y: Any, w: Any, delta: Any) -> Any:
        # y: (N,), w: (M, 2), delta: (2,)
        # Encode: P = Solve(y, H(y))
        # Need H(y).

        # Implement Davies-Meyer construction: H(x) = E_x(0) ^ x
        # This provides a robust random oracle construction from AES-128
        # suitable for the OKVS encoding steps.

        # 1. Expand input items to use as keys/seeds for AES
        # y is (N,) u64. res is (N, 1, 2) u64.
        def _reshape_seeds(items: Any) -> Any:
            # items (N,) u64.
            # We need (N, 2) seeds.
            # Pad with 0?
            lo = items
            hi = jnp.zeros_like(items)
            return jnp.stack([lo, hi], axis=1)  # (N, 2)

        seeds = tensor.run_jax(_reshape_seeds, y)
        res_exp = field.aes_expand(seeds, 1)  # (N, 1, 2)

        def _davies_meyer(enc: Any, s: Any) -> Any:
            # enc: (N, 1, 2) u64
            # s: (N, 2) u64
            enc_flat = enc.reshape(enc.shape[0], 2)
            return jnp.bitwise_xor(enc_flat, s)

        h_y = tensor.run_jax(_davies_meyer, res_exp, seeds)

        # Solve OKVS
        # keys=y, values=h_y
        p_storage = field.solve_okvs(y, h_y, m=M)

        # Mask
        q_storage = field.add(p_storage, w)

        return q_storage

    # Execute on Receiver
    q_shared = simp.pcall_static(
        (receiver,), _recv_ops, receiver_items, w_receiver, delta_receiver
    )

    # Extract Q (sent to Sender)
    q_for_sender = (
        q_shared  # Already on Receiver? No, returned by pcall_static on Recv rank.
    )
    # Actually pcall_static returns a Distributed Object (MPType).
    # We need to shuffle it to Sender.

    # We explicitly mask the shuffle:
    q_sender_view = simp.shuffle_static(q_for_sender, {sender: receiver})

    # 4. Sender Decode & Check
    def _sender_ops(x: Any, q: Any, u: Any, v: Any) -> tuple[Any, Any]:
        # x: (N,), q: (M, 2), u: (M, 2), v: (M, 2)

        # Decode Ops
        s_decoded = field.decode_okvs(x, q)
        v_decoded = field.decode_okvs(x, v)

        # Helper: Hash items
        # Use Davies-Meyer: H(x) = E_x(0) ^ x
        def _reshape_seeds(items: Any) -> Any:
            lo = items
            hi = jnp.zeros_like(items)
            return jnp.stack([lo, hi], axis=1)  # (N, 2)

        seeds_x = tensor.run_jax(_reshape_seeds, x)
        res_exp_x = field.aes_expand(seeds_x, 1)  # (N, 1, 2)

        def _davies_meyer(enc: Any, s: Any) -> Any:
            enc_flat = enc.reshape(enc.shape[0], 2)
            return jnp.bitwise_xor(enc_flat, s)

        h_x = tensor.run_jax(_davies_meyer, res_exp_x, seeds_x)

        # T = S ^ V ^ H(x)
        t_val = field.add(s_decoded, v_decoded)
        t_val = field.add(t_val, h_x)
        s_u = field.decode_okvs(x, u)

        return t_val, s_u

    # Arguments must match: sender_items(x), q_sender_view(q), u_sender(u), v_sender(v)
    t_val_sender, u_star_sender = simp.pcall_static(
        (sender,), _sender_ops, sender_items, q_sender_view, u_sender, v_sender
    )

    # 5. Shared Secret Verification (OPRF Output)
    return t_val_sender, u_star_sender, delta_receiver
