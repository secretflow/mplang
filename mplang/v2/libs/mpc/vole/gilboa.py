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

"""Vector Oblivious Linear Evaluation (VOLE) Protocol.

Implements the Gilboa protocol for VOLE over GF(2^k).
Global SIMP implementation.
"""

from collections.abc import Callable
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

import mplang.v2.edsl as el
import mplang.v2.libs.mpc.ot.extension as ot
from mplang.v2.dialects import field, simp, tensor


def vole(
    sender: int,
    receiver: int,
    n: int,
    u_provider: Callable[[], el.Object],
    delta_provider: Callable[[], el.Object],
    return_secrets: bool = False,
) -> tuple[el.Object, el.Object] | tuple[el.Object, el.Object, el.Object, el.Object]:
    """Execute VOLE Protocol (Gilboa).

    Args:
        sender: Rank of Sender.
        receiver: Rank of Receiver.
        n: Vector length.
        u_provider: Callable running on Sender returning u (N, 2).
        delta_provider: Callable running on Receiver returning delta (2,).
        return_secrets: If True, returns (v, w, u, delta).

    Returns:
        If return_secrets=False:
            v: Vector on Sender (N, 2).
            w: Vector on Receiver (N, 2).
        If return_secrets=True:
            v, w, u, delta
    """
    K = 128

    # 1. Receiver decomp Delta
    def _recv_prep() -> tuple[el.Object, el.Object]:
        delta = delta_provider()

        # Decompose
        # delta is (2,) u64.
        # Run JAX to unpack
        def _unpack(d: Any) -> Any:
            return jnp.unpackbits(d.view(jnp.uint8), bitorder="little")

        bits_u8 = tensor.run_jax(_unpack, delta)  # (128,) u8
        # Reshape to (128, 1) using run_jax for XLA optimization
        bits_reshaped = tensor.run_jax(lambda x: x.reshape(128, 1), bits_u8)
        return delta, bits_reshaped

    delta_and_bits = simp.pcall_static((receiver,), _recv_prep)
    # Extract
    delta_recv = simp.pcall_static((receiver,), lambda x: x[0], delta_and_bits)
    delta_bits = simp.pcall_static((receiver,), lambda x: x[1], delta_and_bits)

    # 2. Run IKNP OT Core
    # Returns t (Sender), q (Receiver), s (Sender)
    # Note: In standard IKNP, Receiver chooses. Sender gets keys.
    # Here VOLE Receiver chooses (delta bits).
    # So VOLE Receiver is OT Receiver.
    # We need 128 OTs for Gilboa. Result is (128, 128) matrices.
    t_matrix_128, q_matrix_128, s_choices = ot.iknp_core(
        delta_bits, sender, receiver, K
    )

    # t_matrix_128: (128, 128) - 128 OT seeds, each 128 bits wide
    # These are the "Seeds" for the Gilboa extension.
    # Sender has T (128 seeds).
    # Receiver has Q (128 seeds).
    # Wait, IKNP usage usually:
    # Q = T ^ (choices * S).
    # Row i of Q is Q_i = T_i ^ (c_i * S).
    # c_i is delta_i.
    # S is the base OT choice vector (global secret S).

    # We need:
    # Sender has S_{i,0}, S_{i,1}.
    # Receiver has S_{i, d_i}.
    #
    # IKNP gives:
    # Col j of Q = Col j of T ^ (c * S_j) ? No.

    # Let's map IKNP output to Gilboa needs.
    # IKNP gives:
    # For generated OT i (0..127):
    #   Sender holds T[i] (block).
    #   Receiver holds Q[i] (block).
    #   Q[i] = T[i] ^ (c[i] * S).
    #   Where S is the Base OT Choice (held by Sender of IKNP = Sender of VOLE).
    #   Wait, Sender acts as Receiver in BaseOT usually.
    #   In `ot_extension.py`: `s` (base choices) is on Sender.
    #   So Q[i] = T[i] ^ (delta_i * s).

    # This gives us CORRELATED SEEDS.
    # Sender has T[i] and s.
    # Receiver has Q[i].

    # Gilboa needs:
    # Sender sends u * x^i masked.
    # We can use T[i] and (T[i]^s) as the seeds for random strings?
    #
    # Q[i] is ONE seed.
    # If delta_i = 0, Q[i] = T[i].
    # If delta_i = 1, Q[i] = T[i] ^ s.

    # So Sender has two seeds for bit i:
    # Seed0 = T[i]
    # Seed1 = T[i] ^ s

    # This is perfect! IKNP *is* ROT.

    # 3. Expansion
    # Sender expands:
    # V0_i = PRG(T[i], N)
    # V1_i = PRG(T[i] ^ s, N)

    # Receiver expands:
    # W_i = PRG(Q[i], N)
    # Note: W_i = V0_i if delta_i=0
    #       W_i = V1_i if delta_i=1

    # Sender computes correction:
    # M_i = V0_i ^ V1_i ^ (u * x^i)
    # M_i = PRG(T) ^ PRG(T^s) ^ (u * x^i)

    # Receiver computes:
    # result_i = W_i ^ (delta_i * M_i)
    #          = V_{delta_i} ^ (delta_i * (V0^V1^term))
    # if d=0: W = V0. Res = V0. Correct.
    # if d=1: W = V1. Res = V1 ^ V0 ^ V1 ^ term = V0 ^ term.
    # Wait.
    # We want result = V0 + ... ?
    # Gilboa: v = Sum(V0).
    # w = v + u*delta.
    #
    # If d=0: Res = V0.
    # If d=1: Res = V0 ^ term.
    # Sum(Res) = Sum(V0) ^ Sum(d_i * term) = v ^ (u * Sum(d_i x^i)) = v + u*delta.
    # Correct.

    # Implementation:

    # Capture U on Sender
    def _sender_wrapper() -> el.Object:
        u = u_provider()
        return u

    u_loc_captured = simp.pcall_static((sender,), _sender_wrapper)

    m_corrections, v_sender = simp.pcall_static(
        (sender,), _sender_round, t_matrix_128, s_choices, u_loc_captured, n
    )

    # Shuffle M to Receiver
    from jax.tree_util import tree_map

    m_recv = tree_map(
        lambda x: simp.shuffle_static(x, {receiver: sender}), m_corrections
    )

    w_receiver = simp.pcall_static(
        (receiver,), _recv_round, q_matrix_128, m_recv, delta_bits, n
    )

    if return_secrets:
        return v_sender, w_receiver, u_loc_captured, delta_recv
    else:
        return v_sender, w_receiver


# A. Expand (Sender)
def _sender_round(
    t_loc: el.Object, s_loc: el.Object, u_loc: el.Object, n: int
) -> tuple[el.Object, el.Object]:
    # t_loc: (128, 128)
    # s_loc: (128,)
    # u_loc: (N, 2)

    # 0. Prep Seeds
    def _prep_sender_seeds(t: Any, s: Any) -> tuple[Any, Any]:
        # t: (128, 128) bits
        # s: (128,) bits
        t_seeds = jnp.packbits(t, axis=-1)  # (128, 16) uint8
        s_bytes = jnp.packbits(s, axis=-1)  # (16,)
        s_broad = jnp.expand_dims(s_bytes, 0)  # (1, 16)
        t_xor_s_seeds = jnp.bitwise_xor(t_seeds, s_broad)
        return t_seeds, t_xor_s_seeds

    t_seeds, t_s_seeds = tensor.run_jax(_prep_sender_seeds, t_loc, s_loc)
    t_seeds = cast(el.Object, t_seeds)
    t_s_seeds = cast(el.Object, t_s_seeds)

    # 1. Expand
    v0_expanded = field.aes_expand(t_seeds, n)
    v1_expanded = field.aes_expand(t_s_seeds, n)

    # 2. Compute term = u * powers using Field Arithmetic
    # Vectorized Version:
    # u_loc: (N, 2)
    # powers: (128, 2)
    # term: (128, N, 2) = u_loc * p_broad

    # Generate Powers of X (128, 2) CONSTANT
    # 1, x, x^2 ...
    powers_list = []
    for i in range(128):
        lo, hi = 0, 0
        if i < 64:
            lo = 1 << i
        else:
            hi = 1 << (i - 64)
        powers_list.append([lo, hi])
    powers_arr = np.array(powers_list, dtype=np.uint64)
    powers_const = tensor.constant(powers_arr)

    # Broadcast for Vectorized Mul
    # u_loc: (N, 2) -> (1, N, 2) -> (128, N, 2)
    # powers: (128, 2) -> (128, 1, 2) -> (128, N, 2)

    def _broadcast_inputs(u_val: Any, p_val: Any) -> tuple[Any, Any]:
        # u: (N, 2)
        # p: (128, 2)
        n_ = u_val.shape[0]

        # Tile U: (128, N, 2)
        u_broad = jnp.tile(u_val[None, :, :], (128, 1, 1))

        # Tile P: (128, N, 2)
        p_broad = jnp.tile(p_val[:, None, :], (1, n_, 1))

        return u_broad, p_broad

    u_vec, p_vec = tensor.run_jax(_broadcast_inputs, u_loc, powers_const)

    # Single Batched Mul
    term_val = field.mul(u_vec, p_vec)  # (128, N, 2)

    # 3. Compute Corrections
    def _sender_calc(v0: Any, v1: Any, term: Any) -> tuple[Any, Any]:
        # v0: (128, N, 2)
        # v1: (128, N, 2)
        # term: (128, N, 2)

        m_out = v0 ^ v1 ^ term

        # v_out sum
        v_out = v0[0]
        for i in range(1, 128):
            v_out = v_out ^ v0[i]

        return m_out, v_out

    m_corr, v = tensor.run_jax(_sender_calc, v0_expanded, v1_expanded, term_val)
    return cast(el.Object, m_corr), cast(el.Object, v)


# B. Expand & Reconstruct (Receiver)
def _recv_round(
    q_loc: el.Object, m_loc: el.Object, d_bits: el.Object, n: int
) -> el.Object:
    # 0. Prep Seeds
    def _prep_recv_seeds(q: Any) -> Any:
        return jnp.packbits(q, axis=-1)

    q_seeds = tensor.run_jax(_prep_recv_seeds, q_loc)

    # 1. AES Expand
    w_expanded = field.aes_expand(q_seeds, n)  # (128, N, 2)

    # 2. Reconstruct
    def _recv_calc(w_exp: Any, m_val: Any, d_b: Any) -> Any:
        # w_exp: (128, N, 2)
        # m_val: (128, N, 2)
        # d_b: (128, 1) bits from earlier

        d_flat = d_b.reshape(128)
        # Mask M
        # m_val is u64. d_flat is u8(?).
        mask = d_flat.reshape(128, 1, 1).astype(bool)
        m_masked = jnp.where(mask, m_val, jnp.zeros_like(m_val))

        res_i = w_exp ^ m_masked
        # w_final = jnp.bitwise_xor.reduce(res_i, axis=0)
        w_final = res_i[0]
        for i in range(1, 128):
            w_final = w_final ^ res_i[i]

        return w_final

    return cast(el.Object, tensor.run_jax(_recv_calc, w_expanded, m_loc, d_bits))


def _gen_powers_of_x_jax(dummy: Any, k: int = 128) -> Any:
    # JAX version for use inside run_jax (returns jnp.array)
    # dummy is required for run_jax tracing anchor
    rows = []
    for i in range(k):
        lo, hi = 0, 0
        if i < 64:
            lo = 1 << i
        else:
            hi = 1 << (i - 64)
        rows.append([lo, hi])
    return jnp.array(rows, dtype=jnp.uint64)
