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

"""Oblivious Pseudorandom Function (OPRF).

Implements KKRT-style OPRF based on OT Extension.
Ref: https://eprint.iacr.org/2016/799.pdf
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.edsl as el
from mplang.v2.dialects import simp, tensor
from mplang.v2.libs.mpc.ot import extension as ot_extension


def eval_oprf(
    receiver_inputs: el.Object,  # (N, 16) bytes
    sender: int,
    receiver: int,
    num_items: int,
) -> tuple[el.Object, el.Object]:
    """Evaluate OPRF on receiver's inputs using KKRT-style protocol.

    Protocol Overview:
    ──────────────────────────────────────────────────────────────────────────
    This implements a simplified KKRT OPRF using IKNP OT Extension as the base.

    Parties:
    - Sender: Has secret key (T matrix, s vector) from IKNP
    - Receiver: Has inputs x₁, ..., xₙ and gets PRF outputs

    Key Relations (IKNP):
    ──────────────────────────────────────────────────────────────────────────
    Let:
        Q[i]: (K,) bit vector - receiver's OT output for row i
        T[i]: (K,) bit vector - sender's OT output for row i
        s:    (K,) bit vector - sender's secret (random)
        c[i]: 1 bit - receiver's choice bit for row i

    IKNP Correlation:
        T[i][j] = Q[i][j] ⊕ (c[i] · s[j])    for all j ∈ [0, K)

    Where ⊕ is XOR and · is AND.
    This means: if c[i] = 1: T[i] = Q[i] ⊕ s
                if c[i] = 0: T[i] = Q[i]

    Simplified OPRF Construction:
    ──────────────────────────────────────────────────────────────────────────
    Choice bits: c[i] = encode(x_i)[0]  (first bit of item encoding)

    Receiver output:  PRF(x_i) = pack(Q[i])  (just pack the Q matrix row)
    Sender can eval:  PRF(y) = pack(T[row(y)])  (pack corresponding T row)

    When x_i == y and they map to same row: outputs match due to IKNP relation.

    Note: Full KKRT uses Cuckoo hashing to map items to rows. This simplified
    version assumes sequential mapping (item i uses row i).

    Args:
        receiver_inputs: (N, 16) byte tensor of receiver's inputs
        sender: Rank of sender party
        receiver: Rank of receiver party
        num_items: Number of items N

    Returns:
        Tuple of:
        - sender_key: (T, s) tuple on sender - T is (N, K) bit matrix, s is (K,)
        - receiver_outputs: (N, 32) byte tensor of OPRF outputs on receiver (SHA256)
    """
    K = 128  # Security parameter (OT extension width)

    # ═════════════════════════════════════════════════════════════════════════
    # Step 1: Encode receiver's inputs to choice bits for IKNP
    # ═════════════════════════════════════════════════════════════════════════
    # For each input x_i, we need K choice bits for IKNP OT Extension.
    # We use a deterministic encoding: unpack bytes to bits.

    def encode_inputs(inputs: el.Object) -> el.Object:
        """Encode (N, 16) byte inputs to (N, K) bit codes.

        Each 16-byte input is unpacked to 128 bits.
        These bits serve as the receiver's OT choices.
        """

        def _encode(x: Any) -> Any:
            # x: (N, 16) uint8 array
            # Unpack each byte to 8 bits: (N, 16) -> (N, 128)
            unpacked = jnp.unpackbits(x, axis=1)  # (N, 128)
            # Ensure exactly K bits
            return unpacked[:, :K].astype(jnp.uint8)  # (N, K)

        return cast(el.Object, tensor.run_jax(_encode, inputs))

    choice_codes = simp.pcall_static((receiver,), encode_inputs, receiver_inputs)
    # choice_codes: (N, K) bit matrix on receiver

    # ═════════════════════════════════════════════════════════════════════════
    # Step 2: Extract first bit of each code as IKNP choice bits
    # ═════════════════════════════════════════════════════════════════════════
    # Simplified: use only first bit of encoding as OT choice
    # Full KKRT would use all K bits differently

    # ═════════════════════════════════════════════════════════════════════════
    # Step 3: Run IKNP OT Extension to generate correlated matrices Q and T
    # ═════════════════════════════════════════════════════════════════════════
    # IKNP generates:
    #   Q: (N, K) on receiver - one K-bit row per item
    #   T: (N, K) on sender   - correlated via T[i] = Q[i] ⊕ (choice[i] · s)
    #   s: (K,) on sender     - random secret vector

    # Pass full K-bit codes as choice bits (N, K)
    t_matrix, q_matrix, s = ot_extension.iknp_core(
        choice_codes, sender, receiver, num_items
    )
    # t_matrix: (N, K) on sender
    # q_matrix: (N, K) on receiver
    # s: (K,) on sender

    # ═════════════════════════════════════════════════════════════════════════
    # Step 4: Compute OPRF outputs
    # ═════════════════════════════════════════════════════════════════════════
    # Simplified KKRT:
    #   Receiver: output_i = pack(Q[i])  (pack 128 bits to 16 bytes)
    #   Sender:   can later compute pack(T[i]) for matching items

    def compute_receiver_outputs(q: el.Object, codes: el.Object) -> el.Object:
        """Compute receiver's OPRF outputs by packing Q matrix rows.

        Args:
            q: (N, K) bit matrix Q from IKNP
            codes: (N, K) bit codes (not used in simplified version)

        Returns:
            (N, 16) packed bytes - OPRF output for each input
        """

        def _process(q_mat: Any, code_mat: Any) -> Any:
            # q_mat: (N, K=128) bits
            # Pack each row from 128 bits to 16 bytes
            packed = jnp.packbits(q_mat, axis=1)  # (N, 16) uint8
            return packed

        packed_q = cast(el.Object, tensor.run_jax(_process, q, codes))

        # Security Fix: Hash the OT output to implement a Random Oracle
        # OPRF = H(OT_output, input_tweaks...)
        # Here we use the shared vec_hash utility which handles domain separation.
        return ot_extension.vec_hash(packed_q, domain_sep=0x0CDF, num_rows=num_items)

    receiver_outputs = simp.pcall_static(
        (receiver,), compute_receiver_outputs, q_matrix, choice_codes
    )
    # receiver_outputs: (N, 32) on receiver

    # ═════════════════════════════════════════════════════════════════════════
    # Step 5: Package sender's key for later PRF evaluation
    # ═════════════════════════════════════════════════════════════════════════
    # Sender keeps (T, s) to evaluate PRF on any input later
    sender_key = simp.pcall_static((sender,), lambda t, s_: (t, s_), t_matrix, s)
    # sender_key: tuple (T, s) on sender where T is (N,K), s is (K,)

    return sender_key, receiver_outputs


# =============================================================================
# KKRT OPRF Sender Evaluation (Vectorized)
# =============================================================================
#
# KKRT Formula:
# ─────────────────────────────────────────────────────────────────────────────
#   For sender with key (T, s) and input y:
#     code_y = encode(y)              # K bits
#     output = pack(T[row] XOR (code_y * s))
#
#   For receiver with Q matrix and input x:
#     code_x = encode(x)              # K bits
#     output = pack(Q[row] XOR code_x)
#
#   When x == y:
#     T[row] XOR (code_x * s) == Q[row] XOR code_x  ✅ (due to IKNP correlation)
# =============================================================================


def sender_eval_prf_batch(
    sender_key: el.Object,  # Tuple (t_matrix, s) on sender
    sender_items: el.Object,  # (M, 16) bytes - items to evaluate
    sender: int,
    num_items: int,
) -> el.Object:
    """Evaluate PRF on sender's side for a batch of items.

    Args:
        sender_key: The key tuple (t_matrix, s) from eval_oprf.
        sender_items: (M, 16) byte tensor of sender's items.
        sender: Rank of sender party.
        num_items: Number of items M (must be provided).

    Returns:
        (M, 32) byte tensor of PRF outputs on sender.
    """
    K = 128

    def compute_sender_outputs(key: el.Object, items: el.Object) -> el.Object:
        """Compute sender's PRF outputs using KKRT formula."""

        def _eval(key_tuple: Any, x: Any) -> Any:
            t_matrix, s = key_tuple
            M = x.shape[0]
            N = t_matrix.shape[0]

            # Encode items to get choice bits
            # Unpack: (M, 16) -> (M, 128) bits
            codes = jnp.unpackbits(x, axis=1)[:, :K]  # (M, K)

            # Compute (codes · s) for each item
            # Masking s with item codes ensures result depends on EVERY bit
            # codes: (M, K), s: (K,) -> broadcast to (M, K)
            code_masked = jnp.where(codes, s, 0).astype(t_matrix.dtype)

            # Use row i for item i
            M_clipped = min(M, N)
            t_rows = t_matrix[:M_clipped]  # (M_clipped, K)

            # KKRT: output = T[i] XOR (first_bit[i] · s)
            xored = jnp.bitwise_xor(t_rows, code_masked[:M_clipped])  # (M_clipped, K)

            # Pack to bytes
            packed = jnp.packbits(xored, axis=1)  # (M_clipped, 16)

            # Pad if needed
            if M > N:
                padding = jnp.zeros((M - N, 16), dtype=packed.dtype)
                packed = jnp.concatenate([packed, padding], axis=0)

            return packed

        raw_outputs = cast(el.Object, tensor.run_jax(_eval, key, items))

        return ot_extension.vec_hash(raw_outputs, domain_sep=0x0CDF, num_rows=num_items)

    return cast(
        el.Object,
        simp.pcall_static((sender,), compute_sender_outputs, sender_key, sender_items),
    )


def sender_eval_prf(
    sender_key: el.Object,  # Tuple (t_matrix, s) on sender
    candidate: el.Object,  # (16,) bytes to evaluate
    sender: int,
) -> el.Object:
    """Evaluate PRF on sender's side for a single candidate.

    This allows sender to compute PRF(k, y) for any y.

    Args:
        sender_key: The key tuple from eval_oprf.
        candidate: A single 16-byte input to evaluate.
        sender: Rank of sender party.

    Returns:
        (32,) byte tensor of PRF output on sender.
    """
    K = 128

    def _eval(key: el.Object, cand: el.Object) -> el.Object:
        def _compute(key_tuple: Any, c: Any) -> Any:
            t_matrix, s = key_tuple

            # Encode candidate to K bits
            code = jnp.unpackbits(c)[:K]  # (K,)

            # KKRT formula: output = pack(t_row XOR (code * s))
            t_row = t_matrix[0]  # (K,) - use first row
            code_masked = jnp.bitwise_and(code, s)  # (K,)
            xored = jnp.bitwise_xor(t_row, code_masked)  # (K,)

            # Pack to bytes
            packed = jnp.packbits(xored)  # (16,)

            # Reshape to (1, 16) for vec_hash
            return packed.reshape(1, 16)

        raw_out_batch = cast(el.Object, tensor.run_jax(_compute, key, cand))

        # Use batched hash with num_rows=1
        hashed_batch = ot_extension.vec_hash(
            raw_out_batch, domain_sep=0x0CDF, num_rows=1
        )

        # Flatten back to (32,) using slice to avoid extra run_jax node
        return tensor.slice_tensor(hashed_batch, (0, 0), (32,))

    return cast(el.Object, simp.pcall_static((sender,), _eval, sender_key, candidate))
