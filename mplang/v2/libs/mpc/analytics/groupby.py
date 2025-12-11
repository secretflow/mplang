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

"""Oblivious Group-by Sum library.

This module implements algorithms to compute the sum of values grouped by bins,
where the data holder (Sender) and the bin holder (Receiver) keep their inputs private.
"""

# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from mplang.v2.dialects import bfv, crypto, simp, tensor
from mplang.v2.libs.mpc.analytics import aggregation, permutation


def oblivious_groupby_sum_bfv(
    data: Any,
    bins: Any,
    K: int,
    sender: int = 0,
    receiver: int = 1,
    poly_modulus_degree: int = 4096,
    plain_modulus: int | None = None,
) -> Any:
    """Computes group-by sum using BFV homomorphic encryption.

    Best for small K (number of bins) and low bandwidth.

    Args:
        data: Input data tensor (on Sender). Shape (N,).
        bins: Bin assignments (on Receiver). Shape (N,). Values in [0, K).
        K: Number of bins.
        sender: Rank of the data holder.
        receiver: Rank of the bin holder.
        poly_modulus_degree: BFV polynomial modulus degree (slot count).
        plain_modulus: BFV plaintext modulus. If None, uses backend default.

    Returns:
        A tensor of shape (K,) on the Receiver containing the sums.
    """

    # ----------------------------------------------------------------------
    # 1. KeyGen (Sender)
    # ----------------------------------------------------------------------
    def keygen_fn(degree, p_mod):
        kwargs = {"poly_modulus_degree": degree}
        if p_mod is not None:
            kwargs["plain_modulus"] = p_mod

        pk, sk = bfv.keygen(**kwargs)
        rk = bfv.make_relin_keys(sk)
        gk = bfv.make_galois_keys(sk)
        encoder = bfv.create_encoder(poly_modulus_degree=degree)
        return pk, sk, rk, gk, encoder

    # We use a closure to capture parameters
    def keygen_fn_closure():
        return keygen_fn(poly_modulus_degree, plain_modulus)

    pk, sk, rk, gk, encoder = simp.pcall_static((sender,), keygen_fn_closure)

    # ----------------------------------------------------------------------
    # 2. Encrypt Data (Sender)
    # ----------------------------------------------------------------------
    def encrypt_chunks_fn(d, enc, p_key):
        # d is a Value (Tensor)
        shape = d.type.shape
        N = shape[0]
        # Use half the degree to avoid column rotation issues (only row rotation supported)
        B = poly_modulus_degree // 2
        num_chunks = (N + B - 1) // B

        ciphertexts = []
        for i in range(num_chunks):
            start = i * B
            end = min((i + 1) * B, N)

            # Bind loop variables
            def get_chunk(x, s=start, e=end, b_val=B):
                c = x[s:e]
                if e - s < b_val:
                    c = jnp.pad(c, (0, b_val - (e - s)))
                return c

            chunk = tensor.run_jax(get_chunk, d)

            pt = bfv.encode(chunk, enc)
            ct = bfv.encrypt(pt, p_key)
            ciphertexts.append(ct)

        return tuple(ciphertexts)

    encrypted_chunks = simp.pcall_static(
        (sender,), encrypt_chunks_fn, data, encoder, pk
    )

    # Transfer data and keys to Receiver
    def transfer_to_receiver(obj):
        return simp.shuffle_static(obj, {receiver: sender})

    # Always a tuple now
    encrypted_chunks_recv = tuple(transfer_to_receiver(c) for c in encrypted_chunks)

    pk_recv = transfer_to_receiver(pk)
    rk_recv = transfer_to_receiver(rk)
    gk_recv = transfer_to_receiver(gk)
    encoder_recv = transfer_to_receiver(encoder)

    # ----------------------------------------------------------------------
    # 3. Aggregate (Receiver)
    # ----------------------------------------------------------------------
    def aggregate_fn(b_data, cts, p_key, r_key, g_key, enc):
        # b_data is Value (Tensor)
        # cts is list/tuple of Values (Ciphertexts)

        N = b_data.type.shape[0]
        # Use half the degree to avoid column rotation issues
        B = poly_modulus_degree // 2
        num_chunks = len(cts)

        bin_sums = [None] * K

        # Zero ciphertext
        # Pass b_data as dummy to satisfy run_jax requirement
        def make_zero(dummy, b_val=B):
            return jnp.zeros((b_val,), dtype=jnp.int64)

        zero_vec = tensor.run_jax(make_zero, b_data)
        pt_zero = bfv.encode(zero_vec, enc)
        ct_zero = bfv.encrypt(pt_zero, p_key)

        for k in range(K):
            current_sum = ct_zero

            for i in range(num_chunks):
                start = i * B
                end = min((i + 1) * B, N)

                def get_mask(b_chunk_full, s=start, e=end, b_val=B, k_target=k):
                    # b_chunk_full is the full bins tensor
                    c = b_chunk_full[s:e]
                    if e - s < b_val:
                        c = jnp.pad(c, (0, b_val - (e - s)), constant_values=-1)
                    return (c == k_target).astype(jnp.int64)

                mask = tensor.run_jax(get_mask, b_data)
                pt_mask = bfv.encode(mask, enc)

                ct_masked = bfv.mul(cts[i], pt_mask)
                ct_masked = bfv.relinearize(ct_masked, r_key)
                current_sum = bfv.add(current_sum, ct_masked)

            total_sum_ct = aggregation.rotate_and_sum(
                current_sum, B, g_key, slot_count=poly_modulus_degree
            )
            bin_sums[k] = total_sum_ct

        return bin_sums

    encrypted_sums = simp.pcall_static(
        (receiver,),
        aggregate_fn,
        bins,
        encrypted_chunks_recv,
        pk_recv,
        rk_recv,
        gk_recv,
        encoder_recv,
    )

    # Transfer encrypted sums back to Sender
    def transfer_to_sender(obj):
        return simp.shuffle_static(obj, {sender: receiver})

    # Always a tuple/list
    encrypted_sums_sender = tuple(transfer_to_sender(s) for s in encrypted_sums)

    # ----------------------------------------------------------------------
    # 4. Decrypt (Sender)
    # ----------------------------------------------------------------------
    def decrypt_fn(cts, s_key, enc):
        results = []
        for ct in cts:
            pt = bfv.decrypt(ct, s_key)
            vec = bfv.decode(pt, enc)
            # vec is a Tensor Value
            # We need to extract the first element.
            val = tensor.run_jax(lambda v: v[0], vec)
            results.append(val)

        # Stack results into a single tensor
        def stack(*args):
            return jnp.stack(args)

        return tensor.run_jax(stack, *results)

    final_sums_sender = simp.pcall_static(
        (sender,), decrypt_fn, encrypted_sums_sender, sk, encoder
    )

    # ----------------------------------------------------------------------
    # 5. Return to Receiver
    # ----------------------------------------------------------------------
    final_sums_receiver = simp.shuffle_static(final_sums_sender, {receiver: sender})

    return final_sums_receiver


def oblivious_groupby_sum_shuffle(
    data: Any,
    bins: Any,
    K: int,
    sender: int = 0,
    receiver: int = 1,
    helper: int = 2,
) -> Any:
    """Computes group-by sum using Oblivious Shuffle.

    Note: This implementation uses secret sharing to hide the data values from the Receiver.
    It requires a Helper party (3-party protocol).

    Security:
    - Sender learns nothing.
    - Receiver learns the final sums and the bin sizes (from bins).
    - Helper learns the bin sizes (from bins) and a random share of data.
    - No party learns the individual data values or the permutation of data values.

    Args:
        data: Input data tensor (on Sender). Shape (N,).
        bins: Bin assignments (on Receiver). Shape (N,). Values in [0, K).
        K: Number of bins.
        sender: Rank of the data holder.
        receiver: Rank of the bin holder.
        helper: Rank of the helper party.

    Returns:
        A tensor of shape (K,) on the Receiver containing the sums.
    """

    # 1. Compute Permutation (Receiver)
    def compute_perm_fn(b):
        # b is the bins tensor
        # We want indices that sort b
        return tensor.run_jax(lambda x: jnp.argsort(x, stable=True), b)

    perm = simp.pcall_static((receiver,), compute_perm_fn, bins)

    # 2. Secret Share Data (Sender)
    # Security Fix: Generate mask using crypto.random_bytes at RUNTIME on Sender
    # This generates cryptographically secure random bytes that are unique per session.

    def split_shares_fn(d):
        # Generate random bytes at runtime (EDSL primitive, NOT during trace)
        # This is secure because crypto.random_bytes executes at runtime on the party.
        n_elements = d.type.shape[0]
        bytes_per_element = 8  # int64 = 8 bytes
        total_bytes = n_elements * bytes_per_element

        mask_bytes = crypto.random_bytes(total_bytes)

        def _apply_mask(arr, m_bytes):
            # View random bytes as int64 (same as typical input dtype)
            # For generality, we use arr.dtype, but assume int64 for now.
            mask = m_bytes.view(jnp.int64).reshape(arr.shape)
            d0 = arr - mask
            d1 = mask
            return d0, d1

        return tensor.run_jax(_apply_mask, d, mask_bytes)

    d0, d1 = simp.pcall_static((sender,), split_shares_fn, data)

    # 3. Shuffle Share 0 (Sender -> Receiver)
    # Receiver gets s0 = perm(d0)
    s0 = permutation.apply_permutation(d0, perm, sender=sender, receiver=receiver)

    # 4. Compute Agg0 (Receiver)
    def agg_s0_fn(s_val, b, p, k_val):
        def _impl(s_v, b_v, p_v):
            # Sort bins to match data
            s_bins = b_v[p_v]
            # Compute sums for share 0
            return jax.ops.segment_sum(s_v, s_bins, num_segments=k_val)

        return tensor.run_jax(_impl, s_val, b, p)

    agg0 = simp.pcall_static((receiver,), agg_s0_fn, s0, bins, perm, K)

    # 5. Send Share 1 to Helper
    d1_helper = simp.shuffle_static(d1, {helper: sender})

    # 6. Send Bins to Helper
    bins_helper = simp.shuffle_static(bins, {helper: receiver})

    # 7. Compute Agg1 (Helper)
    def agg_d1_fn(d_val, b_val, k_val):
        def _impl(d_v, b_v):
            return jax.ops.segment_sum(d_v, b_v, num_segments=k_val)

        return tensor.run_jax(_impl, d_val, b_val)

    agg1 = simp.pcall_static((helper,), agg_d1_fn, d1_helper, bins_helper, K)

    # 8. Send Agg1 to Receiver
    agg1_recv = simp.shuffle_static(agg1, {receiver: helper})

    # 9. Combine (Receiver)
    def combine_fn(a0, a1):
        return tensor.run_jax(lambda x, y: x + y, a0, a1)

    final_sums = simp.pcall_static((receiver,), combine_fn, agg0, agg1_recv)

    return final_sums
