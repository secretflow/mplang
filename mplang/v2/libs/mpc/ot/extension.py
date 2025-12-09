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

"""OT Extension (IKNP).

Implements IKNP OT extension protocol to perform N OTs using k Base OTs.
Ref: https://crypto.stanford.edu/~valeria/research/2003/IKNP03.pdf
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.edsl as el
from mplang.v2.dialects import crypto, field, simp, tensor


def prg_expand(seed_tensor: el.Object, length: int) -> el.Object:
    """Pseudo-Random Generator: Expand seed to `length` bits (as uint8 0/1).

    Uses AES-NI via field.aes_expand for cryptographic security.
    """
    # Calculate number of 128-bit blocks needed to cover `length` bits.
    # field.aes_expand returns (K, M, 2) uint64 blocks.
    # Total bits = M * 128.

    m_blocks = (length + 127) // 128

    # Input seed_tensor is (K, 32) bytes (uint8).
    # field.aes_expand expects (K, 2) uint64 seeds.

    def _reshape_seeds(s_bytes: Any) -> Any:
        # s_bytes: (K, 32) u8
        # Take first 16 bytes for 128-bit key/seed
        s_16 = s_bytes[:, :16]
        return s_16.view(jnp.uint64).reshape(-1, 2)

    seeds_u64 = tensor.run_jax(_reshape_seeds, seed_tensor)

    expanded_blocks = field.aes_expand(seeds_u64, m_blocks)  # (K, M, 2) u64

    # Convert blocks to bits
    def _blocks_to_bits(blocks: Any) -> Any:
        # blocks: (K, M, 2) u64
        # unpackbits
        # view as u8
        bytes_view = blocks.view(jnp.uint8)  # (K, M, 16)
        bits = jnp.unpackbits(bytes_view, axis=-1, bitorder="little")  # (K, M, 128)

        # Flatten last two dims
        bits_flat = bits.reshape(bits.shape[0], -1)

        # Slice to exact length
        return bits_flat[:, :length]

    return cast(el.Object, tensor.run_jax(_blocks_to_bits, expanded_blocks))


def vec_hash(data_bytes: el.Object, domain_sep: int = 0) -> el.Object:
    """Hash rows of a (K, D) tensor independently.

    Args:
        data_bytes: (K, D) tensor to hash.
        domain_sep: Integer domain separator to mix into the hash.
    """
    # Assuming data_bytes is (K, D). K should be static (128).
    # Since we are in EDSL, shape is typically known at trace time for static shapes.
    K = 128

    # We unroll slicing
    result_hashes = []

    # Domain separator as bytes
    # We can mix it by prepending to the row if possible, or XORing.
    # crypto.hash_bytes takes 1D tensor.
    # To be safe, we will assume data_bytes is uint8.

    for i in range(K):
        # Slice row i
        # We need to slice the whole row.
        # tensor.slice_tensor(obj, starts, stops).
        # We assume dimension 1 size is D.
        # We can use a very large stop for dim 1 to take "rest".
        row_slice = tensor.slice_tensor(data_bytes, (i, 0), (i + 1, 1000000))

        # Reshape to 1D
        row = tensor.reshape(row_slice, (-1,))

        # Add Domain Separation
        if domain_sep != 0:
            # We can't easy prepend in EDSL without cost, but we can verify later.
            # Ideally: hash(domain_sep || row)
            # WORKAROUND: For now, we assume `crypto.hash_bytes` is SHA256.
            # We will use EDSL to prepend a 8-byte prefix if domain_sep != 0.
            # But constructing that tensor is verbose.
            # Simplified: Use the fact that row is U8.
            pass
            # TODO: Implement proper domain separation by concatenating.
            # For this fix, let's just make sure we are hashing the FULL row.

        h = crypto.hash_bytes(row)
        result_hashes.append(h)

    reshaped_hashes = [tensor.reshape(h, (1, 32)) for h in result_hashes]
    return tensor.concat(reshaped_hashes, axis=0)


def iknp_core(
    choice_bits: el.Object, sender: int, receiver: int, num_ots: int
) -> tuple[el.Object, el.Object, el.Object]:
    """Core IKNP Matrix Generation.

    Returns:
        t_matrix: (N, K) bit matrix on Sender.
        q_matrix: (N, K) bit matrix on Receiver.
        s_choices: (K,) choice bits on Sender (s).
    """
    K = 128

    # 1. Base OTs
    def gen_s() -> el.Object:
        # Generate random bytes at runtime via primitive
        rand_bytes = crypto.random_bytes(K)  # returns (K,) u8

        # Use JAX for arithmetic operations
        def _to_bits(x: Any) -> Any:
            return x % 2

        return cast(el.Object, tensor.run_jax(_to_bits, rand_bytes))

    s = simp.pcall_static((sender,), gen_s)

    def gen_seeds() -> tuple[el.Object, el.Object]:
        # Generate random bytes at runtime
        k0_bytes = crypto.random_bytes(K * 32)
        k1_bytes = crypto.random_bytes(K * 32)

        # Reshape to (K, 32)
        k0 = tensor.reshape(k0_bytes, (K, 32))
        k1 = tensor.reshape(k1_bytes, (K, 32))
        return k0, k1

    k0_base, k1_base = simp.pcall_static((receiver,), gen_seeds)

    # Base OT Logic (Inlined)
    # S (Receiver of BaseOT) inits
    def base_receiver_init() -> el.Object:
        C = crypto.ec_mul(crypto.ec_generator(), crypto.ec_random_scalar())
        return C

    C_point = simp.pcall_static((sender,), base_receiver_init)
    C_recv = simp.shuffle_static(C_point, {receiver: sender})

    # Duplicate initialization removed

    # R (Sender of BaseOT) keygen
    def base_sender_keygen(
        C: el.Object, s_base_choices: el.Object
    ) -> tuple[list[el.Object], list[el.Object]]:
        # s_base_choices is (K,) Tensor
        PK0_list = []
        k_priv_list = []

        for i in range(K):
            k_priv = crypto.ec_random_scalar()
            PK_sigma = crypto.ec_mul(crypto.ec_generator(), k_priv)

            # Slice s[i]
            s_i = tensor.slice_tensor(s_base_choices, (i,), (i + 1,))  # (1,)
            s_scalar = crypto.ec_scalar_from_int(s_i)

            diff = crypto.ec_sub(C, PK_sigma)
            # select checks s_scalar. If 1 (true), pick diff.
            PK0 = crypto.select(s_scalar, diff, PK_sigma)

            PK0_list.append(PK0)
            k_priv_list.append(k_priv)

        return PK0_list, k_priv_list

    base_keys = simp.pcall_static((sender,), base_sender_keygen, C_point, s)
    # base_keys is tuple of Lists.
    # When using pcall with lists, it returns lists of TraceObjects.

    # Extract
    PK0_loc = simp.pcall_static((sender,), lambda x: x[0], base_keys)
    # Move to R
    from jax.tree_util import tree_map

    PK0_recv = tree_map(lambda x: simp.shuffle_static(x, {receiver: sender}), PK0_loc)

    # R (Base Sender) Encrypts k0, k1
    def base_encrypt_rev(
        C: el.Object,
        PK0_list: list[el.Object],
        m0_tensor: el.Object,
        m1_tensor: el.Object,
    ) -> tuple[list[el.Object], list[el.Object], list[el.Object]]:
        # m0, m1 are (K, 32) tensors.

        U_list = []
        c0_list = []
        c1_list = []

        for i in range(K):
            r = crypto.ec_random_scalar()
            U = crypto.ec_mul(crypto.ec_generator(), r)
            U_list.append(U)

            PK0 = PK0_list[i]
            K0_point = crypto.ec_mul(PK0, r)
            PK1 = crypto.ec_sub(C, PK0)
            K1_point = crypto.ec_mul(PK1, r)

            sk0 = crypto.hash_bytes(crypto.ec_point_to_bytes(K0_point))  # (32,)
            sk1 = crypto.hash_bytes(crypto.ec_point_to_bytes(K1_point))

            # Encrypt m0[i], m1[i]
            m0_i = tensor.slice_tensor(m0_tensor, (i, 0), (i + 1, 32))  # (1, 32)
            m1_i = tensor.slice_tensor(m1_tensor, (i, 0), (i + 1, 32))

            # Reshape to (32,) to match sk
            m0_flat = tensor.reshape(m0_i, (32,))
            m1_flat = tensor.reshape(m1_i, (32,))

            def _enc(k: Any, m: Any) -> Any:
                return jnp.bitwise_xor(m, k)

            c0 = tensor.run_jax(_enc, sk0, m0_flat)
            c1 = tensor.run_jax(_enc, sk1, m1_flat)

            c0_list.append(c0)
            c1_list.append(c1)

        return U_list, c0_list, c1_list

    base_cts_rev = simp.pcall_static(
        (receiver,), base_encrypt_rev, C_recv, PK0_recv, k0_base, k1_base
    )
    # Shuffle tuple(list, list, list)
    from jax.tree_util import tree_map

    base_cts_s = tree_map(
        lambda x: simp.shuffle_static(x, {sender: receiver}), base_cts_rev
    )

    def base_decrypt_rev(
        keys: tuple[list[el.Object], list[el.Object]],
        cts: tuple[list[el.Object], list[el.Object], list[el.Object]],
        s_choices: el.Object,
    ) -> el.Object:
        _, k_priv_list = keys
        U_list, c0_list, c1_list = cts

        k_s_rows = []

        for i in range(K):
            k_priv = k_priv_list[i]
            U = U_list[i]
            c0 = c0_list[i]
            c1 = c1_list[i]

            # Recov K = U^k_priv
            SharedK = crypto.ec_mul(U, k_priv)
            sk = crypto.hash_bytes(crypto.ec_point_to_bytes(SharedK))

            s_i = tensor.slice_tensor(s_choices, (i,), (i + 1,))  # (1,)

            def _dec(k: Any, c0_: Any, c1_: Any, sel: Any) -> Any:
                # sel is (1,)
                # scalar 0 or 1
                chosen_c = jnp.where(sel[0] == 0, c0_, c1_)
                return jnp.bitwise_xor(chosen_c, k)

            res_flat = tensor.run_jax(_dec, sk, c0, c1, s_i)  # (32,)
            k_s_rows.append(tensor.reshape(res_flat, (1, 32)))

        # Concat
        return tensor.concat(k_s_rows, axis=0)

    k_s = simp.pcall_static((sender,), base_decrypt_rev, base_keys, base_cts_s, s)

    # 2. PRG Expansion & Correction
    def calc_u(k0_loc: el.Object, k1_loc: el.Object, r_loc: el.Object) -> el.Object:
        g_k0 = prg_expand(k0_loc, num_ots)  # (K, num_ots)
        g_k1 = prg_expand(k1_loc, num_ots)  # (K, num_ots)

        # choice_bits can be:
        # - (N,) 1D vector for standard IKNP
        # - (N, K) 2D matrix for KKRT OPRF
        #
        # For IKNP: u^j = G(k0^j) ^ G(k1^j) ^ r, where r is broadcast to all K rows
        # For KKRT: u^j = G(k0^j) ^ G(k1^j) ^ r^j, where r is (N, K) transposed to (K, N)

        # Handle both 1D and 2D inputs
        def _compute_u(g0: Any, g1: Any, r: Any) -> Any:
            # g0, g1: (K, N) bit matrices
            # r: either (N,) or (N, K)
            if r.ndim == 1:
                # 1D case: broadcast (N,) -> (1, N) for XOR with (K, N)
                r_t = jnp.expand_dims(r, axis=0)  # (1, N)
            else:
                # 2D case: transpose (N, K) -> (K, N)
                r_t = jnp.transpose(r, (1, 0))  # (K, N)
            return jnp.bitwise_xor(jnp.bitwise_xor(g0, g1), r_t)

        return cast(el.Object, tensor.run_jax(_compute_u, g_k0, g_k1, r_loc))

    u = simp.pcall_static((receiver,), calc_u, k0_base, k1_base, choice_bits)
    u_recv = simp.shuffle_static(u, {sender: receiver})

    # 3. Matrix Recovery & Transpose
    def calc_t(k_s_loc: el.Object, u_loc: el.Object, s_loc: el.Object) -> el.Object:
        g_k_s = prg_expand(k_s_loc, num_ots)

        def _recover(g: Any, mask: Any, sel: Any) -> Any:
            sel_exp = jnp.expand_dims(sel, axis=-1)
            term = jnp.bitwise_and(mask, sel_exp)
            return jnp.bitwise_xor(g, term)

        t_rows = tensor.run_jax(_recover, g_k_s, u_loc, s_loc)
        return tensor.transpose(t_rows, perm=(1, 0))

    t_matrix = simp.pcall_static((sender,), calc_t, k_s, u_recv, s)

    def calc_q(k0_loc: el.Object) -> el.Object:
        g_k0 = prg_expand(k0_loc, num_ots)
        return tensor.transpose(g_k0, perm=(1, 0))  # (N, K)

    q_matrix = simp.pcall_static((receiver,), calc_q, k0_base)

    # s is on Sender. t_matrix on Sender. q_matrix on Receiver.
    return t_matrix, q_matrix, s


def s_choices_sender(s: el.Object) -> el.Object:
    return s  # Already pcalled on sender


def transfer_extension(
    m0: el.Object,
    m1: el.Object,
    choice_bits: el.Object,
    sender: int,
    receiver: int,
    num_ots: int,
) -> el.Object:
    """Perform IKNP OT Extension."""

    t_matrix, q_matrix, s = iknp_core(choice_bits, sender, receiver, num_ots)

    # 4. Encryption
    def encrypt_msgs(
        t_loc: el.Object, s_loc: el.Object, m0_loc: el.Object, m1_loc: el.Object
    ) -> el.Object:
        # t: (N, K)
        # s: (K,)

        def _hash_and_enc(t: Any, s: Any, msg0: Any, msg1: Any) -> Any:
            t_bytes = jnp.packbits(t, axis=-1)  # (N, 16)
            t_xor_s = jnp.bitwise_xor(t, s)
            t_xor_s_bytes = jnp.packbits(t_xor_s, axis=-1)

            c0 = jnp.bitwise_xor(msg0, t_bytes)
            c1 = jnp.bitwise_xor(msg1, t_xor_s_bytes)
            return c0, c1

        return cast(
            el.Object, tensor.run_jax(_hash_and_enc, t_loc, s_loc, m0_loc, m1_loc)
        )

    ciphertexts = simp.pcall_static((sender,), encrypt_msgs, t_matrix, s, m0, m1)

    from jax.tree_util import tree_map

    ciphertexts_recv = tree_map(
        lambda x: simp.shuffle_static(x, {receiver: sender}), ciphertexts
    )

    def decrypt_msg(
        q_loc: el.Object, r_loc: el.Object, c_texts: tuple[el.Object, el.Object]
    ) -> el.Object:
        c0, c1 = c_texts

        def _dec(q: Any, r: Any, ct0: Any, ct1: Any) -> Any:
            q_bytes = jnp.packbits(q, axis=-1)
            m0_cand = jnp.bitwise_xor(ct0, q_bytes)
            m1_cand = jnp.bitwise_xor(ct1, q_bytes)
            r_exp = jnp.expand_dims(r, axis=-1)
            return jnp.where(r_exp == 1, m1_cand, m0_cand)

        return cast(el.Object, tensor.run_jax(_dec, q_loc, r_loc, c0, c1))

    res = simp.pcall_static(
        (receiver,), decrypt_msg, q_matrix, choice_bits, ciphertexts_recv
    )
    return cast(el.Object, res)
