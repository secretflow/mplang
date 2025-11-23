"""Oblivious Transfer (OT) library.

This module implements OT logic using the `crypto` dialect (ECC + Hash + SymEnc).
It implements the Naor-Pinkas 1-out-of-2 OT protocol.

Protocol: Naor-Pinkas 1-out-of-2 OT
Security: Computational security based on ECDH.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import mplang2.edsl.typing as elt
from mplang2.dialects import crypto, simp, tensor


def _receiver_keygen_scalar(C_point, b):
    # b is selection bit (0 or 1)
    # k is private key (random scalar)
    k = crypto.ec_random_scalar()
    G = crypto.ec_generator()
    PK_sigma = crypto.ec_mul(G, k)

    # PK0 = PK_sigma if b=0 else C - PK_sigma
    # We use arithmetic selection for Points:
    # PK0 = PK_sigma + b * (C - 2*PK_sigma)

    b_scalar = crypto.ec_scalar_from_int(b)

    # 2 * PK_sigma
    # We use scalar 2
    two_tensor = tensor.constant(np.array(2, dtype=np.int64))
    two_scalar = crypto.ec_scalar_from_int(two_tensor)

    two_PK_sigma = crypto.ec_mul(PK_sigma, two_scalar)
    diff = crypto.ec_sub(C_point, two_PK_sigma)
    term = crypto.ec_mul(diff, b_scalar)
    PK0 = crypto.ec_add(PK_sigma, term)

    return PK0, k


def _sender_derive_keys(C_point, PK0_point):
    # PK1 = C - PK0
    PK1_point = crypto.ec_sub(C_point, PK0_point)

    def derive_key(PK):
        # Ephemeral key r
        r = crypto.ec_random_scalar()
        G = crypto.ec_generator()
        U = crypto.ec_mul(G, r)  # U = g^r

        # Shared secret K = PK^r
        K_point = crypto.ec_mul(PK, r)
        return U, K_point

    U0, K0 = derive_key(PK0_point)
    U1, K1 = derive_key(PK1_point)

    return U0, K0, U1, K1


def _receiver_derive_key(U0, U1, PK0, k, b):
    b_scalar = crypto.ec_scalar_from_int(b)

    # Select U (Point arithmetic)
    # U = U0 + b*(U1-U0)
    diff_U = crypto.ec_sub(U1, U0)
    term_U = crypto.ec_mul(diff_U, b_scalar)
    U = crypto.ec_add(U0, term_U)

    # Recover Shared Secret K = U^k
    K_point = crypto.ec_mul(U, k)
    return K_point
    # We assume target_type is handled by the primitive or not needed for scalar.
    return crypto.sym_decrypt(sym_key, V, target_type)


def transfer(m0: Any, m1: Any, choice: Any, sender: int, receiver: int) -> Any:
    """Perform 1-out-of-2 Oblivious Transfer (Naor-Pinkas).

    Args:
        m0: Message 0 (on Sender).
        m1: Message 1 (on Sender).
        choice: Selection bit 0 or 1 (on Receiver).
        sender: Rank of the sender.
        receiver: Rank of the receiver.

    Returns:
        The selected message (on Receiver).
    """

    # --- Step 1: Sender Initialization ---
    def sender_init_fn():
        # C is a random point: C = r * G
        r = crypto.ec_random_scalar()
        G = crypto.ec_generator()
        C = crypto.ec_mul(G, r)
        return C

    C = simp.pcall_static((sender,), sender_init_fn)

    # Move C to Receiver
    C_recv = simp.shuffle_static(C, {receiver: sender})

    # Infer target type from m0
    target_type = m0.type
    if isinstance(target_type, elt.MPType):
        target_type = target_type.value_type

    # --- Step 1: Receiver Key Generation ---
    def receiver_keygen_fn(C_point, b):
        # Check if inputs are tensors
        is_tensor = isinstance(b.type, elt.TensorType)

        if is_tensor:
            return tensor.elementwise(_receiver_keygen_scalar, C_point, b)
        else:
            return _receiver_keygen_scalar(C_point, b)

    # Returns (PK0, k) on receiver
    keys_recv = simp.pcall_static((receiver,), receiver_keygen_fn, C_recv, choice)

    # Extract PK0 to send back
    def get_pk0(pair):
        return pair[0]

    PK0_to_send = simp.pcall_static((receiver,), get_pk0, keys_recv)
    PK0_sender = simp.shuffle_static(PK0_to_send, {sender: receiver})

    # --- Step 3: Sender Encryption ---
    def sender_encrypt_fn(C_point, PK0_point, msg0, msg1):
        # 1. Derive keys (Scalar arithmetic on Points)
        # Use elementwise to handle TensorType(PointType) inputs
        # elementwise returns a PyTree (tuple of Tensors)
        keys_tuple = tensor.elementwise(_sender_derive_keys, C_point, PK0_point)
        U0, K0, U1, K1 = keys_tuple

        # 2. Convert keys to bytes and hash (Vectorized ops)
        # K0, K1 are TensorType(PointType, shape)
        # point_to_bytes handles TensorType input
        K0_bytes = crypto.ec_point_to_bytes(K0)
        K1_bytes = crypto.ec_point_to_bytes(K1)

        sym_key0 = crypto.hash_bytes(K0_bytes)
        sym_key1 = crypto.hash_bytes(K1_bytes)

        # 3. Encrypt (Vectorized op)
        # sym_encrypt handles TensorType inputs
        ct0 = crypto.sym_encrypt(sym_key0, msg0)
        ct1 = crypto.sym_encrypt(sym_key1, msg1)

        return U0, ct0, U1, ct1

    ciphertexts = simp.pcall_static((sender,), sender_encrypt_fn, C, PK0_sender, m0, m1)

    # Move ciphertexts to Receiver
    # ciphertexts is a tuple, so we map shuffle over it
    from jax.tree_util import tree_map

    ciphertexts_recv = tree_map(
        lambda x: simp.shuffle_static(x, {receiver: sender}), ciphertexts
    )

    # --- Step 4: Receiver Decryption ---
    def receiver_decrypt_fn(c_texts, keys, b):
        # b is selection bit
        # keys is (PK0, k)
        PK0, k = keys
        U0, V0, U1, V1 = c_texts

        # 1. Derive shared secret (Scalar arithmetic on Points)
        # Use elementwise to handle TensorType inputs
        K_point = tensor.elementwise(_receiver_derive_key, U0, U1, PK0, k, b)

        # 2. Derive key (Vectorized ops)
        K_bytes = crypto.ec_point_to_bytes(K_point)
        sym_key = crypto.hash_bytes(K_bytes)

        # 3. Select ciphertext (Vectorized op)
        # V0, V1 are ciphertexts (TensorType(u8, ...))
        # b is selection bit (TensorType(i64, ...))
        V = crypto.select(b, V1, V0)

        # 4. Decrypt (Vectorized op)
        return crypto.sym_decrypt(sym_key, V, target_type)

    result = simp.pcall_static(
        (receiver,), receiver_decrypt_fn, ciphertexts_recv, keys_recv, choice
    )

    return result
