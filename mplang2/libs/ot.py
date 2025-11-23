"""Oblivious Transfer (OT) library.

This module implements OT logic using the `crypto` dialect (ECC + Hash + SymEnc).
It implements the Naor-Pinkas 1-out-of-2 OT protocol.

Protocol: Naor-Pinkas 1-out-of-2 OT
Security: Computational security based on ECDH.
"""

from __future__ import annotations

from typing import Any

from mplang2.dialects import crypto, simp


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

    # --- Step 2: Receiver Key Generation ---
    def receiver_keygen_fn(C_point, b):
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
        import numpy as np

        from mplang2.dialects import tensor

        two_tensor = tensor.constant(np.array(2, dtype=np.int64))
        two_scalar = crypto.ec_scalar_from_int(two_tensor)

        two_PK_sigma = crypto.ec_mul(PK_sigma, two_scalar)
        diff = crypto.ec_sub(C_point, two_PK_sigma)
        term = crypto.ec_mul(diff, b_scalar)
        PK0 = crypto.ec_add(PK_sigma, term)

        return PK0, k

    # Returns (PK0, k) on receiver
    keys_recv = simp.pcall_static((receiver,), receiver_keygen_fn, C_recv, choice)

    # Extract PK0 to send back
    def get_pk0(pair):
        return pair[0]

    PK0_to_send = simp.pcall_static((receiver,), get_pk0, keys_recv)
    PK0_sender = simp.shuffle_static(PK0_to_send, {sender: receiver})

    # --- Step 3: Sender Encryption ---
    def sender_encrypt_fn(C_point, PK0_point, msg0, msg1):
        # PK1 = C - PK0
        PK1_point = crypto.ec_sub(C_point, PK0_point)

        def encrypt_msg(PK, m):
            # Ephemeral key r
            r = crypto.ec_random_scalar()
            G = crypto.ec_generator()
            U = crypto.ec_mul(G, r)  # U = g^r

            # Shared secret K = PK^r
            K_point = crypto.ec_mul(PK, r)

            # Derive symmetric key
            K_bytes = crypto.ec_point_to_bytes(K_point)
            sym_key = crypto.hash_bytes(K_bytes)

            # Encrypt message
            ciphertext = crypto.sym_encrypt(sym_key, m)
            return U, ciphertext

        U0, V0 = encrypt_msg(PK0_point, msg0)
        U1, V1 = encrypt_msg(PK1_point, msg1)

        return (U0, V0, U1, V1)

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
        _, k = keys
        U0, V0, U1, V1 = c_texts

        # We need to select (U, V) based on b.
        # U = U0 if b=0 else U1
        # V = V0 if b=0 else V1

        b_scalar = crypto.ec_scalar_from_int(b)

        # Select U (Point arithmetic)
        # U = U0 + b*(U1-U0)
        diff_U = crypto.ec_sub(U1, U0)
        term_U = crypto.ec_mul(diff_U, b_scalar)
        U = crypto.ec_add(U0, term_U)

        # Select V (Bytes/Tensor arithmetic)
        # V = V0*(1-b) + V1*b
        # Use crypto.select to avoid run_jax dynamic shape issues
        V = crypto.select(b, V1, V0)

        # Recover Shared Secret K = U^k
        K_point = crypto.ec_mul(U, k)

        # Derive key
        K_bytes = crypto.ec_point_to_bytes(K_point)
        sym_key = crypto.hash_bytes(K_bytes)

        # Decrypt
        # We use a placeholder type for now as we don't have the original type info easily available
        # in this context without passing it.
        from mplang2.edsl.typing import TensorType, i64

        target_type = TensorType(i64, (1,))  # Placeholder

        return crypto.sym_decrypt(sym_key, V, target_type)

    result = simp.pcall_static(
        (receiver,), receiver_decrypt_fn, ciphertexts_recv, keys_recv, choice
    )

    return result
