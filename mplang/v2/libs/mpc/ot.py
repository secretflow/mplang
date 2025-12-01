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

"""Oblivious Transfer (OT) library.

This module implements OT logic using the `crypto` dialect (ECC + Hash + SymEnc).
It implements the Naor-Pinkas 1-out-of-2 OT protocol.

Protocol: Naor-Pinkas 1-out-of-2 OT
Security: Computational security based on ECDH.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects import crypto, simp, tensor


def _receiver_keygen_scalar(
    C_point: el.Object, b: el.Object
) -> tuple[el.Object, el.Object]:
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


def _sender_derive_keys(
    C_point: el.Object, PK0_point: el.Object
) -> tuple[el.Object, el.Object, el.Object, el.Object]:
    # PK1 = C - PK0
    PK1_point = crypto.ec_sub(C_point, PK0_point)

    def derive_key(PK: el.Object) -> tuple[el.Object, el.Object]:
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


def _receiver_derive_key(
    U0: el.Object, U1: el.Object, PK0: el.Object, k: el.Object, b: el.Object
) -> el.Object:
    b_scalar = crypto.ec_scalar_from_int(b)

    # Select U (Point arithmetic)
    # U = U0 + b*(U1-U0)
    diff_U = crypto.ec_sub(U1, U0)
    term_U = crypto.ec_mul(diff_U, b_scalar)
    U = crypto.ec_add(U0, term_U)

    # Recover Shared Secret K = U^k
    K_point = crypto.ec_mul(U, k)
    return K_point


def transfer(
    m0: el.MPObject,
    m1: el.MPObject,
    choice: el.MPObject,
    sender: int,
    receiver: int,
) -> el.MPObject:
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
    assert isinstance(m0, el.Object) and isinstance(m0.type, elt.MPType)
    assert isinstance(m1, el.Object) and isinstance(m1.type, elt.MPType)
    assert isinstance(choice, el.Object) and isinstance(choice.type, elt.MPType)

    # --- Step 1: Sender Initialization ---
    def sender_init_fn() -> el.Object:
        # C is a random point: C = r * G
        r = crypto.ec_random_scalar()
        G = crypto.ec_generator()
        C = crypto.ec_mul(G, r)
        return C

    C = simp.pcall_static((sender,), sender_init_fn)

    # Move C to Receiver
    C_recv = simp.shuffle_static(C, {receiver: sender})

    # Infer target type from m0
    m0_type = m0.type
    if isinstance(m0_type, elt.MPType):
        val_type = m0_type.value_type
    else:
        val_type = m0_type

    # Since we use tensor.elementwise, we need the element type for decryption
    if isinstance(val_type, elt.TensorType):
        target_type = val_type.element_type
    else:
        target_type = val_type

    # --- Step 1: Receiver Key Generation ---
    def receiver_keygen_fn(C_point: el.Object, b: el.Object) -> el.Object:
        res: el.Object = cast(
            el.Object, tensor.elementwise(_receiver_keygen_scalar, C_point, b)
        )
        return res  # type: ignore[no-any-return]

    # Returns (PK0, k) on receiver
    keys_recv = simp.pcall_static((receiver,), receiver_keygen_fn, C_recv, choice)

    # Extract PK0 to send back
    def get_pk0(pair: Any) -> el.Object:
        return cast(el.Object, pair[0])

    PK0_to_send = simp.pcall_static((receiver,), get_pk0, keys_recv)
    PK0_sender = simp.shuffle_static(PK0_to_send, {sender: receiver})

    # --- Step 3: Sender Encryption ---
    def sender_encrypt_fn(
        C_point: el.Object, PK0_point: el.Object, msg0: el.Object, msg1: el.Object
    ) -> Any:
        def encrypt_elementwise(
            c: Any, pk0: Any, m0: Any, m1: Any
        ) -> tuple[Any, Any, Any, Any]:
            u0, k0, u1, k1 = _sender_derive_keys(c, pk0)

            kb0 = crypto.ec_point_to_bytes(k0)
            kb1 = crypto.ec_point_to_bytes(k1)

            sk0 = crypto.hash_bytes(kb0)
            sk1 = crypto.hash_bytes(kb1)

            c0 = crypto.sym_encrypt(sk0, m0)
            c1 = crypto.sym_encrypt(sk1, m1)
            return u0, c0, u1, c1

        return tensor.elementwise(encrypt_elementwise, C_point, PK0_point, msg0, msg1)

    ciphertexts = simp.pcall_static((sender,), sender_encrypt_fn, C, PK0_sender, m0, m1)

    # Move ciphertexts to Receiver
    # ciphertexts is a tuple, so we map shuffle over it
    from jax.tree_util import tree_map

    ciphertexts_recv = tree_map(
        lambda x: simp.shuffle_static(x, {receiver: sender}), ciphertexts
    )

    # --- Step 4: Receiver Decryption ---
    def receiver_decrypt_fn(c_texts: Any, keys: Any, b: el.Object) -> el.Object:
        # b is selection bit
        # keys is (PK0, k)
        PK0, k = keys
        U0, V0, U1, V1 = c_texts

        def decrypt_elementwise(
            u0: Any, v0: Any, u1: Any, v1: Any, pk0: Any, k_priv: Any, sel: Any
        ) -> Any:
            k_pt = _receiver_derive_key(u0, u1, pk0, k_priv, sel)
            kb = crypto.ec_point_to_bytes(k_pt)
            sk = crypto.hash_bytes(kb)
            v = crypto.select(sel, v1, v0)
            return crypto.sym_decrypt(sk, v, target_type)

        res = tensor.elementwise(decrypt_elementwise, U0, V0, U1, V1, PK0, k, b)
        return cast(el.Object, res)

    result = simp.pcall_static(
        (receiver,), receiver_decrypt_fn, ciphertexts_recv, keys_recv, choice
    )

    res_obj: el.Object = cast(el.Object, result)
    return res_obj
