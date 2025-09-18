# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Crypto frontend operations: operation signatures, types, and high-level semantics.

Scope and contracts:
- This module defines portable API shapes; it does not implement cryptography.
- Backends execute the operations and must meet the security semantics required
    by the deployment (confidentiality, authenticity, correctness, etc.).
- The enc/dec API in this frontend uses a conventional 12-byte nonce prefix
    (ciphertext = nonce || payload), and dec expects that format. Other security
    properties (e.g., AEAD) are backend responsibilities.
"""

from __future__ import annotations

import math

from mplang.core.dtype import UINT8
from mplang.core.mpobject import MPObject
from mplang.core.tensor import TensorType
from mplang.frontend.base import femod

_CRYPTO_MOD = femod("crypto")


@_CRYPTO_MOD.typed_op(pfunc_name="crypto.keygen")
def keygen(*, length: int = 32) -> TensorType:
    """Generate random bytes for symmetric keys or generic randomness.

    API: keygen(length: int = 32) -> key: u8[length]

    Notes:
    - Frontend defines the type/shape; backend provides randomness.
    - Raises ValueError when length <= 0.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    return TensorType(UINT8, (length,))


@_CRYPTO_MOD.typed_op(pfunc_name="crypto.enc")
def enc(plaintext: MPObject, key: MPObject) -> TensorType:
    """Symmetric encryption.

    API: enc(plaintext: u8[N], key: u8[M]) -> ciphertext: u8[N + 12]
    """
    pt_ty = TensorType.from_obj(plaintext)
    if pt_ty.dtype != UINT8:
        raise TypeError("enc expects UINT8 plaintext")
    if len(pt_ty.shape) != 1:
        raise TypeError("enc expects 1-D plaintext")
    return TensorType(UINT8, (pt_ty.shape[0] + 12,))


@_CRYPTO_MOD.typed_op(pfunc_name="crypto.dec")
def dec(ciphertext: MPObject, key: MPObject) -> TensorType:
    """Symmetric decryption.

    API: dec(ciphertext: u8[N + 12], key: u8[M]) -> plaintext: u8[N]
    """
    ct_ty = TensorType.from_obj(ciphertext)
    if ct_ty.dtype != UINT8:
        raise TypeError("dec expects UINT8 ciphertext")
    if len(ct_ty.shape) != 1 or ct_ty.shape[0] < 12:
        raise TypeError("dec expects 1-D ciphertext with nonce")
    return TensorType(UINT8, (ct_ty.shape[0] - 12,))


@_CRYPTO_MOD.typed_op(pfunc_name="crypto.pack")
def pack(x: MPObject) -> TensorType:
    """Pack any tensor into a contiguous byte vector (C-order).

    API: pack(x: T[...]) -> bytes: u8[P], where P = sizeof(T) * prod(shape or (1,)).
    """
    x_ty = TensorType.from_obj(x)
    elem_count = 1 if len(x_ty.shape) == 0 else math.prod(x_ty.shape)
    itemsize = x_ty.dtype.numpy_dtype().itemsize
    return TensorType(UINT8, (int(elem_count) * int(itemsize),))


@_CRYPTO_MOD.typed_op(pfunc_name="crypto.unpack")
def unpack(b: MPObject, *, out_ty: TensorType) -> TensorType:
    """Unpack a byte vector into a tensor specified by a TensorType.

    API: unpack(bytes: u8[P], out_ty: TensorType) -> T[...]
    """
    b_ty = TensorType.from_obj(b)
    if b_ty.dtype != UINT8 or len(b_ty.shape) != 1:
        raise TypeError("unpack expects UINT8 1-D byte vector")
    return out_ty


@_CRYPTO_MOD.typed_op(pfunc_name="crypto.kem_keygen")
def kem_keygen(*, suite: str = "x25519") -> tuple[TensorType, TensorType]:
    """KEM-style keypair generation: returns (sk, pk) bytes."""
    sk_ty = TensorType(UINT8, (32,))
    pk_ty = TensorType(UINT8, (32,))
    return sk_ty, pk_ty


@_CRYPTO_MOD.typed_op(pfunc_name="crypto.kem_derive")
def kem_derive(sk: MPObject, peer_pk: MPObject, *, suite: str = "x25519") -> TensorType:
    """KEM-style shared secret derivation: returns secret bytes."""
    _ = TensorType.from_obj(sk)
    _ = TensorType.from_obj(peer_pk)
    return TensorType(UINT8, (32,))


@_CRYPTO_MOD.typed_op(pfunc_name="crypto.hkdf")
def hkdf(secret: MPObject, *, info: str) -> TensorType:
    """HKDF-style key derivation: returns a 32-byte key."""
    _ = TensorType.from_obj(secret)
    return TensorType(UINT8, (32,))
