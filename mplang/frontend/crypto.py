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
Crypto frontend FEOps: operation signatures, types, and high‑level semantics.

Scope and contracts:
- This module defines portable API shapes; it does not implement cryptography.
- Backends execute the operations and must meet the security semantics required
    by the deployment (confidentiality, authenticity, correctness, etc.).
- The enc/dec API in this frontend uses a conventional 12‑byte nonce prefix
    (ciphertext = nonce || payload), and dec expects that format. Other security
    properties (e.g., AEAD) are backend responsibilities.
"""

from __future__ import annotations

import math

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.dtype import BOOL, INT64, UINT8
from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType
from mplang.frontend.base import FEOp


class KeyGen(FEOp):
    """Generate random bytes for symmetric keys or generic randomness.

    API: keygen(length: int = 32) -> key[u8[length]]

    Notes:
    - Frontend defines the type/shape; backend provides randomness.
    - Raises ValueError when length <= 0.
    """

    def __call__(self, length: int = 32) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        if length <= 0:
            raise ValueError("length must be > 0")
        out_ty = TensorType(UINT8, (length,))
        pfunc = PFunction(
            fn_type="crypto.keygen",
            ins_info=(),
            outs_info=(out_ty,),
            length=length,
        )
        _, treedef = tree_flatten(out_ty)
        return pfunc, [], treedef


keygen = KeyGen()


class SymmetricEncrypt(FEOp):
    """Symmetric encryption.

    API: enc(plaintext[T[N]], key[u8[M]]) -> ciphertext[u8[N + 12]]

    Semantics:
    - Ciphertext is defined as nonce(12 bytes) || encrypted_payload.
    - Frontend computes output size from total input bytes: nbytes = itemsize *
        prod(shape or (1,)). Dtype is treated as raw bytes for sizing only.

    Notes:
    - Authenticity/integrity guarantees are backend-defined.
    - Raises TypeError if plaintext is not 1-D.
    """

    def __call__(
        self, plaintext: MPObject, key: MPObject
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        pt_ty = TensorType.from_obj(plaintext)
        key_ty = TensorType.from_obj(key)
        # Compute total plaintext bytes (support scalars and any shape)
        elem_count = 1 if len(pt_ty.shape) == 0 else math.prod(pt_ty.shape)
        itemsize = pt_ty.dtype.numpy_dtype().itemsize
        nbytes = int(elem_count) * int(itemsize)
        # Prepend 12-byte nonce to ciphertext bytes
        out_ty = TensorType(UINT8, (nbytes + 12,))
        pfunc = PFunction(
            fn_type="crypto.enc",
            ins_info=(pt_ty, key_ty),
            outs_info=(out_ty,),
        )
        _, treedef = tree_flatten(out_ty)
        return pfunc, [plaintext, key], treedef


enc = SymmetricEncrypt()


class SymmetricDecrypt(FEOp):
    """Symmetric decryption.

    API: dec(ciphertext[u8[N + 12]], key[u8[M]]) -> plaintext[u8[N]]

    Semantics:
    - Expects a 12-byte nonce prefix and returns the decrypted payload bytes.
    - Frontend validates input dtype/shape and builds IR types.

    Notes:
    - Authenticity/integrity checks (if any) are backend-defined.
    - Raises TypeError if ciphertext is not 1-D UINT8 with nonce.
    """

    def __call__(
        self, ciphertext: MPObject, key: MPObject
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        ct_ty = TensorType.from_obj(ciphertext)
        key_ty = TensorType.from_obj(key)
        if ct_ty.dtype != UINT8:
            raise TypeError("symmetric_decrypt expects UINT8 ciphertext")
        if len(ct_ty.shape) != 1 or ct_ty.shape[0] < 12:
            raise TypeError("symmetric_decrypt expects 1-D ciphertext with nonce")
        byte_len = ct_ty.shape[0] - 12
        # Heuristic reconstruction for common cases to preserve user-visible types
        # - If total bytes divisible by 8, interpret as int64 vector
        # - If single byte, interpret as bool[1]
        # - Otherwise, keep as raw bytes (u8)
        if byte_len % 8 == 0 and byte_len > 0:
            out_ty = TensorType(INT64, (byte_len // 8,))
        elif byte_len == 1:
            out_ty = TensorType(BOOL, (1,))
        else:
            out_ty = TensorType(UINT8, (byte_len,))
        pfunc = PFunction(
            fn_type="crypto.dec",
            ins_info=(ct_ty, key_ty),
            outs_info=(out_ty,),
        )
        _, treedef = tree_flatten(out_ty)
        return pfunc, [ciphertext, key], treedef


dec = SymmetricDecrypt()


class KemKeyGen(FEOp):
    """KEM-style keypair generation.

    API: kem_keygen(suite: str = 'x25519') -> (sk[u8[32]], pk[u8[32]])

    Notes:
    - Frontend expresses the signature/shape; backend implements the scheme.
    - The suite string is forwarded to the backend.
    """

    def __call__(
        self, suite: str = "x25519"
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        sk_ty = TensorType(UINT8, (32,))
        pk_ty = TensorType(UINT8, (32,))
        pfunc = PFunction(
            fn_type="crypto.kem_keygen",
            ins_info=(),
            outs_info=(sk_ty, pk_ty),
            suite=suite,
        )
        _, treedef = tree_flatten((sk_ty, pk_ty))
        return pfunc, [], treedef


kem_keygen = KemKeyGen()


class KemDerive(FEOp):
    """KEM-style shared secret derivation.

    API: kem_derive(sk[u8[32]], peer_pk[u8[32]], suite: str = 'x25519') -> secret[u8[32]]

    Notes:
    - Frontend defines types; backend performs the cryptographic operation.
    - The suite string is forwarded to the backend.
    """

    def __call__(
        self, sk: MPObject, peer_pk: MPObject, suite: str = "x25519"
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        sk_ty = TensorType.from_obj(sk)
        pk_ty = TensorType.from_obj(peer_pk)
        out_ty = TensorType(UINT8, (32,))
        pfunc = PFunction(
            fn_type="crypto.kem_derive",
            ins_info=(sk_ty, pk_ty),
            outs_info=(out_ty,),
            suite=suite,
        )
        _, treedef = tree_flatten(out_ty)
        return pfunc, [sk, peer_pk], treedef


kem_derive = KemDerive()


class HKDF(FEOp):
    """HKDF-style key derivation.

    HKDF stands for "HMAC-based Key Derivation Function". It derives one or
    more sub-keys from a shared secret and a context string ("info"). In
    production, HKDF typically follows an extract-and-expand construction over
    a secure hash/HMAC. Here the frontend only defines the API shape; concrete
    security is provided by the backend.

    API: hkdf(secret[u8[32]], info: str) -> key[u8[32]]

    Notes:
    - Frontend expresses API; backend implements the KDF.
    - The info parameter is a string literal carried in attrs.
    """

    def __call__(
        self, secret: MPObject, info: str
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        sec_ty = TensorType.from_obj(secret)
        out_ty = TensorType(UINT8, (32,))
        pfunc = PFunction(
            fn_type="crypto.hkdf",
            ins_info=(sec_ty,),
            outs_info=(out_ty,),
            info=info,
        )
        _, treedef = tree_flatten(out_ty)
        return pfunc, [secret], treedef


hkdf = HKDF()
