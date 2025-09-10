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

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.dtype import UINT8
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

    API: enc(plaintext[u8[N]], key[u8[M]]) -> ciphertext[u8[N + 12]]

    Semantics:
    - Ciphertext is defined as nonce(12 bytes) || encrypted_payload.
    - Frontend validates input dtype/shape (1-D UINT8) and builds IR types.

    Notes:
    - Authenticity/integrity guarantees are backend-defined.
    - Raises TypeError if plaintext is not 1-D UINT8.
    """

    def __call__(
        self, plaintext: MPObject, key: MPObject
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        pt_ty = TensorType.from_obj(plaintext)
        key_ty = TensorType.from_obj(key)
        if pt_ty.dtype != UINT8:
            # Keep mock simple: only support u8 arrays
            raise TypeError("symmetric_encrypt expects UINT8 plaintext")
        if len(pt_ty.shape) != 1:
            raise TypeError("symmetric_encrypt expects 1-D plaintext")
        # Prepend 12-byte nonce
        out_ty = TensorType(UINT8, (pt_ty.shape[0] + 12,))
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
        out_ty = TensorType(UINT8, (ct_ty.shape[0] - 12,))
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

    API: hkdf(secret[u8[32]], info[u8[M]]) -> key[u8[32]]

    Notes:
    - Frontend expresses API; backend implements the KDF.
    """

    def __call__(
        self, secret: MPObject, info: MPObject
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        sec_ty = TensorType.from_obj(secret)
        info_ty = TensorType.from_obj(info)
        out_ty = TensorType(UINT8, (32,))
        pfunc = PFunction(
            fn_type="crypto.hkdf",
            ins_info=(sec_ty, info_ty),
            outs_info=(out_ty,),
        )
        _, treedef = tree_flatten(out_ty)
        return pfunc, [secret, info], treedef


hkdf = HKDF()
