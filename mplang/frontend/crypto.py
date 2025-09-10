"""
Crypto frontend FEOps (mock implementation backend).

Notes:
- This frontend describes operation signatures, types, and high-level semantics; it
    does not implement cryptography. Security properties depend on the selected backend.
- Backends are responsible for providing secure algorithms (e.g., AEAD for enc/dec;
    real KEM/ECDH/HPKE and HKDF for key agreement/derivation) and for validating inputs.
- This repository also includes a tutorial/mock backend for local demos that is NOT
    secure. When using that backend, `enc/dec` is a hash-based stream cipher without
    authentication and KEM/HKDF are simplified. Production deployments should use a
    proper backend implementation.
    - Stream cipher is hash-based XOR and NOT AEAD (no integrity/authenticity).
    - KEM ops and HKDF are placeholders (not real ECDH/HPKE/HKDF).
- This frontend describes types/attrs; execution is handled by backend/crypto.
- The mock backend uses a blake2b-based keystream (NOT secure) with a random
    12-byte nonce prepended to the ciphertext.
"""

from __future__ import annotations

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.dtype import UINT8
from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType
from mplang.frontend.base import FEOp


class KeyGen(FEOp):
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
    """KEM-style keypair generation (frontend semantic; backend may mock).

    API: kem_keygen(suite: str = 'x25519') -> (sk[u8[32]], pk[u8[32]])
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
    """HKDF-style key derivation (frontend semantic; backend may mock).

    API: hkdf(secret[u8[32]], info[u8[M]]) -> key[u8[32]]
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
