"""
Crypto frontend FEOps (mock implementation backend).

API:
- keygen(length: int = 32) -> key[u8[length]]
- enc(plaintext[u8[N]], key[u8[K]]) -> ciphertext[u8[N+NONCE]]
- dec(ciphertext[u8[N]], key[u8[K]]) -> plaintext[u8[N-NONCE]]

Notes:
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
