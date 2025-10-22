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

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core import UINT8, TensorType
from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction
from mplang.ops.base import stateless_mod

_CRYPTO_MOD = stateless_mod("crypto")


def _get_algo_overhead(algo: str) -> int:
    """Get ciphertext overhead for a given encryption algorithm.

    Args:
        algo: Encryption algorithm identifier

    Returns:
        int: Number of overhead bytes added to plaintext length
    """
    overhead_map = {
        "aes-ctr": 16,  # nonce only (legacy compatibility)
        "aes-gcm": 28,  # nonce(12) + tag(16) for AES-GCM
        "sm4-gcm": 28,  # nonce(12) + tag(16) for SM4-GCM
    }

    if algo not in overhead_map:
        # return unknown overhead as -1
        return -1
    return overhead_map[algo]


@_CRYPTO_MOD.simple_op()
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


@_CRYPTO_MOD.op_def()
def enc(
    plaintext: MPObject, key: MPObject, algo: str = "aes-ctr"
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Symmetric encryption with algorithm-aware output sizing.

    API: enc(plaintext: u8[N], key: u8[M], *, algo: str = "aes-ctr") -> ciphertext: u8[N + overhead]

    Supported algorithms and overhead:
    - "aes-ctr": 16 bytes (nonce only, legacy compatibility)
    - "aes-gcm": 28 bytes (nonce + 16-byte authentication tag)
    - "sm4-gcm": 28 bytes (nonce + 16-byte authentication tag)

    The algo parameter is stored in the PFunction attributes for backend use.
    """
    pt_ty = plaintext
    if pt_ty.dtype != UINT8:
        raise TypeError("enc expects UINT8 plaintext")
    if len(pt_ty.shape) != 1:
        raise TypeError("enc expects 1-D plaintext")

    # Validate and get overhead for the specified algorithm
    overhead = _get_algo_overhead(algo)
    length = pt_ty.shape[0]
    if length >= 0 and overhead >= 0:
        outs_info = (TensorType(UINT8, (length + overhead,)),)
    else:
        # Unknown length or overhead, return dynamic length
        outs_info = (TensorType(UINT8, (-1,)),)

    ins_info = (TensorType.from_obj(pt_ty), TensorType.from_obj(key))
    pfunc = PFunction(
        fn_type="crypto.enc",
        ins_info=ins_info,
        outs_info=outs_info,
        algo=algo,
    )
    _, treedef = tree_flatten(outs_info[0])
    return pfunc, [plaintext, key], treedef


@_CRYPTO_MOD.op_def()
def dec(
    ciphertext: MPObject, key: MPObject, algo: str = "aes-ctr"
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Symmetric decryption with algorithm-aware input sizing.

    API: dec(ciphertext: u8[N + overhead], key: u8[M], *, algo: str = "aes-ctr") -> plaintext: u8[N]

    Supported algorithms and overhead:
    - "aes-ctr": 16 bytes (nonce only, legacy compatibility)
    - "aes-gcm": 28 bytes (nonce + 16-byte authentication tag)
    - "sm4-gcm": 28 bytes (nonce + 16-byte authentication tag)

    The algo parameter is stored in the PFunction attributes for backend use.
    Backend is responsible for parsing the ciphertext format according to algo.
    """
    ct_ty = ciphertext
    if ct_ty.dtype != UINT8:
        raise TypeError("dec expects UINT8 ciphertext")
    if len(ct_ty.shape) != 1:
        raise TypeError("dec expects 1-D ciphertext")

    # Validate and get overhead for the specified algorithm
    overhead = _get_algo_overhead(algo)
    length = ct_ty.shape[0]

    # Validate minimum ciphertext length
    if length >= 0 and overhead >= 0 and length < overhead:
        raise TypeError(
            f"dec expects ciphertext with at least {overhead} bytes for algo='{algo}', but got {length} bytes"
        )

    # Compute output plaintext length
    if length >= 0 and overhead >= 0:
        outs_info = (TensorType(UINT8, (length - overhead,)),)
    else:
        # Unknown length or overhead, return dynamic length
        outs_info = (TensorType(UINT8, (-1,)),)

    ins_info = (TensorType.from_obj(ct_ty), TensorType.from_obj(key))
    pfunc = PFunction(
        fn_type="crypto.dec",
        ins_info=ins_info,
        outs_info=outs_info,
        algo=algo,
    )
    _, treedef = tree_flatten(outs_info[0])
    return pfunc, [ciphertext, key], treedef


@_CRYPTO_MOD.simple_op()
def kem_keygen(*, suite: str = "x25519") -> tuple[TensorType, TensorType]:
    """KEM-style keypair generation: returns (sk, pk) bytes."""
    sk_ty = TensorType(UINT8, (32,))
    pk_ty = TensorType(UINT8, (32,))
    return sk_ty, pk_ty


@_CRYPTO_MOD.simple_op()
def kem_derive(
    sk: TensorType, peer_pk: TensorType, *, suite: str = "x25519"
) -> TensorType:
    """KEM-style shared secret derivation: returns secret bytes."""
    _ = sk
    _ = peer_pk
    return TensorType(UINT8, (32,))


@_CRYPTO_MOD.simple_op()
def hkdf(secret: TensorType, *, info: str) -> TensorType:
    """HKDF-style key derivation: returns a 32-byte key."""
    _ = secret
    return TensorType(UINT8, (32,))
