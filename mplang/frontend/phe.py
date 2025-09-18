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

"""PHE (Partially Homomorphic Encryption) frontend operations."""

from typing import Any

from mplang.core.dtype import COMPLEX128
from mplang.core.mpobject import MPObject
from mplang.core.mptype import TensorType
from mplang.frontend.base import femod

_PHE_MOD = femod("phe")


@_PHE_MOD.typed_op(name="keygen", pfunc_name="phe.keygen")
def keygen(
    *, scheme: str = "paillier", key_size: int = 2048, **_kwargs: Any
) -> tuple[TensorType, TensorType]:
    """Generate a PHE key pair: returns (public_key, private_key)."""
    # For keys, dtype is meaningless as they shouldn't be used for computation,
    # so we choose a relatively uncommon dtype to satisfy the type system
    public_key_ty = TensorType(COMPLEX128, (1,))
    private_key_ty = TensorType(COMPLEX128, (1,))
    return public_key_ty, private_key_ty


@_PHE_MOD.typed_op(name="encrypt", pfunc_name="phe.encrypt")
def encrypt(plaintext: MPObject, public_key: MPObject, **_kwargs: Any) -> TensorType:
    """Encrypt plaintext using PHE public key: returns ciphertext with same semantic type as plaintext."""
    plaintext_ty = TensorType.from_obj(plaintext)
    _ = TensorType.from_obj(public_key)
    return plaintext_ty


@_PHE_MOD.typed_op(name="add", pfunc_name="phe.add")
def add(operand1: MPObject, operand2: MPObject, **_kwargs: Any) -> TensorType:
    """Add two PHE operands (semantics depend on backend representation)."""
    op1_ty = TensorType.from_obj(operand1)
    _ = TensorType.from_obj(operand2)
    return op1_ty


@_PHE_MOD.typed_op(name="mul", pfunc_name="phe.mul")
def mul(ciphertext: MPObject, plaintext: MPObject, **_kwargs: Any) -> TensorType:
    """Multiply a PHE ciphertext with a plaintext value (ciphertext dtype preserved)."""
    ct_ty = TensorType.from_obj(ciphertext)
    pt_ty = TensorType.from_obj(plaintext)
    if ct_ty.dtype.is_floating and pt_ty.dtype.is_floating:
        raise ValueError(
            "PHE multiplication does not support float x float operations due to truncation requirements. "
            "Consider using mixed types (float x int) or integer types instead."
        )
    return ct_ty


@_PHE_MOD.typed_op(name="decrypt", pfunc_name="phe.decrypt")
def decrypt(ciphertext: MPObject, private_key: MPObject, **_kwargs: Any) -> TensorType:
    """Decrypt ciphertext using PHE private key: returns plaintext with same semantic type as ciphertext."""
    ciphertext_ty = TensorType.from_obj(ciphertext)
    _ = TensorType.from_obj(private_key)
    return ciphertext_ty
