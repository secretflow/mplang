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

from mplang.core.dtype import UINT8
from mplang.core.tensor import TensorType
from mplang.frontend.base import stateless_mod

_PHE_MOD = stateless_mod("phe")


@_PHE_MOD.simple_op()
def keygen(
    *, scheme: str = "paillier", key_size: int = 2048
) -> tuple[TensorType, TensorType]:
    """Generate a PHE key pair: returns (public_key, private_key).

    Keys are represented with a sentinel TensorType UINT8[(-1, 0)] to indicate
    non-structural, backend-only handles. Runtime validation will treat this
    shape as an opaque placeholder and skip dtype/shape checks.
    """
    key_spec = TensorType(UINT8, (-1, 0))
    return key_spec, key_spec


@_PHE_MOD.simple_op()
def encrypt(plaintext: TensorType, public_key: TensorType) -> TensorType:
    """Encrypt plaintext using PHE public key: returns ciphertext with same semantic type as plaintext."""
    _ = public_key
    return plaintext


@_PHE_MOD.simple_op()
def add(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Add two PHE operands (semantics depend on backend representation)."""
    _ = operand2
    return operand1


@_PHE_MOD.simple_op()
def mul(ciphertext: TensorType, plaintext: TensorType) -> TensorType:
    """Multiply a PHE ciphertext with a plaintext value (ciphertext dtype preserved)."""
    if ciphertext.dtype.is_floating and plaintext.dtype.is_floating:
        raise ValueError(
            "PHE multiplication does not support float x float operations due to truncation requirements. "
            "Consider using mixed types (float x int) or integer types instead."
        )
    return ciphertext


@_PHE_MOD.simple_op()
def decrypt(ciphertext: TensorType, private_key: TensorType) -> TensorType:
    """Decrypt ciphertext using PHE private key: returns plaintext with same semantic type as ciphertext."""
    _ = private_key
    return ciphertext
