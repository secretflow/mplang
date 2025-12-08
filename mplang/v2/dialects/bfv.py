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

"""BFV (Brakerski-Fan-Vercauteren) dialect for the EDSL.

BFV is a Fully Homomorphic Encryption (FHE) scheme that supports exact arithmetic
on integers. A key feature of BFV in this dialect is its SIMD (Single Instruction,
Multiple Data) capability, where a single ciphertext encrypts a vector of integers
(packed into "slots").

Design principles:
- **SIMD-first**: The fundamental unit of data is a packed vector (Plaintext/Ciphertext),
  not a scalar.
- **Explicit Management**: Relinearization and rotation are explicit operations to
  give users control over noise and performance.
- **Type Safety**: Distinguishes between Plaintext (encoded polynomial) and
  Ciphertext (encrypted polynomial).

Type System Rationale:
    The BFV dialect models data as `Encrypted[Vector[T]]`, where:
    - `Vector[T]`: Represents the logical layout of data in SIMD slots.
    - `T`: Must be an IntegerType (e.g., i64, u32). BFV does not support floating point.
      If you need to encrypt floats, you must quantize them to integers first, or use CKKS.
    - `Encrypted[...]`: Represents the cryptographic wrapper.

    Why `Vector[T]` and not `Vector[BigInt]`?
    - **Optimization**: Knowing the exact integer width (e.g., i32 vs i64) allows the
      compiler to choose optimal encryption parameters (Plaintext Modulus `t`).
    - **Semantics**: Preserves signed/unsigned semantics and bitwidth constraints.

Architecture:
    Tensor[Integer, (N,)]  (1D Vector)
        ↓ encode(encoder)
    Plaintext (Packed Polynomial) -> Wraps Vector[Integer, N]
        ↓ encrypt(pk)
    Ciphertext (Encrypted Polynomial) -> Wraps Vector[Integer, N]
        ↓ add/mul (SIMD operations)
    Ciphertext
        ↓ decrypt(sk)
    Plaintext
        ↓ decode(encoder)
    Tensor[Integer, (N,)]

Example:
```python
from mplang.v2.dialects import tensor, bfv
import mplang.v2.edsl.typing as elt
import numpy as np

# 1. Setup
# poly_modulus_degree=4096 means 4096 slots
pk, sk = bfv.keygen(poly_modulus_degree=4096)
relin_keys = bfv.make_relin_keys(sk)
encoder = bfv.create_encoder(poly_modulus_degree=4096)

# 2. Data (Vectors)
v1 = tensor.constant(np.array([1, 2, 3, 4], dtype=np.int64))
v2 = tensor.constant(np.array([10, 20, 30, 40], dtype=np.int64))

# 3. Encode & Encrypt (SIMD Packing)
pt1 = bfv.encode(v1, encoder)
ct1 = bfv.encrypt(pt1, pk)

pt2 = bfv.encode(v2, encoder)
ct2 = bfv.encrypt(pt2, pk)

# 4. Computation
# Element-wise multiplication of the underlying vectors
ct_prod = bfv.mul(ct1, ct2)
# Relinearize to reduce ciphertext size after multiplication
ct_prod = bfv.relinearize(ct_prod, relin_keys)

# 5. Decrypt
pt_res = bfv.decrypt(ct_prod, sk)
res = bfv.decode(pt_res, encoder)  # Returns Tensor
```
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, cast

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.edsl import serde

# ==============================================================================
# --- Type Definitions
# ==============================================================================

KeyKind = Literal["Public", "Private", "Relin", "Galois"]


@serde.register_class
class KeyType(elt.BaseType):
    """Type for BFV keys."""

    def __init__(self, kind: KeyKind, poly_modulus_degree: int = 4096):
        self.scheme = "bfv"
        self.kind = kind
        self.poly_modulus_degree = poly_modulus_degree

    def __str__(self) -> str:
        return f"BFV{self.kind}Key[N={self.poly_modulus_degree}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KeyType):
            return False
        return (
            self.kind == other.kind
            and self.poly_modulus_degree == other.poly_modulus_degree
        )

    def __hash__(self) -> int:
        return hash(("BFVKeyType", self.kind, self.poly_modulus_degree))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "bfv.KeyType"

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "poly_modulus_degree": self.poly_modulus_degree,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> KeyType:
        return cls(kind=data["kind"], poly_modulus_degree=data["poly_modulus_degree"])


@serde.register_class
class PlaintextType(elt.BaseType):
    """Represents a BFV plaintext (a polynomial encoding a vector of integers).

    In the EDSL type system, this wraps a VectorType which describes the
    logical data layout (SIMD slots).
    """

    def __init__(self, vector_type: elt.VectorType):
        self.vector_type = vector_type

    @property
    def slots(self) -> int:
        return self.vector_type.size

    def __str__(self) -> str:
        return f"BFVPlaintext[{self.vector_type}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlaintextType):
            return False
        return self.vector_type == other.vector_type

    def __hash__(self) -> int:
        return hash(("BFVPlaintextType", self.vector_type))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "bfv.PlaintextType"

    def to_json(self) -> dict[str, Any]:
        return {"vector_type": serde.to_json(self.vector_type)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PlaintextType:
        vt = serde.from_json(data["vector_type"])
        if not isinstance(vt, elt.VectorType):
            raise TypeError(f"Expected VectorType, got {type(vt)}")
        return cls(vector_type=vt)


@serde.register_class
class CiphertextType(elt.BaseType, elt.EncryptedTrait):
    """Represents a BFV ciphertext (encrypting a Plaintext)."""

    def __init__(self, vector_type: elt.VectorType):
        self._scheme = "bfv"
        self.vector_type = vector_type

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def poly_modulus_degree(self) -> int:
        return self.vector_type.size

    def __str__(self) -> str:
        return f"BFVCiphertext[{self.vector_type}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CiphertextType):
            return False
        return self.vector_type == other.vector_type

    def __hash__(self) -> int:
        return hash(("BFVCiphertextType", self.vector_type))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "bfv.CiphertextType"

    def to_json(self) -> dict[str, Any]:
        return {"vector_type": serde.to_json(self.vector_type)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CiphertextType:
        vt = serde.from_json(data["vector_type"])
        if not isinstance(vt, elt.VectorType):
            raise TypeError(f"Expected VectorType, got {type(vt)}")
        return cls(vector_type=vt)


# Opaque types
@serde.register_class
class EncoderType(elt.BaseType):
    """Type for BFV BatchEncoder."""

    def __init__(self, poly_modulus_degree: int):
        self.poly_modulus_degree = poly_modulus_degree

    def __str__(self) -> str:
        return f"BFVEncoder[N={self.poly_modulus_degree}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EncoderType):
            return False
        return self.poly_modulus_degree == other.poly_modulus_degree

    def __hash__(self) -> int:
        return hash(("BFVEncoder", self.poly_modulus_degree))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "bfv.EncoderType"

    def to_json(self) -> dict[str, Any]:
        return {"poly_modulus_degree": self.poly_modulus_degree}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EncoderType:
        return cls(poly_modulus_degree=data["poly_modulus_degree"])


# ==============================================================================
# --- Key Management
# ==============================================================================

keygen_p = el.Primitive[tuple[el.Object, el.Object]]("bfv.keygen")
make_relin_keys_p = el.Primitive[el.Object]("bfv.make_relin_keys")
make_galois_keys_p = el.Primitive[el.Object]("bfv.make_galois_keys")


@keygen_p.def_abstract_eval
def _keygen_ae(
    *,
    poly_modulus_degree: int = 4096,
    plain_modulus: int = 1032193,
) -> tuple[KeyType, KeyType]:
    """Generate Public and Private keys."""
    return (
        KeyType("Public", poly_modulus_degree),
        KeyType("Private", poly_modulus_degree),
    )


@make_relin_keys_p.def_abstract_eval
def _make_relin_keys_ae(sk: KeyType) -> KeyType:
    """Generate Relinearization keys from Secret Key."""
    if not isinstance(sk, KeyType) or sk.kind != "Private":
        raise TypeError(f"Expected BFV PrivateKey, got {sk}")
    return KeyType("Relin", sk.poly_modulus_degree)


@make_galois_keys_p.def_abstract_eval
def _make_galois_keys_ae(sk: KeyType) -> KeyType:
    """Generate Galois keys (for rotation) from Secret Key."""
    if not isinstance(sk, KeyType) or sk.kind != "Private":
        raise TypeError(f"Expected BFV PrivateKey, got {sk}")
    return KeyType("Galois", sk.poly_modulus_degree)


# ==============================================================================
# --- Encoding / Decoding (SIMD Packing)
# ==============================================================================

create_encoder_p = el.Primitive[el.Object]("bfv.create_encoder")
encode_p = el.Primitive[el.Object]("bfv.encode")
batch_encode_p = el.Primitive[el.Object]("bfv.batch_encode")
decode_p = el.Primitive[el.Object]("bfv.decode")


@create_encoder_p.def_abstract_eval
def _create_encoder_ae(*, poly_modulus_degree: int = 4096) -> EncoderType:
    return EncoderType(poly_modulus_degree)


@encode_p.def_abstract_eval
def _encode_ae(tensor: elt.TensorType, encoder: EncoderType) -> PlaintextType:
    """Pack a 1D Tensor of integers into a BFV Plaintext."""
    if not isinstance(encoder, EncoderType):
        raise TypeError(f"Expected BFVEncoder, got {encoder}")

    if not isinstance(tensor, elt.TensorType):
        raise TypeError(f"Expected Tensor input, got {tensor}")

    # Check 1D
    if tensor.rank != 1:
        raise ValueError(
            f"BFV encode currently only supports 1D Tensors, got rank {tensor.rank}"
        )

    # Check Integer type
    if not isinstance(tensor.element_type, elt.IntegerType):
        raise TypeError(
            f"BFV supports integer arithmetic only. Expected Tensor[Integer], got Tensor[{tensor.element_type}]"
        )

    # In a real implementation, we'd check if tensor size <= poly_modulus_degree
    # For abstract eval, we assume N=4096 as default or infer from context if possible.
    return PlaintextType(elt.Vector(tensor.element_type, encoder.poly_modulus_degree))


def _infer_batch_encode_output_types(
    tensor: elt.TensorType,
    encoder: elt.BaseType,
    key: elt.BaseType,
) -> tuple[PlaintextType, ...]:
    if not isinstance(encoder, EncoderType):
        raise TypeError(f"Expected BFVEncoder, got {encoder}")

    if not isinstance(tensor, elt.TensorType):
        raise TypeError(f"Expected Tensor input, got {tensor}")

    if tensor.rank != 2:
        raise ValueError(
            f"BFV batch_encode input must be 2D Tensor, got rank {tensor.rank}"
        )

    N = tensor.shape[0]
    if N is None:
        raise ValueError(
            "BFV batch_encode requires static first dimension for tensor input"
        )

    # Check Integer type
    if not isinstance(tensor.element_type, elt.IntegerType):
        raise TypeError(
            f"BFV supports integer arithmetic only. Expected Tensor[Integer], got Tensor[{tensor.element_type}]"
        )

    return tuple(
        PlaintextType(elt.Vector(tensor.element_type, encoder.poly_modulus_degree))
        for _ in range(N)
    )


@batch_encode_p.def_trace
def _batch_encode_trace(
    tensor: el.Object,
    encoder: el.Object,
    key: el.Object,
) -> tuple[el.Object, ...]:
    from mplang.v2.edsl.tracer import TraceObject, Tracer

    ctx = el.get_current_context()
    if not isinstance(ctx, Tracer):
        raise RuntimeError("batch_encode must be called within a Tracer context")

    encoder_type = encoder.type
    key_type = key.type

    output_types = _infer_batch_encode_output_types(tensor.type, encoder_type, key_type)
    inputs = [tensor, encoder, key]

    input_values = [cast(TraceObject, obj)._graph_value for obj in inputs]

    # 3. Add Op
    result_values = ctx.graph.add_op(
        batch_encode_p.name,
        input_values,
        output_types,
        attrs={},
    )

    # 4. Wrap results
    return tuple(
        TraceObject(val, ctx)
        for val, typ in zip(result_values, output_types, strict=True)
    )


@decode_p.def_abstract_eval
def _decode_ae(plain: PlaintextType, encoder: EncoderType) -> elt.TensorType:
    """Unpack a BFV Plaintext back into a 1D Tensor."""
    if not isinstance(encoder, EncoderType):
        raise TypeError(f"Expected BFVEncoder, got {encoder}")
    if not isinstance(plain, PlaintextType):
        raise TypeError(f"Expected BFVPlaintext, got {plain}")

    # Returns a 1D tensor of i64 (default assumption for BFV)
    # The shape is technically (slots,), but we might not know slots exactly here
    # if it wasn't tracked perfectly.
    return elt.TensorType(plain.vector_type.element_type, (plain.slots,))


# ==============================================================================
# --- Encryption / Decryption
# ==============================================================================

encrypt_p = el.Primitive[el.Object]("bfv.encrypt")
decrypt_p = el.Primitive[el.Object]("bfv.decrypt")


@encrypt_p.def_abstract_eval
def _encrypt_ae(plain: PlaintextType, pk: KeyType) -> CiphertextType:
    if not isinstance(plain, PlaintextType):
        raise TypeError(f"Expected BFVPlaintext, got {plain}")
    if not isinstance(pk, KeyType) or pk.kind != "Public":
        raise TypeError(f"Expected BFV PublicKey, got {pk}")
    return CiphertextType(plain.vector_type)


@decrypt_p.def_abstract_eval
def _decrypt_ae(ct: CiphertextType, sk: KeyType) -> PlaintextType:
    if not isinstance(ct, CiphertextType):
        raise TypeError(f"Expected BFVCiphertext, got {ct}")
    if not isinstance(sk, KeyType) or sk.kind != "Private":
        raise TypeError(f"Expected BFV PrivateKey, got {sk}")
    return PlaintextType(ct.vector_type)


# ==============================================================================
# --- Arithmetic Operations
# ==============================================================================

add_p = el.Primitive[el.Object]("bfv.add")
sub_p = el.Primitive[el.Object]("bfv.sub")
mul_p = el.Primitive[el.Object]("bfv.mul")
relinearize_p = el.Primitive[el.Object]("bfv.relinearize")


def _check_arithmetic_operands(lhs: Any, rhs: Any) -> None:
    """Helper to validate operands for arithmetic."""
    valid_types = (CiphertextType, PlaintextType)
    if not (isinstance(lhs, valid_types) and isinstance(rhs, valid_types)):
        raise TypeError(
            f"Operands must be BFVCiphertext or BFVPlaintext, got {lhs}, {rhs}"
        )
    # At least one must be ciphertext
    if not (isinstance(lhs, CiphertextType) or isinstance(rhs, CiphertextType)):
        raise TypeError("At least one operand must be a Ciphertext")


@add_p.def_abstract_eval
def _add_ae(lhs: Any, rhs: Any) -> CiphertextType:
    _check_arithmetic_operands(lhs, rhs)
    # Result inherits properties from the ciphertext operand
    ct = lhs if isinstance(lhs, CiphertextType) else rhs
    return CiphertextType(ct.vector_type)


@sub_p.def_abstract_eval
def _sub_ae(lhs: Any, rhs: Any) -> CiphertextType:
    _check_arithmetic_operands(lhs, rhs)
    ct = lhs if isinstance(lhs, CiphertextType) else rhs
    return CiphertextType(ct.vector_type)


@mul_p.def_abstract_eval
def _mul_ae(lhs: Any, rhs: Any) -> CiphertextType:
    _check_arithmetic_operands(lhs, rhs)
    ct = lhs if isinstance(lhs, CiphertextType) else rhs
    # Note: Multiplication increases noise and potentially size (if CT*CT)
    # But the type remains CiphertextType.
    return CiphertextType(ct.vector_type)


@relinearize_p.def_abstract_eval
def _relinearize_ae(ct: CiphertextType, rk: KeyType) -> CiphertextType:
    if not isinstance(ct, CiphertextType):
        raise TypeError(f"Expected BFVCiphertext, got {ct}")
    if not isinstance(rk, KeyType) or rk.kind != "Relin":
        raise TypeError(f"Expected BFV RelinKeys, got {rk}")
    return CiphertextType(ct.vector_type)


# ==============================================================================
# --- Rotation
# ==============================================================================

rotate_p = el.Primitive[el.Object]("bfv.rotate")


@rotate_p.def_abstract_eval
def _rotate_ae(ct: CiphertextType, gk: KeyType, *, steps: int) -> CiphertextType:
    if not isinstance(ct, CiphertextType):
        raise TypeError(f"Expected BFVCiphertext, got {ct}")
    if not isinstance(gk, KeyType) or gk.kind != "Galois":
        raise TypeError(f"Expected BFV GaloisKeys, got {gk}")
    return CiphertextType(ct.vector_type)


rotate_columns_p = el.Primitive[el.Object]("bfv.rotate_columns")


@rotate_columns_p.def_abstract_eval
def _rotate_columns_ae(ct: CiphertextType, gk: KeyType) -> CiphertextType:
    """Swap the two rows in SIMD batching (row 0 <-> row 1)."""
    if not isinstance(ct, CiphertextType):
        raise TypeError(f"Expected BFVCiphertext, got {ct}")
    if not isinstance(gk, KeyType) or gk.kind != "Galois":
        raise TypeError(f"Expected BFV GaloisKeys, got {gk}")
    return CiphertextType(ct.vector_type)


# ==============================================================================
# --- User API
# ==============================================================================


def keygen(
    poly_modulus_degree: int = 4096,
    plain_modulus: int = 1032193,
) -> tuple[el.Object, el.Object]:
    """Generate BFV Public and Secret keys.

    Args:
        poly_modulus_degree: Degree of polynomial modulus (N). Determines slot count.
                             Must be power of 2 (e.g., 4096, 8192).
        plain_modulus: Explicit plaintext modulus (integer). Default is 1032193.

    Returns:
        (PublicKey, SecretKey)
    """
    return keygen_p.bind(
        poly_modulus_degree=poly_modulus_degree,
        plain_modulus=plain_modulus,
    )


def make_relin_keys(secret_key: el.Object) -> el.Object:
    """Generate Relinearization Keys from Secret Key."""
    return make_relin_keys_p.bind(secret_key)


def make_galois_keys(secret_key: el.Object) -> el.Object:
    """Generate Galois Keys (for rotation) from Secret Key."""
    return make_galois_keys_p.bind(secret_key)


def create_encoder(poly_modulus_degree: int = 4096) -> el.Object:
    """Create a BatchEncoder for SIMD packing."""
    return create_encoder_p.bind(poly_modulus_degree=poly_modulus_degree)


def encode(tensor: el.Object, encoder: el.Object) -> el.Object:
    """Pack a 1D Tensor of integers into a BFV Plaintext."""
    return encode_p.bind(tensor, encoder)


def batch_encode(
    tensor: el.Object,
    encoder: el.Object,
    key: el.Object,
) -> el.Object:
    """Pack a 2D Tensor of integers into BFV Plaintexts (Batched)."""
    return batch_encode_p.bind(tensor, encoder, key)


def decode(plain: el.Object, encoder: el.Object) -> el.Object:
    """Unpack a BFV Plaintext back into a 1D Tensor."""
    return decode_p.bind(plain, encoder)


def encrypt(plain: el.Object, public_key: el.Object) -> el.Object:
    """Encrypt a Plaintext into a Ciphertext."""
    return encrypt_p.bind(plain, public_key)


def decrypt(ciphertext: el.Object, secret_key: el.Object) -> el.Object:
    """Decrypt a Ciphertext into a Plaintext."""
    return decrypt_p.bind(ciphertext, secret_key)


def add(lhs: el.Object, rhs: el.Object) -> el.Object:
    """Homomorphic Addition (SIMD)."""
    return add_p.bind(lhs, rhs)


def sub(lhs: el.Object, rhs: el.Object) -> el.Object:
    """Homomorphic Subtraction (SIMD)."""
    return sub_p.bind(lhs, rhs)


def mul(lhs: el.Object, rhs: el.Object) -> el.Object:
    """Homomorphic Multiplication (SIMD).

    Note: Multiplying two ciphertexts increases the ciphertext size (e.g., size 2 -> 3).
    Use `relinearize` afterwards to reduce it back to size 2 for further multiplications.
    """
    return mul_p.bind(lhs, rhs)


def relinearize(ciphertext: el.Object, relin_keys: el.Object) -> el.Object:
    """Relinearize ciphertext (reduce size after multiplication)."""
    return relinearize_p.bind(ciphertext, relin_keys)


def rotate(ciphertext: el.Object, steps: int, galois_keys: el.Object) -> el.Object:
    """Cyclic rotation of the encrypted vector slots within each row.

    Args:
        ciphertext: The ciphertext to rotate.
        steps: Number of steps to rotate. Positive = left, Negative = right.
               Must be in range (-slot_count/2, slot_count/2).
        galois_keys: Keys required for rotation.
    """
    return rotate_p.bind(ciphertext, galois_keys, steps=steps)


def rotate_columns(ciphertext: el.Object, galois_keys: el.Object) -> el.Object:
    """Swap the two rows in SIMD batching (row 0 <-> row 1).

    In BFV batching with slot_count = N, slots are arranged as:
    - Row 0: slots 0 to N/2 - 1
    - Row 1: slots N/2 to N - 1

    rotate_columns swaps these two rows.

    Args:
        ciphertext: The ciphertext to rotate.
        galois_keys: Keys required for rotation.
    """
    return rotate_columns_p.bind(ciphertext, galois_keys)


__all__ = [
    "CiphertextType",
    "EncoderType",
    "KeyType",
    "PlaintextType",
    "add",
    "create_encoder",
    "decode",
    "decrypt",
    "encode",
    "encrypt",
    "keygen",
    "make_galois_keys",
    "make_relin_keys",
    "mul",
    "relinearize",
    "rotate",
    "rotate_columns",
    "sub",
]
