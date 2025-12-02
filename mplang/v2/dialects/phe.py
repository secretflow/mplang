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

"""PHE (Partially Homomorphic Encryption) dialect for the EDSL.

Design principles:
- Separate encoding from encryption for semantic clarity
- Element-level primitives operate on encoded integers
- Reuse `tensor.elementwise` to lift primitives across tensors
- Provide ergonomic wrappers for common workflows

Architecture:
    Source Type (f64, i32, etc.)
        ↓ encode(encoder)
    Encoded Integer (i64)
        ↓ encrypt(pk)
    Ciphertext (CiphertextType)
        ↓ homomorphic operations
    Ciphertext (CiphertextType)
        ↓ decrypt(sk)
    Encoded Integer (i64)
        ↓ decode(encoder)
    Source Type (f64, i32, etc.)

Example:
```python
from mplang.v2.dialects import tensor, phe
import mplang.v2.edsl.typing as elt
import numpy as np

# 1. Generate keys (cryptographic only)
pk, sk = phe.keygen()

# 2. Create encoder (encoding parameters)
encoder = phe.create_encoder(dtype=elt.f64, fxp_bits=16)

# 3. Encode data
x = tensor.constant(np.array([1.0, 2.0, 3.0]))
y = tensor.constant(np.array([4.0, 5.0, 6.0]))
x_enc = phe.encode(x, encoder)  # f64 → i64
y_enc = phe.encode(y, encoder)  # f64 → i64

# 4. Encrypt
ct_x = phe.encrypt(x_enc, pk)  # i64 → CiphertextType
ct_y = phe.encrypt(y_enc, pk)  # i64 → CiphertextType

# 5. Homomorphic operations
ct_sum = phe.add(ct_x, ct_y)  # CiphertextType + CiphertextType

# 6. Decrypt and decode
sum_enc = phe.decrypt(ct_sum, sk)  # CiphertextType → i64
result = phe.decode(sum_enc, encoder)  # i64 → f64
```

For convenience, auto wrappers combine encode+encrypt and decrypt+decode:
```python
ct = phe.encrypt_auto(x, encoder, pk)
result = phe.decrypt_auto(ct, encoder, sk)
```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects import tensor

# ==============================================================================
# --- Type Definitions
# ==============================================================================


class KeyType(elt.BaseType):
    """Type for PHE keys carrying scheme information."""

    def __init__(self, scheme: str, is_public: bool):
        self.scheme = scheme
        self.is_public = is_public

    def __str__(self) -> str:
        kind = "P" if self.is_public else "S"
        return f"{kind}Key[{self.scheme}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KeyType):
            return False
        return self.scheme == other.scheme and self.is_public == other.is_public

    def __hash__(self) -> int:
        return hash((self.scheme, self.is_public))


class PlaintextType(elt.ScalarType):
    """Represents an encoded integer ready for PHE encryption.

    This type wraps the underlying integer representation (typically i64 or i128)
    to distinguish it from regular integers. This ensures type safety by preventing
    accidental encryption of raw integers or arithmetic between encoded and raw values.
    """

    def __init__(self, bitwidth: int = 64):
        self.bitwidth = bitwidth

    def __str__(self) -> str:
        return f"PT[i{self.bitwidth}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlaintextType):
            return False
        return self.bitwidth == other.bitwidth

    def __hash__(self) -> int:
        return hash(("PlaintextType", self.bitwidth))


class CiphertextType(elt.ScalarType, elt.EncryptedTrait):
    """Represents a single scalar value encrypted with a PHE scheme.

    Inherits from ScalarType, so it can be used as a tensor element type.
    """

    def __init__(self, scheme: str):
        self._scheme = scheme

    @property
    def scheme(self) -> str:
        return self._scheme

    def __str__(self) -> str:
        return f"CT[{self._scheme}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CiphertextType):
            return False
        return self._scheme == other._scheme

    def __hash__(self) -> int:
        return hash(("CiphertextType", self._scheme))


# Opaque types for PHE (singleton instances)
EncoderType: elt.CustomType = elt.CustomType("Encoder")

# ==============================================================================
# --- Key Management Operations
# ==============================================================================

keygen_p = el.Primitive[tuple[el.Object, el.Object]]("phe.keygen")


@keygen_p.def_abstract_eval
def _keygen_ae(
    *,
    scheme: str = "paillier",
    key_size: int = 2048,
) -> tuple[KeyType, KeyType]:
    """Generate PHE key pair (cryptographic parameters only).

    Args:
        scheme: PHE scheme name (e.g., "paillier", "elgamal")
        key_size: Key size in bits (default: 2048)

    Returns:
        Tuple of (PublicKey, PrivateKey) with scheme info
    """
    return (KeyType(scheme, True), KeyType(scheme, False))


# ==============================================================================
# --- Encoder Operations
# ==============================================================================

create_encoder_p = el.Primitive[el.Object]("phe.create_encoder")
encode_p = el.Primitive[el.Object]("phe.encode")
decode_p = el.Primitive[el.Object]("phe.decode")


@create_encoder_p.def_abstract_eval
def _create_encoder_ae(
    *,
    dtype: elt.ScalarType,
    fxp_bits: int = 16,
    max_value: int | None = None,
) -> elt.CustomType:
    """Create PHE encoder for type conversion and fixed-point encoding.

    Args:
        dtype: Source data type (f32, f64, i32, i64, etc.)
        fxp_bits: Fixed-point fractional bits for float types (default: 16)
        max_value: Optional maximum value for range checking

    Returns:
        EncoderType configured for the specified dtype
    """
    if not isinstance(dtype, elt.ScalarType):
        raise TypeError(f"dtype must be ScalarType, got {type(dtype).__name__}")
    return EncoderType


@encode_p.def_abstract_eval
def _encode_ae(value: elt.ScalarType, encoder: elt.CustomType) -> PlaintextType:
    """Encode scalar value to fixed-point integer representation.

    Args:
        value: Source value (f32, f64, i32, etc.)
        encoder: PHE encoder with encoding parameters

    Returns:
        Encoded integer (PlaintextType)

    Raises:
        TypeError: If encoder is not EncoderType
    """
    if encoder != EncoderType:
        raise TypeError(f"Expected Encoder, got {encoder}")
    if not isinstance(value, elt.ScalarType):
        raise TypeError(f"Can only encode ScalarType, got {value}")
    # Return sufficient integer type for encoded values
    return PlaintextType(bitwidth=64)


@decode_p.def_abstract_eval
def _decode_ae(encoded: PlaintextType, encoder: elt.CustomType) -> elt.ScalarType:
    """Decode fixed-point integer back to original scalar type.

    Args:
        encoded: Encoded integer value
        encoder: PHE encoder (contains target dtype)

    Returns:
        Decoded value in original type (inferred from encoder's dtype)

    Raises:
        TypeError: If encoder is not EncoderType or encoded is not PlaintextType
    """
    if encoder != EncoderType:
        raise TypeError(f"Expected Encoder, got {encoder}")
    if not isinstance(encoded, PlaintextType):
        raise TypeError(f"Can only decode PlaintextType, got {encoded}")
    # In real implementation, would extract dtype from encoder attrs
    # For now, return a default (this will be improved with attr introspection)
    return elt.f64


# ==============================================================================
# --- Encryption/Decryption Operations (Integer only)
# ==============================================================================

encrypt_p = el.Primitive[el.Object]("phe.encrypt")
decrypt_p = el.Primitive[el.Object]("phe.decrypt")


@encrypt_p.def_abstract_eval
def _encrypt_ae(encoded: PlaintextType, pk: KeyType) -> CiphertextType:
    """Encrypt encoded integer using PHE public key.

    Args:
        encoded: Encoded integer (from phe.encode)
        pk: PHE public key

    Returns:
        CiphertextType - encrypted integer

    Raises:
        TypeError: If input is not PlaintextType or pk is not PublicKey
    """
    if not isinstance(pk, KeyType) or not pk.is_public:
        raise TypeError(f"Expected PublicKey, got {pk}")
    if not isinstance(encoded, PlaintextType):
        raise TypeError(f"Can only encrypt PlaintextType, got {encoded}")
    return CiphertextType(pk.scheme)


@decrypt_p.def_abstract_eval
def _decrypt_ae(ct: CiphertextType, sk: KeyType) -> PlaintextType:
    """Decrypt ciphertext to encoded integer using PHE private key.

    Args:
        ct: Encrypted integer
        sk: PHE private key

    Returns:
        Decrypted encoded integer

    Raises:
        TypeError: If ct is not CiphertextType or sk is not PrivateKey
    """
    if not isinstance(sk, KeyType) or sk.is_public:
        raise TypeError(f"Expected PrivateKey, got {sk}")
    if not isinstance(ct, CiphertextType):
        raise TypeError(f"Expected CiphertextType, got {ct}")
    # We assume it decrypts to i64 (standard encoded integer)
    return PlaintextType(bitwidth=64)


# ==============================================================================
# --- Element-level Homomorphic Operations
# ==============================================================================

add_cc_p = el.Primitive[el.Object]("phe.add_cc")
add_cp_p = el.Primitive[el.Object]("phe.add_cp")
mul_cp_p = el.Primitive[el.Object]("phe.mul_cp")


@add_cc_p.def_abstract_eval
def _add_cc_ae(operand1: CiphertextType, operand2: CiphertextType) -> CiphertextType:
    """Ciphertext + ciphertext → ciphertext."""
    if not isinstance(operand1, CiphertextType) or not isinstance(
        operand2, CiphertextType
    ):
        raise TypeError(f"Expected CiphertextType operands, got {operand1}, {operand2}")
    if operand1 != operand2:
        raise TypeError(f"Scheme mismatch: {operand1} vs {operand2}")
    return operand1


@add_cp_p.def_abstract_eval
def _add_cp_ae(ciphertext: CiphertextType, plaintext: PlaintextType) -> CiphertextType:
    """Ciphertext + plaintext → ciphertext."""
    if not isinstance(ciphertext, CiphertextType):
        raise TypeError(f"Expected CiphertextType ciphertext, got {ciphertext}")
    if not isinstance(plaintext, PlaintextType):
        raise TypeError(
            f"Plaintext operand must be PlaintextType (encoded), got {plaintext}"
        )
    return ciphertext


@mul_cp_p.def_abstract_eval
def _mul_cp_ae(ciphertext: CiphertextType, plaintext: PlaintextType) -> CiphertextType:
    """Element-level homomorphic scalar multiplication.

    Args:
        ciphertext: Encrypted value
        plaintext: Encoded integer scalar

    Returns:
        Encrypted product
    """
    if not isinstance(ciphertext, CiphertextType):
        raise TypeError(f"Expected CiphertextType ciphertext, got {ciphertext}")
    if not isinstance(plaintext, PlaintextType):
        raise TypeError(
            f"Plaintext operand must be PlaintextType (encoded), got {plaintext}"
        )
    return ciphertext


# ==============================================================================
# --- User-facing API
# ==============================================================================


def keygen(
    scheme: str = "paillier",
    key_size: int = 2048,
) -> tuple[el.Object, el.Object]:
    """Generate PHE key pair (cryptographic parameters only).

    Encoding parameters (fxp_bits, max_value) are now separate via create_encoder().

    Args:
        scheme: PHE scheme name (default: "paillier")
                Supported: "paillier", "elgamal", "okamoto-uchiyama"
        key_size: Key size in bits (default: 2048)
                  Larger keys = more security but slower computation

    Returns:
        Tuple of (public_key, private_key)

    Example:
        >>> # Basic usage
        >>> pk, sk = phe.keygen()
        >>>
        >>> # Higher security
        >>> pk, sk = phe.keygen(key_size=4096)
    """
    return keygen_p.bind(scheme=scheme, key_size=key_size)


def create_encoder(
    dtype: elt.ScalarType,
    fxp_bits: int = 16,
    max_value: int | None = None,
) -> el.Object:
    """Create PHE encoder for value encoding/decoding.

    Encoders are independent of keys and handle type conversion and
    fixed-point representation for homomorphic operations.

    Args:
        dtype: Source data type (e.g., elt.f64, elt.i32)
               Determines encoding/decoding behavior
        fxp_bits: Fixed-point fractional bits for float types (default: 16)
                  Higher = more precision but smaller value range
                  Example: fxp_bits=16 means precision ≈ 1/65536
        max_value: Optional maximum absolute value for overflow checking
                   Example: max_value=2**32 ensures |encoded_value| < 2**32

    Returns:
        PHE encoder configured for the specified dtype

    Example:
        >>> import mplang.v2.edsl.typing as elt
        >>>
        >>> # Float encoder with 16-bit fractional precision
        >>> encoder_f64 = phe.create_encoder(dtype=elt.f64, fxp_bits=16)
        >>>
        >>> # Higher precision for sensitive computations
        >>> encoder_hp = phe.create_encoder(dtype=elt.f64, fxp_bits=32)
        >>>
        >>> # Integer encoder (no fixed-point needed)
        >>> encoder_i32 = phe.create_encoder(dtype=elt.i32)
    """
    attrs: dict[str, Any] = {
        "dtype": dtype,
        "fxp_bits": fxp_bits,
    }
    if max_value is not None:
        attrs["max_value"] = max_value
    return create_encoder_p.bind(**attrs)


def _has_tensor_args(*objs: el.Object) -> bool:
    """Check whether any argument carries a TensorType."""
    return any(isinstance(obj.type, elt.TensorType) for obj in objs)


class OperandInfo(NamedTuple):
    """Classification of operand for PHE operation dispatch."""

    is_tensor: bool
    is_encrypted: bool
    scalar_type: elt.BaseType | None


def _inspect_operand(obj: el.Object) -> OperandInfo:
    """Classify operand layout/security for dispatch."""
    obj_type = obj.type
    if isinstance(obj_type, elt.TensorType):
        elem = obj_type.element_type
        if isinstance(elem, CiphertextType):
            return OperandInfo(True, True, None)
        if isinstance(elem, elt.ScalarType):
            return OperandInfo(True, False, elem)
        raise TypeError(
            f"PHE operations support Tensor[ScalarType] or Tensor[CiphertextType], got Tensor[{elem}]"
        )
    if isinstance(obj_type, CiphertextType):
        return OperandInfo(False, True, None)
    if isinstance(obj_type, elt.ScalarType):
        return OperandInfo(False, False, obj_type)
    raise TypeError(f"PHE operations expect Scalar or Tensor operands, got {obj_type}")


BinaryFn = Callable[[el.Object, el.Object], el.Object]


def _apply_binary(fn: BinaryFn, lhs: el.Object, rhs: el.Object) -> el.Object:
    """Apply scalar primitive, lifting to tensor.elementwise when needed."""
    if _has_tensor_args(lhs, rhs):
        return tensor.elementwise(fn, lhs, rhs)
    return fn(lhs, rhs)


def _add_cp(ciphertext: el.Object, plaintext: el.Object) -> el.Object:
    """Ciphertext ⊕ plaintext helper (order enforced)."""
    return _apply_binary(add_cp_p.bind, ciphertext, plaintext)


def _mul_cp(ciphertext: el.Object, plaintext: el.Object) -> el.Object:
    """Ciphertext ⊗ plaintext helper (order enforced)."""
    return _apply_binary(mul_cp_p.bind, ciphertext, plaintext)


def encode(value: el.Object, encoder: el.Object) -> el.Object:
    """Encode scalar value to fixed-point integer representation.

    Args:
        value: Source value (scalar or tensor)
        encoder: PHE encoder (from create_encoder)

    Returns:
        Encoded integer with same structure as input

    Example:
        >>> x = tensor.constant(3.14)  # f64
        >>> encoder = phe.create_encoder(dtype=elt.f64, fxp_bits=16)
        >>> x_enc = phe.encode(x, encoder)  # i64 (encoded as 205887)
    """
    if _has_tensor_args(value):
        return tensor.elementwise(encode_p.bind, value, encoder)
    return encode_p.bind(value, encoder)


def decode(encoded: el.Object, encoder: el.Object) -> el.Object:
    """Decode fixed-point integer back to original scalar type.

    Args:
        encoded: Encoded integer (from encode or decrypt)
        encoder: PHE encoder (same as used for encoding)

    Returns:
        Decoded value in original type

    Example:
        >>> encoded = phe.encode(x, encoder)
        >>> result = phe.decode(encoded, encoder)  # Back to f64
    """
    if _has_tensor_args(encoded):
        return tensor.elementwise(decode_p.bind, encoded, encoder)
    return decode_p.bind(encoded, encoder)


def encrypt(encoded: el.Object, public_key: el.Object) -> el.Object:
    """Encrypt encoded integer using PHE public key.

    Note: Input must be encoded first via phe.encode().

    Args:
        encoded: Encoded integer (from phe.encode)
        public_key: PHE public key

    Returns:
        Encrypted integer

    Example:
        >>> x_enc = phe.encode(x, encoder)
        >>> ct = phe.encrypt(x_enc, pk)  # i64 → PHECiphertext
    """
    if _has_tensor_args(encoded):
        return tensor.elementwise(encrypt_p.bind, encoded, public_key)
    return encrypt_p.bind(encoded, public_key)


def decrypt(ciphertext: el.Object, private_key: el.Object) -> el.Object:
    """Decrypt ciphertext to encoded integer using PHE private key.

    Note: Output is still encoded; use phe.decode() to get original type.

    Args:
        ciphertext: Encrypted value
        private_key: PHE private key

    Returns:
        Decrypted encoded integer

    Example:
        >>> ct_sum = phe.add(ct1, ct2)
        >>> sum_enc = phe.decrypt(ct_sum, sk)  # PHECiphertext → i64
        >>> result = phe.decode(sum_enc, encoder)  # i64 → f64
    """
    if _has_tensor_args(ciphertext):
        return tensor.elementwise(decrypt_p.bind, ciphertext, private_key)
    return decrypt_p.bind(ciphertext, private_key)


def encrypt_auto(
    value: el.Object, encoder: el.Object, public_key: el.Object
) -> el.Object:
    """Convenience: encode + encrypt in one step.

    Args:
        value: Source value (any scalar type)
        encoder: PHE encoder
        public_key: PHE public key

    Returns:
        Encrypted value

    Example:
        >>> ct = phe.encrypt_auto(x, encoder, pk)
        >>> # Equivalent to:
        >>> # ct = phe.encrypt(phe.encode(x, encoder), pk)
    """
    encoded = encode(value, encoder)
    return encrypt(encoded, public_key)


def decrypt_auto(
    ciphertext: el.Object, encoder: el.Object, private_key: el.Object
) -> el.Object:
    """Convenience: decrypt + decode in one step.

    Args:
        ciphertext: Encrypted value
        encoder: PHE encoder (same as used for encoding)
        private_key: PHE private key

    Returns:
        Decrypted value in original type

    Example:
        >>> result = phe.decrypt_auto(ct, encoder, sk)
        >>> # Equivalent to:
        >>> # result = phe.decode(phe.decrypt(ct, sk), encoder)
    """
    decoded = decrypt(ciphertext, private_key)
    return decode(decoded, encoder)


def add(lhs: el.Object, rhs: el.Object) -> el.Object:
    """Homomorphic addition.

    Supports:
        Ciphertext + Ciphertext → Ciphertext  (ciphertext + ciphertext)
        Ciphertext + T → Ciphertext      (ciphertext + plaintext)
        T + Ciphertext → Ciphertext      (plaintext + ciphertext)

    Args:
        lhs: Left operand (encrypted or plaintext)
        rhs: Right operand (encrypted or plaintext)

    Returns:
        Encrypted sum

    Raises:
        TypeError: If no operand is encrypted or types mismatch
    """
    lhs_info = _inspect_operand(lhs)
    rhs_info = _inspect_operand(rhs)

    if not (lhs_info.is_encrypted or rhs_info.is_encrypted):
        raise TypeError("phe.add requires at least one ciphertext operand")

    # CT + CT
    if lhs_info.is_encrypted and rhs_info.is_encrypted:
        return _apply_binary(add_cc_p.bind, lhs, rhs)

    # CT + PT or PT + CT
    if lhs_info.is_encrypted:
        return _add_cp(lhs, rhs)
    return _add_cp(rhs, lhs)


def mul_plain(lhs: el.Object, rhs: el.Object) -> el.Object:
    """Homomorphic multiplication: ciphertext × plaintext (encoded integer).

    Supports:
        Ciphertext × i64 → Ciphertext  (ciphertext × encoded plaintext)
        i64 × Ciphertext → Ciphertext  (encoded plaintext × ciphertext)

    Args:
        lhs: Left operand (one must be encrypted, other must be encoded integer)
        rhs: Right operand

    Returns:
        Encrypted product

    Raises:
        TypeError: If both operands are encrypted or both are plaintext

    Note:
        - Ciphertext × ciphertext is not supported (would require FHE)
        - Plaintext must be encoded integer (use phe.encode first)
        - For float multiplication, may need truncation to maintain precision

    Example:
        >>> ct = phe.encrypt(phe.encode(x, encoder), pk)
        >>> y_enc = phe.encode(y, encoder)
        >>> ct_prod = phe.mul_plain(ct, y_enc)
    """
    lhs_info = _inspect_operand(lhs)
    rhs_info = _inspect_operand(rhs)

    # CT * PT
    if lhs_info.is_encrypted and not rhs_info.is_encrypted:
        return _mul_cp(lhs, rhs)
    # PT * CT
    if rhs_info.is_encrypted and not lhs_info.is_encrypted:
        return _mul_cp(rhs, lhs)
    # CT * CT (not supported)
    if lhs_info.is_encrypted and rhs_info.is_encrypted:
        raise TypeError(
            "phe.mul_plain supports ciphertext * plaintext only, not CT * CT. "
            "Ciphertext * ciphertext requires FHE."
        )
    # PT * PT (invalid)
    raise TypeError("phe.mul_plain requires at least one ciphertext operand")


__all__ = [
    "CiphertextType",
    # Types
    "EncoderType",
    "KeyType",
    "PlaintextType",
    # User API
    "add",
    # Primitives
    "add_cc_p",
    "add_cp_p",
    "create_encoder",
    "create_encoder_p",
    "decode",
    "decode_p",
    "decrypt",
    "decrypt_auto",
    "decrypt_p",
    "encode",
    "encode_p",
    "encrypt",
    "encrypt_auto",
    "encrypt_p",
    "keygen",
    "keygen_p",
    "mul_cp_p",
    "mul_plain",
]
