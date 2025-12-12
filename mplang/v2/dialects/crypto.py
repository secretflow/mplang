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

"""Crypto dialect for the EDSL.

Provides cryptographic primitives including ECC, Hashing, and Symmetric Encryption.
"""

from __future__ import annotations

from typing import Any, ClassVar

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.edsl import serde

# ==============================================================================
# --- Type Definitions
# ==============================================================================


@serde.register_class
class PointType(elt.BaseType):
    """Type for an ECC Point."""

    def __init__(self, curve: str = "secp256k1"):
        self.curve = curve

    def __str__(self) -> str:
        return f"Point[{self.curve}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointType):
            return False
        return self.curve == other.curve

    def __hash__(self) -> int:
        return hash(("PointType", self.curve))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "crypto.PointType"

    def to_json(self) -> dict[str, Any]:
        return {"curve": self.curve}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PointType:
        return cls(curve=data["curve"])


@serde.register_class
class ScalarType(elt.BaseType):
    """Type for an ECC Scalar (integer modulo curve order)."""

    def __init__(self, curve: str = "secp256k1"):
        self.curve = curve

    def __str__(self) -> str:
        return f"Scalar[{self.curve}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScalarType):
            return False
        return self.curve == other.curve

    def __hash__(self) -> int:
        return hash(("ScalarType", self.curve))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "crypto.ScalarType"

    def to_json(self) -> dict[str, Any]:
        return {"curve": self.curve}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ScalarType:
        return cls(curve=data["curve"])


@serde.register_class
class PrivateKeyType(elt.BaseType):
    """Type for a KEM private key."""

    def __init__(self, suite: str = "x25519"):
        self.suite = suite

    def __str__(self) -> str:
        return f"PrivateKey[{self.suite}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrivateKeyType):
            return False
        return self.suite == other.suite

    def __hash__(self) -> int:
        return hash(("PrivateKeyType", self.suite))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "crypto.PrivateKeyType"

    def to_json(self) -> dict[str, Any]:
        return {"suite": self.suite}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PrivateKeyType:
        return cls(suite=data["suite"])


@serde.register_class
class PublicKeyType(elt.BaseType):
    """Type for a KEM public key."""

    def __init__(self, suite: str = "x25519"):
        self.suite = suite

    def __str__(self) -> str:
        return f"PublicKey[{self.suite}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PublicKeyType):
            return False
        return self.suite == other.suite

    def __hash__(self) -> int:
        return hash(("PublicKeyType", self.suite))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "crypto.PublicKeyType"

    def to_json(self) -> dict[str, Any]:
        return {"suite": self.suite}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PublicKeyType:
        return cls(suite=data["suite"])


@serde.register_class
class SymmetricKeyType(elt.BaseType):
    """Type for a symmetric encryption key (e.g., from KEM derive)."""

    def __init__(self, suite: str = "x25519"):
        self.suite = suite

    def __str__(self) -> str:
        return f"SymmetricKey[{self.suite}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymmetricKeyType):
            return False
        return self.suite == other.suite

    def __hash__(self) -> int:
        return hash(("SymmetricKeyType", self.suite))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "crypto.SymmetricKeyType"

    def to_json(self) -> dict[str, Any]:
        return {"suite": self.suite}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SymmetricKeyType:
        return cls(suite=data["suite"])


# ==============================================================================
# --- Primitives
# ==============================================================================

# ECC
generator_p = el.Primitive[el.Object]("crypto.ec_generator")
mul_p = el.Primitive[el.Object]("crypto.ec_mul")
add_p = el.Primitive[el.Object]("crypto.ec_add")
sub_p = el.Primitive[el.Object]("crypto.ec_sub")
point_to_bytes_p = el.Primitive[el.Object]("crypto.ec_point_to_bytes")
random_scalar_p = el.Primitive[el.Object]("crypto.ec_random_scalar")
scalar_from_int_p = el.Primitive[el.Object]("crypto.ec_scalar_from_int")

# Symmetric / Hash
hash_p = el.Primitive[el.Object]("crypto.hash")
hash_batch_p = el.Primitive[el.Object]("crypto.hash_batch")
sym_encrypt_p = el.Primitive[el.Object]("crypto.sym_encrypt")
sym_decrypt_p = el.Primitive[el.Object]("crypto.sym_decrypt")
select_p = el.Primitive[el.Object]("crypto.select")

# KEM (Key Encapsulation Mechanism)
kem_keygen_p = el.Primitive[tuple[el.Object, el.Object]]("crypto.kem_keygen")
kem_derive_p = el.Primitive[el.Object]("crypto.kem_derive")

# Randomness
random_bytes_p = el.Primitive[el.Object]("crypto.random_bytes")


# ==============================================================================
# --- Abstract Evaluation (Type Inference)
# ==============================================================================


@generator_p.def_abstract_eval
def _generator_ae(curve: str = "secp256k1") -> PointType:
    return PointType(curve)


@mul_p.def_abstract_eval
def _mul_ae(point: PointType, scalar: ScalarType) -> PointType:
    return PointType(point.curve)


@add_p.def_abstract_eval
def _add_ae(p1: PointType, p2: PointType) -> PointType:
    return PointType(p1.curve)


@sub_p.def_abstract_eval
def _sub_ae(p1: PointType, p2: PointType) -> PointType:
    return PointType(p1.curve)


@point_to_bytes_p.def_abstract_eval
def _pt_to_bytes_ae(point: elt.BaseType) -> elt.TensorType:
    if isinstance(point, elt.TensorType):
        # Vectorized behavior: Tensor[Point, shape] -> Tensor[u8, shape + (65,)]
        return elt.TensorType(elt.u8, (*point.shape, 65))
    return elt.TensorType(elt.u8, (65,))


@random_scalar_p.def_abstract_eval
def _random_scalar_ae(curve: str = "secp256k1") -> ScalarType:
    return ScalarType(curve)


@scalar_from_int_p.def_abstract_eval
def _scalar_from_int_ae(
    val: elt.TensorType | elt.IntegerType, curve: str = "secp256k1"
) -> ScalarType:
    return ScalarType(curve)


@hash_p.def_abstract_eval
def _hash_ae(data: elt.BaseType) -> elt.TensorType:
    # Strictly single output (blob hash)
    return elt.TensorType(elt.u8, (32,))


@hash_batch_p.def_abstract_eval
def _hash_batch_ae(data: elt.BaseType) -> elt.TensorType:
    # Explicit batch hashing: Input (..., D) -> Output (..., 32)
    # Hashes the last dimension D bytes.
    if not isinstance(data, elt.TensorType):
        raise TypeError(f"hash_batch requires TensorType, got {data}")

    # data.shape is tuple[int | None, ...]
    shape = data.shape
    if len(shape) < 2:
        # Fallback/Edge case: (D,) -> (32,)
        # One could argue this should be an error for *batch* primitive,
        # but allowing it provides consistency for (N=1, D).
        return elt.TensorType(elt.u8, (32,))

    # Batch shape is everything except last dim
    batch_shape = shape[:-1]
    return elt.TensorType(elt.u8, (*batch_shape, 32))


@sym_encrypt_p.def_abstract_eval
def _sym_encrypt_ae(key: elt.BaseType, plaintext: elt.BaseType) -> elt.TensorType:
    # Dynamic shape for ciphertext
    return elt.TensorType(elt.u8, (-1,))


@sym_decrypt_p.def_abstract_eval
def _sym_decrypt_ae(
    key: elt.BaseType, ciphertext: elt.BaseType, target_type: elt.BaseType
) -> elt.BaseType:
    return target_type


@select_p.def_abstract_eval
def _select_ae(
    cond: elt.BaseType, true_val: elt.BaseType, false_val: elt.BaseType
) -> elt.BaseType:
    return true_val


@kem_keygen_p.def_abstract_eval
def _kem_keygen_ae(suite: str = "x25519") -> tuple[PrivateKeyType, PublicKeyType]:
    return (PrivateKeyType(suite), PublicKeyType(suite))


@kem_derive_p.def_abstract_eval
def _kem_derive_ae(
    private_key: PrivateKeyType, public_key: PublicKeyType
) -> SymmetricKeyType:
    suite = getattr(private_key, "suite", "x25519")
    return SymmetricKeyType(suite)


@random_bytes_p.def_abstract_eval
def _random_bytes_ae(length: int) -> elt.TensorType:
    return elt.TensorType(elt.u8, (length,))


# ==============================================================================
# --- Helper Functions (Ops)
# ==============================================================================


def ec_generator(curve: str = "secp256k1") -> el.Object:
    """Get the generator point G for the curve."""
    return generator_p.bind(curve=curve)


def ec_mul(point: el.Object, scalar: el.Object) -> el.Object:
    """Scalar multiplication: point * scalar."""
    return mul_p.bind(point, scalar)


def ec_add(p1: el.Object, p2: el.Object) -> el.Object:
    """Point addition: p1 + p2."""
    return add_p.bind(p1, p2)


def ec_sub(p1: el.Object, p2: el.Object) -> el.Object:
    """Point subtraction: p1 - p2."""
    return sub_p.bind(p1, p2)


def ec_point_to_bytes(point: el.Object) -> el.Object:
    """Serialize point to bytes."""
    return point_to_bytes_p.bind(point)


def ec_random_scalar(curve: str = "secp256k1") -> el.Object:
    """Generate a random scalar."""
    return random_scalar_p.bind(curve=curve)


def ec_scalar_from_int(val: el.Object, curve: str = "secp256k1") -> el.Object:
    """Convert an integer tensor to a scalar."""
    return scalar_from_int_p.bind(val, curve=curve)


def hash_bytes(data: el.Object) -> el.Object:
    """Hash bytes (SHA256). Returns 32-byte tensor."""
    return hash_p.bind(data)


def hash_batch(data: el.Object) -> el.Object:
    """Hash each row of a tensor independently.

    Treats the last dimension as the data to hash.
    Input: (N, D) -> Output: (N, 32)
    Input: (B, N, D) -> Output: (B, N, 32)
    """
    return hash_batch_p.bind(data)


def sym_encrypt(key: el.Object, plaintext: el.Object) -> el.Object:
    """Symmetric encrypt (XOR stream or AES-GCM)."""
    return sym_encrypt_p.bind(key, plaintext)


def sym_decrypt(
    key: el.Object, ciphertext: el.Object, target_type: elt.BaseType
) -> el.Object:
    """Symmetric decrypt."""
    return sym_decrypt_p.bind(key, ciphertext, target_type=target_type)


def select(cond: el.Object, true_val: el.Object, false_val: el.Object) -> el.Object:
    """Select between two values based on condition."""
    return select_p.bind(cond, true_val, false_val)


def kem_keygen(suite: str = "x25519") -> tuple[el.Object, el.Object]:
    """Generate a KEM key pair (private_key, public_key).

    Args:
        suite: The KEM suite to use (e.g., "x25519", "kyber768")

    Returns:
        A tuple of (private_key, public_key)
    """
    return kem_keygen_p.bind(suite=suite)


def kem_derive(private_key: el.Object, public_key: el.Object) -> el.Object:
    """Derive a symmetric key from a private key and a public key (ECDH).

    Args:
        private_key: The local private key
        public_key: The remote party's public key

    Returns:
        A symmetric key suitable for use with sym_encrypt/sym_decrypt
    """
    return kem_derive_p.bind(private_key, public_key)


def random_bytes(length: int) -> el.Object:
    """Generate cryptographically secure random bytes at runtime.

    Args:
        length: Number of bytes to generate.

    Returns:
        (length,) uint8 Tensor.
    """
    return random_bytes_p.bind(length=length)


def random_tensor(shape: tuple[int, ...], dtype: elt.ScalarType) -> el.Object:
    """Generate cryptographically secure random tensor at runtime.

    This is a helper function that composes `random_bytes` with `tensor.run_jax`
    to produce a tensor of the specified shape and dtype.

    Args:
        shape: Output tensor shape (e.g., (100,) or (10, 16)).
        dtype: Element type (e.g., elt.u64, elt.i32, elt.f32).

    Returns:
        Tensor[dtype, shape] with CSPRNG values.

    Example:
        >>> # Generate 100 random uint64 values
        >>> x = crypto.random_tensor((100,), elt.u64)
        >>> # Generate 10x16 random int32 matrix
        >>> y = crypto.random_tensor((10, 16), elt.i32)
    """
    import math
    from typing import cast

    from mplang.v2.dialects import dtypes, tensor

    # Get byte size from numpy dtype
    np_dtype = dtypes.to_numpy(dtype)
    element_bytes = np_dtype.itemsize
    total_elements = math.prod(shape)
    total_bytes = total_elements * element_bytes

    raw = random_bytes(total_bytes)

    jax_dtype = dtypes.to_jax(dtype)

    def _view_reshape(b: Any) -> Any:
        return b.view(jax_dtype).reshape(shape)

    return cast(el.Object, tensor.run_jax(_view_reshape, raw))


def random_bits(n: int) -> el.Object:
    """Generate n cryptographically secure random bits at runtime.

    Each bit is stored as a uint8 with value 0 or 1 (unpacked representation).

    Args:
        n: Number of random bits to generate.

    Returns:
        (n,) uint8 Tensor with values 0 or 1.

    Example:
        >>> # Generate 1024 random bits for OT selection
        >>> choice_bits = crypto.random_bits(1024)
    """
    from typing import cast

    import jax.numpy as jnp

    from mplang.v2.dialects import tensor

    # Generate enough bytes to cover n bits
    num_bytes = (n + 7) // 8
    raw = random_bytes(num_bytes)

    def _unpack_and_slice(b: Any, n: int = n) -> Any:
        bits = jnp.unpackbits(b, bitorder="little")
        return bits[:n]

    return cast(el.Object, tensor.run_jax(_unpack_and_slice, raw))


# --- Bytes <-> Point Conversions ---

bytes_to_point_p = el.Primitive[el.Object]("crypto.ec_bytes_to_point")


@bytes_to_point_p.def_abstract_eval
def _bytes_to_point_ae(b: elt.TensorType) -> PointType:
    return PointType("secp256k1")


def ec_bytes_to_point(b: el.Object) -> el.Object:
    """
    Deserialize bytes to an ECC point.

    Args:
        b: A (65,) uint8 Tensor representing an uncompressed point in SEC1 format.
           The first byte must be 0x04, followed by 32 bytes for X and 32 bytes for Y.

    Returns:
        An ECC point object corresponding to the input bytes.

    Raises:
        ValueError: If the input is not a valid 65-byte uncompressed point representation.

    Example:
        >>> # Example: Deserialize a point from bytes
        >>> point_bytes = jnp.array([0x04] + [0x01] * 32 + [0x02] * 32, dtype=jnp.uint8)
        >>> point = crypto.ec_bytes_to_point(point_bytes)
    """
    return bytes_to_point_p.bind(b)
