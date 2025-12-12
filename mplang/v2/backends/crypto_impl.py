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

"""Crypto backend implementation using cryptography and coincurve."""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from typing import Any, ClassVar

import coincurve
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import crypto
from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import Interpreter
from mplang.v2.runtime.value import Value, WrapValue

# =============================================================================
# BytesValue - Wrapper for raw bytes (keys, hashes, ciphertexts)
# =============================================================================


@serde.register_class
class BytesValue(WrapValue[bytes]):
    """Runtime value wrapping raw bytes.

    Used for cryptographic data like:
    - Hash outputs (32 bytes for SHA-256)
    - Symmetric keys (32 bytes for AES-256)
    - Ciphertexts (variable length)
    - EC point serializations
    """

    _serde_kind: ClassVar[str] = "crypto_impl.BytesValue"

    def _convert(self, data: Any) -> bytes:
        if isinstance(data, BytesValue):
            return data.unwrap()
        if isinstance(data, bytes):
            return data
        if isinstance(data, (bytearray, memoryview)):
            return bytes(data)
        # Handle numpy arrays
        if hasattr(data, "tobytes"):
            return bytes(data.tobytes())  # type: ignore[union-attr]
        raise TypeError(f"Cannot convert {type(data).__name__} to bytes")

    def to_json(self) -> dict[str, Any]:
        return {"data": base64.b64encode(self._data).decode("ascii")}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BytesValue:
        return cls(base64.b64decode(data["data"]))


# =============================================================================
# ECC Point Wrapper (secp256k1)
# =============================================================================


@serde.register_class
class ECPointValue(WrapValue[bytes]):
    """Wrapper for coincurve.PublicKey representing an elliptic curve point.

    This wraps the external coincurve library's PublicKey type to provide
    proper serialization support via the Value base class.
    """

    _serde_kind: ClassVar[str] = "crypto_impl.ECPointValue"

    def _convert(self, data: Any) -> bytes:
        if isinstance(data, ECPointValue):
            return data.unwrap()
        if isinstance(data, bytes):
            return data
        if isinstance(data, coincurve.PublicKey):
            return data.format(compressed=True)
        raise TypeError(f"Expected bytes or coincurve.PublicKey, got {type(data)}")

    @property
    def key_bytes(self) -> bytes:
        return self._data

    def to_json(self) -> dict[str, Any]:
        return {"data": base64.b64encode(self._data).decode("ascii")}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ECPointValue:
        return cls(base64.b64decode(data["data"]))

    @property
    def coincurve_key(self) -> coincurve.PublicKey:
        """Get the underlying coincurve.PublicKey object."""
        return coincurve.PublicKey(self._data)

    @classmethod
    def from_coincurve(cls, pk: coincurve.PublicKey) -> ECPointValue:
        """Create ECPointValue from a coincurve.PublicKey."""
        return cls(pk)


# --- ECC Impl (Coincurve) ---

# secp256k1 order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


@crypto.generator_p.def_impl
def generator_impl(interpreter: Interpreter, op: Operation) -> ECPointValue:
    # Compressed G
    g_bytes = bytes.fromhex(
        "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    )
    return ECPointValue(g_bytes)


@crypto.mul_p.def_impl
def mul_impl(
    interpreter: Interpreter,
    op: Operation,
    point: ECPointValue | None,
    scalar: int | TensorValue,
) -> ECPointValue | None:
    # scalar can be:
    # - int: from ec_random_scalar or ec_scalar_from_int
    # - TensorValue: shouldn't happen but handle for robustness
    # - numpy scalar: from inside elementwise (shouldn't reach here as mul is not in elementwise)
    s_val: int
    if isinstance(scalar, TensorValue):
        raw = scalar.unwrap()
        if hasattr(raw, "item"):
            s_val = int(raw.item())
        else:
            s_val = int(raw)
    elif isinstance(scalar, (int, np.integer)):
        s_val = int(scalar)
    else:
        raise TypeError(
            f"mul_impl scalar must be int or TensorValue, got {type(scalar).__name__}"
        )

    s_val = s_val % N

    if s_val == 0:
        return None

    if point is None:
        return None

    # coincurve multiply expects bytes
    s_bytes = s_val.to_bytes(32, "big")
    result = point.coincurve_key.multiply(s_bytes)
    return ECPointValue.from_coincurve(result)


@crypto.add_p.def_impl
def add_impl(
    interpreter: Interpreter,
    op: Operation,
    p1: ECPointValue | None,
    p2: ECPointValue | None,
) -> ECPointValue | None:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    result = p1.coincurve_key.combine([p2.coincurve_key])
    return ECPointValue.from_coincurve(result)


@crypto.sub_p.def_impl
def sub_impl(
    interpreter: Interpreter,
    op: Operation,
    p1: ECPointValue | None,
    p2: ECPointValue | None,
) -> ECPointValue | None:
    # p1 - p2 = p1 + (-p2)
    if p2 is None:
        return p1

    # Negate p2 by multiplying by (N-1)
    neg_scalar = (N - 1).to_bytes(32, "big")
    neg_p2 = p2.coincurve_key.multiply(neg_scalar)

    if p1 is None:
        return ECPointValue.from_coincurve(neg_p2)

    result = p1.coincurve_key.combine([neg_p2])
    return ECPointValue.from_coincurve(result)


@crypto.random_scalar_p.def_impl
def random_scalar_impl(interpreter: Interpreter, op: Operation) -> int:
    return int.from_bytes(os.urandom(32), "big") % N


@crypto.scalar_from_int_p.def_impl
def scalar_from_int_impl(
    interpreter: Interpreter, op: Operation, val: TensorValue | int
) -> int:
    """Convert a tensor/scalar value to an EC scalar (int).

    val can be:
    - TensorValue: wrapping a scalar numpy array
    - int/bool: direct Python integer or boolean
    - numpy scalar (np.integer, np.bool_): from inside elementwise operations
    """
    if isinstance(val, TensorValue):
        raw = val.unwrap()
        if hasattr(raw, "item"):
            return int(raw.item())
        return int(raw)
    elif isinstance(val, (int, bool, np.integer, np.bool_)):
        return int(val)
    else:
        raise TypeError(
            f"scalar_from_int val must be TensorValue or int-like, "
            f"got {type(val).__name__}"
        )


@crypto.point_to_bytes_p.def_impl
def point_to_bytes_impl(
    interpreter: Interpreter, op: Operation, point: ECPointValue | None
) -> TensorValue:
    if point is None:
        # Infinity / Identity -> Zeros (65 bytes to match uncompressed format)
        arr = np.zeros(65, dtype=np.uint8)
        return TensorValue(arr)

    # Returns 65 bytes (uncompressed)
    b = point.coincurve_key.format(compressed=False)
    arr = np.frombuffer(b, dtype=np.uint8).copy()
    return TensorValue(arr)


@crypto.bytes_to_point_p.def_impl
def bytes_to_point_impl(
    interpreter: Interpreter, op: Operation, b: TensorValue | BytesValue
) -> ECPointValue:
    if isinstance(b, TensorValue):
        raw = b.unwrap().tobytes()
    elif isinstance(b, BytesValue):
        raw = b.unwrap()
    else:
        raise TypeError(
            f"bytes_to_point expects TensorValue or BytesValue, got {type(b)}"
        )

    return ECPointValue(raw)


# --- Sym / Hash Impl ---


@crypto.hash_p.def_impl
def hash_impl(interpreter: Interpreter, op: Operation, data: Value) -> Value:
    """Hash input data using SHA-256 (strict single blob)."""
    # data can be BytesValue or TensorValue
    if isinstance(data, BytesValue):
        d = data.unwrap()
    elif isinstance(data, TensorValue):
        # Flatten and hash as single blob
        d = data.unwrap().tobytes()
    else:
        raise TypeError(
            f"hash expects BytesValue or TensorValue, got {type(data).__name__}"
        )

    h = hashlib.sha256(d).digest()
    arr = np.frombuffer(h, dtype=np.uint8)
    return TensorValue(arr)


@crypto.hash_batch_p.def_impl
def hash_batch_impl(interpreter: Interpreter, op: Operation, data: Value) -> Value:
    """Hash data treating last dimension as bytes (explicit batching)."""
    if not isinstance(data, TensorValue):
        raise TypeError(f"hash_batch requires TensorValue, got {type(data)}")

    arr_in = data.unwrap()

    # Handle scalar / 0D / 1D case simply
    if arr_in.ndim <= 1:
        d = arr_in.tobytes()
        h = hashlib.sha256(d).digest()
        return TensorValue(np.frombuffer(h, dtype=np.uint8))

    # Batch case: (B1, B2, ..., D)
    batch_shape = arr_in.shape[:-1]
    D = arr_in.shape[-1]

    flat_in = arr_in.reshape(-1, D)
    num_items = flat_in.shape[0]

    hashes = []
    for i in range(num_items):
        row_bytes = flat_in[i].tobytes()
        hashes.append(hashlib.sha256(row_bytes).digest())

    flat_out = np.frombuffer(b"".join(hashes), dtype=np.uint8).reshape(num_items, 32)
    arr_out = flat_out.reshape(*batch_shape, 32)

    return TensorValue(arr_out)


@crypto.sym_encrypt_p.def_impl
def sym_encrypt_impl(
    interpreter: Interpreter,
    op: Operation,
    key: SymmetricKeyValue | BytesValue,
    plaintext: Any,
) -> BytesValue:
    """Encrypt plaintext using AES-GCM with the given symmetric key.

    The plaintext can be any JSON-serializable value (Value subclasses,
    numpy arrays, scalars, etc.). This supports both high-level API usage
    (with TensorValue) and elementwise operations (with raw scalars).
    """
    # Get raw key bytes - strict type checking
    if isinstance(key, SymmetricKeyValue):
        k = key.key_bytes
    elif isinstance(key, BytesValue):
        k = key.unwrap()
    elif isinstance(key, TensorValue):
        k = key.unwrap().tobytes()
    else:
        raise TypeError(
            f"sym_encrypt key must be SymmetricKeyValue, BytesValue, or TensorValue, "
            f"got {type(key).__name__}"
        )

    # Serialize the plaintext using secure JSON serde
    # serde.dumps handles Value subclasses, numpy arrays, scalars, etc.
    pt_bytes = serde.dumps(plaintext)

    # AES-GCM encryption
    aesgcm = AESGCM(k)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, pt_bytes, None)

    # Result: nonce + ct
    return BytesValue(nonce + ct)


@crypto.sym_decrypt_p.def_impl
def sym_decrypt_impl(
    interpreter: Interpreter,
    op: Operation,
    key: SymmetricKeyValue | BytesValue,
    ciphertext: BytesValue,
    target_type: Any = None,
) -> Any:
    """Decrypt ciphertext using AES-GCM with the given symmetric key.

    Returns the original plaintext value that was encrypted. The type depends
    on what was encrypted - could be a Value subclass (TensorValue, BytesValue),
    a numpy array, or a scalar (int, float, etc.) when used in elementwise ops.
    """
    # Get raw key bytes - strict type checking
    if isinstance(key, SymmetricKeyValue):
        k = key.key_bytes
    elif isinstance(key, BytesValue):
        k = key.unwrap()
    elif isinstance(key, TensorValue):
        k = key.unwrap().tobytes()
    else:
        raise TypeError(
            f"sym_decrypt key must be SymmetricKeyValue, BytesValue, or TensorValue, "
            f"got {type(key).__name__}"
        )

    # Get ciphertext bytes - strict type checking
    if not isinstance(ciphertext, BytesValue):
        raise TypeError(
            f"sym_decrypt ciphertext must be BytesValue, "
            f"got {type(ciphertext).__name__}"
        )
    ct_full = ciphertext.unwrap()

    # Extract nonce and decrypt
    nonce = ct_full[:12]
    ct = ct_full[12:]

    aesgcm = AESGCM(k)
    pt_bytes = aesgcm.decrypt(nonce, ct, None)

    # Deserialize back using secure JSON serde
    # Returns the original type that was encrypted
    return serde.loads(pt_bytes)


@crypto.select_p.def_impl
def select_impl(
    interpreter: Interpreter,
    op: Operation,
    cond: TensorValue | int,
    true_val: Value,
    false_val: Value,
) -> Value:
    # Handle both TensorValue and raw scalar (from elementwise)
    c: int
    if isinstance(cond, TensorValue):
        raw = cond.unwrap()
        if hasattr(raw, "item"):
            c = int(raw.item())
        else:
            c = int(raw)
    else:
        c = int(cond)
    return true_val if c else false_val


# ==============================================================================
# --- KEM (Key Encapsulation Mechanism) Implementations
# ==============================================================================


@serde.register_class
@dataclass
class PrivateKeyValue(Value):
    """Runtime representation of a KEM private key.

    This wraps the raw key bytes from a real cryptographic implementation
    (e.g., X25519). The actual cryptographic operations use the `cryptography`
    library which provides secure, audited implementations.
    """

    _serde_kind: ClassVar[str] = "crypto_impl.PrivateKeyValue"

    suite: str
    key_bytes: bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PrivateKeyValue:
        return cls(
            suite=data["suite"],
            key_bytes=base64.b64decode(data["key_bytes"]),
        )


@serde.register_class
@dataclass
class PublicKeyValue(Value):
    """Runtime representation of a KEM public key.

    This wraps the raw key bytes from a real cryptographic implementation.
    """

    _serde_kind: ClassVar[str] = "crypto_impl.PublicKeyValue"

    suite: str
    key_bytes: bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PublicKeyValue:
        return cls(
            suite=data["suite"],
            key_bytes=base64.b64decode(data["key_bytes"]),
        )


@serde.register_class
@dataclass
class SymmetricKeyValue(Value):
    """Runtime representation of a symmetric encryption key.

    This wraps the raw key bytes derived from ECDH key exchange.
    The key is used with AES-256-GCM for authenticated encryption.
    """

    _serde_kind: ClassVar[str] = "crypto_impl.SymmetricKeyValue"

    suite: str
    key_bytes: bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SymmetricKeyValue:
        return cls(
            suite=data["suite"],
            key_bytes=base64.b64decode(data["key_bytes"]),
        )


@crypto.kem_keygen_p.def_impl
def kem_keygen_impl(
    interpreter: Interpreter, op: Operation, suite: str = "x25519"
) -> tuple[PrivateKeyValue, PublicKeyValue]:
    """Generate a KEM key pair."""
    if suite == "x25519":
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

        private_key = X25519PrivateKey.generate()
        public_key = private_key.public_key()

        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
            PublicFormat,
        )

        sk_bytes = private_key.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )
        pk_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

        return (
            PrivateKeyValue(suite=suite, key_bytes=sk_bytes),
            PublicKeyValue(suite=suite, key_bytes=pk_bytes),
        )
    else:
        # Fallback to random bytes for unknown suites
        sk_bytes = os.urandom(32)
        pk_bytes = os.urandom(32)
        return (
            PrivateKeyValue(suite=suite, key_bytes=sk_bytes),
            PublicKeyValue(suite=suite, key_bytes=pk_bytes),
        )


@crypto.kem_derive_p.def_impl
def kem_derive_impl(
    interpreter: Interpreter,
    op: Operation,
    private_key: PrivateKeyValue,
    public_key: PublicKeyValue,
) -> SymmetricKeyValue:
    """Derive a symmetric key using ECDH."""
    suite = getattr(private_key, "suite", "x25519")

    if suite == "x25519":
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey,
            X25519PublicKey,
        )

        sk = X25519PrivateKey.from_private_bytes(private_key.key_bytes)
        pk = X25519PublicKey.from_public_bytes(public_key.key_bytes)
        shared_secret = sk.exchange(pk)

        return SymmetricKeyValue(suite=suite, key_bytes=shared_secret)
    else:
        # Fallback for unknown suites: XOR the key bytes (not cryptographically secure)
        sk_bytes = private_key.key_bytes
        pk_bytes = public_key.key_bytes
        secret = bytes(a ^ b for a, b in zip(sk_bytes, pk_bytes, strict=True))
        return SymmetricKeyValue(suite=suite, key_bytes=secret)


@crypto.random_bytes_p.def_impl
def random_bytes_impl(interpreter: Interpreter, op: Operation) -> TensorValue:
    """Generate random bytes using os.urandom."""
    # Length is passed as attribute
    length = op.attrs["length"]

    if not isinstance(length, int):
        raise TypeError(f"random_bytes length must be int, got {type(length)}")

    b = os.urandom(length)
    arr = np.frombuffer(b, dtype=np.uint8).copy()
    return TensorValue(arr)
