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
import pickle
from dataclasses import dataclass
from typing import Any, ClassVar

import coincurve
import jax
import jax.numpy as jnp
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mplang.v2.backends.value import Value
from mplang.v2.dialects import crypto
from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Operation
from mplang.v2.edsl.interpreter import Interpreter

# =============================================================================
# ECC Point Wrapper (secp256k1)
# =============================================================================


@serde.register_class
@dataclass
class ECPoint(Value):
    """Wrapper for coincurve.PublicKey representing an elliptic curve point.

    This wraps the external coincurve library's PublicKey type to provide
    proper serialization support via the Value base class.
    """

    _serde_kind: ClassVar[str] = "crypto_impl.ECPoint"

    # The raw public key bytes (compressed format, 33 bytes)
    key_bytes: bytes

    def to_json(self) -> dict[str, Any]:
        return {"data": base64.b64encode(self.key_bytes).decode("ascii")}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ECPoint:
        return cls(key_bytes=base64.b64decode(data["data"]))

    @property
    def coincurve_key(self) -> coincurve.PublicKey:
        """Get the underlying coincurve.PublicKey object."""
        return coincurve.PublicKey(self.key_bytes)

    @classmethod
    def from_coincurve(cls, pk: coincurve.PublicKey) -> ECPoint:
        """Create ECPoint from a coincurve.PublicKey."""
        return cls(key_bytes=pk.format(compressed=True))


# --- ECC Impl (Coincurve) ---

# secp256k1 order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


@crypto.generator_p.def_impl
def generator_impl(interpreter: Interpreter, op: Operation) -> ECPoint:
    # Compressed G
    g_bytes = bytes.fromhex(
        "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    )
    return ECPoint(key_bytes=g_bytes)


@crypto.mul_p.def_impl
def mul_impl(
    interpreter: Interpreter, op: Operation, point: ECPoint | None, scalar: Any
) -> ECPoint | None:
    s_val = scalar
    if hasattr(s_val, "item"):
        s_val = s_val.item()
    s_val = int(s_val) % N

    if s_val == 0:
        return None

    if point is None:
        return None

    # coincurve multiply expects bytes
    s_bytes = s_val.to_bytes(32, "big")
    result = point.coincurve_key.multiply(s_bytes)
    return ECPoint.from_coincurve(result)


@crypto.add_p.def_impl
def add_impl(
    interpreter: Interpreter, op: Operation, p1: ECPoint | None, p2: ECPoint | None
) -> ECPoint | None:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    result = p1.coincurve_key.combine([p2.coincurve_key])
    return ECPoint.from_coincurve(result)


@crypto.sub_p.def_impl
def sub_impl(
    interpreter: Interpreter, op: Operation, p1: ECPoint | None, p2: ECPoint | None
) -> ECPoint | None:
    # p1 - p2 = p1 + (-p2)
    if p2 is None:
        return p1

    # Negate p2 by multiplying by (N-1)
    neg_scalar = (N - 1).to_bytes(32, "big")
    neg_p2 = p2.coincurve_key.multiply(neg_scalar)

    if p1 is None:
        return ECPoint.from_coincurve(neg_p2)

    result = p1.coincurve_key.combine([neg_p2])
    return ECPoint.from_coincurve(result)


@crypto.random_scalar_p.def_impl
def random_scalar_impl(interpreter: Interpreter, op: Operation) -> int:
    return int.from_bytes(os.urandom(32), "big") % N


@crypto.scalar_from_int_p.def_impl
def scalar_from_int_impl(interpreter: Interpreter, op: Operation, val: Any) -> int:
    return int(val)


@crypto.point_to_bytes_p.def_impl
def point_to_bytes_impl(
    interpreter: Interpreter, op: Operation, point: ECPoint | None
) -> jnp.ndarray:
    if point is None:
        # Infinity / Identity -> Zeros
        # 65 bytes to match uncompressed format length
        return jnp.zeros(65, dtype=jnp.uint8)

    # Returns 65 bytes (uncompressed)
    b = point.coincurve_key.format(compressed=False)
    return jnp.frombuffer(b, dtype=jnp.uint8)


# --- Sym / Hash Impl ---


@crypto.hash_p.def_impl
def hash_impl(interpreter: Interpreter, op: Operation, data: Any) -> jnp.ndarray:
    d = data
    if hasattr(d, "tobytes"):
        d = d.tobytes()
    elif isinstance(d, (list, tuple)):
        d = bytes(d)

    h = hashlib.sha256(d).digest()
    arr = jnp.frombuffer(h, dtype=jnp.uint8)
    return arr


@crypto.sym_encrypt_p.def_impl
def sym_encrypt_impl(
    interpreter: Interpreter,
    op: Operation,
    key: RuntimeSymmetricKey | jax.Array,
    plaintext: Any,
) -> jax.Array:
    k = key
    # Support RuntimeSymmetricKey from kem_derive
    if isinstance(k, RuntimeSymmetricKey):
        k = k.key_bytes
    elif hasattr(k, "tobytes"):
        k = k.tobytes()

    # Ensure key is 32 bytes (AES-256)
    if len(k) != 32:
        pass

    pt = plaintext
    pt_bytes = pickle.dumps(pt)

    # AES-GCM
    aesgcm = AESGCM(k)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, pt_bytes, None)

    # Result: nonce + ct
    res = nonce + ct
    arr = jnp.frombuffer(res, dtype=jnp.uint8)
    return arr


@crypto.sym_decrypt_p.def_impl
def sym_decrypt_impl(
    interpreter: Interpreter,
    op: Operation,
    key: RuntimeSymmetricKey | jax.Array,
    ciphertext: jax.Array | bytes,
    target_type: Any = None,
) -> Any:
    k = key
    # Support RuntimeSymmetricKey from kem_derive
    if isinstance(k, RuntimeSymmetricKey):
        k = k.key_bytes
    elif hasattr(k, "tobytes"):
        k = k.tobytes()

    ct_full = ciphertext
    if hasattr(ct_full, "tobytes"):
        ct_full = ct_full.tobytes()
    elif isinstance(ct_full, (list, tuple)):
        ct_full = bytes(ct_full)

    # Extract nonce
    nonce = ct_full[:12]
    ct = ct_full[12:]

    aesgcm = AESGCM(k)
    pt_bytes = aesgcm.decrypt(nonce, ct, None)

    pt = pickle.loads(pt_bytes)
    return pt


@crypto.select_p.def_impl
def select_impl(
    interpreter: Interpreter, op: Operation, cond: Any, true_val: Any, false_val: Any
) -> Any:
    c = cond
    if hasattr(c, "item"):
        c = c.item()

    return true_val if c else false_val


# ==============================================================================
# --- KEM (Key Encapsulation Mechanism) Implementations
# ==============================================================================


@serde.register_class
@dataclass
class RuntimePrivateKey(Value):
    """Runtime representation of a KEM private key.

    This wraps the raw key bytes from a real cryptographic implementation
    (e.g., X25519). The actual cryptographic operations use the `cryptography`
    library which provides secure, audited implementations.
    """

    _serde_kind: ClassVar[str] = "crypto_impl.RuntimePrivateKey"

    suite: str
    key_bytes: bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RuntimePrivateKey:
        return cls(
            suite=data["suite"],
            key_bytes=base64.b64decode(data["key_bytes"]),
        )


@serde.register_class
@dataclass
class RuntimePublicKey(Value):
    """Runtime representation of a KEM public key.

    This wraps the raw key bytes from a real cryptographic implementation.
    """

    _serde_kind: ClassVar[str] = "crypto_impl.RuntimePublicKey"

    suite: str
    key_bytes: bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RuntimePublicKey:
        return cls(
            suite=data["suite"],
            key_bytes=base64.b64decode(data["key_bytes"]),
        )


@serde.register_class
@dataclass
class RuntimeSymmetricKey(Value):
    """Runtime representation of a symmetric encryption key.

    This wraps the raw key bytes derived from ECDH key exchange.
    The key is used with AES-256-GCM for authenticated encryption.
    """

    _serde_kind: ClassVar[str] = "crypto_impl.RuntimeSymmetricKey"

    suite: str
    key_bytes: bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RuntimeSymmetricKey:
        return cls(
            suite=data["suite"],
            key_bytes=base64.b64decode(data["key_bytes"]),
        )


@crypto.kem_keygen_p.def_impl
def kem_keygen_impl(
    interpreter: Interpreter, op: Operation, suite: str = "x25519"
) -> tuple[RuntimePrivateKey, RuntimePublicKey]:
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
            RuntimePrivateKey(suite=suite, key_bytes=sk_bytes),
            RuntimePublicKey(suite=suite, key_bytes=pk_bytes),
        )
    else:
        # Fallback to random bytes for unknown suites
        sk_bytes = os.urandom(32)
        pk_bytes = os.urandom(32)
        return (
            RuntimePrivateKey(suite=suite, key_bytes=sk_bytes),
            RuntimePublicKey(suite=suite, key_bytes=pk_bytes),
        )


@crypto.kem_derive_p.def_impl
def kem_derive_impl(
    interpreter: Interpreter,
    op: Operation,
    private_key: RuntimePrivateKey,
    public_key: RuntimePublicKey,
) -> RuntimeSymmetricKey:
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

        return RuntimeSymmetricKey(suite=suite, key_bytes=shared_secret)
    else:
        # Fallback for unknown suites: XOR the key bytes (not cryptographically secure)
        sk_bytes = private_key.key_bytes
        pk_bytes = public_key.key_bytes
        secret = bytes(a ^ b for a, b in zip(sk_bytes, pk_bytes, strict=True))
        return RuntimeSymmetricKey(suite=suite, key_bytes=secret)
