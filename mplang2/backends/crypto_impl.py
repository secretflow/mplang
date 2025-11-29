"""Crypto backend implementation using cryptography and coincurve."""

import hashlib
import os
import pickle
from dataclasses import dataclass
from typing import Any

import coincurve
import jax.numpy as jnp
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mplang2.dialects import crypto

# --- ECC Impl (Coincurve) ---

# secp256k1 order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


@crypto.generator_p.def_impl
def generator_impl(interpreter: Any, op: Any) -> Any:
    # Compressed G
    g_bytes = bytes.fromhex(
        "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    )
    return coincurve.PublicKey(g_bytes)


@crypto.mul_p.def_impl
def mul_impl(interpreter: Any, op: Any, point: Any, scalar: Any) -> Any:
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
    return point.multiply(s_bytes)


@crypto.add_p.def_impl
def add_impl(interpreter: Any, op: Any, p1: Any, p2: Any) -> Any:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return p1.combine([p2])


@crypto.sub_p.def_impl
def sub_impl(interpreter: Any, op: Any, p1: Any, p2: Any) -> Any:
    # p1 - p2 = p1 + (-p2)
    if p2 is None:
        return p1

    # Negate p2
    # coincurve doesn't have direct negate?
    # We can multiply by -1 (N-1)
    neg_scalar = (N - 1).to_bytes(32, "big")
    neg_p2 = p2.multiply(neg_scalar)

    if p1 is None:
        return neg_p2

    return p1.combine([neg_p2])


@crypto.random_scalar_p.def_impl
def random_scalar_impl(interpreter: Any, op: Any) -> Any:
    return int.from_bytes(os.urandom(32), "big") % N


@crypto.scalar_from_int_p.def_impl
def scalar_from_int_impl(interpreter: Any, op: Any, val: Any) -> Any:
    return int(val)


@crypto.point_to_bytes_p.def_impl
def point_to_bytes_impl(interpreter: Any, op: Any, point: Any) -> Any:
    if point is None:
        # Infinity / Identity -> Zeros
        # 65 bytes to match uncompressed format length?
        # Or 64? Previous was 64.
        # coincurve uncompressed is 65.
        # Let's return 65 zeros.
        return jnp.zeros(65, dtype=jnp.uint8)

    # Returns 65 bytes (uncompressed)
    b = point.format(compressed=False)
    return jnp.frombuffer(b, dtype=jnp.uint8)


# --- Sym / Hash Impl ---


@crypto.hash_p.def_impl
def hash_impl(interpreter: Any, op: Any, data: Any) -> Any:
    d = data
    if hasattr(d, "tobytes"):
        d = d.tobytes()
    elif isinstance(d, (list, tuple)):
        d = bytes(d)

    h = hashlib.sha256(d).digest()
    arr = jnp.frombuffer(h, dtype=jnp.uint8)
    return arr


@crypto.sym_encrypt_p.def_impl
def sym_encrypt_impl(interpreter: Any, op: Any, key: Any, plaintext: Any) -> Any:
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
    interpreter: Any,
    op: Any,
    key: Any,
    ciphertext: Any,
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
    interpreter: Any, op: Any, cond: Any, true_val: Any, false_val: Any
) -> Any:
    c = cond
    if hasattr(c, "item"):
        c = c.item()

    return true_val if c else false_val


# ==============================================================================
# --- KEM (Key Encapsulation Mechanism) Implementations
# ==============================================================================


@dataclass
class RuntimePrivateKey:
    """Runtime representation of a KEM private key.

    This wraps the raw key bytes from a real cryptographic implementation
    (e.g., X25519). The actual cryptographic operations use the `cryptography`
    library which provides secure, audited implementations.
    """

    suite: str
    key_bytes: bytes


@dataclass
class RuntimePublicKey:
    """Runtime representation of a KEM public key.

    This wraps the raw key bytes from a real cryptographic implementation.
    """

    suite: str
    key_bytes: bytes


@dataclass
class RuntimeSymmetricKey:
    """Runtime representation of a symmetric encryption key.

    This wraps the raw key bytes derived from ECDH key exchange.
    The key is used with AES-256-GCM for authenticated encryption.
    """

    suite: str
    key_bytes: bytes


@crypto.kem_keygen_p.def_impl
def kem_keygen_impl(interpreter: Any, op: Any, suite: str = "x25519") -> Any:
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
    interpreter: Any, op: Any, private_key: Any, public_key: Any
) -> Any:
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
