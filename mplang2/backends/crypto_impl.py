"""Crypto backend implementation using cryptography and coincurve."""

import hashlib
import os
import pickle
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
    if hasattr(k, "tobytes"):
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
    if hasattr(k, "tobytes"):
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
