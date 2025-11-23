"""Crypto backend implementation using cryptography and coincurve."""

import hashlib
import os
import pickle
from typing import Any

import coincurve
import jax.numpy as jnp
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mplang2.dialects import crypto
from mplang2.edsl.registry import register_impl as _register_impl_fn


def register_impl(opcode: str):
    def wrapper(fn):
        _register_impl_fn(opcode, fn)
        return fn

    return wrapper


# --- ECC Impl (Coincurve) ---

# secp256k1 order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


@register_impl(crypto.generator_p.name)
def generator_impl(interpreter: Any, op: Any) -> Any:
    # Compressed G
    g_bytes = bytes.fromhex(
        "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    )
    return coincurve.PublicKey(g_bytes)


@register_impl(crypto.mul_p.name)
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


@register_impl(crypto.add_p.name)
def add_impl(interpreter: Any, op: Any, p1: Any, p2: Any) -> Any:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return p1.combine([p2])


@register_impl(crypto.sub_p.name)
def sub_impl(interpreter: Any, op: Any, p1: Any, p2: Any) -> Any:
    # p1 - p2 = p1 + (p2 * -1)
    if p2 is None:
        return p1

    neg_1 = (N - 1).to_bytes(32, "big")
    p2_neg = p2.multiply(neg_1)

    if p1 is None:
        return p2_neg

    return p1.combine([p2_neg])


@register_impl(crypto.random_scalar_p.name)
def random_scalar_impl(interpreter: Any, op: Any) -> Any:
    return int.from_bytes(os.urandom(32), "big") % N


@register_impl(crypto.scalar_from_int_p.name)
def scalar_from_int_impl(interpreter: Any, op: Any, val: Any) -> Any:
    return int(val)


@register_impl(crypto.point_to_bytes_p.name)
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


@register_impl(crypto.hash_p.name)
def hash_impl(interpreter: Any, op: Any, data: Any) -> Any:
    d = data
    if hasattr(d, "tobytes"):
        d = d.tobytes()
    elif isinstance(d, (list, tuple)):
        d = bytes(d)

    h = hashlib.sha256(d).digest()
    arr = jnp.frombuffer(h, dtype=jnp.uint8)
    return arr


@register_impl(crypto.sym_encrypt_p.name)
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


@register_impl(crypto.sym_decrypt_p.name)
def sym_decrypt_impl(
    interpreter: Any,
    op: Any,
    key: Any,
    ciphertext: Any,
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


@register_impl(crypto.select_p.name)
def select_impl(
    interpreter: Any, op: Any, cond: Any, true_val: Any, false_val: Any
) -> Any:
    c = cond
    if hasattr(c, "item"):
        c = c.item()

    return true_val if c else false_val
