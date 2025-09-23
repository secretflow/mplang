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

from __future__ import annotations

import os

import numpy as np

from mplang.backend.base import backend_kernel, cur_kctx
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType
from mplang.utils.crypto import blake2b

__all__ = [
    # flat kernels only; handler shim below for backward compatibility errors
]


def _get_rng():
    """Get (and lazily create) per-rank RNG for crypto kernels.

    Seed rule matches legacy handler: MPLANG_CRYPTO_SEED + rank*7919
    """
    kctx = cur_kctx()
    pocket = kctx.state.setdefault("crypto", {})
    rng = pocket.get("rng")
    if rng is None:
        seed = int(os.environ.get("MPLANG_CRYPTO_SEED", "0")) + kctx.rank * 7919
        rng = np.random.default_rng(seed)
        pocket["rng"] = rng
    return rng


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    # WARNING (INSECURE): hash-based keystream (key||nonce||counter)
    out = bytearray()
    counter = 0
    while len(out) < length:
        chunk = blake2b(key + nonce + counter.to_bytes(4, "little"))
        out.extend(chunk)
        counter += 1
    return bytes(out[:length])


@backend_kernel("crypto.keygen")
def _crypto_keygen(pfunc: PFunction, args: tuple) -> tuple:
    length = int(pfunc.attrs.get("length", 32))
    rng = _get_rng()
    key = rng.integers(0, 256, size=(length,), dtype=np.uint8)
    return (key,)


@backend_kernel("crypto.enc")
def _crypto_encrypt(pfunc: PFunction, args: tuple) -> tuple:
    # args: (pt_bytes, key)
    if len(args) != 2:
        raise ValueError("crypto.enc expects (pt_bytes, key)")
    pt_bytes = np.asarray(args[0], dtype=np.uint8)
    key = np.asarray(args[1], dtype=np.uint8)
    rng = _get_rng()
    nonce = rng.integers(0, 256, size=(12,), dtype=np.uint8)
    stream = np.frombuffer(
        _keystream(key.tobytes(), nonce.tobytes(), pt_bytes.size), dtype=np.uint8
    )
    ct = (pt_bytes ^ stream).astype(np.uint8)
    out = np.concatenate([nonce, ct]).astype(np.uint8)
    return (out,)


@backend_kernel("crypto.dec")
def _crypto_decrypt(pfunc: PFunction, args: tuple) -> tuple:
    if len(args) != 2:
        raise ValueError("crypto.dec expects (ct_with_nonce, key)")
    ct_with_nonce = np.asarray(args[0], dtype=np.uint8)
    key = np.asarray(args[1], dtype=np.uint8)
    nonce = ct_with_nonce[:12]
    ct = ct_with_nonce[12:]
    stream = np.frombuffer(
        _keystream(key.tobytes(), nonce.tobytes(), len(ct)), dtype=np.uint8
    )
    pt_bytes = (ct ^ stream).astype(np.uint8)
    return (pt_bytes,)


@backend_kernel("crypto.pack")
def _crypto_pack(pfunc: PFunction, args: tuple) -> tuple:
    if len(args) != 1:
        raise ValueError("crypto.pack expects a single argument")
    x_any = np.asarray(args[0])
    out = np.frombuffer(x_any.tobytes(order="C"), dtype=np.uint8)
    return (out,)


@backend_kernel("crypto.unpack")
def _crypto_unpack(pfunc: PFunction, args: tuple) -> tuple:
    if len(args) != 1:
        raise ValueError("crypto.unpack expects a single byte tensor argument")
    b = np.asarray(args[0], dtype=np.uint8)
    assert len(pfunc.outs_info) == 1
    out_ty_any = pfunc.outs_info[0]
    if not isinstance(out_ty_any, TensorType):
        raise TypeError("unpack outs_info must be TensorType")
    out_ty = out_ty_any
    np_dtype = out_ty.dtype.numpy_dtype()
    shape = tuple(out_ty.shape)
    expected = (
        int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        if len(shape) > 0
        else np.dtype(np_dtype).itemsize
    )
    if b.size != expected:
        raise ValueError(
            f"unpack size mismatch: got {b.size} bytes, expect {expected} for {np_dtype} {shape}"
        )
    arr = np.frombuffer(b.tobytes(), dtype=np_dtype).reshape(shape)
    return (arr,)


@backend_kernel("crypto.kem_keygen")
def _crypto_kem_keygen(pfunc: PFunction, args: tuple) -> tuple:
    rng = _get_rng()
    sk = rng.integers(0, 256, size=(32,), dtype=np.uint8)
    pk = np.frombuffer(blake2b(sk.tobytes())[:32], dtype=np.uint8)
    return (sk, pk)


@backend_kernel("crypto.kem_derive")
def _crypto_kem_derive(pfunc: PFunction, args: tuple) -> tuple:
    if len(args) != 2:
        raise ValueError("crypto.kem_derive expects (sk, peer_pk)")
    sk = np.asarray(args[0], dtype=np.uint8)
    peer_pk = np.asarray(args[1], dtype=np.uint8)
    self_pk = np.frombuffer(blake2b(sk.tobytes())[:32], dtype=np.uint8)
    xored = (self_pk ^ peer_pk).astype(np.uint8)
    secret = np.frombuffer(blake2b(xored.tobytes())[:32], dtype=np.uint8)
    return (secret,)


@backend_kernel("crypto.hkdf")
def _crypto_hkdf(pfunc: PFunction, args: tuple) -> tuple:
    if len(args) != 1:
        raise ValueError("crypto.hkdf expects (secret,)")
    secret = np.asarray(args[0], dtype=np.uint8)
    info_str = str(pfunc.attrs.get("info", ""))
    info = info_str.encode("utf-8")
    out = np.frombuffer(blake2b(secret.tobytes() + info)[:32], dtype=np.uint8)
    return (out,)
