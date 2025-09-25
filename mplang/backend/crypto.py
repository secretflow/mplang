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
from typing import Any

import numpy as np

from mplang.backend.base import cur_kctx, kernel_def
from mplang.core.pfunc import PFunction
from mplang.utils.crypto import blake2b

__all__: list[str] = []  # flat kernels only


def _get_rng() -> np.random.Generator:
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


@kernel_def("crypto.keygen")
def _crypto_keygen(pfunc: PFunction) -> Any:
    length = int(pfunc.attrs.get("length", 32))
    rng = _get_rng()
    key = rng.integers(0, 256, size=(length,), dtype=np.uint8)
    return key


@kernel_def("crypto.enc")
def _crypto_encrypt(pfunc: PFunction, pt_bytes: Any, key: Any) -> Any:
    pt_bytes = np.asarray(pt_bytes, dtype=np.uint8)
    key = np.asarray(key, dtype=np.uint8)
    rng = _get_rng()
    nonce = rng.integers(0, 256, size=(12,), dtype=np.uint8)
    stream = np.frombuffer(
        _keystream(key.tobytes(), nonce.tobytes(), pt_bytes.size), dtype=np.uint8
    )
    ct = (pt_bytes ^ stream).astype(np.uint8)
    out = np.concatenate([nonce, ct]).astype(np.uint8)
    return out


@kernel_def("crypto.dec")
def _crypto_decrypt(pfunc: PFunction, ct_with_nonce: Any, key: Any) -> Any:
    ct_with_nonce = np.asarray(ct_with_nonce, dtype=np.uint8)
    key = np.asarray(key, dtype=np.uint8)
    nonce = ct_with_nonce[:12]
    ct = ct_with_nonce[12:]
    stream = np.frombuffer(
        _keystream(key.tobytes(), nonce.tobytes(), len(ct)), dtype=np.uint8
    )
    pt_bytes = (ct ^ stream).astype(np.uint8)
    return pt_bytes


@kernel_def("crypto.kem_keygen")
def _crypto_kem_keygen(pfunc: PFunction) -> Any:
    rng = _get_rng()
    sk = rng.integers(0, 256, size=(32,), dtype=np.uint8)
    pk = np.frombuffer(blake2b(sk.tobytes())[:32], dtype=np.uint8)
    return (sk, pk)


@kernel_def("crypto.kem_derive")
def _crypto_kem_derive(pfunc: PFunction, sk: Any, peer_pk: Any) -> Any:
    sk = np.asarray(sk, dtype=np.uint8)
    peer_pk = np.asarray(peer_pk, dtype=np.uint8)
    self_pk = np.frombuffer(blake2b(sk.tobytes())[:32], dtype=np.uint8)
    xored = (self_pk ^ peer_pk).astype(np.uint8)
    secret = np.frombuffer(blake2b(xored.tobytes())[:32], dtype=np.uint8)
    return secret


@kernel_def("crypto.hkdf")
def _crypto_hkdf(pfunc: PFunction, secret: Any) -> Any:
    secret = np.asarray(secret, dtype=np.uint8)
    info_str = str(pfunc.attrs.get("info", ""))
    info = info_str.encode("utf-8")
    out = np.frombuffer(blake2b(secret.tobytes() + info)[:32], dtype=np.uint8)
    return out
