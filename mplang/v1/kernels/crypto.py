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

from mplang.v1.core import PFunction
from mplang.v1.kernels.base import cur_kctx, kernel_def
from mplang.v1.kernels.value import TensorValue
from mplang.v1.utils.crypto import blake2b

__all__: list[str] = []  # No public exports currently


def _get_rng() -> np.random.Generator:
    """Get (and lazily create) per-rank RNG for crypto kernels.

    Runtime state is untyped, so we narrow the type explicitly for mypy.
    """
    kctx = cur_kctx()
    rt = kctx.runtime
    rng_obj = rt.get_state("crypto.rng")
    if rng_obj is None:
        seed = int(os.environ.get("MPLANG_CRYPTO_SEED", "0")) + kctx.rank * 7919
        rng_obj = np.random.default_rng(seed)
        rt.set_state("crypto.rng", rng_obj)
    assert isinstance(rng_obj, np.random.Generator)  # narrow
    return rng_obj


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    # WARNING (INSECURE): hash-based keystream (key||nonce||counter)
    out = bytearray()
    while len(out) < length:
        chunk = blake2b(key + nonce)
        out.extend(chunk)
    return bytes(out[:length])


@kernel_def("crypto.keygen")
def _crypto_keygen(pfunc: PFunction) -> TensorValue:
    length = int(pfunc.attrs.get("length", 32))
    rng = _get_rng()
    key = rng.integers(0, 256, size=(length,), dtype=np.uint8)
    return TensorValue(key)


@kernel_def("crypto.enc")
def _crypto_encrypt(
    pfunc: PFunction, pt_bytes: TensorValue, key: TensorValue
) -> TensorValue:
    pt_bytes_np = pt_bytes.to_numpy().astype(np.uint8, copy=False)
    key_np = key.to_numpy().astype(np.uint8, copy=False)
    rng = _get_rng()
    nonce = rng.integers(0, 256, size=(16,), dtype=np.uint8)
    stream = np.frombuffer(
        _keystream(key_np.tobytes(), nonce.tobytes(), pt_bytes_np.size), dtype=np.uint8
    )
    ct = (pt_bytes_np ^ stream).astype(np.uint8)
    out = np.concatenate([nonce, ct]).astype(np.uint8)
    return TensorValue(out)


@kernel_def("crypto.dec")
def _crypto_decrypt(
    pfunc: PFunction, ct_with_nonce: TensorValue, key: TensorValue
) -> TensorValue:
    ct_np = ct_with_nonce.to_numpy().astype(np.uint8, copy=False)
    key_np = key.to_numpy().astype(np.uint8, copy=False)
    nonce = ct_np[:16]
    ct = ct_np[16:]
    stream = np.frombuffer(
        _keystream(key_np.tobytes(), nonce.tobytes(), len(ct)), dtype=np.uint8
    )
    pt_bytes = (ct ^ stream).astype(np.uint8)
    return TensorValue(pt_bytes)


@kernel_def("crypto.kem_keygen")
def _crypto_kem_keygen(pfunc: PFunction) -> tuple[TensorValue, TensorValue]:
    rng = _get_rng()
    sk = rng.integers(0, 256, size=(32,), dtype=np.uint8)
    pk_bytes = blake2b(sk.tobytes())[:32]
    pk = np.frombuffer(pk_bytes, dtype=np.uint8)
    return (TensorValue(sk), TensorValue(pk))


@kernel_def("crypto.kem_derive")
def _crypto_kem_derive(
    pfunc: PFunction, sk: TensorValue, peer_pk: TensorValue
) -> TensorValue:
    sk_np = sk.to_numpy().astype(np.uint8, copy=False)
    peer_pk_np = peer_pk.to_numpy().astype(np.uint8, copy=False)

    self_pk_bytes = blake2b(sk_np.tobytes())[:32]
    self_pk_arr = np.frombuffer(self_pk_bytes, dtype=np.uint8)
    xored = (self_pk_arr ^ peer_pk_np).astype(np.uint8)
    secret = np.frombuffer(blake2b(xored.tobytes())[:32], dtype=np.uint8)
    return TensorValue(secret)


@kernel_def("crypto.hkdf")
def _crypto_hkdf(pfunc: PFunction, secret: TensorValue) -> TensorValue:
    secret_np = secret.to_numpy().astype(np.uint8, copy=False)
    info_str = str(pfunc.attrs.get("info", ""))
    info = info_str.encode("utf-8")
    out = np.frombuffer(blake2b(secret_np.tobytes() + info)[:32], dtype=np.uint8)
    return TensorValue(out)
