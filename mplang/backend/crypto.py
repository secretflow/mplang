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

from mplang.core.pfunc import PFunction, TensorHandler
from mplang.core.tensor import TensorLike, TensorType
from mplang.utils.crypto import blake2b


class CryptoHandler(TensorHandler):
    """WARNING: Mock crypto implementation for demos/tests only.

    Security notes:
    - Stream cipher is a hash-based XOR keystream (not AEAD): INSECURE.
        No integrity or authenticity. Nonce misuse is not prevented.
    - KEM keygen/derive are placeholders, not real ECDH/HPKE; suite is ignored.
    - HKDF is a simple blake2b(secret||info) truncation; not a real HKDF.
    """

    KEY_GEN = "crypto.keygen"
    ENC = "crypto.enc"
    DEC = "crypto.dec"
    PACK = "crypto.pack"
    UNPACK = "crypto.unpack"
    KEM_KEYGEN = "crypto.kem_keygen"
    KEM_DERIVE = "crypto.kem_derive"
    HKDF = "crypto.hkdf"

    def setup(self, rank: int) -> None:
        seed = int(os.environ.get("MPLANG_CRYPTO_SEED", "0")) + rank * 7919
        self._rng = np.random.default_rng(seed)

    def teardown(self) -> None: ...

    def list_fn_names(self) -> list[str]:
        return [
            self.KEY_GEN,
            self.ENC,
            self.DEC,
            self.PACK,
            self.UNPACK,
            self.KEM_KEYGEN,
            self.KEM_DERIVE,
            self.HKDF,
        ]

    def _keystream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        # WARNING (INSECURE): hash-based keystream (key||nonce||counter)
        out = bytearray()
        counter = 0
        while len(out) < length:
            chunk = blake2b(key + nonce + counter.to_bytes(4, "little"))
            out.extend(chunk)
            counter += 1
        return bytes(out[:length])

    def _execute_keygen(self, pfunc: PFunction) -> list[TensorLike]:
        length = int(pfunc.attrs.get("length", 32))
        key = self._rng.integers(0, 256, size=(length,), dtype=np.uint8)
        return [key]

    def _execute_decrypt(self, args: list[TensorLike]) -> list[TensorLike]:
        # WARNING: No authenticity/integrity check prior to decryption.
        ct_with_nonce = np.asarray(args[0], dtype=np.uint8)
        key = np.asarray(args[1], dtype=np.uint8)
        nonce = ct_with_nonce[:12]
        ct = ct_with_nonce[12:]
        stream = np.frombuffer(
            self._keystream(key.tobytes(), nonce.tobytes(), len(ct)), dtype=np.uint8
        )
        pt_bytes = (ct ^ stream).astype(np.uint8)
        return [pt_bytes]

    def _execute_encrypt(self, args: list[TensorLike]) -> list[TensorLike]:
        # WARNING: Not AEAD. Ciphertext has no auth tag.
        pt_bytes = np.asarray(args[0], dtype=np.uint8)
        key = np.asarray(args[1], dtype=np.uint8)
        nonce = self._rng.integers(0, 256, size=(12,), dtype=np.uint8)
        stream = np.frombuffer(
            self._keystream(key.tobytes(), nonce.tobytes(), pt_bytes.size),
            dtype=np.uint8,
        )
        ct = (pt_bytes ^ stream).astype(np.uint8)
        out = np.concatenate([nonce, ct]).astype(np.uint8)
        return [out]

    def _execute_pack(self, args: list[TensorLike]) -> list[TensorLike]:
        x_any = np.asarray(args[0])
        out = np.frombuffer(x_any.tobytes(order="C"), dtype=np.uint8)
        return [out]

    def _execute_unpack(
        self, args: list[TensorLike], pfunc: PFunction
    ) -> list[TensorLike]:
        b = np.asarray(args[0], dtype=np.uint8)
        # Read desired output type from outs_info
        assert len(pfunc.outs_info) == 1
        out_ty_any = pfunc.outs_info[0]
        assert isinstance(out_ty_any, TensorType), "unpack outs_info must be TensorType"
        out_ty = out_ty_any
        # Map dtype to numpy dtype
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
        arr = np.frombuffer(b.tobytes(), dtype=np_dtype)
        arr = arr.reshape(shape)
        return [arr]

    def _execute_kem_keygen(self, pfunc: PFunction) -> list[TensorLike]:
        # WARNING: Mock KEM keypair. public = H(sk) for symmetric demo only.
        # Suite attribute is ignored.
        sk = self._rng.integers(0, 256, size=(32,), dtype=np.uint8)
        pk = np.frombuffer(blake2b(sk.tobytes())[:32], dtype=np.uint8)
        return [sk, pk]

    def _execute_kem_derive(
        self, args: list[TensorLike], pfunc: PFunction
    ) -> list[TensorLike]:
        sk = np.asarray(args[0], dtype=np.uint8)
        peer_pk = np.asarray(args[1], dtype=np.uint8)
        # WARNING: Insecure symmetric mock derive (NOT real ECDH/HPKE):
        # self_pk = H(sk); shared = H(self_pk XOR peer_pk)
        self_pk = np.frombuffer(blake2b(sk.tobytes())[:32], dtype=np.uint8)
        xored = (self_pk ^ peer_pk).astype(np.uint8)
        secret = np.frombuffer(blake2b(xored.tobytes())[:32], dtype=np.uint8)
        return [secret]

    def _execute_hkdf(
        self, args: list[TensorLike], pfunc: PFunction
    ) -> list[TensorLike]:
        secret = np.asarray(args[0], dtype=np.uint8)
        info_str = str(pfunc.attrs.get("info", ""))
        info = info_str.encode("utf-8")
        # WARNING: Mock HKDF using blake2b(secret||info) truncation.
        out = np.frombuffer(blake2b(secret.tobytes() + info)[:32], dtype=np.uint8)
        return [out]

    def execute(self, pfunc: PFunction, args: list[TensorLike]) -> list[TensorLike]:
        if pfunc.fn_type == self.KEY_GEN:
            return self._execute_keygen(pfunc)
        if pfunc.fn_type == self.ENC:
            return self._execute_encrypt(args)
        if pfunc.fn_type == self.DEC:
            return self._execute_decrypt(args)
        if pfunc.fn_type == self.PACK:
            return self._execute_pack(args)
        if pfunc.fn_type == self.UNPACK:
            return self._execute_unpack(args, pfunc)
        if pfunc.fn_type == self.KEM_KEYGEN:
            return self._execute_kem_keygen(pfunc)
        if pfunc.fn_type == self.KEM_DERIVE:
            return self._execute_kem_derive(args, pfunc)
        if pfunc.fn_type == self.HKDF:
            return self._execute_hkdf(args, pfunc)
        raise ValueError(f"Unsupported function type: {pfunc.fn_type}")
