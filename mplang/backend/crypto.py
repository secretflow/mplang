from __future__ import annotations

import os

import numpy as np

from mplang.core.pfunc import PFunction, TensorHandler
from mplang.core.tensor import TensorLike
from mplang.utils.crypto import blake2b


class CryptoHandler(TensorHandler):
    KEY_GEN = "crypto.keygen"
    ENC = "crypto.enc"
    DEC = "crypto.dec"

    def setup(self, rank: int) -> None:
        seed = int(os.environ.get("MPLANG_CRYPTO_SEED", "0")) + rank * 7919
        self._rng = np.random.default_rng(seed)

    def teardown(self) -> None: ...

    def list_fn_names(self) -> list[str]:
        return [self.KEY_GEN, self.ENC, self.DEC]

    def _keystream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        # Insecure mock: derive keystream by hashing (key || nonce || counter)
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

    def _execute_encrypt(self, args: list[TensorLike]) -> list[TensorLike]:
        pt, key = (
            np.asarray(args[0], dtype=np.uint8),
            np.asarray(args[1], dtype=np.uint8),
        )
        nonce = self._rng.integers(0, 256, size=(12,), dtype=np.uint8)
        stream = np.frombuffer(
            self._keystream(key.tobytes(), nonce.tobytes(), len(pt)), dtype=np.uint8
        )
        ct = (pt ^ stream).astype(np.uint8)
        out = np.concatenate([nonce, ct]).astype(np.uint8)
        return [out]

    def _execute_decrypt(self, args: list[TensorLike]) -> list[TensorLike]:
        ct_with_nonce, key = (
            np.asarray(args[0], dtype=np.uint8),
            np.asarray(args[1], dtype=np.uint8),
        )
        nonce = ct_with_nonce[:12]
        ct = ct_with_nonce[12:]
        stream = np.frombuffer(
            self._keystream(key.tobytes(), nonce.tobytes(), len(ct)), dtype=np.uint8
        )
        pt = (ct ^ stream).astype(np.uint8)
        return [pt]

    def execute(self, pfunc: PFunction, args: list[TensorLike]) -> list[TensorLike]:
        if pfunc.fn_type == self.KEY_GEN:
            return self._execute_keygen(pfunc)
        if pfunc.fn_type == self.ENC:
            return self._execute_encrypt(args)
        if pfunc.fn_type == self.DEC:
            return self._execute_decrypt(args)
        raise ValueError(f"Unsupported function type: {pfunc.fn_type}")
