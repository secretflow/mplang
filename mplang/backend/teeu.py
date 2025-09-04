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


from typing import Any

from mplang.core.mptype import TableLike, TensorLike
from mplang.core.pfunc import HybridHandler, PFunction
from mplang.crypto.rsa import EncryptedTable, EncryptedTensor, RSAEncryptor
from mplang.runtime.executor.tee_attestation import TEEKeyManager


class TEEHandler(HybridHandler):
    def __init__(self, key_mgr: TEEKeyManager, session_id: str) -> None:
        super().__init__()

        self._key_mgr = key_mgr
        self._session_id = session_id

    def setup(self, rank: int) -> None: ...
    def teardown(self) -> None: ...
    def list_fn_names(self) -> list[str]:
        return [
            "tee.encrypt",
            "tee.decrypt",
        ]

    def execute(
        self,
        pfunc: PFunction,
        args: list[TensorLike | TableLike],
    ) -> list[TensorLike | TableLike]:
        """Execute PHE operations."""
        if pfunc.fn_type == "tee.encrypt":
            return self._execute_encrypt(pfunc, args)
        elif pfunc.fn_type == "tee.decrypt":
            return self._execute_decrypt(pfunc, args)
        else:
            raise ValueError(f"Unsupported TEE function type: {pfunc.fn_type}")

    def _execute_encrypt(
        self, pfunc: PFunction, args: list[TensorLike | TableLike]
    ) -> list[TensorLike | TableLike]:
        if len(args) != 1:
            raise ValueError("Encryption expects exactly one arguments: plaintext")

        plaintext = args[0]
        attrs: dict[str, Any] = dict(pfunc.attrs or {})
        target_rank = attrs.get("to_rank")
        tee_rank = attrs.get("tee_rank")
        assert target_rank is not None

        if target_rank == tee_rank:
            public_key = self._key_mgr.get_tee_pub_key(self._session_id)
        else:
            public_key = self._key_mgr.get_peer_pub_key(self._session_id, target_rank)

        try:
            rsa = RSAEncryptor()
            rsa.load_keys(public_key_pem=public_key)
            if isinstance(plaintext, TensorLike):
                encrypted_tensor = rsa.encrypt_tensor(plaintext)
                return [encrypted_tensor]
            elif isinstance(plaintext, TableLike):
                encrypted_table = rsa.encrypt_table(plaintext)
                return [encrypted_table]
            else:
                raise ValueError(
                    "First argument must be TensorLike or TableLike instance"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to encrypt data: {e}") from e

    def _execute_decrypt(
        self, pfunc: PFunction, args: list[TensorLike | TableLike]
    ) -> list[TensorLike | TableLike]:
        if len(args) != 1:
            raise ValueError("Decryption expects exactly one arguments: ciphertext")

        ciphertext = args[0]

        try:
            rsa = RSAEncryptor()
            rsa.load_keys(
                private_key_pem=self._key_mgr.get_or_create_self_key_pair(
                    self._session_id
                )[1]
            )
            if isinstance(ciphertext, EncryptedTensor):
                plaintext_np = rsa.decrypt_tensor(ciphertext)
                return [plaintext_np]
            elif isinstance(ciphertext, EncryptedTable):
                plaintext_table = rsa.decrypt_table(ciphertext)
                return [plaintext_table]
            else:
                raise ValueError(
                    "First argument must be TensorLike or TableLike instance"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to decrypt data: {e}") from e
