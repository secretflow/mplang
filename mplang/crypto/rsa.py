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

"""RSA encryption utilities for TensorLike and TableLike data."""

import base64
from typing import Any

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)

from mplang.core.table import TableLike
from mplang.core.tensor import TensorLike

__all__ = ["EncryptedTable", "EncryptedTensor", "RSAEncryptor"]


class EncryptedTensor:
    """Container for encrypted tensor data."""

    def __init__(
        self,
        encrypted_data: list[str],
        original_shape: tuple[int, ...],
        original_dtype: str,
    ):
        self.encrypted_data = encrypted_data
        self.original_shape = original_shape
        self.original_dtype = original_dtype

    @property
    def dtype(self) -> str:
        """Implement TensorLike protocol - returns the original dtype."""
        return self.original_dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Implement TensorLike protocol - returns the original shape."""
        return self.original_shape

    def __repr__(self) -> str:
        return (
            f"EncryptedTensor(shape={self.original_shape}, dtype={self.original_dtype})"
        )


class EncryptedTable:
    """Container for encrypted table data."""

    def __init__(
        self,
        encrypted_data: dict[str, list[str]],
        original_columns: list[str],
        original_dtypes: dict[str, str],
    ):
        self.encrypted_data = encrypted_data
        self.original_columns = original_columns
        self.original_dtypes = original_dtypes

    @property
    def dtypes(self) -> dict[str, str]:
        """Implement TableLike protocol - returns the original dtypes."""
        return self.original_dtypes

    @property
    def columns(self) -> list[str]:
        """Implement TableLike protocol - returns the original columns."""
        return self.original_columns

    def __repr__(self) -> str:
        return f"EncryptedTable(columns={self.original_columns}, rows={len(next(iter(self.encrypted_data.values()))) if self.encrypted_data else 0})"


class RSAEncryptor:
    """RSA encryption utilities for tensor and table data."""

    def __init__(self, key_size: int = 2048):
        """
        Initialize RSA encryptor with given key size.

        Args:
            key_size: RSA key size in bits (default: 2048)
        """
        self.key_size = key_size
        self._private_key: rsa.RSAPrivateKey | None = None
        self._public_key: rsa.RSAPublicKey | None = None

    def generate_keys(self) -> tuple[bytes, bytes]:
        """
        Generate new RSA key pair.

        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        self._private_key = private_key
        self._public_key = private_key.public_key()

        return private_pem, public_pem

    def load_keys(
        self,
        private_key_pem: str | bytes | None = None,
        public_key_pem: str | bytes | None = None,
    ) -> None:
        """
        Load RSA keys from PEM format strings.

        Args:
            private_key_pem: Private key in PEM format as string
            public_key_pem: Public key in PEM format as string

        Example:
            >>> private_pem = (
            ...     "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"
            ... )
            >>> public_pem = "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
            >>> encryptor.load_keys(
            ...     private_key_pem=private_pem, public_key_pem=public_pem
            ... )
        """
        if private_key_pem:
            private_key_data = (
                private_key_pem.encode("utf-8") if isinstance(private_key_pem, str) else private_key_pem
            )
            private_key = load_pem_private_key(private_key_data, password=None)
            if not isinstance(private_key, rsa.RSAPrivateKey):
                raise ValueError("Private key is not an RSA key")
            self._private_key = private_key
            self._public_key = private_key.public_key()

        if public_key_pem and not self._public_key:
            public_key_data = (
                public_key_pem.encode("utf-8") if isinstance(public_key_pem, str) else public_key_pem
            )
            public_key = load_pem_public_key(public_key_data)
            if not isinstance(public_key, rsa.RSAPublicKey):
                raise ValueError("Public key is not an RSA key")
            self._public_key = public_key

    def load_pem_file(
        self, private_key_file: str | None = None, public_key_file: str | None = None
    ) -> None:
        """
        Load RSA keys from PEM files.

        Args:
            private_key_file: Path to private key PEM file
            public_key_file: Path to public key PEM file
        """
        if private_key_file:
            with open(private_key_file, "rb") as f:
                self.load_keys(private_key_pem=f.read())

        if public_key_file and not self._public_key:
            with open(public_key_file, "rb") as f:
                self.load_keys(public_key_pem=f.read())

    def save_pem_file(
        self, private_key_file: str | None = None, public_key_file: str | None = None
    ) -> None:
        """
        Save RSA keys to PEM files.

        Args:
            private_key_file: Path to save private key PEM file
            public_key_file: Path to save public key PEM file
        """
        if private_key_file and self._private_key:
            private_pem = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            with open(private_key_file, "wb") as f:
                f.write(private_pem)

        if public_key_file and self._public_key:
            public_pem = self._public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            with open(public_key_file, "wb") as f:
                f.write(public_pem)

    def _encrypt_chunk(self, data: bytes) -> str:
        """Encrypt a single chunk of data."""
        if not self._public_key:
            raise ValueError("Public key not loaded")

        encrypted = self._public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return base64.b64encode(encrypted).decode("utf-8")

    def _decrypt_chunk(self, encrypted_data: str) -> bytes:
        """Decrypt a single chunk of data."""
        if not self._private_key:
            raise ValueError("Private key not loaded")

        encrypted_bytes = base64.b64decode(encrypted_data.encode("utf-8"))
        return self._private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def encrypt_tensor(self, tensor: TensorLike) -> EncryptedTensor:
        """
        Encrypt tensor data using RSA.

        Args:
            tensor: TensorLike object to encrypt

        Returns:
            EncryptedTensor containing encrypted data
        """
        if not self._public_key:
            raise ValueError("Public key not loaded")

        # Convert tensor to numpy array
        if hasattr(tensor, "numpy"):
            np_array = tensor.numpy()
        else:
            np_array = np.asarray(tensor)

        # Flatten and serialize
        flat_data = np_array.flatten()
        serialized = flat_data.astype(np.float64).tobytes()

        # Split into chunks for RSA encryption (max 190 bytes for 2048-bit key)
        chunk_size = (self.key_size // 8) - 2 * 32 - 2
        encrypted_chunks = []

        for i in range(0, len(serialized), chunk_size):
            chunk = serialized[i : i + chunk_size]
            encrypted_chunks.append(self._encrypt_chunk(chunk))

        return EncryptedTensor(
            encrypted_data=encrypted_chunks,
            original_shape=np_array.shape,
            original_dtype=str(np_array.dtype),
        )

    def decrypt_tensor(self, encrypted_tensor: EncryptedTensor) -> np.ndarray:
        """
        Decrypt tensor data using RSA.

        Args:
            encrypted_tensor: EncryptedTensor to decrypt

        Returns:
            Decrypted numpy array
        """
        if not self._private_key:
            raise ValueError("Private key not loaded")

        # Decrypt all chunks
        decrypted_bytes = b"".join(
            self._decrypt_chunk(chunk) for chunk in encrypted_tensor.encrypted_data
        )

        # Convert back to numpy array
        flat_data = np.frombuffer(decrypted_bytes, dtype=np.float64)

        # Convert to original dtype and reshape
        original_dtype = np.dtype(encrypted_tensor.original_dtype)
        result = flat_data.astype(original_dtype)

        return result.reshape(encrypted_tensor.original_shape)

    def encrypt_table(self, table: TableLike) -> EncryptedTable:
        """
        Encrypt table data using RSA.

        Args:
            table: TableLike object to encrypt

        Returns:
            EncryptedTable containing encrypted data
        """
        if not self._public_key:
            raise ValueError("Public key not loaded")

        # Convert to pandas DataFrame
        if hasattr(table, "to_pandas"):
            df = table.to_pandas()
        else:
            df = pd.DataFrame(table)

        encrypted_data = {}
        original_dtypes = {}

        # Encrypt each column separately
        for column in df.columns:
            series = df[column]
            original_dtypes[str(column)] = str(series.dtype)

            # Handle different data types
            if pd.api.types.is_numeric_dtype(series.dtype):
                data = series.astype(float).values
                serialized = data.astype(np.float64).tobytes()
            else:
                # For string/object types, encode as UTF-8
                data = series.astype(str).values
                serialized = "|".join(data).encode("utf-8")

            # Split into chunks for RSA encryption
            chunk_size = 190
            encrypted_chunks = []

            for i in range(0, len(serialized), chunk_size):
                chunk = serialized[i : i + chunk_size]
                encrypted_chunks.append(self._encrypt_chunk(chunk))

            encrypted_data[str(column)] = encrypted_chunks

        return EncryptedTable(
            encrypted_data=encrypted_data,
            original_columns=list(df.columns),
            original_dtypes=original_dtypes,
        )

    def decrypt_table(self, encrypted_table: EncryptedTable) -> pd.DataFrame:
        """
        Decrypt table data using RSA.

        Args:
            encrypted_table: EncryptedTable to decrypt

        Returns:
            Decrypted pandas DataFrame
        """
        if not self._private_key:
            raise ValueError("Private key not loaded")

        decrypted_columns = {}

        for column, encrypted_chunks in encrypted_table.encrypted_data.items():
            # Decrypt all chunks
            decrypted_bytes = b"".join(
                self._decrypt_chunk(chunk) for chunk in encrypted_chunks
            )

            original_dtype = encrypted_table.original_dtypes[column]

            if pd.api.types.is_numeric_dtype(np.dtype(original_dtype)):
                # Numeric data
                flat_data = np.frombuffer(decrypted_bytes, dtype=np.float64)
                decrypted_columns[column] = flat_data.astype(original_dtype)
            else:
                # String data
                decoded = decrypted_bytes.decode("utf-8")
                values = decoded.split("|")
                decrypted_columns[column] = pd.Series(values, dtype=original_dtype)

        return pd.DataFrame(decrypted_columns)

    def export_keys(self) -> dict[str, str]:
        """Export keys as base64 strings for storage/transmission."""
        keys = {}

        if self._private_key:
            private_pem = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            keys["private_key"] = base64.b64encode(private_pem).decode("utf-8")

        if self._public_key:
            public_pem = self._public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            keys["public_key"] = base64.b64encode(public_pem).decode("utf-8")

        return keys

    def import_keys(self, keys: dict[str, str]) -> None:
        """Import keys from base64 strings."""
        if "private_key" in keys:
            private_pem = base64.b64decode(keys["private_key"].encode("utf-8"))
            private_key = load_pem_private_key(private_pem, password=None)
            if not isinstance(private_key, rsa.RSAPrivateKey):
                raise ValueError("Private key is not an RSA key")
            self._private_key = private_key
            self._public_key = private_key.public_key()

        if "public_key" in keys and not self._private_key:
            public_key = load_pem_public_key(
                base64.b64decode(keys["public_key"].encode("utf-8"))
            )
            if not isinstance(public_key, rsa.RSAPublicKey):
                raise ValueError("Public key is not an RSA key")
            self._public_key = public_key
