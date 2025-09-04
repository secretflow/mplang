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

import binascii

import numpy as np
import pandas as pd
import pytest

from mplang.crypto.rsa import EncryptedTensor, RSAEncryptor


class TestRSAEncryption:
    def test_tensor_encryption(self) -> None:
        """Test RSA encryption for tensor data."""
        # Create test data
        tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

        # Create encryptor and generate keys
        encryptor = RSAEncryptor(key_size=2048)
        _private_key, _public_key = encryptor.generate_keys()

        # Test encryption
        encrypted = encryptor.encrypt_tensor(tensor)
        assert str(encrypted.original_shape) == str(tensor.shape)
        assert encrypted.original_dtype == "float32"

        # Test decryption
        decrypted = encryptor.decrypt_tensor(encrypted)
        np.testing.assert_array_equal(decrypted, tensor)

    def test_table_encryption(self) -> None:
        """Test RSA encryption for table data."""
        # Create test data
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4],
            "float_col": [1.1, 2.2, 3.3, 4.4],
            "str_col": ["a", "b", "c", "d"],
        })

        # Create encryptor and generate keys
        encryptor = RSAEncryptor(key_size=2048)
        _private_key, _public_key = encryptor.generate_keys()

        # Test encryption
        encrypted = encryptor.encrypt_table(df)
        assert len(encrypted.original_columns) == 3
        assert "int_col" in encrypted.original_dtypes
        assert "float_col" in encrypted.original_dtypes
        assert "str_col" in encrypted.original_dtypes

        # Test decryption
        decrypted = encryptor.decrypt_table(encrypted)
        pd.testing.assert_frame_equal(decrypted, df)

    def test_key_export_import(self) -> None:
        """Test key export and import functionality."""
        # Create test data
        tensor = np.array([1, 2, 3, 4, 5])

        # Generate keys
        encryptor1 = RSAEncryptor(key_size=1024)  # Smaller for testing
        encryptor1.generate_keys()

        # Export keys
        keys = encryptor1.export_keys()
        assert "private_key" in keys
        assert "public_key" in keys

        # Import keys into new encryptor
        encryptor2 = RSAEncryptor()
        encryptor2.import_keys(keys)

        # Test encryption/decryption with imported keys
        encrypted = encryptor2.encrypt_tensor(tensor)
        decrypted = encryptor2.decrypt_tensor(encrypted)
        np.testing.assert_array_equal(decrypted, tensor)

    def test_encryption_without_keys(self) -> None:
        """Test error handling for encryption without keys."""
        tensor = np.array([1, 2, 3])
        encryptor = RSAEncryptor()

        with pytest.raises(ValueError, match="Public key not loaded"):
            encryptor.encrypt_tensor(tensor)

    def test_decryption_without_keys(self) -> None:
        """Test error handling for decryption without keys."""
        tensor = np.array([1, 2, 3])
        encryptor = RSAEncryptor(key_size=1024)
        encryptor.generate_keys()

        encrypted = encryptor.encrypt_tensor(tensor)

        # Create new encryptor without keys
        encryptor2 = RSAEncryptor()
        with pytest.raises(ValueError, match="Private key not loaded"):
            encryptor2.decrypt_tensor(encrypted)

    def test_type_mismatch_error(self) -> None:
        """Test error handling for type mismatches."""
        tensor = np.array([1, 2, 3])
        df = pd.DataFrame({"a": [1, 2, 3]})

        encryptor = RSAEncryptor(key_size=1024)
        encryptor.generate_keys()

        # Encrypt tensor
        encrypted_tensor = encryptor.encrypt_tensor(tensor)

        # Try to decrypt tensor data as table (should fail)
        with pytest.raises((AttributeError, ValueError, binascii.Error)):
            encryptor.decrypt_table(encrypted_tensor)

        # Encrypt table
        encrypted_table = encryptor.encrypt_table(df)

        # Try to decrypt table data as tensor (should fail)
        with pytest.raises((AttributeError, ValueError, binascii.Error)):
            encryptor.decrypt_tensor(encrypted_table)

    def test_medium_tensor_encryption(self) -> None:
        """Test encryption of medium-sized tensors."""
        # Create medium tensor (small enough for RSA)
        tensor = np.random.randn(5, 3).astype(np.float32)

        encryptor = RSAEncryptor(key_size=2048)
        encryptor.generate_keys()

        # Test encryption/decryption
        encrypted = encryptor.encrypt_tensor(tensor)
        decrypted = encryptor.decrypt_tensor(encrypted)

        np.testing.assert_array_equal(decrypted, tensor)

    def test_mixed_dtype_table_encryption(self) -> None:
        """Test encryption of tables with mixed data types."""
        df = pd.DataFrame({
            "int8_col": np.array([1, 2, 3], dtype=np.int8),
            "float64_col": np.array([1.1, 2.2, 3.3], dtype=np.float64),
            "bool_col": [True, False, True],
            "str_col": ["hello", "world", "test"],
        })

        encryptor = RSAEncryptor(key_size=1024)
        encryptor.generate_keys()

        # Test encryption/decryption
        encrypted = encryptor.encrypt_table(df)
        decrypted = encryptor.decrypt_table(encrypted)

        pd.testing.assert_frame_equal(decrypted, df)

    def test_empty_data_encryption(self) -> None:
        """Test encryption of empty data structures."""
        # Empty tensor
        empty_tensor = np.array([])

        # Empty DataFrame
        empty_df = pd.DataFrame()

        encryptor = RSAEncryptor(key_size=1024)
        encryptor.generate_keys()

        # Test empty tensor
        encrypted_tensor = encryptor.encrypt_tensor(empty_tensor)
        decrypted_tensor = encryptor.decrypt_tensor(encrypted_tensor)
        np.testing.assert_array_equal(decrypted_tensor, empty_tensor)

        # Test empty DataFrame
        encrypted_df = encryptor.encrypt_table(empty_df)
        decrypted_df = encryptor.decrypt_table(encrypted_df)
        pd.testing.assert_frame_equal(decrypted_df, empty_df)

    def test_rsa_integration(self) -> None:
        """Test full integration with RSA encryption."""
        # Test with tensor
        tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)

        encryptor = RSAEncryptor(key_size=1024)
        encryptor.generate_keys()

        # Encrypt
        encrypted = encryptor.encrypt_tensor(tensor)
        assert isinstance(encrypted, EncryptedTensor)

        # Decrypt
        decrypted = encryptor.decrypt_tensor(encrypted)
        np.testing.assert_array_equal(decrypted, tensor)

        # Test with table
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})

        encrypted2 = encryptor.encrypt_table(df)
        decrypted2 = encryptor.decrypt_table(encrypted2)
        pd.testing.assert_frame_equal(decrypted2, df)

    def test_pem_file_operations(self) -> None:
        """Test PEM file import/export operations."""
        import os
        import tempfile

        # Create test data
        tensor = np.array([1, 2, 3])

        with tempfile.TemporaryDirectory() as temp_dir:
            private_file = os.path.join(temp_dir, "private.pem")
            public_file = os.path.join(temp_dir, "public.pem")

            # Generate and save keys
            encryptor1 = RSAEncryptor(key_size=1024)
            encryptor1.generate_keys()
            encryptor1.save_pem_file(private_file, public_file)

            # Verify files exist
            assert os.path.exists(private_file)
            assert os.path.exists(public_file)

            # Load keys from files
            encryptor2 = RSAEncryptor()
            encryptor2.load_pem_file(private_file, public_file)

            # Test encryption/decryption
            encrypted = encryptor2.encrypt_tensor(tensor)
            decrypted = encryptor2.decrypt_tensor(encrypted)
            np.testing.assert_array_equal(decrypted, tensor)

    def test_pem_string_usage(self) -> None:
        """Test direct PEM string usage."""
        tensor = np.array([1, 2, 3])

        # Generate keys
        encryptor1 = RSAEncryptor(key_size=1024)
        private_pem_bytes, public_pem_bytes = encryptor1.generate_keys()

        # Convert to strings for string usage test
        private_pem_str = private_pem_bytes.decode("utf-8")
        public_pem_str = public_pem_bytes.decode("utf-8")

        # Use PEM strings directly
        encryptor2 = RSAEncryptor()
        encryptor2.load_keys(
            private_key_pem=private_pem_str, public_key_pem=public_pem_str
        )

        # Test functionality
        encrypted = encryptor2.encrypt_tensor(tensor)
        decrypted = encryptor2.decrypt_tensor(encrypted)
        np.testing.assert_array_equal(decrypted, tensor)

    def test_non_rsa_key_error(self) -> None:
        """Test error handling for non-RSA keys."""
        encryptor = RSAEncryptor()

        # Test with invalid private key
        invalid_private = (
            "-----BEGIN PRIVATE KEY-----\nINVALID\n-----END PRIVATE KEY-----"
        )
        with pytest.raises(Exception):
            encryptor.load_keys(private_key_pem=invalid_private)

        # Test with invalid public key
        invalid_public = "-----BEGIN PUBLIC KEY-----\nINVALID\n-----END PUBLIC KEY-----"
        with pytest.raises(Exception):
            encryptor.load_keys(public_key_pem=invalid_public)
