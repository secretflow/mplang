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

"""Tests for crypto backend implementation (Interpreter mode execution).

NOTE: Dialect type tests are in tests/mplang.v2/dialects/test_crypto.py
"""

import numpy as np
import pytest

import mplang.v2.backends.crypto_impl  # noqa: F401 - Register implementations
import mplang.v2.edsl.typing as elt
from mplang.v2.backends.crypto_impl import (
    BytesValue,
    PrivateKeyValue,
    PublicKeyValue,
    SymmetricKeyValue,
)
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import crypto, tensor
from mplang.v2.runtime.interpreter import Interpreter


def _unwrap(val):
    """Unwrap Value subclass to raw data."""
    if isinstance(val, (TensorValue, BytesValue)):
        return val.unwrap()
    return val


class TestKEMExecution:
    """Test KEM execution with interpreter."""

    def test_kem_keygen_execution(self):
        """Test kem_keygen generates valid key pair."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")

            # Check types
            assert isinstance(sk.runtime_obj, PrivateKeyValue)
            assert isinstance(pk.runtime_obj, PublicKeyValue)

            # Check suite
            assert sk.runtime_obj.suite == "x25519"
            assert pk.runtime_obj.suite == "x25519"

            # Check key sizes (X25519 uses 32-byte keys)
            assert len(sk.runtime_obj.key_bytes) == 32
            assert len(pk.runtime_obj.key_bytes) == 32

    def test_kem_keygen_different_keys(self):
        """Test kem_keygen generates different keys each time."""
        with Interpreter():
            sk1, pk1 = crypto.kem_keygen("x25519")
            sk2, pk2 = crypto.kem_keygen("x25519")

            # Keys should be different
            assert sk1.runtime_obj.key_bytes != sk2.runtime_obj.key_bytes
            assert pk1.runtime_obj.key_bytes != pk2.runtime_obj.key_bytes

    def test_kem_derive_execution(self):
        """Test kem_derive produces symmetric key."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            symmetric_key = crypto.kem_derive(sk, pk)

            # Check type
            assert isinstance(symmetric_key.runtime_obj, SymmetricKeyValue)
            assert symmetric_key.runtime_obj.suite == "x25519"

            # Check key size
            assert len(symmetric_key.runtime_obj.key_bytes) == 32

    def test_kem_derive_ecdh_property(self):
        """Test ECDH key exchange: both parties derive same key."""
        with Interpreter():
            # Alice and Bob generate key pairs
            alice_sk, alice_pk = crypto.kem_keygen("x25519")
            bob_sk, bob_pk = crypto.kem_keygen("x25519")

            # Each party derives symmetric key using their sk and other's pk
            alice_key = crypto.kem_derive(alice_sk, bob_pk)
            bob_key = crypto.kem_derive(bob_sk, alice_pk)

            # Both should derive the same key (ECDH property)
            assert alice_key.runtime_obj.key_bytes == bob_key.runtime_obj.key_bytes


class TestSymmetricEncryption:
    """Test symmetric encryption with KEM-derived keys."""

    def test_encrypt_decrypt_basic(self):
        """Test basic encrypt/decrypt roundtrip."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            key = crypto.kem_derive(sk, pk)

            # Create message using tensor.constant
            message = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
            message_obj = tensor.constant(message)

            # Encrypt
            ciphertext = crypto.sym_encrypt(key, message_obj)
            assert len(_unwrap(ciphertext.runtime_obj)) > len(
                message
            )  # Ciphertext includes nonce

            # Decrypt
            plaintext = crypto.sym_decrypt(
                key, ciphertext, elt.TensorType(elt.u8, (5,))
            )
            np.testing.assert_array_equal(_unwrap(plaintext.runtime_obj), message)

    def test_encrypt_decrypt_string_message(self):
        """Test encrypt/decrypt with string-like message."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            key = crypto.kem_derive(sk, pk)

            # "Hello" as bytes, using tensor.constant
            message = np.array([72, 101, 108, 108, 111], dtype=np.uint8)
            message_obj = tensor.constant(message)

            ciphertext = crypto.sym_encrypt(key, message_obj)
            plaintext = crypto.sym_decrypt(
                key, ciphertext, elt.TensorType(elt.u8, (5,))
            )

            assert bytes(_unwrap(plaintext.runtime_obj)).decode() == "Hello"

    def test_encrypt_decrypt_different_keys_fail(self):
        """Test decryption with wrong key fails."""
        with Interpreter():
            # Generate two different key pairs
            sk1, pk1 = crypto.kem_keygen("x25519")
            sk2, pk2 = crypto.kem_keygen("x25519")

            key1 = crypto.kem_derive(sk1, pk1)
            key2 = crypto.kem_derive(sk2, pk2)

            # Encrypt with key1
            message = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
            message_obj = tensor.constant(message)
            ciphertext = crypto.sym_encrypt(key1, message_obj)

            # Decrypt with key2 should fail
            with pytest.raises(Exception):  # AES-GCM will raise on auth failure
                crypto.sym_decrypt(key2, ciphertext, elt.TensorType(elt.u8, (5,)))

    def test_encrypt_produces_different_ciphertext(self):
        """Test encryption with same key produces different ciphertext (due to nonce)."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            key = crypto.kem_derive(sk, pk)

            message = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
            message_obj = tensor.constant(message)

            ct1 = crypto.sym_encrypt(key, message_obj)
            ct2 = crypto.sym_encrypt(key, message_obj)

            # Ciphertexts should be different due to random nonce
            assert _unwrap(ct1.runtime_obj) != _unwrap(ct2.runtime_obj)

    def test_encrypt_decrypt_keyword_only_params(self):
        """Test that algo parameter is properly passed as keyword-only."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            key = crypto.kem_derive(sk, pk)

            message = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
            message_obj = tensor.constant(message)

            # Test keyword-only algo parameter in encrypt
            ciphertext = crypto.sym_encrypt(key, message_obj, algo="aes-gcm")
            assert isinstance(ciphertext.runtime_obj, BytesValue)

            # Test keyword-only parameters in decrypt (target_type and algo)
            plaintext = crypto.sym_decrypt(
                key,
                ciphertext,
                target_type=elt.TensorType(elt.u8, (5,)),
                algo="aes-gcm",
            )
            np.testing.assert_array_equal(_unwrap(plaintext.runtime_obj), message)

    def test_encrypt_decrypt_default_algo(self):
        """Test that default algo parameter works correctly."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            key = crypto.kem_derive(sk, pk)

            message = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
            message_obj = tensor.constant(message)

            # Should work without explicit algo (defaults to "aes-gcm")
            ciphertext = crypto.sym_encrypt(key, message_obj)
            plaintext = crypto.sym_decrypt(
                key, ciphertext, target_type=elt.TensorType(elt.u8, (5,))
            )
            np.testing.assert_array_equal(_unwrap(plaintext.runtime_obj), message)


class TestDigitalEnvelope:
    """Test complete digital envelope workflow."""

    def test_digital_envelope_alice_to_bob(self):
        """Test digital envelope: Alice sends encrypted message to Bob."""
        with Interpreter():
            # Setup: Both parties generate key pairs
            alice_sk, alice_pk = crypto.kem_keygen("x25519")
            bob_sk, bob_pk = crypto.kem_keygen("x25519")

            # Alice wants to send a message to Bob
            # Step 1: Alice derives symmetric key using her sk and Bob's pk
            alice_key = crypto.kem_derive(alice_sk, bob_pk)

            # Step 2: Alice encrypts the message using tensor.constant
            secret_message = np.array(
                [83, 101, 99, 114, 101, 116], dtype=np.uint8
            )  # "Secret"
            msg_obj = tensor.constant(secret_message)
            ciphertext = crypto.sym_encrypt(alice_key, msg_obj)

            # Alice sends (alice_pk, ciphertext) to Bob

            # Step 3: Bob derives the same symmetric key
            bob_key = crypto.kem_derive(bob_sk, alice_pk)

            # Step 4: Bob decrypts
            plaintext = crypto.sym_decrypt(
                bob_key, ciphertext, elt.TensorType(elt.u8, (6,))
            )

            # Verify
            np.testing.assert_array_equal(
                _unwrap(plaintext.runtime_obj), secret_message
            )
            assert bytes(_unwrap(plaintext.runtime_obj)).decode() == "Secret"

    def test_bidirectional_communication(self):
        """Test bidirectional encrypted communication."""
        with Interpreter():
            # Both parties generate key pairs
            alice_sk, alice_pk = crypto.kem_keygen("x25519")
            bob_sk, bob_pk = crypto.kem_keygen("x25519")

            # Derive shared key (same for both due to ECDH)
            alice_key = crypto.kem_derive(alice_sk, bob_pk)
            bob_key = crypto.kem_derive(bob_sk, alice_pk)

            # Alice sends to Bob using tensor.constant
            msg_a = np.array([65, 66, 67], dtype=np.uint8)  # "ABC"
            msg_a_obj = tensor.constant(msg_a)
            ct_a = crypto.sym_encrypt(alice_key, msg_a_obj)
            pt_a = crypto.sym_decrypt(bob_key, ct_a, elt.TensorType(elt.u8, (3,)))
            np.testing.assert_array_equal(_unwrap(pt_a.runtime_obj), msg_a)

            # Bob sends to Alice using tensor.constant
            msg_b = np.array([88, 89, 90], dtype=np.uint8)  # "XYZ"
            msg_b_obj = tensor.constant(msg_b)
            ct_b = crypto.sym_encrypt(bob_key, msg_b_obj)
            pt_b = crypto.sym_decrypt(alice_key, ct_b, elt.TensorType(elt.u8, (3,)))
            np.testing.assert_array_equal(_unwrap(pt_b.runtime_obj), msg_b)


class TestHKDF:
    """Test HKDF key derivation function (RFC 5869, NIST SP 800-56C)."""

    def test_hkdf_basic(self):
        """Test basic HKDF derivation from SymmetricKeyValue."""
        with Interpreter():
            # Generate ECDH shared secret
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            # Derive key with HKDF
            derived_key = crypto.hkdf(shared_secret, "test/context/v1")

            # Verify type and length
            assert isinstance(derived_key.runtime_obj, SymmetricKeyValue)
            assert len(derived_key.runtime_obj.key_bytes) == 32  # AES-256

    def test_hkdf_suite_format(self):
        """Test that suite follows 'hkdf-{hash_algo}' format."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            # Default SHA-256
            key1 = crypto.hkdf(shared_secret, "test/info")
            assert key1.runtime_obj.suite == "hkdf-sha256"

            # Explicit SHA-256
            key2 = crypto.hkdf(shared_secret, "test/info", hash_algo="sha256")
            assert key2.runtime_obj.suite == "hkdf-sha256"

    def test_hkdf_different_info(self):
        """Test domain separation: different info → different keys."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            key1 = crypto.hkdf(shared_secret, "context-A")
            key2 = crypto.hkdf(shared_secret, "context-B")

            # Must be different due to different info
            assert key1.runtime_obj.key_bytes != key2.runtime_obj.key_bytes

    def test_hkdf_deterministic(self):
        """Test determinism: same input → same output."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            key1 = crypto.hkdf(shared_secret, "test/info")
            key2 = crypto.hkdf(shared_secret, "test/info")

            # Identical input must produce identical output
            assert key1.runtime_obj.key_bytes == key2.runtime_obj.key_bytes

    def test_hkdf_from_tensor_value(self):
        """Test HKDF accepts TensorValue (raw bytes) as input."""
        with Interpreter():
            import os

            # Create 32-byte secret as TensorValue
            secret_bytes = os.urandom(32)
            secret_tensor = tensor.constant(np.frombuffer(secret_bytes, dtype=np.uint8))

            # Derive key from raw bytes
            derived_key = crypto.hkdf(secret_tensor, info="test/raw-bytes")

            assert isinstance(derived_key.runtime_obj, SymmetricKeyValue)
            assert len(derived_key.runtime_obj.key_bytes) == 32
            assert derived_key.runtime_obj.suite == "hkdf-sha256"

    def test_hkdf_empty_info_runtime_error(self):
        """Test that empty info raises ValueError at runtime (if it bypasses trace-time check)."""
        import pytest

        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            # This should be caught at trace time, but test runtime as well
            with pytest.raises(ValueError, match="non-empty 'info' parameter"):
                crypto.hkdf(shared_secret, info="")

    def test_hkdf_unsupported_hash_algo(self):
        """Test that unsupported hash algorithms raise NotImplementedError."""
        import pytest

        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            with pytest.raises(
                NotImplementedError,
                match="hash algorithm 'sha512' is not yet implemented",
            ):
                crypto.hkdf(shared_secret, info="test/info", hash_algo="sha512")

    def test_hkdf_hash_algo_normalization(self):
        """Test that hash_algo is normalized correctly (lowercase, no hyphens)."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            # All these should normalize to "sha256" and produce same result
            key1 = crypto.hkdf(shared_secret, info="test", hash_algo="SHA-256")
            key2 = crypto.hkdf(shared_secret, info="test", hash_algo="sha_256")
            key3 = crypto.hkdf(shared_secret, info="test", hash_algo="Sha256")

            # All should have same suite after normalization
            assert key1.runtime_obj.suite == "hkdf-sha256"
            assert key2.runtime_obj.suite == "hkdf-sha256"
            assert key3.runtime_obj.suite == "hkdf-sha256"

            # Keys should be identical (deterministic with same inputs)
            assert key1.runtime_obj.key_bytes == key2.runtime_obj.key_bytes
            assert key2.runtime_obj.key_bytes == key3.runtime_obj.key_bytes

    def test_hkdf_keyword_only_params(self):
        """Test that info and hash_algo are properly passed as keyword arguments."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            # Verify keyword-only parameter passing works
            key = crypto.hkdf(shared_secret, info="mplang/device/tee/v2")
            assert isinstance(key.runtime_obj, SymmetricKeyValue)
            assert key.runtime_obj.suite == "hkdf-sha256"

            # Verify with explicit hash_algo
            key2 = crypto.hkdf(
                shared_secret, info="mplang/device/tee/v2", hash_algo="sha256"
            )
            assert key2.runtime_obj.suite == "hkdf-sha256"

            # Keys with same parameters should match
            assert key.runtime_obj.key_bytes == key2.runtime_obj.key_bytes

    def test_hkdf_with_sym_encrypt(self):
        """Test HKDF-derived key works with symmetric encryption (roundtrip)."""
        with Interpreter():
            # Derive session key via HKDF
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)
            session_key = crypto.hkdf(shared_secret, "test/encryption/v1")

            # Encrypt message
            message = tensor.constant(np.array([1, 2, 3, 4, 5], dtype=np.uint8))
            ciphertext = crypto.sym_encrypt(session_key, message)

            # Decrypt message
            decrypted = crypto.sym_decrypt(
                session_key, ciphertext, elt.TensorType(elt.u8, (5,))
            )

            # Verify roundtrip correctness
            np.testing.assert_array_equal(
                _unwrap(decrypted.runtime_obj), _unwrap(message.runtime_obj)
            )

    def test_hkdf_empty_info_error(self):
        """Test that empty info string raises ValueError."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            with pytest.raises(ValueError, match="non-empty 'info' parameter"):
                crypto.hkdf(shared_secret, "")

    def test_hkdf_unsupported_hash_error(self):
        """Test that unsupported hash algorithm raises NotImplementedError at runtime."""
        with Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            shared_secret = crypto.kem_derive(sk, pk)

            # Unsupported hash algorithm is caught at runtime (execution time)
            with pytest.raises(NotImplementedError, match="not yet implemented"):
                crypto.hkdf(shared_secret, info="test/info", hash_algo="sha512")

    def test_hkdf_tee_session_scenario(self):
        """Test complete TEE session establishment scenario (ECDH + HKDF)."""
        with Interpreter():
            # Simulate two-party key agreement (PPU ↔ TEE)
            # Party A (PPU)
            sk_a, pk_a = crypto.kem_keygen("x25519")
            # Party B (TEE)
            sk_b, pk_b = crypto.kem_keygen("x25519")

            # Both sides perform ECDH
            shared_a = crypto.kem_derive(sk_a, pk_b)  # A derives with B's pk
            shared_b = crypto.kem_derive(sk_b, pk_a)  # B derives with A's pk

            # Both sides derive session key with HKDF using same info
            sess_a = crypto.hkdf(shared_a, "mplang/device/tee/v2")
            sess_b = crypto.hkdf(shared_b, "mplang/device/tee/v2")

            # Session keys must be identical (same ECDH secret + same info)
            assert sess_a.runtime_obj.key_bytes == sess_b.runtime_obj.key_bytes
            assert sess_a.runtime_obj.suite == "hkdf-sha256"
            assert sess_b.runtime_obj.suite == "hkdf-sha256"


class TestAlgoParameter:
    """Test algo parameter validation in sym_encrypt/sym_decrypt."""

    def setup_method(self):
        self.interpreter = Interpreter()
        self.interpreter.__enter__()
        sk, pk = crypto.kem_keygen("x25519")
        self.key = crypto.kem_derive(sk, pk)

    def teardown_method(self):
        self.interpreter.__exit__(None, None, None)

    def test_default_algo_works(self):
        """Test default algo='aes-gcm' works without explicit parameter."""

        message = np.array([1, 2, 3], dtype=np.uint8)
        message_obj = tensor.constant(message)

        # Use default algo (should be aes-gcm)
        ciphertext = crypto.sym_encrypt(self.key, message_obj)
        plaintext = crypto.sym_decrypt(
            self.key, ciphertext, elt.TensorType(elt.u8, (3,))
        )
        np.testing.assert_array_equal(_unwrap(plaintext.runtime_obj), message)

    def test_explicit_aes_gcm_algo(self):
        """Test explicitly passing algo='aes-gcm' works."""

        message = np.array([4, 5, 6], dtype=np.uint8)
        message_obj = tensor.constant(message)

        # Explicitly pass algo='aes-gcm'
        ciphertext = crypto.sym_encrypt(self.key, message_obj, algo="aes-gcm")
        plaintext = crypto.sym_decrypt(
            self.key, ciphertext, elt.TensorType(elt.u8, (3,)), algo="aes-gcm"
        )
        np.testing.assert_array_equal(_unwrap(plaintext.runtime_obj), message)

    def test_unsupported_algo_encrypt_fails(self):
        """Test that unsupported algo raises ValueError during encryption."""

        message = np.array([7, 8, 9], dtype=np.uint8)
        message_obj = tensor.constant(message)

        # Try unsupported algorithm
        with pytest.raises(ValueError, match="Unsupported encryption algorithm"):
            crypto.sym_encrypt(self.key, message_obj, algo="aes-ctr")

    def test_unsupported_algo_decrypt_fails(self):
        """Test that unsupported algo raises ValueError during decryption."""

        message = np.array([10, 11, 12], dtype=np.uint8)
        message_obj = tensor.constant(message)

        # Encrypt with default algo
        ciphertext = crypto.sym_encrypt(self.key, message_obj)

        # Try to decrypt with unsupported algorithm
        with pytest.raises(ValueError, match="Unsupported decryption algorithm"):
            crypto.sym_decrypt(
                self.key, ciphertext, elt.TensorType(elt.u8, (3,)), algo="sm4-gcm"
            )
