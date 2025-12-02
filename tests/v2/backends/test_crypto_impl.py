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
import mplang.v2.dialects.crypto as crypto
import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.backends.crypto_impl import (
    RuntimePrivateKey,
    RuntimePublicKey,
    RuntimeSymmetricKey,
)


class TestKEMExecution:
    """Test KEM execution with interpreter."""

    def test_kem_keygen_execution(self):
        """Test kem_keygen generates valid key pair."""
        with el.Interpreter():
            sk, pk = crypto.kem_keygen("x25519")

            # Check types
            assert isinstance(sk.runtime_obj, RuntimePrivateKey)
            assert isinstance(pk.runtime_obj, RuntimePublicKey)

            # Check suite
            assert sk.runtime_obj.suite == "x25519"
            assert pk.runtime_obj.suite == "x25519"

            # Check key sizes (X25519 uses 32-byte keys)
            assert len(sk.runtime_obj.key_bytes) == 32
            assert len(pk.runtime_obj.key_bytes) == 32

    def test_kem_keygen_different_keys(self):
        """Test kem_keygen generates different keys each time."""
        with el.Interpreter():
            sk1, pk1 = crypto.kem_keygen("x25519")
            sk2, pk2 = crypto.kem_keygen("x25519")

            # Keys should be different
            assert sk1.runtime_obj.key_bytes != sk2.runtime_obj.key_bytes
            assert pk1.runtime_obj.key_bytes != pk2.runtime_obj.key_bytes

    def test_kem_derive_execution(self):
        """Test kem_derive produces symmetric key."""
        with el.Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            symmetric_key = crypto.kem_derive(sk, pk)

            # Check type
            assert isinstance(symmetric_key.runtime_obj, RuntimeSymmetricKey)
            assert symmetric_key.runtime_obj.suite == "x25519"

            # Check key size
            assert len(symmetric_key.runtime_obj.key_bytes) == 32

    def test_kem_derive_ecdh_property(self):
        """Test ECDH key exchange: both parties derive same key."""
        with el.Interpreter():
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
        with el.Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            key = crypto.kem_derive(sk, pk)

            # Create message
            message = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
            message_obj = el.InterpObject(message, elt.TensorType(elt.u8, (5,)))

            # Encrypt
            ciphertext = crypto.sym_encrypt(key, message_obj)
            assert len(ciphertext.runtime_obj) > len(
                message
            )  # Ciphertext includes nonce

            # Decrypt
            plaintext = crypto.sym_decrypt(
                key, ciphertext, elt.TensorType(elt.u8, (5,))
            )
            np.testing.assert_array_equal(plaintext.runtime_obj, message)

    def test_encrypt_decrypt_string_message(self):
        """Test encrypt/decrypt with string-like message."""
        with el.Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            key = crypto.kem_derive(sk, pk)

            # "Hello" as bytes
            message = np.array([72, 101, 108, 108, 111], dtype=np.uint8)
            message_obj = el.InterpObject(message, elt.TensorType(elt.u8, (5,)))

            ciphertext = crypto.sym_encrypt(key, message_obj)
            plaintext = crypto.sym_decrypt(
                key, ciphertext, elt.TensorType(elt.u8, (5,))
            )

            assert bytes(plaintext.runtime_obj).decode() == "Hello"

    def test_encrypt_decrypt_different_keys_fail(self):
        """Test decryption with wrong key fails."""
        with el.Interpreter():
            # Generate two different key pairs
            sk1, pk1 = crypto.kem_keygen("x25519")
            sk2, pk2 = crypto.kem_keygen("x25519")

            key1 = crypto.kem_derive(sk1, pk1)
            key2 = crypto.kem_derive(sk2, pk2)

            # Encrypt with key1
            message = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
            message_obj = el.InterpObject(message, elt.TensorType(elt.u8, (5,)))
            ciphertext = crypto.sym_encrypt(key1, message_obj)

            # Decrypt with key2 should fail
            with pytest.raises(Exception):  # AES-GCM will raise on auth failure
                crypto.sym_decrypt(key2, ciphertext, elt.TensorType(elt.u8, (5,)))

    def test_encrypt_produces_different_ciphertext(self):
        """Test encryption with same key produces different ciphertext (due to nonce)."""
        with el.Interpreter():
            sk, pk = crypto.kem_keygen("x25519")
            key = crypto.kem_derive(sk, pk)

            message = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
            message_obj = el.InterpObject(message, elt.TensorType(elt.u8, (5,)))

            ct1 = crypto.sym_encrypt(key, message_obj)
            ct2 = crypto.sym_encrypt(key, message_obj)

            # Ciphertexts should be different due to random nonce
            assert not np.array_equal(ct1.runtime_obj, ct2.runtime_obj)


class TestDigitalEnvelope:
    """Test complete digital envelope workflow."""

    def test_digital_envelope_alice_to_bob(self):
        """Test digital envelope: Alice sends encrypted message to Bob."""
        with el.Interpreter():
            # Setup: Both parties generate key pairs
            alice_sk, alice_pk = crypto.kem_keygen("x25519")
            bob_sk, bob_pk = crypto.kem_keygen("x25519")

            # Alice wants to send a message to Bob
            # Step 1: Alice derives symmetric key using her sk and Bob's pk
            alice_key = crypto.kem_derive(alice_sk, bob_pk)

            # Step 2: Alice encrypts the message
            secret_message = np.array(
                [83, 101, 99, 114, 101, 116], dtype=np.uint8
            )  # "Secret"
            msg_obj = el.InterpObject(secret_message, elt.TensorType(elt.u8, (6,)))
            ciphertext = crypto.sym_encrypt(alice_key, msg_obj)

            # Alice sends (alice_pk, ciphertext) to Bob

            # Step 3: Bob derives the same symmetric key
            bob_key = crypto.kem_derive(bob_sk, alice_pk)

            # Step 4: Bob decrypts
            plaintext = crypto.sym_decrypt(
                bob_key, ciphertext, elt.TensorType(elt.u8, (6,))
            )

            # Verify
            np.testing.assert_array_equal(plaintext.runtime_obj, secret_message)
            assert bytes(plaintext.runtime_obj).decode() == "Secret"

    def test_bidirectional_communication(self):
        """Test bidirectional encrypted communication."""
        with el.Interpreter():
            # Both parties generate key pairs
            alice_sk, alice_pk = crypto.kem_keygen("x25519")
            bob_sk, bob_pk = crypto.kem_keygen("x25519")

            # Derive shared key (same for both due to ECDH)
            alice_key = crypto.kem_derive(alice_sk, bob_pk)
            bob_key = crypto.kem_derive(bob_sk, alice_pk)

            # Alice sends to Bob
            msg_a = np.array([65, 66, 67], dtype=np.uint8)  # "ABC"
            msg_a_obj = el.InterpObject(msg_a, elt.TensorType(elt.u8, (3,)))
            ct_a = crypto.sym_encrypt(alice_key, msg_a_obj)
            pt_a = crypto.sym_decrypt(bob_key, ct_a, elt.TensorType(elt.u8, (3,)))
            np.testing.assert_array_equal(pt_a.runtime_obj, msg_a)

            # Bob sends to Alice
            msg_b = np.array([88, 89, 90], dtype=np.uint8)  # "XYZ"
            msg_b_obj = el.InterpObject(msg_b, elt.TensorType(elt.u8, (3,)))
            ct_b = crypto.sym_encrypt(bob_key, msg_b_obj)
            pt_b = crypto.sym_decrypt(alice_key, ct_b, elt.TensorType(elt.u8, (3,)))
            np.testing.assert_array_equal(pt_b.runtime_obj, msg_b)


class TestKEMWithRawTensorKey:
    """Test symmetric encryption with raw tensor keys (not from KEM)."""

    def test_encrypt_with_raw_key(self):
        """Test encryption with raw 32-byte tensor key."""
        with el.Interpreter():
            # Create a raw 32-byte key
            raw_key = np.random.randint(0, 256, size=32, dtype=np.uint8)
            key_obj = el.InterpObject(raw_key, elt.TensorType(elt.u8, (32,)))

            # Create message
            message = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
            msg_obj = el.InterpObject(message, elt.TensorType(elt.u8, (5,)))

            # Encrypt and decrypt
            ciphertext = crypto.sym_encrypt(key_obj, msg_obj)
            plaintext = crypto.sym_decrypt(
                key_obj, ciphertext, elt.TensorType(elt.u8, (5,))
            )

            np.testing.assert_array_equal(plaintext.runtime_obj, message)
