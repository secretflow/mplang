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
