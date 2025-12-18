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

"""Tests for TEE backend implementation (mock TEE)."""

import numpy as np
import pytest

import mplang.v2.backends.tee_impl  # noqa: F401 - Register implementations
import mplang.v2.edsl.typing as elt
from mplang.v2.backends.tee_impl import MockQuoteValue
from mplang.v2.dialects import crypto, tee, tensor
from mplang.v2.runtime.interpreter import Interpreter


class TestMockQuoteDataStructure:
    """Test MockQuoteValue data structure directly."""

    def test_mock_quote_roundtrip(self):
        """Test MockQuoteValue serialization/deserialization."""
        pk = bytes(np.random.randint(0, 256, size=32, dtype=np.uint8))

        quote = MockQuoteValue(platform="sgx", bound_pk=pk, suite="x25519")
        data = quote.to_bytes()

        recovered = MockQuoteValue.from_bytes(data)
        assert recovered.platform == "sgx"
        assert recovered.bound_pk == pk
        assert recovered.suite == "x25519"


class TestMockTEEExecution:
    """Test mock TEE backend execution via Interpreter."""

    def test_quote_gen_execution(self):
        """Test quote_gen primitive execution produces correct type."""
        with Interpreter():
            _sk, pk = crypto.kem_keygen("x25519")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(pk)

            # Verify type propagation
            assert isinstance(quote.type, tee.QuoteType)

    def test_attest_execution(self):
        """Test attest primitive execution produces correct type.

        Note: After simplification, attest returns a PublicKeyValue (at runtime)
        with PublicKeyType (at type level), not AttestedKeyType.
        """
        with Interpreter():
            _sk, pk = crypto.kem_keygen("x25519")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(pk)

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                attested_pk = tee.attest(quote, expected_curve="x25519")

            # Type level still shows AttestedKeyType (for type checking)
            # But runtime returns PublicKeyValue directly
            assert isinstance(attested_pk.type, tee.AttestedKeyType)
            assert attested_pk.type.curve == "x25519"

    def test_full_attestation_workflow(self):
        """Test complete attestation workflow: keygen -> quote_gen -> attest."""
        with Interpreter():
            _sk, pk = crypto.kem_keygen("x25519")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(pk)

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                attested_pk = tee.attest(quote, expected_curve="x25519")

            # Verify types propagate correctly through the workflow
            assert isinstance(attested_pk.type, tee.AttestedKeyType)


class TestTEEWithCryptoIntegration:
    """Test TEE + crypto integration via Interpreter."""

    def test_attestation_then_key_exchange(self):
        """Test TEE attestation followed by symmetric key derivation."""
        with Interpreter():
            # TEE side: generate keypair and quote
            tee_sk, tee_pk = crypto.kem_keygen("x25519")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(tee_pk)

            # Verifier side: attest and generate own keypair
            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                attested_pk = tee.attest(quote, expected_curve="x25519")

            _verifier_sk, verifier_pk = crypto.kem_keygen("x25519")

            # TEE derives symmetric key
            tee_key = crypto.kem_derive(tee_sk, verifier_pk)

            # Verify types
            assert isinstance(attested_pk.type, tee.AttestedKeyType)
            assert isinstance(tee_key.type, crypto.SymmetricKeyType)

    def test_encrypt_decrypt_after_attestation(self):
        """Test symmetric encryption/decryption after attestation workflow."""
        with Interpreter():
            # Setup: both parties generate keypairs
            tee_sk, tee_pk = crypto.kem_keygen("x25519")
            verifier_sk, verifier_pk = crypto.kem_keygen("x25519")

            # TEE generates quote (attestation)
            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(tee_pk)

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                _attested_pk = tee.attest(quote, expected_curve="x25519")

            # Both parties derive symmetric keys (ECDH)
            tee_key = crypto.kem_derive(tee_sk, verifier_pk)
            verifier_key = crypto.kem_derive(verifier_sk, tee_pk)

            # Encrypt with TEE's key, decrypt with verifier's key
            message = np.array([72, 101, 108, 108, 111], dtype=np.uint8)  # "Hello"
            message_obj = tensor.constant(message)
            ciphertext = crypto.sym_encrypt(tee_key, message_obj)
            plaintext = crypto.sym_decrypt(
                verifier_key, ciphertext, elt.TensorType(elt.u8, (5,))
            )

            # Verify decryption succeeds (same key due to ECDH)
            np.testing.assert_array_equal(plaintext.runtime_obj.unwrap(), message)

            # Verify types
            assert isinstance(tee_key.type, crypto.SymmetricKeyType)
            assert isinstance(verifier_key.type, crypto.SymmetricKeyType)
