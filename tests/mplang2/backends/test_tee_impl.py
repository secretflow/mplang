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

import mplang2.backends.crypto_impl
import mplang2.backends.tee_impl  # noqa: F401 - Register implementations
import mplang2.dialects.crypto as crypto
import mplang2.dialects.tee as tee
import mplang2.edsl as el
import mplang2.edsl.typing as elt
from mplang2.backends.tee_impl import (
    MockAttestedKey,
    MockMeasurement,
    MockQuote,
    attested_key_to_bytes,
    measurement_to_bytes,
)


class TestMockQuoteDataStructure:
    """Test MockQuote data structure directly."""

    def test_mock_quote_roundtrip(self):
        """Test MockQuote serialization/deserialization."""
        pk = np.random.randint(0, 256, size=32, dtype=np.uint8)
        measurement = np.random.randint(0, 256, size=32, dtype=np.uint8)

        quote = MockQuote(platform="sgx", bound_pk=pk, measurement=measurement)
        data = quote.to_bytes()

        recovered = MockQuote.from_bytes(data)
        assert recovered.platform == "sgx"
        np.testing.assert_array_equal(recovered.bound_pk, pk)
        np.testing.assert_array_equal(recovered.measurement, measurement)


class TestMockTEEExecution:
    """Test mock TEE backend execution via Interpreter."""

    def test_quote_gen_execution(self):
        """Test quote_gen primitive execution produces correct type."""
        with el.Interpreter():
            _sk, pk = crypto.kem_keygen("x25519")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(pk, platform="mock")

            # Verify type propagation
            assert isinstance(quote.type, tee.QuoteType)
            assert quote.type.platform == "mock"

    def test_attest_execution(self):
        """Test attest primitive execution produces correct type."""
        with el.Interpreter():
            _sk, pk = crypto.kem_keygen("x25519")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(pk, platform="sgx")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                attested_pk = tee.attest(quote, expected_curve="x25519")

            # Verify type propagation
            assert isinstance(attested_pk.type, tee.AttestedKeyType)
            assert attested_pk.type.platform == "sgx"
            assert attested_pk.type.curve == "x25519"

    def test_get_measurement_execution(self):
        """Test get_measurement primitive execution produces correct type."""
        with el.Interpreter():
            _sk, pk = crypto.kem_keygen("x25519")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(pk)

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                measurement = tee.get_measurement(quote)

            # Verify type propagation
            assert isinstance(measurement.type, tee.MeasurementType)
            assert measurement.type.platform == "mock"

    def test_full_attestation_workflow(self):
        """Test complete attestation workflow: keygen -> quote_gen -> attest."""
        with el.Interpreter():
            _sk, pk = crypto.kem_keygen("x25519")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(pk, platform="tdx")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                attested_pk = tee.attest(quote, expected_curve="x25519")

            # Verify types propagate correctly through the workflow
            assert isinstance(attested_pk.type, tee.AttestedKeyType)
            assert attested_pk.type.platform == "tdx"


class TestTEEHelperFunctions:
    """Test helper functions for converting TEE types."""

    def test_attested_key_to_bytes(self):
        """Test converting AttestedKey to bytes for crypto operations."""
        pk = np.array([1, 2, 3, 4] * 8, dtype=np.uint8)
        ak = MockAttestedKey(platform="sgx", curve="x25519", public_key=pk)

        result = attested_key_to_bytes(ak)
        np.testing.assert_array_equal(result, pk)

    def test_measurement_to_bytes(self):
        """Test converting Measurement to bytes."""
        hash_bytes = np.array([0xDE, 0xAD, 0xBE, 0xEF] * 8, dtype=np.uint8)
        m = MockMeasurement(platform="sgx", hash_bytes=hash_bytes)

        result = measurement_to_bytes(m)
        np.testing.assert_array_equal(result, hash_bytes)


class TestTEEWithCryptoIntegration:
    """Test TEE + crypto integration via Interpreter."""

    def test_attestation_then_key_exchange(self):
        """Test TEE attestation followed by symmetric key derivation."""
        with el.Interpreter():
            # TEE side: generate keypair and quote
            tee_sk, tee_pk = crypto.kem_keygen("x25519")

            with pytest.warns(UserWarning, match="Insecure mock TEE"):
                quote = tee.quote_gen(tee_pk, platform="sgx")

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
        with el.Interpreter():
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
            message_obj = el.InterpObject(message, elt.TensorType(elt.u8, (5,)))
            ciphertext = crypto.sym_encrypt(tee_key, message_obj)
            plaintext = crypto.sym_decrypt(
                verifier_key, ciphertext, elt.TensorType(elt.u8, (5,))
            )

            # Verify decryption succeeds (same key due to ECDH)
            np.testing.assert_array_equal(plaintext.runtime_obj, message)

            # Verify types
            assert isinstance(tee_key.type, crypto.SymmetricKeyType)
            assert isinstance(verifier_key.type, crypto.SymmetricKeyType)
