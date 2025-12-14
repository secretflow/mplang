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

"""Tests for crypto dialect types and type inference.

NOTE: Backend execution tests are in tests/mplang.v2/backends/test_crypto_impl.py
"""

import mplang.v2.edsl as el
from mplang.v2.dialects import crypto


class TestKEMTypes:
    """Test KEM type definitions."""

    def test_private_key_type_default(self):
        """Test PrivateKeyType with default suite."""
        pk_type = crypto.PrivateKeyType()
        assert pk_type.suite == "x25519"
        assert str(pk_type) == "PrivateKey[x25519]"

    def test_private_key_type_custom_suite(self):
        """Test PrivateKeyType with custom suite."""
        pk_type = crypto.PrivateKeyType(suite="kyber768")
        assert pk_type.suite == "kyber768"
        assert str(pk_type) == "PrivateKey[kyber768]"

    def test_private_key_type_equality(self):
        """Test PrivateKeyType equality and hashing."""
        pk1 = crypto.PrivateKeyType("x25519")
        pk2 = crypto.PrivateKeyType("x25519")
        pk3 = crypto.PrivateKeyType("kyber768")

        assert pk1 == pk2
        assert pk1 != pk3
        assert hash(pk1) == hash(pk2)
        assert hash(pk1) != hash(pk3)

    def test_public_key_type_default(self):
        """Test PublicKeyType with default suite."""
        pk_type = crypto.PublicKeyType()
        assert pk_type.suite == "x25519"
        assert str(pk_type) == "PublicKey[x25519]"

    def test_public_key_type_custom_suite(self):
        """Test PublicKeyType with custom suite."""
        pk_type = crypto.PublicKeyType(suite="kyber768")
        assert pk_type.suite == "kyber768"
        assert str(pk_type) == "PublicKey[kyber768]"

    def test_public_key_type_equality(self):
        """Test PublicKeyType equality and hashing."""
        pk1 = crypto.PublicKeyType("x25519")
        pk2 = crypto.PublicKeyType("x25519")
        pk3 = crypto.PublicKeyType("kyber768")

        assert pk1 == pk2
        assert pk1 != pk3
        assert hash(pk1) == hash(pk2)
        assert hash(pk1) != hash(pk3)

    def test_symmetric_key_type_default(self):
        """Test SymmetricKeyType with default suite."""
        sk_type = crypto.SymmetricKeyType()
        assert sk_type.suite == "x25519"
        assert str(sk_type) == "SymmetricKey[x25519]"

    def test_symmetric_key_type_custom_suite(self):
        """Test SymmetricKeyType with custom suite."""
        sk_type = crypto.SymmetricKeyType(suite="kyber768")
        assert sk_type.suite == "kyber768"
        assert str(sk_type) == "SymmetricKey[kyber768]"

    def test_symmetric_key_type_equality(self):
        """Test SymmetricKeyType equality and hashing."""
        sk1 = crypto.SymmetricKeyType("x25519")
        sk2 = crypto.SymmetricKeyType("x25519")
        sk3 = crypto.SymmetricKeyType("kyber768")

        assert sk1 == sk2
        assert sk1 != sk3
        assert hash(sk1) == hash(sk2)
        assert hash(sk1) != hash(sk3)


class TestKEMTypeInference:
    """Test KEM type inference (abstract_eval)."""

    def test_kem_keygen_type_inference(self):
        """Test kem_keygen returns correct types."""
        tracer = el.Tracer()

        def fn():
            sk, pk = crypto.kem_keygen("x25519")
            return sk, pk

        traced = tracer.run(fn)
        outputs = traced.graph.outputs

        assert len(outputs) == 2
        assert isinstance(outputs[0].type, crypto.PrivateKeyType)
        assert outputs[0].type.suite == "x25519"
        assert isinstance(outputs[1].type, crypto.PublicKeyType)
        assert outputs[1].type.suite == "x25519"

    def test_kem_keygen_custom_suite(self):
        """Test kem_keygen with custom suite."""
        tracer = el.Tracer()

        def fn():
            sk, pk = crypto.kem_keygen("kyber768")
            return sk, pk

        traced = tracer.run(fn)
        outputs = traced.graph.outputs

        assert outputs[0].type.suite == "kyber768"
        assert outputs[1].type.suite == "kyber768"

    def test_kem_derive_type_inference(self):
        """Test kem_derive returns SymmetricKeyType."""
        tracer = el.Tracer()

        def fn():
            sk, pk = crypto.kem_keygen("x25519")
            symmetric_key = crypto.kem_derive(sk, pk)
            return symmetric_key

        traced = tracer.run(fn)
        output = traced.graph.outputs[0]

        assert isinstance(output.type, crypto.SymmetricKeyType)
        assert output.type.suite == "x25519"
