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

"""Tests for TEE dialect types and type inference."""

import numpy as np
import pytest

import mplang.v2.edsl as el
from mplang.v2.dialects import crypto, tee


class TestTEETypes:
    """Test TEE type definitions."""

    def test_quote_type_default(self):
        """Test QuoteType with default platform."""
        qt = tee.QuoteType()
        assert qt.platform == "mock"
        assert str(qt) == "TEEQuote[mock]"
        assert repr(qt) == "QuoteType(platform='mock')"

    def test_quote_type_platforms(self):
        """Test QuoteType with different platforms."""
        platforms = ["mock", "sgx", "tdx", "sev"]
        for platform in platforms:
            qt = tee.QuoteType(platform=platform)
            assert qt.platform == platform
            assert str(qt) == f"TEEQuote[{platform}]"

    def test_quote_type_equality(self):
        """Test QuoteType equality and hashing."""
        q1 = tee.QuoteType("sgx")
        q2 = tee.QuoteType("sgx")
        q3 = tee.QuoteType("tdx")

        assert q1 == q2
        assert q1 != q3
        assert hash(q1) == hash(q2)
        assert hash(q1) != hash(q3)

    def test_attested_key_type_default(self):
        """Test AttestedKeyType with defaults."""
        akt = tee.AttestedKeyType()
        assert akt.platform == "mock"
        assert akt.curve == "x25519"
        assert str(akt) == "AttestedKey[mock, x25519]"

    def test_attested_key_type_custom(self):
        """Test AttestedKeyType with custom values."""
        akt = tee.AttestedKeyType(platform="sgx", curve="secp256k1")
        assert akt.platform == "sgx"
        assert akt.curve == "secp256k1"
        assert str(akt) == "AttestedKey[sgx, secp256k1]"

    def test_attested_key_type_equality(self):
        """Test AttestedKeyType equality and hashing."""
        a1 = tee.AttestedKeyType("sgx", "x25519")
        a2 = tee.AttestedKeyType("sgx", "x25519")
        a3 = tee.AttestedKeyType("sgx", "secp256k1")
        a4 = tee.AttestedKeyType("tdx", "x25519")

        assert a1 == a2
        assert a1 != a3  # Different curve
        assert a1 != a4  # Different platform
        assert hash(a1) == hash(a2)

    def test_measurement_type(self):
        """Test MeasurementType."""
        mt = tee.MeasurementType(platform="sgx")
        assert mt.platform == "sgx"
        assert str(mt) == "TEEMeasurement[sgx]"
        assert repr(mt) == "MeasurementType(platform='sgx')"

    def test_measurement_type_equality(self):
        """Test MeasurementType equality."""
        m1 = tee.MeasurementType("sgx")
        m2 = tee.MeasurementType("sgx")
        m3 = tee.MeasurementType("tdx")

        assert m1 == m2
        assert m1 != m3
        assert hash(m1) == hash(m2)


class TestTEETypeInference:
    """Test TEE abstract evaluation (type inference) in Tracer mode."""

    def test_quote_gen_type_inference(self):
        """Test quote_gen returns QuoteType."""

        def workflow():
            _sk, pk = crypto.kem_keygen("x25519")
            quote = tee.quote_gen(pk)
            return quote

        traced = el.trace(workflow)
        graph = traced.graph
        assert len(graph.operations) == 2  # kem_keygen + quote_gen
        quote_op = graph.operations[1]
        assert quote_op.opcode == "tee.quote_gen"

    def test_quote_gen_with_platform(self):
        """Test quote_gen with explicit platform."""

        def workflow():
            _sk, pk = crypto.kem_keygen("x25519")
            quote = tee.quote_gen(pk, platform="sgx")
            return quote

        traced = el.trace(workflow)
        graph = traced.graph

        quote_op = graph.operations[1]
        assert quote_op.attrs["platform"] == "sgx"

    def test_attest_in_graph(self):
        """Test attest creates correct graph operation."""

        def workflow():
            _sk, pk = crypto.kem_keygen("x25519")
            quote = tee.quote_gen(pk, platform="sgx")
            attested_pk = tee.attest(quote, expected_curve="secp256k1")
            return attested_pk

        traced = el.trace(workflow)
        graph = traced.graph

        assert len(graph.operations) == 3  # kem_keygen, quote_gen, attest
        attest_op = graph.operations[2]
        assert attest_op.opcode == "tee.attest"
        assert attest_op.attrs["expected_curve"] == "secp256k1"

    def test_get_measurement_in_graph(self):
        """Test get_measurement creates correct graph operation."""

        def workflow():
            _sk, pk = crypto.kem_keygen("x25519")
            quote = tee.quote_gen(pk, platform="sev")
            measurement = tee.get_measurement(quote)
            return measurement

        traced = el.trace(workflow)
        graph = traced.graph

        assert len(graph.operations) == 3
        measure_op = graph.operations[2]
        assert measure_op.opcode == "tee.get_measurement"

    def test_full_workflow_graph(self):
        """Test complete TEE workflow produces correct graph."""

        def workflow():
            _sk, pk = crypto.kem_keygen("x25519")
            quote = tee.quote_gen(pk, platform="tdx")
            attested_pk = tee.attest(quote)
            measurement = tee.get_measurement(quote)
            return attested_pk, measurement

        traced = el.trace(workflow)
        graph = traced.graph

        opcodes = [op.opcode for op in graph.operations]
        assert "crypto.kem_keygen" in opcodes
        assert "tee.quote_gen" in opcodes
        assert "tee.attest" in opcodes
        assert "tee.get_measurement" in opcodes

    def test_quote_gen_rejects_tensor_type(self):
        """Test quote_gen raises TypeError for non-PublicKeyType inputs."""
        from mplang.v2.dialects import tensor

        def workflow():
            pk = tensor.constant(np.zeros(32, dtype=np.uint8))
            tee.quote_gen(pk)

        with pytest.raises(TypeError, match="expects PublicKeyType"):
            el.trace(workflow)
