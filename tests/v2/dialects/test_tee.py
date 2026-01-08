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

    def test_quote_type(self):
        """Test QuoteType."""
        qt = tee.QuoteType()
        assert str(qt) == "TEEQuote"
        assert repr(qt) == "QuoteType()"

    def test_quote_type_equality(self):
        """Test QuoteType equality and hashing."""
        q1 = tee.QuoteType()
        q2 = tee.QuoteType()

        assert q1 == q2
        assert hash(q1) == hash(q2)

    def test_attested_key_type(self):
        """Test AttestedKeyType."""
        akt = tee.AttestedKeyType()
        assert akt.curve == "x25519"
        assert str(akt) == "AttestedKey[x25519]"

    def test_attested_key_type_custom(self):
        """Test AttestedKeyType with custom values."""
        akt = tee.AttestedKeyType(curve="secp256k1")
        assert akt.curve == "secp256k1"
        assert str(akt) == "AttestedKey[secp256k1]"

    def test_attested_key_type_equality(self):
        """Test AttestedKeyType equality and hashing."""
        a1 = tee.AttestedKeyType("x25519")
        a2 = tee.AttestedKeyType("x25519")
        a3 = tee.AttestedKeyType("secp256k1")

        assert a1 == a2
        assert a1 != a3  # Different curve
        assert hash(a1) == hash(a2)

    def test_measurement_type(self):
        """Test MeasurementType."""
        mt = tee.MeasurementType()
        assert str(mt) == "TEEMeasurement"
        assert repr(mt) == "MeasurementType()"

    def test_measurement_type_equality(self):
        """Test MeasurementType equality."""
        m1 = tee.MeasurementType()
        m2 = tee.MeasurementType()

        assert m1 == m2
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

    def test_attest_in_graph(self):
        """Test attest creates correct graph operation."""

        def workflow():
            _sk, pk = crypto.kem_keygen("x25519")
            quote = tee.quote_gen(pk)
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
            quote = tee.quote_gen(pk)
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
            quote = tee.quote_gen(pk)
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
