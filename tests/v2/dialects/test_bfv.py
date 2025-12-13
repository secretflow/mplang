# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for BFV dialect."""

import numpy as np
import pytest

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects import bfv, tensor


class TestBFVTypes:
    """Test BFV type definitions."""

    def test_key_types(self):
        pk = bfv.KeyType("Public", 4096)
        sk = bfv.KeyType("Private", 4096)
        rk = bfv.KeyType("Relin", 4096)
        gk = bfv.KeyType("Galois", 4096)

        assert str(pk) == "BFVPublicKey[N=4096]"
        assert str(sk) == "BFVPrivateKey[N=4096]"
        assert str(rk) == "BFVRelinKey[N=4096]"
        assert str(gk) == "BFVGaloisKey[N=4096]"
        assert pk != sk
        assert pk.scheme == "bfv"

    def test_plaintext_type(self):
        vec_type = elt.Vector[elt.i64, 4096]
        pt = bfv.PlaintextType(vec_type)
        assert str(pt) == f"BFVPlaintext[{vec_type}]"
        assert isinstance(pt, elt.BaseType)

    def test_ciphertext_type(self):
        vec_type = elt.Vector[elt.i64, 4096]
        ct = bfv.CiphertextType(vec_type)
        assert str(ct) == f"BFVCiphertext[{vec_type}]"
        assert isinstance(ct, elt.BaseType)
        assert isinstance(ct, elt.EncryptedTrait)

    def test_encoder_type(self):
        enc = bfv.EncoderType(poly_modulus_degree=4096)
        assert str(enc) == "BFVEncoder[N=4096]"
        assert enc == bfv.EncoderType(4096)
        assert enc != bfv.EncoderType(8192)


class TestBFVWorkflow:
    """Test complete BFV workflows."""

    def test_basic_simd_arithmetic(self):
        """Test encode -> encrypt -> add/mul -> decrypt -> decode."""

        def workflow():
            # 1. Setup
            pk, sk = bfv.keygen(poly_modulus_degree=4096)
            rk = bfv.make_relin_keys(sk)
            encoder = bfv.create_encoder(poly_modulus_degree=4096)

            # 2. Data
            v1 = tensor.constant(np.array([1, 2, 3, 4], dtype=np.int64))
            v2 = tensor.constant(np.array([10, 20, 30, 40], dtype=np.int64))

            # 3. Encode & Encrypt
            pt1 = bfv.encode(v1, encoder)
            ct1 = bfv.encrypt(pt1, pk)

            pt2 = bfv.encode(v2, encoder)
            ct2 = bfv.encrypt(pt2, pk)

            # 4. Computation
            # (ct1 + ct2) * ct1
            ct_sum = bfv.add(ct1, ct2)
            ct_prod = bfv.mul(ct_sum, ct1)
            ct_res = bfv.relinearize(ct_prod, rk)

            # 5. Decrypt
            pt_res = bfv.decrypt(ct_res, sk)
            res = bfv.decode(pt_res, encoder)
            return res, ct_sum, pt_res

        # Verify graph
        traced = el.trace(workflow)
        graph = traced.graph
        opcodes = [op.opcode for op in graph.operations]
        assert "bfv.keygen" in opcodes
        assert "bfv.make_relin_keys" in opcodes
        assert "bfv.encode" in opcodes
        assert "bfv.encrypt" in opcodes
        assert "bfv.add" in opcodes
        assert "bfv.mul" in opcodes
        assert "bfv.relinearize" in opcodes
        assert "bfv.decrypt" in opcodes
        assert "bfv.decode" in opcodes

        # Verify types
        # graph.outputs corresponds to flattened (res, ct_sum, pt_res)
        res_val = graph.outputs[0]
        ct_sum_val = graph.outputs[1]
        pt_res_val = graph.outputs[2]

        assert isinstance(ct_sum_val.type, bfv.CiphertextType)
        assert isinstance(pt_res_val.type, bfv.PlaintextType)
        assert isinstance(res_val.type, elt.TensorType)

    def test_rotation(self):
        """Test rotation workflow."""

        def workflow():
            pk, sk = bfv.keygen()
            gk = bfv.make_galois_keys(sk)
            encoder = bfv.create_encoder()

            v = tensor.constant(np.array([1, 2, 3, 4], dtype=np.int64))
            pt = bfv.encode(v, encoder)
            ct = bfv.encrypt(pt, pk)

            ct_rot = bfv.rotate(ct, 1, gk)
            return ct_rot

        traced = el.trace(workflow)
        ct_rot = traced.graph.outputs[0]
        assert isinstance(ct_rot.type, bfv.CiphertextType)


class TestBFVTypeChecking:
    """Test type checking in primitives."""

    def test_encode_checks(self):
        """Test encode input validation."""
        encoder = bfv.EncoderType(4096)

        # Fail on float tensor
        f_tensor = elt.TensorType(elt.f32, (10,))
        assert bfv.encode_p._abstract_eval is not None
        with pytest.raises(TypeError, match="integer arithmetic only"):
            bfv.encode_p._abstract_eval(f_tensor, encoder)

        # Fail on 2D tensor
        matrix = elt.TensorType(elt.i64, (10, 10))
        with pytest.raises(ValueError, match="only supports 1D"):
            bfv.encode_p._abstract_eval(matrix, encoder)

    def test_arithmetic_checks(self):
        """Test arithmetic operand validation."""
        ct = bfv.CiphertextType(4096)
        pt = bfv.PlaintextType(4096)
        scalar = elt.i64

        # Valid
        assert bfv.add_p._abstract_eval is not None
        bfv.add_p._abstract_eval(ct, ct)
        bfv.add_p._abstract_eval(ct, pt)

        # Invalid: Scalar operand (must be encoded first)
        with pytest.raises(TypeError, match="must be BFVCiphertext or BFVPlaintext"):
            bfv.add_p._abstract_eval(ct, scalar)

        # Invalid: No ciphertext
        with pytest.raises(TypeError, match="must be a Ciphertext"):
            bfv.add_p._abstract_eval(pt, pt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
