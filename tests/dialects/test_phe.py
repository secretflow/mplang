"""Tests for PHE dialect."""

import numpy as np
import pytest

import mplang2.dialects.phe as phe
import mplang2.dialects.tensor as tensor
import mplang2.edsl as el
import mplang2.edsl.typing as elt


class TestPHEKeyManagement:
    """Test PHE key generation and basic operations."""

    def test_keygen_basic(self):
        """Test basic key generation."""
        with el.Tracer():
            pk, sk = phe.keygen()

            assert isinstance(pk, el.TraceObject)
            assert isinstance(sk, el.TraceObject)
            assert isinstance(pk.type, elt.CustomType)
            assert isinstance(sk.type, elt.CustomType)
            assert pk.type.kind == "PHEPublicKey"
            assert sk.type.kind == "PHEPrivateKey"

    def test_keygen_with_params(self):
        """Test key generation with custom parameters."""
        with el.Tracer() as tracer:
            pk, sk = phe.keygen(
                scheme="paillier", key_size=4096, max_value=2**64, fxp_bits=16
            )

            # Verify operation was created
            graph = tracer.finalize((pk, sk))
            assert len(graph.operations) == 1
            op = graph.operations[0]
            assert op.opcode == "phe.keygen"
            assert op.attrs["scheme"] == "paillier"
            assert op.attrs["key_size"] == 4096
            assert op.attrs["max_value"] == 2**64
            assert op.attrs["fxp_bits"] == 16


class TestPHEEncryptDecrypt:
    """Test PHE encryption and decryption type transformations."""

    def test_encrypt_1d_tensor(self):
        """Test encrypting 1D tensor."""
        with el.Tracer():
            # Create plaintext
            x = tensor.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
            pk, _sk = phe.keygen()

            # Encrypt
            ct = phe.encrypt(x, pk)

            # Verify type transformation: Tensor[f32, (3,)] -> Tensor[HE[f32], (3,)]
            assert isinstance(ct, el.TraceObject)
            assert isinstance(ct.type, elt.TensorType)
            assert isinstance(ct.type.element_type, elt.ScalarHEType)
            assert ct.type.shape == (3,)

            # Verify underlying plaintext type
            assert isinstance(ct.type.element_type.pt_type, elt.ScalarType)
            assert ct.type.element_type.pt_type == elt.f32

    def test_encrypt_2d_tensor(self):
        """Test encrypting 2D tensor."""
        with el.Tracer():
            x = tensor.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            pk, _sk = phe.keygen()

            ct = phe.encrypt(x, pk)

            assert ct.type.shape == (2, 2)
            assert isinstance(ct.type.element_type, elt.ScalarHEType)

    def test_decrypt_tensor(self):
        """Test decrypting tensor."""
        with el.Tracer():
            x = tensor.constant(np.array([1.0, 2.0, 3.0]))
            pk, sk = phe.keygen()

            ct = phe.encrypt(x, pk)
            pt = phe.decrypt(ct, sk)

            # Verify type transformation: Tensor[HE[f64], (3,)] -> Tensor[f64, (3,)]
            assert isinstance(pt.type, elt.TensorType)
            assert isinstance(pt.type.element_type, elt.ScalarType)
            assert pt.type.element_type == elt.f64  # numpy default float64
            assert pt.type.shape == (3,)

    def test_encrypt_decrypt_round_trip(self):
        """Test full encrypt-decrypt round trip."""
        with el.Tracer():
            x = tensor.constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
            pk, sk = phe.keygen()

            ct = phe.encrypt(x, pk)
            result = phe.decrypt(ct, sk)

            # Verify result type matches input type
            assert result.type == x.type


class TestPHEHomomorphicOperations:
    """Test PHE homomorphic operations."""

    def test_add_encrypted_tensors(self):
        """Test element-wise addition of encrypted tensors."""
        with el.Tracer() as tracer:
            x = tensor.constant(np.array([1.0, 2.0, 3.0]))
            y = tensor.constant(np.array([4.0, 5.0, 6.0]))
            pk, _sk = phe.keygen()

            ct_x = phe.encrypt(x, pk)
            ct_y = phe.encrypt(y, pk)
            ct_sum = phe.add(ct_x, ct_y)

            # Verify type: Tensor[HE[f64], (3,)]
            assert isinstance(ct_sum.type, elt.TensorType)
            assert isinstance(ct_sum.type.element_type, elt.ScalarHEType)
            assert ct_sum.type.shape == (3,)

            # Verify operation is tensor.elementwise
            graph = tracer.finalize(ct_sum)
            generic_ops = [
                op for op in graph.operations if op.opcode == "tensor.elementwise"
            ]
            # 2 encrypt operations + 1 add operation = 3 elementwise ops
            assert len(generic_ops) == 3

    def test_mul_scalar_encrypted_tensor(self):
        """Test element-wise scalar multiplication."""
        with el.Tracer():
            x = tensor.constant(np.array([1.0, 2.0, 3.0]))
            scale = tensor.constant(np.array([2.0, 2.0, 2.0]))
            pk, _sk = phe.keygen()

            ct_x = phe.encrypt(x, pk)
            ct_scaled = phe.mul_scalar(ct_x, scale)

            # Verify type: Tensor[HE[f64], (3,)]
            assert isinstance(ct_scaled.type, elt.TensorType)
            assert isinstance(ct_scaled.type.element_type, elt.ScalarHEType)
            assert ct_scaled.type.shape == (3,)

    def test_homomorphic_computation_chain(self):
        """Test chaining homomorphic operations."""
        with el.Tracer():
            x = tensor.constant(np.array([1.0, 2.0, 3.0]))
            y = tensor.constant(np.array([4.0, 5.0, 6.0]))
            scale = tensor.constant(
                np.array([2.0, 2.0, 2.0])
            )  # Use tensor instead of scalar
            pk, sk = phe.keygen()

            # Compute: (x + y) * 2
            ct_x = phe.encrypt(x, pk)
            ct_y = phe.encrypt(y, pk)
            ct_sum = phe.add(ct_x, ct_y)
            ct_result = phe.mul_scalar(ct_sum, scale)

            # Decrypt
            result = phe.decrypt(ct_result, sk)

            # Verify final type
            assert result.type.element_type == elt.f64  # numpy default
            assert result.type.shape == (3,)

    def test_add_requires_matching_shapes(self):
        """Mismatch shapes should raise ValueError."""
        with el.Tracer():
            matrix = tensor.constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
            vector = tensor.constant(np.array([10.0, 20.0]))
            pk, _sk = phe.keygen()

            ct_matrix = phe.encrypt(matrix, pk)  # (2, 2)
            ct_vector = phe.encrypt(vector, pk)  # (2,)

            with pytest.raises(
                ValueError, match="All tensor arguments must have the same shape"
            ):
                phe.add(ct_matrix, ct_vector)

    def test_mul_scalar_requires_matching_shapes(self):
        """Tensor scalar args must share shape."""
        with el.Tracer():
            matrix = tensor.constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
            scalar = tensor.constant(5.0)
            pk, _sk = phe.keygen()

            ct_matrix = phe.encrypt(matrix, pk)  # (2, 2)

            with pytest.raises(
                ValueError, match="All tensor arguments must have the same shape"
            ):
                phe.mul_scalar(ct_matrix, scalar)


class TestPHEWithTensorOps:
    """Test PHE operations combined with tensor structural operations."""

    def test_transpose_encrypted_tensor(self):
        """Test transposing encrypted tensor."""
        with el.Tracer():
            x = tensor.constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            pk, _sk = phe.keygen()

            ct = phe.encrypt(x, pk)  # Tensor[HE[f64], (2, 3)]
            ct_t = tensor.transpose(ct, (1, 0))  # Tensor[HE[f64], (3, 2)]

            # Verify type
            assert isinstance(ct_t.type.element_type, elt.ScalarHEType)
            assert ct_t.type.shape == (3, 2)

    def test_reshape_encrypted_tensor(self):
        """Test reshaping encrypted tensor."""
        with el.Tracer():
            x = tensor.constant(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            pk, _sk = phe.keygen()

            ct = phe.encrypt(x, pk)  # Tensor[HE[f64], (6,)]
            ct_reshaped = tensor.reshape(ct, (2, 3))  # Tensor[HE[f64], (2, 3)]

            assert ct_reshaped.type.shape == (2, 3)
            assert isinstance(ct_reshaped.type.element_type, elt.ScalarHEType)

    def test_concat_encrypted_tensors(self):
        """Test concatenating encrypted tensors."""
        with el.Tracer():
            x = tensor.constant(np.array([1.0, 2.0, 3.0]))
            y = tensor.constant(np.array([4.0, 5.0, 6.0]))
            pk, _sk = phe.keygen()

            ct_x = phe.encrypt(x, pk)
            ct_y = phe.encrypt(y, pk)
            ct_concat = tensor.concat([ct_x, ct_y], axis=0)

            assert ct_concat.type.shape == (6,)
            assert isinstance(ct_concat.type.element_type, elt.ScalarHEType)

    def test_gather_encrypted_tensor(self):
        """Test gathering from encrypted tensor."""
        with el.Tracer():
            x = tensor.constant(np.array([10.0, 20.0, 30.0, 40.0]))
            indices = tensor.constant(np.array([0, 2, 1]))
            pk, _sk = phe.keygen()

            ct = phe.encrypt(x, pk)
            ct_gathered = tensor.gather(ct, indices, axis=0)

            assert ct_gathered.type.shape == (3,)
            assert isinstance(ct_gathered.type.element_type, elt.ScalarHEType)

    def test_complex_workflow(self):
        """Test complex PHE workflow combining multiple operations."""
        with el.Tracer() as tracer:
            # Data preparation
            x = tensor.constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
            y = tensor.constant(np.array([[5.0, 6.0], [7.0, 8.0]]))
            scale = tensor.constant(
                np.array([[0.5, 0.5], [0.5, 0.5]])
            )  # Use tensor instead of scalar

            # Key generation
            pk, sk = phe.keygen(scheme="paillier", key_size=2048)

            # Encrypt
            ct_x = phe.encrypt(x, pk)
            ct_y = phe.encrypt(y, pk)

            # Homomorphic computation: (x + y) * 0.5
            ct_sum = phe.add(ct_x, ct_y)
            ct_result = phe.mul_scalar(ct_sum, scale)

            # Tensor operations on encrypted data
            ct_transposed = tensor.transpose(ct_result, (1, 0))
            ct_reshaped = tensor.reshape(ct_transposed, (4,))

            # Decrypt
            result = phe.decrypt(ct_reshaped, sk)

            # Verify final result
            assert result.type.element_type == elt.f64  # numpy default
            assert result.type.shape == (4,)

            # Verify graph structure
            graph = tracer.finalize(result)
            opcodes = [op.opcode for op in graph.operations]

            assert "phe.keygen" in opcodes
            # encrypt/decrypt operations are wrapped in tensor.elementwise
            assert "tensor.elementwise" in opcodes  # For encrypt/add/mul_scalar/decrypt
            assert "tensor.transpose" in opcodes
            assert "tensor.reshape" in opcodes


class TestPHETypeInference:
    """Test PHE type inference for element-level operations."""

    def test_add_element_type_inference(self):
        """Test that add_cc primitive correctly infers HE[T] + HE[T] -> HE[T]."""
        he_f32 = elt.ScalarHEType(elt.f32)
        result_type = phe.add_cc_p._abstract_eval(he_f32, he_f32)

        assert isinstance(result_type, elt.ScalarHEType)
        assert result_type.pt_type == elt.f32

    def test_mul_scalar_element_type_inference(self):
        """Test that mul_scalar primitive correctly infers HE[T] * T -> HE[T]."""
        he_f32 = elt.ScalarHEType(elt.f32)
        result_type = phe.mul_cp_p._abstract_eval(he_f32, elt.f32)

        assert isinstance(result_type, elt.ScalarHEType)
        assert result_type.pt_type == elt.f32

    def test_add_type_mismatch_error(self):
        """Test that add primitives reject mismatched types."""
        he_f32 = elt.ScalarHEType(elt.f32)
        he_f64 = elt.ScalarHEType(elt.f64)

        with pytest.raises(TypeError, match="Type mismatch"):
            phe.add_cc_p._abstract_eval(he_f32, he_f64)

        with pytest.raises(TypeError, match="Type mismatch"):
            phe.add_cp_p._abstract_eval(he_f32, elt.f64)

    def test_add_requires_encrypted_types(self):
        """Test that add primitives require ciphertext inputs."""
        # Test that primitives validate input types
        with pytest.raises(AttributeError):  # FloatType doesn't have pt_type
            phe.add_cc_p._abstract_eval(elt.f32, elt.f32)  # type: ignore[arg-type]

        with pytest.raises(AttributeError):  # FloatType doesn't have pt_type
            phe.add_cp_p._abstract_eval(elt.f32, elt.f32)  # type: ignore[arg-type]

    def test_mul_scalar_type_mismatch_error(self):
        """Test that mul_scalar rejects mismatched types."""
        he_f32 = elt.ScalarHEType(elt.f32)

        with pytest.raises(TypeError, match="Type mismatch"):
            phe.mul_cp_p._abstract_eval(he_f32, elt.f64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
