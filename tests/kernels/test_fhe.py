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

"""Tests for FHE (Fully Homomorphic Encryption) backend using TenSEAL."""

import pytest
import numpy as np

from mplang.core.dtype import DType
from mplang.core.pfunc import PFunction
from mplang.core.mptype import TensorType
from mplang.kernels.fhe import (
    FHEContext,
    CipherText,
    _fhe_keygen,
    _fhe_encrypt,
    _fhe_decrypt,
    _fhe_add,
    _fhe_mul,
    _fhe_dot,
    _fhe_polyval,
)
from mplang.kernels.base import list_kernels


def _create_test_pfunc(**attrs) -> PFunction:
    """Helper to create a test PFunction with dummy type info."""
    dummy_tensor = TensorType.from_obj(np.array(0.0))
    return PFunction(
        fn_type="test.fhe", ins_info=(), outs_info=(dummy_tensor,), **attrs
    )


class TestFHEKernelRegistry:
    """Test FHE kernel registration."""

    def test_kernel_registry(self):
        """Test that all FHE kernels are properly registered."""
        for name in [
            "fhe.keygen",
            "fhe.encrypt",
            "fhe.decrypt",
            "fhe.add",
            "fhe.mul",
            "fhe.dot",
            "fhe.polyval",
        ]:
            assert name in list_kernels()


class TestFHEContext:
    """Test FHE context generation and management."""

    def test_ckks_context_generation(self):
        """Test CKKS context generation returns private and public contexts."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        result = _fhe_keygen(pfunc)

        assert len(result) == 2
        private_context, public_context = result

        # Check private context
        assert isinstance(private_context, FHEContext)
        assert private_context.scheme == "CKKS"
        assert private_context.is_private is True
        assert private_context.is_public is False

        # Check public context
        assert isinstance(public_context, FHEContext)
        assert public_context.scheme == "CKKS"
        assert public_context.is_private is False
        assert public_context.is_public is True

    def test_bfv_context_generation(self):
        """Test BFV context generation returns private and public contexts."""
        pfunc = _create_test_pfunc(scheme="BFV")
        result = _fhe_keygen(pfunc)

        assert len(result) == 2
        private_context, public_context = result

        # Check private context
        assert isinstance(private_context, FHEContext)
        assert private_context.scheme == "BFV"
        assert private_context.is_private is True
        assert private_context.is_public is False

        # Check public context
        assert isinstance(public_context, FHEContext)
        assert public_context.scheme == "BFV"
        assert public_context.is_private is False
        assert public_context.is_public is True

    def test_context_with_custom_parameters(self):
        """Test context generation with custom parameters."""
        pfunc = _create_test_pfunc(
            scheme="CKKS",
            poly_modulus_degree=4096,
            coeff_mod_bit_sizes=[40, 20, 40],
            global_scale=2**20,
        )
        result = _fhe_keygen(pfunc)

        assert len(result) == 2
        private_context, public_context = result
        assert isinstance(private_context, FHEContext)
        assert private_context.scheme == "CKKS"
        assert private_context.global_scale == 2**20
        assert isinstance(public_context, FHEContext)
        assert public_context.scheme == "CKKS"

    def test_context_serialization(self):
        """Test context serialization and public context."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        result = _fhe_keygen(pfunc)
        private_context, public_context = result

        # Test serialization of private context
        serialized = private_context.serialize()
        assert isinstance(serialized, bytes)

        # Test serialization of public context (should not have secret key)
        serialized_public = public_context.serialize()
        assert isinstance(serialized_public, bytes)

    def test_unsupported_scheme(self):
        """Test error handling for unsupported schemes."""
        pfunc = _create_test_pfunc(scheme="INVALID")
        with pytest.raises(ValueError, match="Unsupported FHE scheme"):
            _fhe_keygen(pfunc)


class TestFHEEncryptDecrypt:
    """Test FHE encryption and decryption operations."""

    @pytest.fixture
    def ckks_context(self):
        """Fixture for CKKS context."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        result = _fhe_keygen(pfunc)
        return result[0]  # Return private context

    @pytest.fixture
    def bfv_context(self):
        """Fixture for BFV context."""
        pfunc = _create_test_pfunc(scheme="BFV")
        result = _fhe_keygen(pfunc)
        return result[0]  # Return private context

    def test_ckks_scalar_encrypt_decrypt(self, ckks_context):
        """Test CKKS scalar encryption and decryption."""
        pfunc = _create_test_pfunc()
        plaintext = np.array(3.14)

        # Encrypt
        result = _fhe_encrypt(pfunc, plaintext, ckks_context)
        assert len(result) == 1
        ciphertext = result[0]
        assert isinstance(ciphertext, CipherText)
        assert ciphertext.scheme == "CKKS"
        assert ciphertext.semantic_shape == ()

        # Decrypt
        result = _fhe_decrypt(pfunc, ciphertext, ckks_context)
        assert len(result) == 1
        decrypted = result[0]
        assert abs(decrypted.item() - 3.14) < 1e-3

    def test_ckks_vector_encrypt_decrypt(self, ckks_context):
        """Test CKKS vector encryption and decryption."""
        pfunc = _create_test_pfunc()
        plaintext = np.array([1.1, 2.2, 3.3])

        # Encrypt
        result = _fhe_encrypt(pfunc, plaintext, ckks_context)
        ciphertext = result[0]
        assert ciphertext.semantic_shape == (3,)

        # Decrypt
        result = _fhe_decrypt(pfunc, ciphertext, ckks_context)
        decrypted = result[0]
        assert decrypted.shape == (3,)
        np.testing.assert_allclose(decrypted, plaintext, atol=1e-3)

    def test_bfv_scalar_encrypt_decrypt(self, bfv_context):
        """Test BFV scalar encryption and decryption."""
        pfunc = _create_test_pfunc()
        plaintext = np.array(42)

        # Encrypt
        result = _fhe_encrypt(pfunc, plaintext, bfv_context)
        ciphertext = result[0]
        assert ciphertext.scheme == "BFV"
        assert ciphertext.semantic_shape == ()

        # Decrypt
        result = _fhe_decrypt(pfunc, ciphertext, bfv_context)
        decrypted = result[0]
        assert decrypted.item() == 42

    def test_bfv_vector_encrypt_decrypt(self, bfv_context):
        """Test BFV vector encryption and decryption."""
        pfunc = _create_test_pfunc()
        plaintext = np.array([10, 20, 30])

        # Encrypt
        result = _fhe_encrypt(pfunc, plaintext, bfv_context)
        ciphertext = result[0]
        assert ciphertext.semantic_shape == (3,)

        # Decrypt
        result = _fhe_decrypt(pfunc, ciphertext, bfv_context)
        decrypted = result[0]
        np.testing.assert_array_equal(decrypted, plaintext)

    def test_bfv_float_encryption_error(self, bfv_context):
        """Test that BFV rejects floating point data."""
        pfunc = _create_test_pfunc()
        plaintext = np.array(3.14)

        with pytest.raises(
            RuntimeError, match="BFV scheme requires integer semantic_dtype"
        ):
            _fhe_encrypt(pfunc, plaintext, bfv_context)

    def test_decrypt_without_secret_key(self, ckks_context):
        """Test decryption fails without secret key."""
        pfunc = _create_test_pfunc()
        plaintext = np.array(3.14)

        # Encrypt with private context
        result = _fhe_encrypt(pfunc, plaintext, ckks_context)
        ciphertext = result[0]

        # Get public context (without secret key)
        public_context = ckks_context.drop_secret_key()

        # Try to decrypt with public context
        with pytest.raises(
            ValueError, match="Context must have secret key for decryption"
        ):
            _fhe_decrypt(pfunc, ciphertext, public_context)

    def test_scheme_mismatch_error(self, ckks_context, bfv_context):
        """Test error when decrypting with mismatched scheme."""
        pfunc = _create_test_pfunc()
        plaintext = np.array([10, 20, 30])

        # Encrypt with BFV
        result = _fhe_encrypt(pfunc, plaintext, bfv_context)
        ciphertext = result[0]

        # Try to decrypt with CKKS context
        with pytest.raises(ValueError, match="Scheme mismatch"):
            _fhe_decrypt(pfunc, ciphertext, ckks_context)


class TestFHEArithmetic:
    """Test FHE arithmetic operations."""

    @pytest.fixture
    def ckks_context(self):
        """Fixture for CKKS context."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        result = _fhe_keygen(pfunc)
        return result[0]  # Return private context

    @pytest.fixture
    def bfv_context(self):
        """Fixture for BFV context."""
        pfunc = _create_test_pfunc(scheme="BFV")
        result = _fhe_keygen(pfunc)
        return result[0]  # Return private context

    def test_ckks_ciphertext_addition(self, ckks_context):
        """Test CKKS ciphertext + ciphertext addition."""
        pfunc = _create_test_pfunc()

        # Encrypt two values
        plaintext1 = np.array([1.1, 2.2])
        plaintext2 = np.array([3.3, 4.4])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Add ciphertexts
        result = _fhe_add(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = plaintext1 + plaintext2
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_ckks_ciphertext_plaintext_addition(self, ckks_context):
        """Test CKKS ciphertext + plaintext addition."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.1, 2.2])
        plaintext2 = np.array([3.3, 4.4])

        ct = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]

        # Add plaintext to ciphertext
        result = _fhe_add(pfunc, ct, plaintext2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = plaintext1 + plaintext2
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_bfv_ciphertext_addition(self, bfv_context):
        """Test BFV ciphertext + ciphertext addition."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([10, 20])
        plaintext2 = np.array([30, 40])

        ct1 = _fhe_encrypt(pfunc, plaintext1, bfv_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, bfv_context)[0]

        # Add ciphertexts
        result = _fhe_add(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, bfv_context)[0]
        expected = plaintext1 + plaintext2
        np.testing.assert_array_equal(decrypted, expected)

    def test_ckks_scalar_multiplication(self, ckks_context):
        """Test CKKS ciphertext * plaintext multiplication."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([1.5, 2.5])
        multiplier = np.array([2.0, 3.0])

        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        # Multiply by plaintext
        result = _fhe_mul(pfunc, ct, multiplier)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = plaintext * multiplier
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_ckks_ciphertext_multiplication(self, ckks_context):
        """Test CKKS ciphertext * ciphertext multiplication."""
        pfunc = _create_test_pfunc()

        # Encrypt two values
        plaintext1 = np.array([2.0, 3.0])
        plaintext2 = np.array([4.0, 5.0])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Multiply ciphertexts
        result = _fhe_mul(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = plaintext1 * plaintext2
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_bfv_scalar_multiplication(self, bfv_context):
        """Test BFV ciphertext * plaintext multiplication."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([10, 20])
        multiplier = np.array([3, 4])

        ct = _fhe_encrypt(pfunc, plaintext, bfv_context)[0]

        # Multiply by plaintext
        result = _fhe_mul(pfunc, ct, multiplier)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, bfv_context)[0]
        expected = plaintext * multiplier
        np.testing.assert_array_equal(decrypted, expected)

    def test_bfv_ciphertext_multiplication(self, bfv_context):
        """Test BFV ciphertext * ciphertext multiplication."""
        pfunc = _create_test_pfunc()

        # Encrypt two values
        plaintext1 = np.array([5, 6])
        plaintext2 = np.array([7, 8])

        ct1 = _fhe_encrypt(pfunc, plaintext1, bfv_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, bfv_context)[0]

        # Multiply ciphertexts
        result = _fhe_mul(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, bfv_context)[0]
        expected = plaintext1 * plaintext2
        np.testing.assert_array_equal(decrypted, expected)

    def test_scalar_operations(self, ckks_context):
        """Test operations with scalar values."""
        pfunc = _create_test_pfunc()

        # Scalar encryption and operations
        plaintext1 = np.array(5.0)
        plaintext2 = np.array(3.0)

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Addition
        result_add = _fhe_add(pfunc, ct1, ct2)[0]
        decrypted_add = _fhe_decrypt(pfunc, result_add, ckks_context)[0]
        assert abs(decrypted_add.item() - 8.0) < 1e-3

        # Multiplication (ciphertext * ciphertext)
        result_mul_ct = _fhe_mul(pfunc, ct1, ct2)[0]
        decrypted_mul_ct = _fhe_decrypt(pfunc, result_mul_ct, ckks_context)[0]
        assert abs(decrypted_mul_ct.item() - 15.0) < 1e-3

        # Multiplication (ciphertext * plaintext)
        multiplier = np.array(2.0)
        result_mul_pt = _fhe_mul(pfunc, ct1, multiplier)[0]
        decrypted_mul_pt = _fhe_decrypt(pfunc, result_mul_pt, ckks_context)[0]
        assert abs(decrypted_mul_pt.item() - 10.0) < 1e-3

    def test_shape_mismatch_errors(self, ckks_context):
        """Test error handling for shape mismatches."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0])
        plaintext2 = np.array([1.0, 2.0, 3.0])  # Different shape

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Addition should fail due to shape mismatch
        with pytest.raises(RuntimeError, match="must have same shape"):
            _fhe_add(pfunc, ct1, ct2)

        # Multiplication should also fail due to shape mismatch
        with pytest.raises(RuntimeError, match="must have same shape"):
            _fhe_mul(pfunc, ct1, ct2)

    def test_scheme_mismatch_in_addition(self, ckks_context, bfv_context):
        """Test error when adding ciphertexts with different schemes."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0])
        plaintext2 = np.array([10, 20])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, bfv_context)[0]

        # Addition should fail due to scheme mismatch
        with pytest.raises(RuntimeError, match="must use same scheme"):
            _fhe_add(pfunc, ct1, ct2)

    def test_scheme_mismatch_in_multiplication(self, ckks_context, bfv_context):
        """Test error when multiplying ciphertexts with different schemes."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0])
        plaintext2 = np.array([10, 20])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, bfv_context)[0]

        # Multiplication should fail due to scheme mismatch
        with pytest.raises(RuntimeError, match="must use same scheme"):
            _fhe_mul(pfunc, ct1, ct2)

    def test_bfv_float_multiplication_error(self, bfv_context):
        """Test that BFV rejects floating point multiplication."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([10, 20])
        multiplier = np.array([1.5, 2.5])  # Float multiplier

        ct = _fhe_encrypt(pfunc, plaintext, bfv_context)[0]

        with pytest.raises(
            RuntimeError, match="BFV scheme only supports integer plaintext"
        ):
            _fhe_mul(pfunc, ct, multiplier)


class TestFHEPublicContext:
    """Test FHE operations using public context (simulating multi-party computation)."""

    @pytest.fixture
    def ckks_contexts(self):
        """Fixture for CKKS private and public contexts."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        result = _fhe_keygen(pfunc)
        return result[0], result[1]  # private_context, public_context

    @pytest.fixture
    def bfv_contexts(self):
        """Fixture for BFV private and public contexts."""
        pfunc = _create_test_pfunc(scheme="BFV")
        result = _fhe_keygen(pfunc)
        return result[0], result[1]  # private_context, public_context

    def test_ckks_public_scalar_encryption(self, ckks_contexts):
        """Test CKKS scalar encryption with public context."""
        private_context, public_context = ckks_contexts
        pfunc = _create_test_pfunc()

        # Party A encrypts with private context
        plaintext_a = np.array(3.14)
        ct_a = _fhe_encrypt(pfunc, plaintext_a, private_context)[0]

        # Party B encrypts with public context (doesn't have secret key)
        plaintext_b = np.array(2.71)
        ct_b = _fhe_encrypt(pfunc, plaintext_b, public_context)[0]

        # Both parties can perform homomorphic addition
        result_ct = _fhe_add(pfunc, ct_a, ct_b)[0]

        # Only party with private key can decrypt
        decrypted = _fhe_decrypt(pfunc, result_ct, private_context)[0]
        expected = plaintext_a + plaintext_b
        assert abs(decrypted.item() - expected.item()) < 1e-3

    def test_ckks_public_1d_vector(self, ckks_contexts):
        """Test CKKS 1D vector encryption with public context."""
        private_context, public_context = ckks_contexts
        pfunc = _create_test_pfunc()

        # Party A's data: 1D float64 vector
        plaintext_a = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        ct_a = _fhe_encrypt(pfunc, plaintext_a, private_context)[0]

        # Party B's data: 1D float64 vector (encrypted with public context)
        plaintext_b = np.array([0.5, 1.0, 1.5], dtype=np.float64)
        ct_b = _fhe_encrypt(pfunc, plaintext_b, public_context)[0]

        # Homomorphic addition
        result_ct = _fhe_add(pfunc, ct_a, ct_b)[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, private_context)[0]
        expected = plaintext_a + plaintext_b
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_ckks_public_2d_matrix(self, ckks_contexts):
        """Test CKKS 2D matrix encryption with public context."""
        private_context, public_context = ckks_contexts
        pfunc = _create_test_pfunc()

        # Party A's data: 2D float32 matrix
        plaintext_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ct_a = _fhe_encrypt(pfunc, plaintext_a, private_context)[0]

        # Party B's data: 2D float32 matrix (encrypted with public context)
        plaintext_b = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
        ct_b = _fhe_encrypt(pfunc, plaintext_b, public_context)[0]

        # Homomorphic addition
        result_ct = _fhe_add(pfunc, ct_a, ct_b)[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, private_context)[0]
        expected = plaintext_a + plaintext_b
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_bfv_public_scalar_encryption(self, bfv_contexts):
        """Test BFV scalar encryption with public context."""
        private_context, public_context = bfv_contexts
        pfunc = _create_test_pfunc()

        # Party A encrypts int64 scalar with private context
        plaintext_a = np.array(100, dtype=np.int64)
        ct_a = _fhe_encrypt(pfunc, plaintext_a, private_context)[0]

        # Party B encrypts int64 scalar with public context
        plaintext_b = np.array(50, dtype=np.int64)
        ct_b = _fhe_encrypt(pfunc, plaintext_b, public_context)[0]

        # Homomorphic addition
        result_ct = _fhe_add(pfunc, ct_a, ct_b)[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, private_context)[0]
        expected = plaintext_a + plaintext_b
        assert decrypted.item() == expected.item()

    def test_bfv_public_1d_vector(self, bfv_contexts):
        """Test BFV 1D vector encryption with public context."""
        private_context, public_context = bfv_contexts
        pfunc = _create_test_pfunc()

        # Party A's data: 1D int32 vector
        plaintext_a = np.array([10, 20, 30], dtype=np.int32)
        ct_a = _fhe_encrypt(pfunc, plaintext_a, private_context)[0]

        # Party B's data: 1D int32 vector (encrypted with public context)
        plaintext_b = np.array([5, 15, 25], dtype=np.int32)
        ct_b = _fhe_encrypt(pfunc, plaintext_b, public_context)[0]

        # Homomorphic multiplication with plaintext
        multiplier = np.array([2, 2, 2], dtype=np.int32)
        ct_a_mul = _fhe_mul(pfunc, ct_a, multiplier)[0]

        # Add the multiplied result with ct_b
        result_ct = _fhe_add(pfunc, ct_a_mul, ct_b)[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, private_context)[0]
        expected = plaintext_a * multiplier + plaintext_b
        np.testing.assert_array_equal(decrypted, expected)

    def test_bfv_public_2d_matrix(self, bfv_contexts):
        """Test BFV 2D matrix encryption with public context."""
        private_context, public_context = bfv_contexts
        pfunc = _create_test_pfunc()

        # Party A's data: 2D int16 matrix
        plaintext_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        ct_a = _fhe_encrypt(pfunc, plaintext_a, private_context)[0]

        # Party B's data: 2D int16 matrix (encrypted with public context)
        plaintext_b = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16)
        ct_b = _fhe_encrypt(pfunc, plaintext_b, public_context)[0]

        # Homomorphic addition
        result_ct = _fhe_add(pfunc, ct_a, ct_b)[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, private_context)[0]
        expected = plaintext_a + plaintext_b
        np.testing.assert_array_equal(decrypted, expected)

    def test_mixed_context_computation(self, ckks_contexts):
        """Test computation with data encrypted by different contexts."""
        private_context, public_context = ckks_contexts
        pfunc = _create_test_pfunc()

        # Three parties with different data
        plaintext1 = np.array([1.0, 2.0, 3.0])
        plaintext2 = np.array([4.0, 5.0, 6.0])
        plaintext3 = np.array([7.0, 8.0, 9.0])

        # Party 1 uses private context
        ct1 = _fhe_encrypt(pfunc, plaintext1, private_context)[0]

        # Party 2 uses public context
        ct2 = _fhe_encrypt(pfunc, plaintext2, public_context)[0]

        # Party 3 also uses public context
        ct3 = _fhe_encrypt(pfunc, plaintext3, public_context)[0]

        # Compute: (ct1 + ct2) + ct3
        temp_ct = _fhe_add(pfunc, ct1, ct2)[0]
        result_ct = _fhe_add(pfunc, temp_ct, ct3)[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, private_context)[0]
        expected = plaintext1 + plaintext2 + plaintext3
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_public_context_ciphertext_multiplication(self, ckks_contexts):
        """Test ciphertext * ciphertext with public context."""
        private_context, public_context = ckks_contexts
        pfunc = _create_test_pfunc()

        # Party A encrypts with private context
        plaintext_a = np.array([2.0, 3.0])
        ct_a = _fhe_encrypt(pfunc, plaintext_a, private_context)[0]

        # Party B encrypts with public context
        plaintext_b = np.array([4.0, 5.0])
        ct_b = _fhe_encrypt(pfunc, plaintext_b, public_context)[0]

        # Homomorphic multiplication (ciphertext * ciphertext)
        result_ct = _fhe_mul(pfunc, ct_a, ct_b)[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, private_context)[0]
        expected = plaintext_a * plaintext_b
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_bfv_public_ciphertext_multiplication(self, bfv_contexts):
        """Test BFV ciphertext * ciphertext with public context."""
        private_context, public_context = bfv_contexts
        pfunc = _create_test_pfunc()

        # Party A encrypts with private context
        plaintext_a = np.array([5, 6], dtype=np.int32)
        ct_a = _fhe_encrypt(pfunc, plaintext_a, private_context)[0]

        # Party B encrypts with public context
        plaintext_b = np.array([7, 8], dtype=np.int32)
        ct_b = _fhe_encrypt(pfunc, plaintext_b, public_context)[0]

        # Homomorphic multiplication (ciphertext * ciphertext)
        result_ct = _fhe_mul(pfunc, ct_a, ct_b)[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, private_context)[0]
        expected = plaintext_a * plaintext_b
        np.testing.assert_array_equal(decrypted, expected)

    def test_public_context_cannot_decrypt(self, ckks_contexts):
        """Test that public context cannot decrypt data."""
        private_context, public_context = ckks_contexts
        pfunc = _create_test_pfunc()

        # Encrypt with public context
        plaintext = np.array([1.0, 2.0, 3.0])
        ciphertext = _fhe_encrypt(pfunc, plaintext, public_context)[0]

        # Try to decrypt with public context (should fail)
        with pytest.raises(
            ValueError, match="Context must have secret key for decryption"
        ):
            _fhe_decrypt(pfunc, ciphertext, public_context)


class TestFHEEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_context_type(self):
        """Test error when passing invalid context type."""
        pfunc = _create_test_pfunc()
        plaintext = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="must be an FHEContext instance"):
            _fhe_encrypt(pfunc, plaintext, "invalid_context")

    def test_invalid_ciphertext_type(self):
        """Test error when passing invalid ciphertext type."""
        pfunc = _create_test_pfunc()
        plaintext = np.array([1.0, 2.0])

        with pytest.raises(
            RuntimeError, match="At least one operand must be a CipherText"
        ):
            _fhe_mul(pfunc, "invalid_ciphertext", plaintext)

    def test_invalid_multiplication_operands(self):
        """Test error when neither operand is a ciphertext for multiplication."""
        pfunc = _create_test_pfunc()

        with pytest.raises(
            RuntimeError, match="At least one operand must be a CipherText"
        ):
            _fhe_mul(pfunc, np.array([1.0]), np.array([2.0]))

    def test_invalid_addition_operands(self):
        """Test error when neither operand is a ciphertext."""
        pfunc = _create_test_pfunc()

        with pytest.raises(
            RuntimeError, match="At least one operand must be a CipherText"
        ):
            _fhe_add(pfunc, np.array([1.0]), np.array([2.0]))


class TestFHEDot:
    """Test FHE dot product operations."""

    @pytest.fixture
    def ckks_context(self):
        """Create CKKS context for testing."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        result = _fhe_keygen(pfunc)
        return result[0]  # Private context

    @pytest.fixture
    def bfv_context(self):
        """Create BFV context for testing."""
        pfunc = _create_test_pfunc(scheme="BFV")
        result = _fhe_keygen(pfunc)
        return result[0]  # Private context

    def test_ckks_dot_1d_ct_ct(self, ckks_context):
        """Test CKKS 1D×1D ciphertext dot product (scalar result)."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0, 3.0])
        plaintext2 = np.array([4.0, 5.0, 6.0])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Dot product: 1*4 + 2*5 + 3*6 = 32
        result = _fhe_dot(pfunc, ct1, ct2)
        result_ct = result[0]

        # Check result shape is scalar
        assert result_ct.semantic_shape == ()

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = np.dot(plaintext1, plaintext2)
        assert abs(decrypted.item() - expected) < 1e-3

    def test_ckks_dot_1d_ct_pt(self, ckks_context):
        """Test CKKS 1D×1D ciphertext and plaintext dot product."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([2.0, 3.0, 4.0])
        plaintext2 = np.array([1.0, 2.0, 3.0])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]

        # Dot product: 2*1 + 3*2 + 4*3 = 20
        result = _fhe_dot(pfunc, ct1, plaintext2)
        result_ct = result[0]

        # Check result shape is scalar
        assert result_ct.semantic_shape == ()

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = np.dot(plaintext1, plaintext2)
        assert abs(decrypted.item() - expected) < 1e-3

    def test_bfv_dot_1d_ct_ct(self, bfv_context):
        """Test BFV 1D×1D ciphertext dot product."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([5, 6, 7])
        plaintext2 = np.array([1, 2, 3])

        ct1 = _fhe_encrypt(pfunc, plaintext1, bfv_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, bfv_context)[0]

        # Dot product: 5*1 + 6*2 + 7*3 = 38
        result = _fhe_dot(pfunc, ct1, ct2)
        result_ct = result[0]

        # Check result shape is scalar
        assert result_ct.semantic_shape == ()

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, bfv_context)[0]
        expected = np.dot(plaintext1, plaintext2)
        assert decrypted.item() == expected

    def test_ckks_dot_2d_1d_ct_pt(self, ckks_context):
        """Test CKKS 2D×1D matrix-vector dot product."""
        pfunc = _create_test_pfunc()

        # Matrix 2×3
        plaintext_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Vector 3
        plaintext_vector = np.array([1.0, 0.0, 1.0])

        ct_matrix = _fhe_encrypt(pfunc, plaintext_matrix, ckks_context)[0]

        # Dot product: [[1,2,3],[4,5,6]] · [1,0,1] = [4, 10]
        result = _fhe_dot(pfunc, ct_matrix, plaintext_vector)
        result_ct = result[0]

        # Check result shape
        assert result_ct.semantic_shape == (2,)

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = np.dot(plaintext_matrix, plaintext_vector)
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_bfv_dot_2d_1d_ct_pt(self, bfv_context):
        """Test BFV 2D×1D matrix-vector dot product."""
        pfunc = _create_test_pfunc()

        plaintext_matrix = np.array([[2, 3], [4, 5]])
        plaintext_vector = np.array([1, 2])

        ct_matrix = _fhe_encrypt(pfunc, plaintext_matrix, bfv_context)[0]

        # Dot product: [[2,3],[4,5]] · [1,2] = [8, 14]
        result = _fhe_dot(pfunc, ct_matrix, plaintext_vector)
        result_ct = result[0]

        # Check result shape
        assert result_ct.semantic_shape == (2,)

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, bfv_context)[0]
        expected = np.dot(plaintext_matrix, plaintext_vector)
        np.testing.assert_array_equal(decrypted, expected)

    def test_ckks_dot_2d_2d_ct_ct(self, ckks_context):
        """Test CKKS 2D×2D matrix-matrix dot product."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        plaintext2 = np.array([[5.0, 6.0], [7.0, 8.0]])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Dot product: [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        result = _fhe_dot(pfunc, ct1, ct2)
        result_ct = result[0]

        # Check result shape
        assert result_ct.semantic_shape == (2, 2)

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = np.dot(plaintext1, plaintext2)
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_dot_dimension_limit(self, ckks_context):
        """Test that dot product rejects tensors beyond 2D."""
        pfunc = _create_test_pfunc()

        # Create a 3D tensor (not supported)
        plaintext_3d = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        plaintext_1d = np.array([1.0, 2.0])

        ct_3d = _fhe_encrypt(pfunc, plaintext_3d, ckks_context)[0]

        with pytest.raises(
            RuntimeError,
            match="TenSEAL only supports dot product for tensors up to 2D×2D",
        ):
            _fhe_dot(pfunc, ct_3d, plaintext_1d)

    def test_dot_shape_mismatch(self, ckks_context):
        """Test dot product with incompatible shapes."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0, 3.0])
        plaintext2 = np.array([4.0, 5.0])  # Different length

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        with pytest.raises(RuntimeError, match="Incompatible dimension"):
            _fhe_dot(pfunc, ct1, ct2)

    def test_dot_scheme_mismatch(self, ckks_context, bfv_context):
        """Test dot product with mismatched schemes."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0])
        plaintext2 = np.array([10, 20])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, bfv_context)[0]

        with pytest.raises(
            RuntimeError, match="CipherText operands must use same scheme"
        ):
            _fhe_dot(pfunc, ct1, ct2)


class TestFHEPolyval:
    """Test FHE polynomial evaluation operations."""

    @pytest.fixture
    def ckks_context(self):
        """Create CKKS context for testing."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        result = _fhe_keygen(pfunc)
        return result[0]  # Private context

    @pytest.fixture
    def bfv_context(self):
        """Create BFV context for testing."""
        pfunc = _create_test_pfunc(scheme="BFV")
        result = _fhe_keygen(pfunc)
        return result[0]  # Private context

    def test_ckks_polyval_scalar(self, ckks_context):
        """Test CKKS polynomial evaluation on scalar."""
        pfunc = _create_test_pfunc()

        # x = 2.0, polynomial: 1 + 2x + 3x^2 = 1 + 4 + 12 = 17
        plaintext = np.array(2.0)
        coeffs = np.array([1.0, 2.0, 3.0])

        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = 1.0 + 2.0 * 2.0 + 3.0 * 2.0 * 2.0
        assert abs(decrypted.item() - expected) < 1e-2

    def test_ckks_polyval_vector(self, ckks_context):
        """Test CKKS polynomial evaluation on vector."""
        pfunc = _create_test_pfunc()

        # x = [1.0, 2.0, 3.0], polynomial: 1 + 2x = [3, 5, 7]
        plaintext = np.array([1.0, 2.0, 3.0])
        coeffs = np.array([1.0, 2.0])

        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = np.polyval([2.0, 1.0], plaintext)  # np.polyval uses reversed coeffs
        np.testing.assert_allclose(decrypted, expected, atol=1e-2)

    def test_bfv_polyval_scalar(self, bfv_context):
        """Test BFV polynomial evaluation on scalar."""
        pfunc = _create_test_pfunc()

        # x = 3, polynomial: 2 + 3x = 2 + 9 = 11
        plaintext = np.array(3)
        coeffs = np.array([2, 3])

        ct = _fhe_encrypt(pfunc, plaintext, bfv_context)[0]

        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, bfv_context)[0]
        expected = 2 + 3 * 3
        assert decrypted.item() == expected

    def test_bfv_polyval_vector(self, bfv_context):
        """Test BFV polynomial evaluation on vector."""
        pfunc = _create_test_pfunc()

        # x = [1, 2, 3], polynomial: 1 + 2x + x^2
        plaintext = np.array([1, 2, 3])
        coeffs = np.array([1, 2, 1])

        ct = _fhe_encrypt(pfunc, plaintext, bfv_context)[0]

        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, bfv_context)[0]
        # For x=1: 1 + 2*1 + 1*1 = 4
        # For x=2: 1 + 2*2 + 1*4 = 9
        # For x=3: 1 + 2*3 + 1*9 = 16
        expected = np.array([4, 9, 16])
        np.testing.assert_array_equal(decrypted, expected)

    def test_polyval_constant(self, ckks_context):
        """Test that constant polynomial (single coefficient) raises error."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([1.0, 2.0, 3.0])
        coeffs = np.array([5.0])  # Constant polynomial (not supported by TenSEAL)

        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        with pytest.raises(RuntimeError, match="Polynomial must have degree >= 1"):
            _fhe_polyval(pfunc, ct, coeffs)

    def test_polyval_high_degree(self, ckks_context):
        """Test polynomial evaluation with higher degree."""
        pfunc = _create_test_pfunc()

        # x = 2.0, polynomial: 1 + x + x^2 + x^3 + x^4 = 1 + 2 + 4 + 8 + 16 = 31
        plaintext = np.array(2.0)
        coeffs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _fhe_decrypt(pfunc, result_ct, ckks_context)[0]
        expected = 1.0 + 2.0 + 4.0 + 8.0 + 16.0
        assert abs(decrypted.item() - expected) < 1e-1

    def test_polyval_empty_coeffs(self, ckks_context):
        """Test that empty coefficients array raises error."""
        pfunc = _create_test_pfunc()

        plaintext = np.array(2.0)
        coeffs = np.array([])

        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        with pytest.raises(RuntimeError, match="Coefficients array cannot be empty"):
            _fhe_polyval(pfunc, ct, coeffs)

    def test_polyval_2d_coeffs(self, ckks_context):
        """Test that 2D coefficients array raises error."""
        pfunc = _create_test_pfunc()

        plaintext = np.array(2.0)
        coeffs = np.array([[1.0, 2.0], [3.0, 4.0]])

        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        with pytest.raises(RuntimeError, match="Coefficients must be 1D array"):
            _fhe_polyval(pfunc, ct, coeffs)

    def test_bfv_polyval_float_coeffs(self, bfv_context):
        """Test that BFV rejects floating point coefficients."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([1, 2, 3])
        coeffs = np.array([1.5, 2.5])  # Float coeffs

        ct = _fhe_encrypt(pfunc, plaintext, bfv_context)[0]

        with pytest.raises(
            RuntimeError, match="BFV scheme only supports integer coefficients"
        ):
            _fhe_polyval(pfunc, ct, coeffs)
