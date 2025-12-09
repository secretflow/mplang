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

"""Tests for FHE Vector backend using TenSEAL CKKSVector/BFVVector.

This test suite validates the vector-based FHE operations which only support
1D data (scalars and vectors).
"""

import numpy as np
import pytest

from mplang.v1.core import PFunction, TensorType
from mplang.v1.kernels.base import list_kernels
from mplang.v1.kernels.fhe import (
    CipherText,
    FHEContext,
    _fhe_add,
    _fhe_decrypt,
    _fhe_dot,
    _fhe_encrypt,
    _fhe_keygen,
    _fhe_mul,
    _fhe_negate,
    _fhe_polyval,
    _fhe_square,
    _fhe_sub,
)


def _as_np(x):
    """Return numpy array/scalar from TensorValue or pass-through numpy."""
    from mplang.v1.kernels.value import TensorValue

    if isinstance(x, TensorValue):
        return x.to_numpy()
    return x


def _create_test_pfunc(**attrs) -> PFunction:
    """Helper to create a test PFunction with dummy type info."""
    dummy_tensor = TensorType.from_obj(np.array(0.0))
    return PFunction(
        fn_type="test.fhe", ins_info=(), outs_info=(dummy_tensor,), **attrs
    )


class TestFHEVecKernelRegistry:
    """Test FHE vector kernel registration."""

    def test_kernel_registry(self):
        """Test that all FHE vector kernels are properly registered."""
        for name in [
            "fhe.keygen",
            "fhe.encrypt",
            "fhe.decrypt",
            "fhe.add",
            "fhe.sub",
            "fhe.mul",
            "fhe.dot",
            "fhe.polyval",
            "fhe.negate",
            "fhe.square",
        ]:
            assert name in list_kernels(), f"Kernel {name} not registered"


class TestFHEVecContext:
    """Test FHE vector context generation and management."""

    def test_ckks_context_generation(self):
        """Test CKKS context generation returns private, public, and eval contexts."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        result = _fhe_keygen(pfunc)

        assert len(result) == 3
        private_context, public_context, eval_context = result

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

        # Check eval context (same as public for TenSEAL)
        assert isinstance(eval_context, FHEContext)
        assert eval_context.scheme == "CKKS"

    def test_bfv_context_generation(self):
        """Test BFV context generation returns private, public, and eval contexts."""
        pfunc = _create_test_pfunc(scheme="BFV")
        result = _fhe_keygen(pfunc)

        assert len(result) == 3
        private_context, public_context, _ = result

        # Check private context
        assert isinstance(private_context, FHEContext)
        assert private_context.scheme == "BFV"
        assert private_context.is_private is True

        # Check public context
        assert isinstance(public_context, FHEContext)
        assert public_context.scheme == "BFV"
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

        assert len(result) == 3
        private_context = result[0]
        assert isinstance(private_context, FHEContext)
        assert private_context.scheme == "CKKS"
        assert private_context.global_scale == 2**20

    def test_context_serialization(self):
        """Test context serialization and public context."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        result = _fhe_keygen(pfunc)
        private_context, public_context, _ = result

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


class TestFHEVecEncryptDecrypt:
    """Test FHE vector encryption and decryption operations."""

    @pytest.fixture
    def ckks_context(self):
        pfunc = _create_test_pfunc(scheme="CKKS")
        return _fhe_keygen(pfunc)[0]  # Private context

    @pytest.fixture
    def bfv_context(self):
        pfunc = _create_test_pfunc(scheme="BFV")
        return _fhe_keygen(pfunc)[0]  # Private context

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
        decrypted = _as_np(result[0])
        assert abs(decrypted.item() - 3.14) < 1e-3

    def test_ckks_vector_encrypt_decrypt(self, ckks_context):
        """Test CKKS 1D vector encryption and decryption."""
        pfunc = _create_test_pfunc()
        plaintext = np.array([1.1, 2.2, 3.3])

        # Encrypt
        result = _fhe_encrypt(pfunc, plaintext, ckks_context)
        ciphertext = result[0]
        assert ciphertext.semantic_shape == (3,)

        # Decrypt
        result = _fhe_decrypt(pfunc, ciphertext, ckks_context)
        decrypted = _as_np(result[0])
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
        decrypted = _as_np(result[0])
        assert decrypted.item() == 42

    def test_bfv_vector_encrypt_decrypt(self, bfv_context):
        """Test BFV 1D vector encryption and decryption."""
        pfunc = _create_test_pfunc()
        plaintext = np.array([10, 20, 30])

        # Encrypt
        result = _fhe_encrypt(pfunc, plaintext, bfv_context)
        ciphertext = result[0]
        assert ciphertext.semantic_shape == (3,)

        # Decrypt
        result = _fhe_decrypt(pfunc, ciphertext, bfv_context)
        decrypted = _as_np(result[0])
        np.testing.assert_array_equal(decrypted, plaintext)

    def test_2d_array_encryption_error(self, ckks_context):
        """Test that 2D arrays are rejected by vector backend."""
        pfunc = _create_test_pfunc()
        plaintext = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(RuntimeError, match="only supports 1D data"):
            _fhe_encrypt(pfunc, plaintext, ckks_context)

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


class TestFHEVecArithmetic:
    """Test FHE vector arithmetic operations."""

    @pytest.fixture
    def ckks_context(self):
        pfunc = _create_test_pfunc(scheme="CKKS")
        return _fhe_keygen(pfunc)[0]

    @pytest.fixture
    def bfv_context(self):
        pfunc = _create_test_pfunc(scheme="BFV")
        return _fhe_keygen(pfunc)[0]

    def test_ckks_ciphertext_addition(self, ckks_context):
        """Test CKKS ciphertext + ciphertext addition."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.1, 2.2])
        plaintext2 = np.array([3.3, 4.4])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Add ciphertexts
        result = _fhe_add(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
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
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = plaintext1 + plaintext2
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_ckks_scalar_addition(self, ckks_context):
        """Test CKKS scalar addition."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array(5.0)
        plaintext2 = np.array(3.0)

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Add scalars
        result = _fhe_add(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        assert abs(decrypted.item() - 8.0) < 1e-3

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
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, bfv_context)[0])
        expected = plaintext1 + plaintext2
        np.testing.assert_array_equal(decrypted, expected)

    def test_ckks_subtraction(self, ckks_context):
        """Test CKKS subtraction."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([5.5, 7.7])
        plaintext2 = np.array([2.2, 3.3])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Subtract ciphertexts
        result = _fhe_sub(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = plaintext1 - plaintext2
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_ckks_scalar_multiplication(self, ckks_context):
        """Test CKKS ciphertext × scalar multiplication."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([2.0, 3.0])
        scalar = 5.0

        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        # Multiply by scalar
        result = _fhe_mul(pfunc, ct, scalar)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = plaintext * scalar
        np.testing.assert_allclose(decrypted, expected, atol=1e-2)

    def test_ckks_ciphertext_multiplication(self, ckks_context):
        """Test CKKS ciphertext × ciphertext multiplication."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([2.0, 3.0])
        plaintext2 = np.array([4.0, 5.0])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Multiply ciphertexts
        result = _fhe_mul(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = plaintext1 * plaintext2
        np.testing.assert_allclose(decrypted, expected, atol=1e-2)

    def test_bfv_scalar_multiplication(self, bfv_context):
        """Test BFV ciphertext × scalar multiplication."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([5, 10])
        scalar = 3

        ct = _fhe_encrypt(pfunc, plaintext, bfv_context)[0]

        # Multiply by scalar
        result = _fhe_mul(pfunc, ct, scalar)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, bfv_context)[0])
        expected = plaintext * scalar
        np.testing.assert_array_equal(decrypted, expected)

    def test_bfv_ciphertext_multiplication(self, bfv_context):
        """Test BFV ciphertext × ciphertext multiplication."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([2, 3])
        plaintext2 = np.array([4, 5])

        ct1 = _fhe_encrypt(pfunc, plaintext1, bfv_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, bfv_context)[0]

        # Multiply ciphertexts
        result = _fhe_mul(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, bfv_context)[0])
        expected = plaintext1 * plaintext2
        np.testing.assert_array_equal(decrypted, expected)

    def test_shape_mismatch_errors(self, ckks_context):
        """Test errors for shape mismatches."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0])
        plaintext2 = np.array([3.0, 4.0, 5.0])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Try to add with mismatched shapes
        with pytest.raises(RuntimeError, match="same shape"):
            _fhe_add(pfunc, ct1, ct2)

    def test_negation(self, ckks_context):
        """Test ciphertext negation."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([1.5, -2.5, 3.5])
        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        # Negate
        result = _fhe_negate(pfunc, ct)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = -plaintext
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_square(self, ckks_context):
        """Test ciphertext squaring."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([2.0, 3.0, -4.0])
        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        # Square
        result = _fhe_square(pfunc, ct)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = plaintext**2
        np.testing.assert_allclose(decrypted, expected, atol=1e-2)


class TestFHEVecDot:
    """Test FHE vector dot product operations (1D only)."""

    @pytest.fixture
    def ckks_context(self):
        pfunc = _create_test_pfunc(scheme="CKKS")
        return _fhe_keygen(pfunc)[0]

    @pytest.fixture
    def bfv_context(self):
        pfunc = _create_test_pfunc(scheme="BFV")
        return _fhe_keygen(pfunc)[0]

    def test_ckks_dot_ct_ct(self, ckks_context):
        """Test CKKS dot product of two encrypted vectors."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0, 3.0])
        plaintext2 = np.array([4.0, 5.0, 6.0])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Dot product
        result = _fhe_dot(pfunc, ct1, ct2)
        result_ct = result[0]

        # Result should be scalar
        assert result_ct.semantic_shape == ()

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = np.dot(plaintext1, plaintext2)
        assert abs(decrypted.item() - expected) < 0.1

    def test_ckks_dot_ct_pt(self, ckks_context):
        """Test CKKS dot product of encrypted and plaintext vectors."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([2.0, 3.0, 4.0])
        plaintext2 = np.array([1.0, 2.0, 3.0])

        ct = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]

        # Dot product with plaintext
        result = _fhe_dot(pfunc, ct, plaintext2)
        result_ct = result[0]

        # Result should be scalar
        assert result_ct.semantic_shape == ()

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = np.dot(plaintext1, plaintext2)
        assert abs(decrypted.item() - expected) < 0.1

    def test_bfv_dot_ct_ct(self, bfv_context):
        """Test BFV dot product of two encrypted vectors."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1, 2, 3])
        plaintext2 = np.array([4, 5, 6])

        ct1 = _fhe_encrypt(pfunc, plaintext1, bfv_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, bfv_context)[0]

        # Dot product
        result = _fhe_dot(pfunc, ct1, ct2)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, bfv_context)[0])
        expected = np.dot(plaintext1, plaintext2)
        assert decrypted.item() == expected

    def test_dot_scalar_error(self, ckks_context):
        """Test that dot product rejects scalars."""
        pfunc = _create_test_pfunc()

        plaintext = np.array(5.0)
        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        # Try dot product with scalar
        with pytest.raises(RuntimeError, match="Dot product requires 1D vectors"):
            _fhe_dot(pfunc, ct, ct)

    def test_dot_dimension_mismatch(self, ckks_context):
        """Test dot product dimension mismatch error."""
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0])
        plaintext2 = np.array([3.0, 4.0, 5.0])

        ct1 = _fhe_encrypt(pfunc, plaintext1, ckks_context)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, ckks_context)[0]

        # Try dot product with mismatched dimensions
        with pytest.raises(RuntimeError, match="Dot product dimension mismatch"):
            _fhe_dot(pfunc, ct1, ct2)


class TestFHEVecPolyval:
    """Test FHE vector polynomial evaluation operations."""

    @pytest.fixture
    def ckks_context(self):
        pfunc = _create_test_pfunc(scheme="CKKS")
        return _fhe_keygen(pfunc)[0]

    @pytest.fixture
    def bfv_context(self):
        pfunc = _create_test_pfunc(scheme="BFV")
        return _fhe_keygen(pfunc)[0]

    def test_ckks_polyval_scalar(self, ckks_context):
        """Test CKKS polynomial evaluation on scalar."""
        pfunc = _create_test_pfunc()

        plaintext = np.array(2.0)
        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        # Evaluate p(x) = 1 + 2x + 3x²
        coeffs = np.array([1.0, 2.0, 3.0])
        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = 1.0 + 2.0 * 2.0 + 3.0 * (2.0**2)  # = 1 + 4 + 12 = 17
        assert abs(decrypted.item() - expected) < 0.1

    def test_ckks_polyval_vector(self, ckks_context):
        """Test CKKS polynomial evaluation on vector."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([1.0, 2.0, 3.0])
        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        # Evaluate p(x) = 2 + 3x
        coeffs = np.array([2.0, 3.0])
        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = 2.0 + 3.0 * plaintext  # Element-wise
        np.testing.assert_allclose(decrypted, expected, atol=0.1)

    def test_bfv_polyval_scalar(self, bfv_context):
        """Test BFV polynomial evaluation on scalar."""
        pfunc = _create_test_pfunc()

        plaintext = np.array(3)
        ct = _fhe_encrypt(pfunc, plaintext, bfv_context)[0]

        # Evaluate p(x) = 1 + 2x
        coeffs = np.array([1, 2])
        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, bfv_context)[0])
        expected = 1 + 2 * 3  # = 7
        assert decrypted.item() == expected

    def test_bfv_polyval_vector(self, bfv_context):
        """Test BFV polynomial evaluation on vector."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([1, 2, 3])
        ct = _fhe_encrypt(pfunc, plaintext, bfv_context)[0]

        # Evaluate p(x) = 5 + 2x
        coeffs = np.array([5, 2])
        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, bfv_context)[0])
        expected = 5 + 2 * plaintext
        np.testing.assert_array_equal(decrypted, expected)

    @pytest.mark.skip(reason="TenSEAL has a bug with constant polynomials (degree 0)")
    def test_polyval_constant(self, ckks_context):
        """Test polynomial evaluation with constant (degree 0).

        NOTE: This test is skipped due to a TenSEAL bug where polyval()
        fails for constant polynomials (single coefficient).
        Error: "vector::reserve"

        For constant polynomials in practice, use scalar multiplication instead:
        result = ct * 0 + constant
        """
        pfunc = _create_test_pfunc()

        plaintext = np.array([1.0, 2.0, 3.0])
        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        # Constant polynomial p(x) = 5
        coeffs = np.array([5.0])
        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify (result should be all 5s)
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])
        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_allclose(decrypted, expected, atol=0.1)

    def test_polyval_sigmoid_approximation(self, ckks_context):
        """Test sigmoid polynomial approximation."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([0.0, 1.0, -1.0])
        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        # Sigmoid approximation: sigmoid(x) ≈ 0.5 + 0.197x - 0.004x³
        coeffs = np.array([0.5, 0.197, 0.0, -0.004])
        result = _fhe_polyval(pfunc, ct, coeffs)
        result_ct = result[0]

        # Decrypt and verify
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, ckks_context)[0])

        # Expected values (approximate)
        # sigmoid(0) ≈ 0.5
        # sigmoid(1) ≈ 0.5 + 0.197 - 0.004 = 0.693
        # sigmoid(-1) ≈ 0.5 - 0.197 + 0.004 = 0.307
        assert abs(decrypted[0] - 0.5) < 0.1
        assert abs(decrypted[1] - 0.693) < 0.1
        assert abs(decrypted[2] - 0.307) < 0.1

    def test_polyval_empty_coeffs_error(self, ckks_context):
        """Test error for empty coefficient array."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([1.0])
        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        coeffs = np.array([])
        with pytest.raises(RuntimeError, match="cannot be empty"):
            _fhe_polyval(pfunc, ct, coeffs)

    def test_polyval_2d_coeffs_error(self, ckks_context):
        """Test error for 2D coefficient array."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([1.0])
        ct = _fhe_encrypt(pfunc, plaintext, ckks_context)[0]

        coeffs = np.array([[1.0, 2.0]])
        with pytest.raises(RuntimeError, match="must be 1D array"):
            _fhe_polyval(pfunc, ct, coeffs)

    def test_bfv_polyval_float_coeffs_error(self, bfv_context):
        """Test that BFV rejects float coefficients."""
        pfunc = _create_test_pfunc()

        plaintext = np.array([1])
        ct = _fhe_encrypt(pfunc, plaintext, bfv_context)[0]

        coeffs = np.array([1.5, 2.5])
        with pytest.raises(RuntimeError, match="BFV scheme requires integer"):
            _fhe_polyval(pfunc, ct, coeffs)


class TestFHEVecPublicContext:
    """Test FHE vector operations using public context (multi-party simulation)."""

    @pytest.fixture
    def ckks_contexts(self):
        pfunc = _create_test_pfunc(scheme="CKKS")
        private_ctx, public_ctx, _ = _fhe_keygen(pfunc)
        return private_ctx, public_ctx

    @pytest.fixture
    def bfv_contexts(self):
        pfunc = _create_test_pfunc(scheme="BFV")
        private_ctx, public_ctx, _ = _fhe_keygen(pfunc)
        return private_ctx, public_ctx

    def test_ckks_public_scalar_encryption(self, ckks_contexts):
        """Test CKKS encryption with public context (scalar)."""
        private_ctx, public_ctx = ckks_contexts
        pfunc = _create_test_pfunc()

        plaintext = np.array(3.14)

        # Encrypt with public context
        ct = _fhe_encrypt(pfunc, plaintext, public_ctx)[0]
        assert isinstance(ct, CipherText)

        # Decrypt with private context
        decrypted = _as_np(_fhe_decrypt(pfunc, ct, private_ctx)[0])
        assert abs(decrypted.item() - 3.14) < 1e-3

    def test_ckks_public_vector(self, ckks_contexts):
        """Test CKKS encryption with public context (vector)."""
        private_ctx, public_ctx = ckks_contexts
        pfunc = _create_test_pfunc()

        plaintext = np.array([1.1, 2.2, 3.3])

        # Encrypt with public context
        ct = _fhe_encrypt(pfunc, plaintext, public_ctx)[0]

        # Decrypt with private context
        decrypted = _as_np(_fhe_decrypt(pfunc, ct, private_ctx)[0])
        np.testing.assert_allclose(decrypted, plaintext, atol=1e-3)

    def test_mixed_context_computation(self, ckks_contexts):
        """Test computation with ciphertexts from different contexts."""
        private_ctx, public_ctx = ckks_contexts
        pfunc = _create_test_pfunc()

        plaintext1 = np.array([1.0, 2.0])
        plaintext2 = np.array([3.0, 4.0])

        # Encrypt one with private, one with public
        ct1 = _fhe_encrypt(pfunc, plaintext1, private_ctx)[0]
        ct2 = _fhe_encrypt(pfunc, plaintext2, public_ctx)[0]

        # Add them
        result_ct = _fhe_add(pfunc, ct1, ct2)[0]

        # Decrypt
        decrypted = _as_np(_fhe_decrypt(pfunc, result_ct, private_ctx)[0])
        expected = plaintext1 + plaintext2
        np.testing.assert_allclose(decrypted, expected, atol=1e-3)

    def test_public_context_cannot_decrypt(self, ckks_contexts):
        """Test that public context cannot decrypt."""
        private_ctx, public_ctx = ckks_contexts
        pfunc = _create_test_pfunc()

        plaintext = np.array([1.0, 2.0])
        ct = _fhe_encrypt(pfunc, plaintext, private_ctx)[0]

        # Try to decrypt with public context
        with pytest.raises(ValueError, match="must have secret key"):
            _fhe_decrypt(pfunc, ct, public_ctx)


class TestFHEVecEdgeCases:
    """Test edge cases and error conditions for vector backend."""

    def test_invalid_context_type(self):
        """Test error for invalid context type."""
        pfunc = _create_test_pfunc()
        plaintext = np.array([1.0])

        with pytest.raises(TypeError, match="Expected FHEContext"):
            _fhe_encrypt(pfunc, plaintext, "not_a_context")

    def test_invalid_ciphertext_type(self):
        """Test error for invalid ciphertext type."""
        pfunc = _create_test_pfunc(scheme="CKKS")
        context = _fhe_keygen(pfunc)[0]

        with pytest.raises(TypeError, match="Expected CipherText"):
            _fhe_decrypt(pfunc, "not_a_ciphertext", context)
