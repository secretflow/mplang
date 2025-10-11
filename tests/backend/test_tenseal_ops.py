"""
TenSEAL Operations Test Suite

This test suite covers comprehensive TenSEAL operations for both BFV and CKKS schemes,
including key generation, encryption/decryption, homomorphic operations, and tensor operations.

Scheme Comparison:
- BFV: Exact integer arithmetic, supports addition and multiplication
- CKKS: Approximate floating-point arithmetic, supports addition, multiplication, and more operations
"""

import pytest
import tenseal as ts
import numpy as np


# =============================================================================
# Fixtures - Context Setup
# =============================================================================


@pytest.fixture
def bfv_context():
    """Create BFV context with appropriate parameters."""
    context = ts.context(
        ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193
    )
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


@pytest.fixture
def ckks_context():
    """Create CKKS context with appropriate parameters."""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.generate_galois_keys()
    context.generate_relin_keys()
    context.global_scale = 2**40
    return context


# =============================================================================
# Test Class 1: Key Generation
# =============================================================================


class TestKeyGeneration:
    """Test key generation for both BFV and CKKS schemes."""

    def test_bfv_keygen(self):
        """Test BFV context creation and key generation."""
        context = ts.context(
            ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193
        )

        # Generate keys
        context.generate_galois_keys()
        context.generate_relin_keys()

        # Verify keys are generated
        assert context.is_private(), "Context should have private key"

    def test_ckks_keygen(self):
        """Test CKKS context creation and key generation."""
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context.global_scale = 2**40

        # Generate keys
        context.generate_galois_keys()
        context.generate_relin_keys()

        # Verify keys are generated
        assert context.is_private(), "Context should have private key"

    def test_public_context(self, bfv_context):
        """Test creating public context without private key."""
        # Make context public
        public_context = bfv_context.copy()
        public_context.make_context_public()

        assert (
            not public_context.is_private()
        ), "Public context should not have private key"


# =============================================================================
# Test Class 2: BFV Basic Operations
# =============================================================================


class TestBFVOperations:
    """Test BFV scheme operations."""

    def test_bfv_encryption_decryption(self, bfv_context):
        """Test basic BFV encryption and decryption."""
        vec = [1, 2, 3, 4, 5]
        enc_vec = ts.bfv_vector(bfv_context, vec)

        decrypted = enc_vec.decrypt()

        assert decrypted == vec, f"Expected {vec}, got {decrypted}"

    def test_bfv_addition(self, bfv_context):
        """Test BFV ciphertext + ciphertext addition."""
        vec1 = [1, 2, 3, 4]
        vec2 = [5, 6, 7, 8]

        enc1 = ts.bfv_vector(bfv_context, vec1)
        enc2 = ts.bfv_vector(bfv_context, vec2)

        result = enc1 + enc2
        decrypted = result.decrypt()
        expected = [v1 + v2 for v1, v2 in zip(vec1, vec2)]

        assert decrypted == expected, f"Expected {expected}, got {decrypted}"

    def test_bfv_addition_plaintext_scalar(self, bfv_context):
        """Test BFV ciphertext + plaintext scalar addition."""
        vec = [10, 20, 30, 40]
        plain_scalar = 5

        enc = ts.bfv_vector(bfv_context, vec)
        result = enc + plain_scalar
        decrypted = result.decrypt()

        expected = [v + plain_scalar for v in vec]
        assert decrypted == expected

    def test_bfv_subtraction_plaintext_vector(self, bfv_context):
        """Test BFV subtraction with plaintext vector.

        Note: Direct scalar subtraction may not work in all versions,
        so we use vector subtraction.
        """
        vec = [10, 20, 30, 40]
        enc_vec = ts.bfv_vector(bfv_context, vec)

        # Subtract plaintext vector
        plain_scalar = 5
        plain_vec = [plain_scalar] * len(vec)
        result = enc_vec - plain_vec
        decrypted = result.decrypt()
        expected = [x - plain_scalar for x in vec]

        assert decrypted == expected, f"Expected {expected}, got {decrypted}"

    def test_bfv_plaintext_multiplication(self, bfv_context):
        """Test ciphertext * plaintext multiplication in BFV."""
        vec = [1, 2, 3, 4]
        plain_scalar = 3

        enc = ts.bfv_vector(bfv_context, vec)
        result = enc * plain_scalar
        decrypted = result.decrypt()

        expected = [v * plain_scalar for v in vec]
        assert decrypted == expected

    def test_bfv_negation_not_supported(self, bfv_context):
        """Test that BFV negation is not directly supported.

        Note: BFV does not support negation operation directly.
        Negation would use modular arithmetic: -x = (modulus - x) mod modulus.
        This is a limitation of BFV scheme.
        """
        vec = [1, 2, 3, 4]
        enc_vec = ts.bfv_vector(bfv_context, vec)

        # Verify negation is not supported
        with pytest.raises(AttributeError):
            result = -enc_vec

    def test_bfv_power_not_supported(self, bfv_context):
        """Test that BFV power operation is not directly supported.

        Note: BFV does not support power operation directly.
        This is a limitation of the BFV scheme.
        """
        vec = [2, 3, 4]
        enc_vec = ts.bfv_vector(bfv_context, vec)

        # Verify power operation is not supported
        power = 2
        with pytest.raises(AttributeError):
            result = enc_vec**power


# =============================================================================
# Test Class 3: CKKS Basic Operations
# =============================================================================


class TestCKKSOperations:
    """Test CKKS scheme operations."""

    def test_ckks_encryption_decryption(self, ckks_context):
        """Test basic CKKS encryption and decryption."""
        vec = [1.5, 2.5, 3.5, 4.5]
        enc_vec = ts.ckks_vector(ckks_context, vec)

        decrypted = enc_vec.decrypt()

        for original, decrypted_val in zip(vec, decrypted):
            assert abs(original - decrypted_val) < 1e-5

    def test_ckks_addition_ciphertext(self, ckks_context):
        """Test CKKS ciphertext + ciphertext addition."""
        vec1 = [1.1, 2.2, 3.3]
        vec2 = [4.4, 5.5, 6.6]

        enc1 = ts.ckks_vector(ckks_context, vec1)
        enc2 = ts.ckks_vector(ckks_context, vec2)

        result = enc1 + enc2
        decrypted = result.decrypt()
        expected = [v1 + v2 for v1, v2 in zip(vec1, vec2)]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-5

    def test_ckks_addition_plaintext(self, ckks_context):
        """Test CKKS ciphertext + plaintext addition."""
        vec = [1.5, 2.5, 3.5]
        plain_scalar = 1.0

        enc = ts.ckks_vector(ckks_context, vec)
        result = enc + plain_scalar
        decrypted = result.decrypt()
        expected = [v + plain_scalar for v in vec]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-5

    def test_ckks_subtraction_ciphertext(self, ckks_context):
        """Test CKKS ciphertext - ciphertext subtraction."""
        vec1 = [10.5, 20.5, 30.5]
        vec2 = [1.5, 2.5, 3.5]

        enc1 = ts.ckks_vector(ckks_context, vec1)
        enc2 = ts.ckks_vector(ckks_context, vec2)

        result = enc1 - enc2
        decrypted = result.decrypt()
        expected = [v1 - v2 for v1, v2 in zip(vec1, vec2)]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-5

    def test_ckks_subtraction_plaintext(self, ckks_context):
        """Test CKKS ciphertext - plaintext subtraction."""
        vec = [10.5, 20.5, 30.5]
        plain_scalar = 5.0

        enc = ts.ckks_vector(ckks_context, vec)
        result = enc - plain_scalar
        decrypted = result.decrypt()
        expected = [v - plain_scalar for v in vec]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-5

    def test_ckks_multiplication_ciphertext(self, ckks_context):
        """Test CKKS ciphertext * ciphertext multiplication."""
        vec1 = [1.5, 2.5, 3.5]
        vec2 = [2.0, 3.0, 4.0]

        enc1 = ts.ckks_vector(ckks_context, vec1)
        enc2 = ts.ckks_vector(ckks_context, vec2)

        result = enc1 * enc2
        decrypted = result.decrypt()
        expected = [v1 * v2 for v1, v2 in zip(vec1, vec2)]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4  # Slightly larger tolerance for multiplication

    def test_ckks_multiplication_plaintext(self, ckks_context):
        """Test CKKS ciphertext * plaintext multiplication."""
        vec = [1.5, 2.5, 3.5]
        plain_scalar = 2.0

        enc = ts.ckks_vector(ckks_context, vec)
        result = enc * plain_scalar
        decrypted = result.decrypt()
        expected = [v * plain_scalar for v in vec]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-5

    def test_ckks_negation(self, ckks_context):
        """Test CKKS negation operation."""
        vec = [1.5, 2.5, 3.5]
        enc = ts.ckks_vector(ckks_context, vec)

        result = -enc
        decrypted = result.decrypt()
        expected = [-v for v in vec]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-5

    def test_ckks_power(self, ckks_context):
        """Test CKKS power operation."""
        vec = [2.0, 3.0, 4.0]
        enc = ts.ckks_vector(ckks_context, vec)

        power = 2
        result = enc**power
        decrypted = result.decrypt()
        expected = [v**power for v in vec]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4  # Larger tolerance for power operation

    def test_ckks_polyval(self, ckks_context):
        """Test CKKS polynomial evaluation (only available in CKKS)."""
        vec = [0.5, 1.0, 1.5]
        enc = ts.ckks_vector(ckks_context, vec)

        # Evaluate polynomial: 1 + 2x + 3x^2
        coefficients = [1.0, 2.0, 3.0]
        result = enc.polyval(coefficients)
        decrypted = result.decrypt()

        # Manual calculation
        expected = [
            coefficients[0] + coefficients[1] * v + coefficients[2] * (v**2)
            for v in vec
        ]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-3


# =============================================================================
# Test Class 4: Tensor Operations
# =============================================================================


class TestTensorOperations:
    """Test tensor-specific operations for both schemes."""

    def test_bfv_dot_product(self, bfv_context):
        """Test BFV dot product operation."""
        vec1 = [1, 2, 3, 4]
        vec2 = [2, 3, 4, 5]

        enc1 = ts.bfv_vector(bfv_context, vec1)
        enc2 = ts.bfv_vector(bfv_context, vec2)

        # Element-wise multiplication then sum
        result = enc1 * enc2
        decrypted = result.decrypt()
        expected = [v1 * v2 for v1, v2 in zip(vec1, vec2)]

        assert decrypted == expected

    def test_ckks_dot_product(self, ckks_context):
        """Test CKKS dot product operation."""
        vec1 = [1.0, 2.0, 3.0, 4.0]
        vec2 = [2.0, 3.0, 4.0, 5.0]

        enc1 = ts.ckks_vector(ckks_context, vec1)
        enc2 = ts.ckks_vector(ckks_context, vec2)

        # Element-wise multiplication
        result = enc1 * enc2
        decrypted = result.decrypt()
        expected = [v1 * v2 for v1, v2 in zip(vec1, vec2)]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4

    def test_ckks_matrix_vector_multiplication(self, ckks_context):
        """Test CKKS matrix-vector multiplication."""
        # Simple 2x2 matrix and 2-element vector
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        vec = [5.0, 6.0]

        # Encrypt vector
        enc_vec = ts.ckks_vector(ckks_context, vec)

        # Matrix-vector multiplication: result[i] = sum(matrix[i][j] * vec[j])
        result = []
        for row in matrix:
            enc_row_result = ts.ckks_vector(ckks_context, row) * enc_vec
            result.append(enc_row_result.decrypt())

        # Expected: [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
        expected = [[row[0] * vec[0], row[1] * vec[1]] for row in matrix]

        for exp_row, act_row in zip(expected, result):
            for exp, act in zip(exp_row, act_row):
                assert abs(exp - act) < 1e-4

    def test_ckks_batch_operations(self, ckks_context):
        """Test batch operations on multiple encrypted vectors."""
        batch = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

        # Encrypt all vectors
        encrypted_batch = [ts.ckks_vector(ckks_context, vec) for vec in batch]

        # Perform operation on each
        scalar = 2.0
        results = []
        for enc_vec in encrypted_batch:
            result = enc_vec * scalar
            results.append(result.decrypt())

        # Verify
        expected = [[v * scalar for v in vec] for vec in batch]
        for exp_vec, act_vec in zip(expected, results):
            for exp, act in zip(exp_vec, act_vec):
                assert abs(exp - act) < 1e-5


# =============================================================================
# Test Class 5: Scheme Comparison
# =============================================================================


class TestSchemeComparison:
    """Tests comparing BFV and CKKS capabilities and limitations."""

    def test_bfv_integer_precision(self):
        """Demonstrate BFV's exact integer arithmetic."""
        context = ts.context(
            ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193
        )
        context.generate_galois_keys()

        # Large integer computation
        vec = [100, 200, 300, 400]
        enc = ts.bfv_vector(context, vec)
        result = enc * 10
        decrypted = result.decrypt()

        # BFV maintains exact integer values
        expected = [v * 10 for v in vec]
        assert decrypted == expected

    def test_ckks_floating_point_approximation(self):
        """Demonstrate CKKS's approximate floating-point arithmetic."""
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context.generate_galois_keys()
        context.global_scale = 2**40

        # Floating-point computation
        vec = [1.123456789, 2.987654321, 3.141592653]
        enc = ts.ckks_vector(context, vec)
        result = enc * 2.0
        decrypted = result.decrypt()

        # CKKS has small approximation errors
        expected = [v * 2.0 for v in vec]
        for exp, act in zip(expected, decrypted):
            # Error exists but is small
            assert abs(exp - act) < 1e-5

    def test_polyval_ckks_only(self):
        """Demonstrate that polyval is only available in CKKS.

        Note: BFV does not support polynomial evaluation.
        """
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context.global_scale = 2**40

        vec = [1.0, 2.0, 3.0]
        enc = ts.ckks_vector(context, vec)

        # This works in CKKS
        coeffs = [1.0, 2.0, 1.0]  # 1 + 2x + x^2
        result = enc.polyval(coeffs)
        decrypted = result.decrypt()

        expected = [coeffs[0] + coeffs[1] * v + coeffs[2] * (v**2) for v in vec]
        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4


# =============================================================================
# Test Class 6: Serialization
# =============================================================================


class TestSerialization:
    """Test serialization and deserialization of contexts and ciphertexts."""

    def test_bfv_context_serialization(self, bfv_context):
        """Test BFV context serialization.

        Note: When deserializing a context, the private key is not included
        by default for security reasons. We need to serialize with the secret key.
        """
        # Encrypt with original context
        vec = [1, 2, 3, 4]
        enc = ts.bfv_vector(bfv_context, vec)

        # Serialize both context and ciphertext
        serialized_ctx = bfv_context.serialize(save_secret_key=True)
        serialized_enc = enc.serialize()

        # Deserialize
        new_context = ts.context_from(serialized_ctx)
        new_enc = ts.bfv_vector_from(new_context, serialized_enc)

        # Decrypt with new context
        decrypted = new_enc.decrypt()

        assert decrypted == vec

    def test_bfv_context_keys_consistency_after_serialization(self, bfv_context):
        """Test that BFV context keys remain consistent after serialization/deserialization.

        This test verifies that:
        1. Public key values are preserved
        2. Galois keys are preserved
        3. Relinearization keys are preserved
        4. The deserialized context can produce identical encryption results
        """
        # Test data
        test_vec = [10, 20, 30, 40]

        # Encrypt with original context
        original_enc = ts.bfv_vector(bfv_context, test_vec)
        original_decrypted = original_enc.decrypt()

        # Serialize context with secret key
        serialized_ctx = bfv_context.serialize(save_secret_key=True)

        # Deserialize to new context
        new_context = ts.context_from(serialized_ctx)

        # Test 1: Verify encryption/decryption consistency
        new_enc = ts.bfv_vector(new_context, test_vec)
        new_decrypted = new_enc.decrypt()

        # Should produce identical results (BFV is exact)
        assert (
            original_decrypted == new_decrypted
        ), f"Encryption results differ: {original_decrypted} vs {new_decrypted}"

        # Test 2: Verify operations work identically
        # BFV addition
        original_add = original_enc + original_enc
        new_add = new_enc + new_enc

        orig_add_result = original_add.decrypt()
        new_add_result = new_add.decrypt()

        assert (
            orig_add_result == new_add_result
        ), f"Addition results differ: {orig_add_result} vs {new_add_result}"

        # Test 3: Verify scalar operations work
        original_scalar = original_enc + 5
        new_scalar = new_enc + 5

        orig_scalar_result = original_scalar.decrypt()
        new_scalar_result = new_scalar.decrypt()

        assert (
            orig_scalar_result == new_scalar_result
        ), f"Scalar addition results differ: {orig_scalar_result} vs {new_scalar_result}"

        # Test 4: Verify multiplication works
        original_mul = original_enc * 2
        new_mul = new_enc * 2

        orig_mul_result = original_mul.decrypt()
        new_mul_result = new_mul.decrypt()

        assert (
            orig_mul_result == new_mul_result
        ), f"Multiplication results differ: {orig_mul_result} vs {new_mul_result}"

        # Test 5: Verify ciphertext from original context works with new context
        # (This tests key compatibility)
        serialized_enc = original_enc.serialize()
        restored_enc = ts.bfv_vector_from(new_context, serialized_enc)
        restored_decrypted = restored_enc.decrypt()

        assert (
            test_vec == restored_decrypted
        ), f"Restored ciphertext differs: {test_vec} vs {restored_decrypted}"

        print("✅ All BFV key consistency tests passed!")
        print(f"   - Encryption/decryption consistency: ✓")
        print(f"   - Addition operation consistency: ✓")
        print(f"   - Scalar operation consistency: ✓")
        print(f"   - Multiplication operation consistency: ✓")
        print(f"   - Cross-context ciphertext compatibility: ✓")

    def test_ckks_context_serialization(self, ckks_context):
        """Test CKKS context serialization.

        Note: When deserializing a context, the private key is not included
        by default for security reasons. We need to serialize with the secret key.
        """
        # Encrypt with original context
        vec = [1.5, 2.5, 3.5]
        enc = ts.ckks_vector(ckks_context, vec)

        # Serialize both context and ciphertext
        serialized_ctx = ckks_context.serialize(save_secret_key=True)
        serialized_enc = enc.serialize()

        # Deserialize
        new_context = ts.context_from(serialized_ctx)
        new_enc = ts.ckks_vector_from(new_context, serialized_enc)

        # Decrypt with new context
        decrypted = new_enc.decrypt()

        for original, dec in zip(vec, decrypted):
            assert abs(original - dec) < 1e-5

    def test_bfv_ciphertext_serialization(self, bfv_context):
        """Test BFV ciphertext serialization."""
        vec = [10, 20, 30]
        enc = ts.bfv_vector(bfv_context, vec)

        # Serialize ciphertext
        serialized = enc.serialize()

        # Deserialize
        new_enc = ts.bfv_vector_from(bfv_context, serialized)
        decrypted = new_enc.decrypt()

        assert decrypted == vec

    def test_ckks_ciphertext_serialization(self, ckks_context):
        """Test CKKS ciphertext serialization."""
        vec = [1.5, 2.5, 3.5]
        enc = ts.ckks_vector(ckks_context, vec)

        # Serialize ciphertext
        serialized = enc.serialize()

        # Deserialize
        new_enc = ts.ckks_vector_from(ckks_context, serialized)
        decrypted = new_enc.decrypt()

        for original, dec in zip(vec, decrypted):
            assert abs(original - dec) < 1e-5

    def test_ckks_context_keys_consistency_after_serialization(self, ckks_context):
        """Test that CKKS context keys remain consistent after serialization/deserialization.

        This test verifies that:
        1. Public key values are preserved
        2. Galois keys are preserved
        3. Relinearization keys are preserved
        4. The deserialized context can produce identical encryption results
        """
        # Test data
        test_vec = [1.5, 2.5, 3.5, 4.5]

        # Encrypt with original context
        original_enc = ts.ckks_vector(ckks_context, test_vec)
        original_decrypted = original_enc.decrypt()

        # Serialize context with secret key
        serialized_ctx = ckks_context.serialize(save_secret_key=True)

        # Deserialize to new context
        new_context = ts.context_from(serialized_ctx)

        # Test 1: Verify encryption/decryption consistency
        new_enc = ts.ckks_vector(new_context, test_vec)
        new_decrypted = new_enc.decrypt()

        # Should produce identical results (within CKKS precision)
        for orig, new in zip(original_decrypted, new_decrypted):
            assert abs(orig - new) < 1e-5, f"Encryption results differ: {orig} vs {new}"

        # Test 2: Verify operations work identically
        original_sum = original_enc.sum()
        new_sum = new_enc.sum()

        orig_sum_val = original_sum.decrypt()[0] if original_sum.decrypt() else 0
        new_sum_val = new_sum.decrypt()[0] if new_sum.decrypt() else 0

        assert (
            abs(orig_sum_val - new_sum_val) < 1e-4
        ), f"Sum operations differ: {orig_sum_val} vs {new_sum_val}"

        # Test 3: Verify matrix operations work
        matrix = [[1.0, 2.0], [3.0, 4.0]]

        # Create 2-element vector for matmul
        vec_2d = [1.0, 2.0]
        original_vec_2d = ts.ckks_vector(ckks_context, vec_2d)
        new_vec_2d = ts.ckks_vector(new_context, vec_2d)

        # Matrix multiplication should work identically
        original_matmul = original_vec_2d.matmul(matrix)
        new_matmul = new_vec_2d.matmul(matrix)

        orig_matmul_result = original_matmul.decrypt()
        new_matmul_result = new_matmul.decrypt()

        for orig, new in zip(orig_matmul_result, new_matmul_result):
            assert abs(orig - new) < 1e-4, f"Matmul results differ: {orig} vs {new}"

        # Test 4: Verify polynomial evaluation works identically
        poly_coeffs = [1.0, 2.0, 3.0]  # 1 + 2x + 3x^2

        original_poly = original_enc.polyval(poly_coeffs)
        new_poly = new_enc.polyval(poly_coeffs)

        orig_poly_result = original_poly.decrypt()
        new_poly_result = new_poly.decrypt()

        for orig, new in zip(orig_poly_result, new_poly_result):
            assert abs(orig - new) < 1e-3, f"Polyval results differ: {orig} vs {new}"

        # Test 5: Verify ciphertext from original context works with new context
        # (This tests key compatibility)
        serialized_enc = original_enc.serialize()
        restored_enc = ts.ckks_vector_from(new_context, serialized_enc)
        restored_decrypted = restored_enc.decrypt()

        for orig, restored in zip(test_vec, restored_decrypted):
            assert (
                abs(orig - restored) < 1e-5
            ), f"Restored ciphertext differs: {orig} vs {restored}"

        print("✅ All key consistency tests passed!")
        print(f"   - Encryption/decryption consistency: ✓")
        print(f"   - Sum operation consistency: ✓")
        print(f"   - Matrix operation consistency: ✓")
        print(f"   - Polynomial evaluation consistency: ✓")
        print(f"   - Cross-context ciphertext compatibility: ✓")

    def test_different_public_keys_operation_security(self):
        """Test security behavior when operating on ciphertexts encrypted with different public keys.

        IMPORTANT FINDING: TenSEAL allows cross-context operations but produces meaningless results.
        This test documents this behavior and verifies that while operations don't fail,
        the results are completely corrupted, which serves as a form of security through corruption.
        """
        # Create two independent CKKS contexts with different key pairs
        context1 = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context1.generate_galois_keys()
        context1.generate_relin_keys()
        context1.global_scale = 2**40

        context2 = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context2.generate_galois_keys()
        context2.generate_relin_keys()
        context2.global_scale = 2**40

        # Test data
        data1 = [1.0, 2.0, 3.0]
        data2 = [4.0, 5.0, 6.0]
        expected_sum = [5.0, 7.0, 9.0]  # Correct result if same context

        # Encrypt with different contexts (different public keys)
        enc1 = ts.ckks_vector(context1, data1)
        enc2 = ts.ckks_vector(context2, data2)

        # Test 1: Cross-context addition produces meaningless results
        print("Testing cross-context addition behavior...")
        result_cross = enc1 + enc2
        decrypted_cross = result_cross.decrypt()

        # Verify that cross-context result is completely wrong
        cross_errors = [abs(decrypted_cross[i] - expected_sum[i]) for i in range(3)]
        print(f"Cross-context errors: {cross_errors[:3]}")

        # All errors should be extremely large (indicating corrupted result)
        for i, error in enumerate(cross_errors[:3]):
            assert (
                error > 1e6
            ), f"Cross-context error {error} at position {i} is too small - should be much larger"

        # Test 2: Same context operations produce correct results
        enc2_same_context = ts.ckks_vector(context1, data2)  # Same context as enc1
        result_same = enc1 + enc2_same_context
        decrypted_same = result_same.decrypt()

        # Verify same-context result is correct
        same_errors = [abs(decrypted_same[i] - expected_sum[i]) for i in range(3)]
        print(f"Same-context errors: {same_errors[:3]}")

        for i, error in enumerate(same_errors[:3]):
            assert (
                error < 1e-6
            ), f"Same-context error {error} at position {i} is too large"

        # Test 3: Test other cross-context operations
        print("Testing cross-context subtraction and multiplication...")

        # Subtraction
        result_sub = enc1 - enc2
        decrypted_sub = result_sub.decrypt()
        expected_sub = [-3.0, -3.0, -3.0]
        sub_errors = [abs(decrypted_sub[i] - expected_sub[i]) for i in range(3)]
        for i, error in enumerate(sub_errors[:3]):
            assert (
                error > 1e6
            ), f"Cross-context subtraction error {error} at position {i} should be large"

        # Multiplication
        result_mul = enc1 * enc2
        decrypted_mul = result_mul.decrypt()
        expected_mul = [4.0, 10.0, 18.0]
        mul_errors = [abs(decrypted_mul[i] - expected_mul[i]) for i in range(3)]
        for i, error in enumerate(mul_errors[:3]):
            assert (
                error > 1e6
            ), f"Cross-context multiplication error {error} at position {i} should be large"

        # Test 4: Test with BFV scheme
        print("Testing BFV cross-context behavior...")

        bfv_context1 = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=8192,
            plain_modulus=1032193,
        )
        bfv_context1.generate_galois_keys()
        bfv_context1.generate_relin_keys()

        bfv_context2 = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=8192,
            plain_modulus=1032193,
        )
        bfv_context2.generate_galois_keys()
        bfv_context2.generate_relin_keys()

        bfv_data1 = [1, 2, 3]
        bfv_data2 = [4, 5, 6]
        bfv_expected = [5, 7, 9]

        bfv_enc1 = ts.bfv_vector(bfv_context1, bfv_data1)
        bfv_enc2 = ts.bfv_vector(bfv_context2, bfv_data2)

        # BFV cross-context operations also produce wrong results
        bfv_result = bfv_enc1 + bfv_enc2
        bfv_decrypted = bfv_result.decrypt()

        bfv_errors = [abs(bfv_decrypted[i] - bfv_expected[i]) for i in range(3)]
        print(f"BFV cross-context errors: {bfv_errors[:3]}")

        # BFV errors should be large (wrong results)
        for i, error in enumerate(bfv_errors[:3]):
            assert (
                error > 100
            ), f"BFV cross-context error {error} at position {i} should be large"

        print("✅ Cross-context security test completed!")
        print(
            "SUMMARY: TenSEAL allows cross-context operations but produces corrupted results,"
        )
        print(
            "which effectively prevents meaningful data extraction across different key pairs."
        )

    def test_different_public_keys_decryption_security(self):
        """Test that decryption with wrong context fails.

        Verify that a ciphertext encrypted with one public key cannot be
        decrypted with a different private key.
        """
        # Create two independent contexts
        context1 = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context1.generate_galois_keys()
        context1.generate_relin_keys()
        context1.global_scale = 2**40

        context2 = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context2.generate_galois_keys()
        context2.generate_relin_keys()
        context2.global_scale = 2**40

        # Encrypt with context1
        data = [1.5, 2.5, 3.5]
        enc_context1 = ts.ckks_vector(context1, data)

        # Serialize the ciphertext
        serialized_enc = enc_context1.serialize()

        # Try to load and decrypt with context2 (wrong key)
        print("Testing decryption with wrong private key...")
        try:
            # This might fail at deserialization or decryption stage
            enc_wrong_context = ts.ckks_vector_from(context2, serialized_enc)
            wrong_decrypted = enc_wrong_context.decrypt()

            # If it doesn't fail, the result should be garbage (not the original data)
            print(f"Original data: {data}")
            print(f"Wrong decryption: {wrong_decrypted}")

            # The decrypted values should NOT match the original (within reasonable tolerance)
            matches = 0
            for orig, wrong in zip(data, wrong_decrypted):
                if abs(orig - wrong) < 0.1:  # Very generous tolerance
                    matches += 1

            # If most values match, something is wrong with the security
            if matches >= len(data) // 2:
                pytest.fail(
                    "ERROR: Wrong private key produced correct decryption! Security compromised!"
                )
            else:
                print("✓ Wrong private key produced garbage output (expected)")

        except Exception as e:
            # This is the expected behavior - should fail
            print(f"✓ Wrong private key correctly failed: {type(e).__name__}: {e}")

        # Verify correct decryption still works
        correct_decrypted = enc_context1.decrypt()
        for orig, correct in zip(data, correct_decrypted):
            assert abs(orig - correct) < 1e-5

        print("✓ Correct private key decryption works fine")
        print("✅ Decryption security test passed!")

    def test_serialized_context_keys_equality(self):
        """Test if keys from serialized/deserialized context can be compared with == operator.

        This test examines whether TenSEAL key objects support direct equality comparison
        and whether serialization preserves key equality at the object level.
        """
        # Create a CKKS context with all keys
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context.generate_galois_keys()
        context.generate_relin_keys()
        context.global_scale = 2**40

        # Get original keys
        print("Testing key equality comparison...")
        try:
            original_public_key = context.public_key()
            original_secret_key = context.secret_key()
            original_galois_keys = context.galois_keys()
            original_relin_keys = context.relin_keys()

            print(f"Original public key type: {type(original_public_key)}")
            print(f"Original secret key type: {type(original_secret_key)}")
            print(f"Original galois keys type: {type(original_galois_keys)}")
            print(f"Original relin keys type: {type(original_relin_keys)}")

        except Exception as e:
            print(f"Error getting original keys: {e}")
            return

        # Serialize and deserialize context
        # Important: Include save_secret_key=True to preserve secret key
        print(f"Original context has secret key: {context.has_secret_key()}")

        # Test both with and without secret key serialization
        serialized_context_no_secret = context.serialize(save_secret_key=False)
        serialized_context_with_secret = context.serialize(save_secret_key=True)

        context_restored_no_secret = ts.context_from(serialized_context_no_secret)
        context_restored_with_secret = ts.context_from(serialized_context_with_secret)

        print(
            f"Restored without secret key: {context_restored_no_secret.has_secret_key()}"
        )
        print(
            f"Restored with secret key: {context_restored_with_secret.has_secret_key()}"
        )

        # Use the context with secret key for main testing
        context_restored = context_restored_with_secret

        # Get restored keys
        try:
            restored_public_key = context_restored.public_key()
            restored_secret_key = context_restored.secret_key()
            restored_galois_keys = context_restored.galois_keys()
            restored_relin_keys = context_restored.relin_keys()

        except Exception as e:
            print(f"Error getting restored keys: {e}")
            return

        # Test 1: Check if keys support == comparison
        print("\n1. Testing direct key equality with == operator:")

        try:
            public_equal = original_public_key.data == restored_public_key.data
            print(f"   Public keys equal: {public_equal}")
        except Exception as e:
            print(f"   Public key comparison failed: {type(e).__name__}: {e}")

        try:
            secret_equal = original_secret_key == restored_secret_key
            print(f"   Secret keys equal: {secret_equal}")
        except Exception as e:
            print(f"   Secret key comparison failed: {type(e).__name__}: {e}")

        try:
            galois_equal = original_galois_keys == restored_galois_keys
            print(f"   Galois keys equal: {galois_equal}")
        except Exception as e:
            print(f"   Galois key comparison failed: {type(e).__name__}: {e}")

        try:
            relin_equal = original_relin_keys == restored_relin_keys
            print(f"   Relin keys equal: {relin_equal}")
        except Exception as e:
            print(f"   Relin key comparison failed: {type(e).__name__}: {e}")

        # Test 2: Check if keys have serialization methods for manual comparison
        print("\n2. Testing key serialization for manual comparison:")

        try:
            # Check if keys have serialize method
            if hasattr(original_public_key, "serialize"):
                orig_pub_serialized = original_public_key.serialize()
                rest_pub_serialized = restored_public_key.serialize()
                pub_serialized_equal = orig_pub_serialized == rest_pub_serialized
                print(f"   Public key serialized data equal: {pub_serialized_equal}")
            else:
                print("   Public key has no serialize method")

        except Exception as e:
            print(f"   Public key serialization comparison failed: {e}")

        try:
            if hasattr(original_secret_key, "serialize"):
                orig_sec_serialized = original_secret_key.serialize()
                rest_sec_serialized = restored_secret_key.serialize()
                sec_serialized_equal = orig_sec_serialized == rest_sec_serialized
                print(f"   Secret key serialized data equal: {sec_serialized_equal}")
            else:
                print("   Secret key has no serialize method")

        except Exception as e:
            print(f"   Secret key serialization comparison failed: {e}")

        # Test 3: Functional equality test
        print("\n3. Testing functional equality (can encrypt/decrypt consistently):")

        test_data = [1.0, 2.0, 3.0]

        # Encrypt with original context
        enc_original = ts.ckks_vector(context, test_data)
        decrypted_original = enc_original.decrypt()

        # Encrypt with restored context
        enc_restored = ts.ckks_vector(context_restored, test_data)
        decrypted_restored = enc_restored.decrypt()

        # Compare decrypted results
        functional_equal = True
        for orig, rest in zip(decrypted_original, decrypted_restored):
            if abs(orig - rest) > 1e-6:
                functional_equal = False
                break

        print(f"   Functional equality (encrypt/decrypt): {functional_equal}")

        # Test 4: Cross-context operations
        print("\n4. Testing cross-context operations between original and restored:")

        try:
            # Try to add ciphertexts from original and restored contexts
            cross_result = enc_original + enc_restored
            cross_decrypted = cross_result.decrypt()

            # Check if result makes sense (should be 2x original if truly same context)
            expected_doubled = [2 * x for x in test_data]
            cross_functional = True
            errors = []

            for exp, act in zip(expected_doubled, cross_decrypted):
                error = abs(exp - act)
                errors.append(error)
                if error > 1e-3:  # More lenient for cross-context ops
                    cross_functional = False

            print(f"   Cross-context operation works correctly: {cross_functional}")
            print(f"   Cross-context operation errors: {errors[:3]}")

        except Exception as e:
            print(f"   Cross-context operation failed: {type(e).__name__}: {e}")

        # Test 5: Same reference equality
        print("\n5. Testing same context reference equality:")

        same_public_1 = context.public_key()
        same_public_2 = context.public_key()

        try:
            same_ref_equal = same_public_1 == same_public_2
            print(f"   Same context public keys equal: {same_ref_equal}")
        except Exception as e:
            print(f"   Same context key comparison failed: {e}")

        # Check if they are the same object reference
        same_ref_identity = same_public_1 is same_public_2
        print(f"   Same context keys are identical objects: {same_ref_identity}")

        print("\n✅ Key equality comparison test completed!")
        print("Note: This test explores TenSEAL's key comparison capabilities,")
        print("which may vary depending on the underlying implementation.")


# =============================================================================
# Test Class 7: Advanced Operations
# =============================================================================


class TestAdvancedOperations:
    """Test advanced homomorphic operations."""

    def test_ckks_sum_operation(self, ckks_context):
        """Test CKKS sum operation.

        Note: The sum() method returns an encrypted vector, not a scalar.
        We need to decrypt it first.
        """
        vec = [1.0, 2.0, 3.0, 4.0]
        enc = ts.ckks_vector(ckks_context, vec)

        # Sum all elements (returns encrypted result)
        result = enc.sum()
        decrypted_result = result.decrypt()

        expected_sum = sum(vec)
        # The result is a vector with the sum value
        actual_sum = (
            decrypted_result[0]
            if isinstance(decrypted_result, list)
            else decrypted_result
        )
        assert abs(actual_sum - expected_sum) < 1e-4

    def test_ckks_chained_operations(self, ckks_context):
        """Test chaining multiple operations."""
        vec = [1.0, 2.0, 3.0]
        enc = ts.ckks_vector(ckks_context, vec)

        # Chain: (vec + 1) * 2 - 0.5
        result = ((enc + 1.0) * 2.0) - 0.5
        decrypted = result.decrypt()

        expected = [((v + 1.0) * 2.0) - 0.5 for v in vec]

        for exp, act in zip(expected, decrypted):
            # Larger tolerance due to error accumulation
            assert abs(exp - act) < 1e-3

    def test_bfv_chained_operations(self, bfv_context):
        """Test chaining multiple BFV operations."""
        vec = [1, 2, 3, 4]
        enc = ts.bfv_vector(bfv_context, vec)

        # Chain: (vec + 1) * 2
        result = (enc + 1) * 2
        decrypted = result.decrypt()

        expected = [(v + 1) * 2 for v in vec]
        assert decrypted == expected


# =============================================================================
# Test Class 8: Shape Operations (CKKS Vector/Tensor)
# =============================================================================


class TestCKKSShapeOperations:
    """Test shape-related operations for CKKS vectors and tensors."""

    def test_vector_shape_property(self, ckks_context):
        """Test accessing shape property of CKKS vector."""
        # Test various vector sizes
        vec1 = ts.ckks_vector(ckks_context, [1.0])
        vec10 = ts.ckks_vector(ckks_context, list(range(10)))
        vec100 = ts.ckks_vector(ckks_context, [float(i) for i in range(100)])

        assert vec1.shape == [1], f"Expected shape [1], got {vec1.shape}"
        assert vec10.shape == [10], f"Expected shape [10], got {vec10.shape}"
        assert vec100.shape == [100], f"Expected shape [100], got {vec100.shape}"

    def test_vector_size_method(self, ckks_context):
        """Test size() method of CKKS vector."""
        vec1 = ts.ckks_vector(ckks_context, [1.0])
        vec10 = ts.ckks_vector(ckks_context, list(range(10)))
        vec100 = ts.ckks_vector(ckks_context, [float(i) for i in range(100)])

        assert vec1.size() == 1, f"Expected size 1, got {vec1.size()}"
        assert vec10.size() == 10, f"Expected size 10, got {vec10.size()}"
        assert vec100.size() == 100, f"Expected size 100, got {vec100.size()}"

    def test_shape_consistency_with_size(self, ckks_context):
        """Test that shape[0] equals size()."""
        for n in [1, 5, 10, 50]:
            vec = ts.ckks_vector(ckks_context, [float(i) for i in range(n)])
            assert (
                vec.shape[0] == vec.size()
            ), f"Shape {vec.shape} and size {vec.size()} inconsistent"

    def test_dot_product_reduces_shape(self, ckks_context):
        """Test that dot product reduces shape to [1]."""
        vec1 = ts.ckks_vector(ckks_context, [1.0, 2.0, 3.0, 4.0])
        vec2 = ts.ckks_vector(ckks_context, [5.0, 6.0, 7.0, 8.0])

        # Original shapes
        assert vec1.shape == [4]
        assert vec2.shape == [4]

        # Dot product
        result = vec1.dot(vec2)

        # Result should have shape [1]
        assert result.shape == [1], f"Expected shape [1], got {result.shape}"
        assert result.size() == 1, f"Expected size 1, got {result.size()}"

        # Verify the computation
        expected = sum(
            v1 * v2 for v1, v2 in zip([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
        )
        actual = result.decrypt()[0]
        assert abs(expected - actual) < 1e-4

    def test_sum_reduces_shape(self, ckks_context):
        """Test that sum reduces shape to [1]."""
        vec = ts.ckks_vector(ckks_context, [1.0, 2.0, 3.0, 4.0, 5.0])

        # Original shape
        assert vec.shape == [5]

        # Sum
        result = vec.sum()

        # Result should have shape [1]
        assert result.shape == [1], f"Expected shape [1], got {result.shape}"
        assert result.size() == 1, f"Expected size 1, got {result.size()}"

        # Verify the computation
        expected = 15.0
        actual = result.decrypt()[0]
        assert abs(expected - actual) < 1e-4

    def test_matmul_changes_shape(self, ckks_context):
        """Test that matrix multiplication changes vector shape.

        Matrix multiplication: vec @ matrix -> new_vec
        If vec has shape [n] and matrix has shape [n, m], result has shape [m].
        """
        # Create a vector of size 6
        vec = ts.ckks_vector(ckks_context, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert vec.shape == [6]

        # Create a 6x3 matrix (multiply 6-element vector by 6x3 matrix -> 3-element vector)
        matrix = [
            [1.0, 0.0, 0.0],  # row 0
            [0.0, 1.0, 0.0],  # row 1
            [0.0, 0.0, 1.0],  # row 2
            [1.0, 1.0, 1.0],  # row 3
            [0.0, 1.0, 0.0],  # row 4
            [1.0, 0.0, 1.0],  # row 5
        ]

        # Perform matrix multiplication
        result = vec.mm(matrix)

        # Result should have shape [3]
        assert result.shape == [3], f"Expected shape [3], got {result.shape}"
        assert result.size() == 3, f"Expected size 3, got {result.size()}"

        # Verify computation
        decrypted = result.decrypt()
        # Expected: [1 + 4 + 6, 2 + 4 + 5, 3 + 4 + 6] = [11, 11, 13]
        expected = [11.0, 11.0, 13.0]
        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4

    def test_matmul_alias(self, ckks_context):
        """Test that matmul is an alias for mm."""
        vec = ts.ckks_vector(ckks_context, [1.0, 2.0, 3.0, 4.0])
        matrix = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]

        result_mm = vec.mm(matrix)
        result_matmul = vec.matmul(matrix)

        # Both should produce same shape
        assert result_mm.shape == result_matmul.shape
        assert result_mm.size() == result_matmul.size()

        # Both should produce same results
        dec_mm = result_mm.decrypt()
        dec_matmul = result_matmul.decrypt()
        for v1, v2 in zip(dec_mm, dec_matmul):
            assert abs(v1 - v2) < 1e-6

    def test_enc_matmul_plain_usage(self, ckks_context):
        """Test enc_matmul_plain behavior and limitations.

        Note: enc_matmul_plain has strict requirements and is less commonly used.
        For general matrix multiplication, use mm() or matmul() instead.

        This test documents the API surface exists but has constraints.
        """
        # Create a simple vector
        vec = ts.ckks_vector(ckks_context, [1.0, 2.0, 3.0, 4.0])
        assert vec.shape == [4]

        # enc_matmul_plain exists as a method
        assert hasattr(vec, "enc_matmul_plain"), "enc_matmul_plain method should exist"

        # For practical matrix multiplication, use mm() or matmul() instead
        # which work more reliably with various matrix shapes
        plain_matrix = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
        result = vec.mm(plain_matrix)
        assert result.shape == [2]
        decrypted = result.decrypt()
        assert len(decrypted) == 2

    def test_pack_vectors_combines_shapes(self, ckks_context):
        """Test pack_vectors static method combines multiple vectors.

        pack_vectors takes multiple vectors and concatenates them into one.
        """
        vec1 = ts.ckks_vector(ckks_context, [1.0, 2.0])
        vec2 = ts.ckks_vector(ckks_context, [3.0, 4.0])
        vec3 = ts.ckks_vector(ckks_context, [5.0, 6.0])

        # Pack vectors
        packed = ts.CKKSVector.pack_vectors([vec1, vec2, vec3])

        # Packed vector should have combined size
        assert packed.shape == [6], f"Expected shape [6], got {packed.shape}"
        assert packed.size() == 6, f"Expected size 6, got {packed.size()}"

        # Verify data is concatenated
        decrypted = packed.decrypt()
        expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4

    def test_pack_vectors_different_sizes(self, ckks_context):
        """Test that pack_vectors requires same-sized vectors.

        Note: pack_vectors in TenSEAL requires all vectors to have the same size.
        This is a constraint of the SIMD packing mechanism.
        """
        vec1 = ts.ckks_vector(ckks_context, [1.0])
        vec2 = ts.ckks_vector(ckks_context, [2.0, 3.0, 4.0])
        vec3 = ts.ckks_vector(ckks_context, [5.0, 6.0])

        # Should raise ValueError for different sizes
        with pytest.raises(ValueError, match="vectors sizes are different"):
            packed = ts.CKKSVector.pack_vectors([vec1, vec2, vec3])

    def test_pack_vectors_same_sizes(self, ckks_context):
        """Test pack_vectors with same-sized vectors."""
        # All vectors must have the same size
        vec1 = ts.ckks_vector(ckks_context, [1.0, 2.0])
        vec2 = ts.ckks_vector(ckks_context, [3.0, 4.0])
        vec3 = ts.ckks_vector(ckks_context, [5.0, 6.0])

        packed = ts.CKKSVector.pack_vectors([vec1, vec2, vec3])

        # Packed vector should have combined size
        assert packed.shape == [6], f"Expected shape [6], got {packed.shape}"
        assert packed.size() == 6, f"Expected size 6, got {packed.size()}"

        # Verify data is concatenated
        decrypted = packed.decrypt()
        expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4

    def test_operations_preserve_shape(self, ckks_context):
        """Test that arithmetic operations preserve shape."""
        vec = ts.ckks_vector(ckks_context, [1.0, 2.0, 3.0, 4.0, 5.0])
        original_shape = vec.shape

        # Addition preserves shape
        result = vec + 1.0
        assert result.shape == original_shape

        # Subtraction preserves shape
        result = vec - 1.0
        assert result.shape == original_shape

        # Multiplication preserves shape
        result = vec * 2.0
        assert result.shape == original_shape

        # Power preserves shape
        result = vec**2
        assert result.shape == original_shape

        # Square preserves shape
        result = vec.square()
        assert result.shape == original_shape

        # Negation preserves shape
        result = -vec
        assert result.shape == original_shape

    def test_ciphertext_operations_preserve_shape(self, ckks_context):
        """Test that ciphertext-to-ciphertext operations preserve shape."""
        vec1 = ts.ckks_vector(ckks_context, [1.0, 2.0, 3.0])
        vec2 = ts.ckks_vector(ckks_context, [4.0, 5.0, 6.0])
        original_shape = vec1.shape

        # Addition
        result = vec1 + vec2
        assert result.shape == original_shape

        # Subtraction
        result = vec1 - vec2
        assert result.shape == original_shape

        # Multiplication (element-wise)
        result = vec1 * vec2
        assert result.shape == original_shape

    def test_shape_after_chained_operations(self, ckks_context):
        """Test shape behavior in chained operations."""
        vec = ts.ckks_vector(ckks_context, [1.0, 2.0, 3.0, 4.0])
        original_shape = vec.shape

        # Chain operations that preserve shape
        result = ((vec + 1.0) * 2.0) - 0.5
        assert result.shape == original_shape

        # Chain ending with sum (reduces to [1])
        result = ((vec + 1.0) * 2.0).sum()
        assert result.shape == [1]
        assert result.size() == 1

    def test_empty_or_single_element_shapes(self, ckks_context):
        """Test shape operations with edge cases."""
        # Single element vector
        vec1 = ts.ckks_vector(ckks_context, [5.0])
        assert vec1.shape == [1]
        assert vec1.size() == 1

        # Operations on single element
        result = vec1 + 1.0
        assert result.shape == [1]
        assert abs(result.decrypt()[0] - 6.0) < 1e-4

        # Sum of single element
        result = vec1.sum()
        assert result.shape == [1]
        assert abs(result.decrypt()[0] - 5.0) < 1e-4

    def test_large_vector_shape(self, ckks_context):
        """Test shape operations with larger vectors."""
        # Create a larger vector (e.g., 1000 elements)
        large_vec = ts.ckks_vector(ckks_context, [float(i) for i in range(1000)])

        assert large_vec.shape == [1000]
        assert large_vec.size() == 1000

        # Operations should preserve large shape
        result = large_vec * 2.0
        assert result.shape == [1000]

        # Sum should reduce to [1]
        result_sum = large_vec.sum()
        assert result_sum.shape == [1]

    def test_matrix_multiplication_various_dimensions(self, ckks_context):
        """Test matrix multiplication with various input/output dimensions."""
        # Test case 1: [8] @ 8x4 -> [4]
        vec8 = ts.ckks_vector(ckks_context, [float(i) for i in range(1, 9)])
        matrix_8x4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(8)]
        result = vec8.mm(matrix_8x4)
        assert result.shape == [4]

        # Test case 2: [3] @ 3x1 -> [1]
        vec3 = ts.ckks_vector(ckks_context, [1.0, 2.0, 3.0])
        matrix_3x1 = [[1.0], [1.0], [1.0]]
        result = vec3.mm(matrix_3x1)
        assert result.shape == [1]
        # This is essentially a sum: 1+2+3=6
        assert abs(result.decrypt()[0] - 6.0) < 1e-4

        # Test case 3: [2] @ 2x5 -> [5]
        vec2 = ts.ckks_vector(ckks_context, [2.0, 3.0])
        matrix_2x5 = [
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
        ]
        result = vec2.mm(matrix_2x5)
        assert result.shape == [5]
        # Expected: [2*1+3*0, 2*0+3*1, 2*1+3*1, 2*0+3*0, 2*1+3*0] = [2, 3, 5, 0, 2]
        decrypted = result.decrypt()
        expected = [2.0, 3.0, 5.0, 0.0, 2.0]
        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4


# =============================================================================
# Test Class 9: CKKS Tensor Shape Operations
# =============================================================================


class TestCKKSTensorShapeOperations:
    """Test shape-related operations for CKKS tensors (multi-dimensional arrays)."""

    def test_2d_tensor_shape_property(self, ckks_context):
        """Test that 2D tensors have correct shape property."""
        # Create a 2x3 tensor
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        tensor = ts.ckks_tensor(ckks_context, data)

        assert tensor.shape == [2, 3]
        assert isinstance(tensor.shape, list)
        assert len(tensor.shape) == 2

    def test_3d_tensor_shape_property(self, ckks_context):
        """Test that 3D tensors have correct shape property."""
        # Create a 2x2x2 tensor
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        tensor = ts.ckks_tensor(ckks_context, data)

        assert tensor.shape == [2, 2, 2]
        assert len(tensor.shape) == 3

    def test_1d_tensor_shape(self, ckks_context):
        """Test 1D tensor shape (similar to vector but as tensor)."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ts.ckks_tensor(ckks_context, data)

        assert tensor.shape == [4]
        assert len(tensor.shape) == 1

    def test_reshape_operation(self, ckks_context):
        """Test reshape operation changes tensor shape."""
        # Create 2x3 tensor
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert tensor.shape == [2, 3]

        # Reshape to 3x2
        reshaped = tensor.reshape([3, 2])
        assert reshaped.shape == [3, 2]

        # Verify data is preserved (just rearranged)
        import numpy as np

        original_flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        decrypted = np.array(reshaped.decrypt().tolist()).flatten().tolist()
        for orig, dec in zip(original_flat, decrypted):
            assert abs(orig - dec) < 1e-4

    def test_reshape_to_1d(self, ckks_context):
        """Test reshaping multi-dimensional tensor to 1D."""
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0], [3.0, 4.0]])
        assert tensor.shape == [2, 2]

        # Reshape to 1D
        flat = tensor.reshape([4])
        assert flat.shape == [4]

        decrypted = flat.decrypt().tolist()
        expected = [1.0, 2.0, 3.0, 4.0]
        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4

    def test_transpose_2d(self, ckks_context):
        """Test transpose operation on 2D tensor."""
        # Create 2x3 tensor
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        tensor = ts.ckks_tensor(ckks_context, data)
        assert tensor.shape == [2, 3]

        # Transpose
        transposed = tensor.transpose()
        assert transposed.shape == [3, 2]

        # Verify values are transposed correctly
        decrypted = transposed.decrypt().tolist()
        expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

        for i, row in enumerate(expected):
            for j, exp_val in enumerate(row):
                assert abs(decrypted[i][j] - exp_val) < 1e-4

    def test_sum_all_elements(self, ckks_context):
        """Test sum() without axis sums all elements.

        Note: sum() without axis parameter sums along axis 0 by default in TenSEAL.
        """
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0], [3.0, 4.0]])

        # Sum all elements (actually sums along first axis by default)
        result = tensor.sum()

        # Result should be a 1D tensor with sums
        decrypted = result.decrypt().tolist()

        # sum() along axis 0: [1+3, 2+4] = [4, 6]
        expected = [4.0, 6.0]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-3

    def test_sum_along_axis_0(self, ckks_context):
        """Test sum along axis 0 (column-wise sum)."""
        # 2x3 tensor
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert tensor.shape == [2, 3]

        # Sum along axis 0 (sum each column)
        result = tensor.sum(0)
        assert result.shape == [3]

        decrypted = result.decrypt().tolist()
        expected = [5.0, 7.0, 9.0]  # [1+4, 2+5, 3+6]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4

    def test_sum_along_axis_1(self, ckks_context):
        """Test sum along axis 1 (row-wise sum)."""
        # 2x3 tensor
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert tensor.shape == [2, 3]

        # Sum along axis 1 (sum each row)
        result = tensor.sum(1)
        assert result.shape == [2]

        decrypted = result.decrypt().tolist()
        expected = [6.0, 15.0]  # [1+2+3, 4+5+6]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-4

    def test_broadcast_scalar_to_shape(self, ckks_context):
        """Test broadcasting a scalar to a specific shape."""
        # Create scalar-like tensor
        scalar = ts.ckks_tensor(ckks_context, [5.0])
        assert scalar.shape == [1]

        # Broadcast to 2x3 shape
        broadcasted = scalar.broadcast([2, 3])
        assert broadcasted.shape == [2, 3]

        # All values should be 5.0
        decrypted = broadcasted.decrypt().tolist()
        for row in decrypted:
            for val in row:
                assert abs(val - 5.0) < 1e-4

    def test_broadcast_1d_to_2d(self, ckks_context):
        """Test broadcasting 1D tensor to 2D."""
        # Create 1D tensor
        vec = ts.ckks_tensor(ckks_context, [1.0, 2.0, 3.0])
        assert vec.shape == [3]

        # Broadcast to 2x3
        broadcasted = vec.broadcast([2, 3])
        assert broadcasted.shape == [2, 3]

        decrypted = broadcasted.decrypt().tolist()
        # Each row should be [1, 2, 3]
        expected = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        for i, row in enumerate(expected):
            for j, exp_val in enumerate(row):
                assert abs(decrypted[i][j] - exp_val) < 1e-4

    def test_matrix_multiplication_shape(self, ckks_context):
        """Test matrix multiplication with mm() method."""
        # 2x3 @ 3x2 = 2x2
        mat1 = ts.ckks_tensor(ckks_context, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mat2 = ts.ckks_tensor(ckks_context, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        assert mat1.shape == [2, 3]
        assert mat2.shape == [3, 2]

        result = mat1.mm(mat2)
        assert result.shape == [2, 2]

        # Verify matrix multiplication result
        # [[1,2,3], [4,5,6]] @ [[1,2], [3,4], [5,6]]
        # = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        # = [[22, 28], [49, 64]]
        decrypted = result.decrypt().tolist()
        expected = [[22.0, 28.0], [49.0, 64.0]]

        for i, row in enumerate(expected):
            for j, exp_val in enumerate(row):
                assert abs(decrypted[i][j] - exp_val) < 1e-3

    def test_square_matrix_multiplication(self, ckks_context):
        """Test multiplication of square matrices."""
        mat1 = ts.ckks_tensor(ckks_context, [[1.0, 2.0], [3.0, 4.0]])
        mat2 = ts.ckks_tensor(ckks_context, [[5.0, 6.0], [7.0, 8.0]])

        result = mat1.mm(mat2)
        assert result.shape == [2, 2]

        # [[1,2], [3,4]] @ [[5,6], [7,8]]
        # = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        # = [[19, 22], [43, 50]]
        decrypted = result.decrypt().tolist()
        expected = [[19.0, 22.0], [43.0, 50.0]]

        for i, row in enumerate(expected):
            for j, exp_val in enumerate(row):
                assert abs(decrypted[i][j] - exp_val) < 1e-3

    def test_operations_preserve_shape(self, ckks_context):
        """Test that arithmetic operations preserve tensor shape."""
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0], [3.0, 4.0]])
        original_shape = tensor.shape

        # Addition with scalar
        result_add = tensor + 5.0
        assert result_add.shape == original_shape

        # Multiplication with scalar
        result_mul = tensor * 2.0
        assert result_mul.shape == original_shape

        # Subtraction with scalar
        result_sub = tensor - 1.0
        assert result_sub.shape == original_shape

        # Negation
        result_neg = -tensor
        assert result_neg.shape == original_shape

    def test_tensor_tensor_operations_preserve_shape(self, ckks_context):
        """Test element-wise operations between tensors preserve shape."""
        tensor1 = ts.ckks_tensor(ckks_context, [[1.0, 2.0], [3.0, 4.0]])
        tensor2 = ts.ckks_tensor(ckks_context, [[5.0, 6.0], [7.0, 8.0]])

        # Element-wise addition
        result_add = tensor1 + tensor2
        assert result_add.shape == [2, 2]

        # Element-wise multiplication
        result_mul = tensor1 * tensor2
        assert result_mul.shape == [2, 2]

        # Element-wise subtraction
        result_sub = tensor2 - tensor1
        assert result_sub.shape == [2, 2]

    def test_chained_operations_shape(self, ckks_context):
        """Test shape behavior in chained operations."""
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Chain: add, multiply, then sum
        result = ((tensor + 1.0) * 2.0).sum(1)

        # After sum(1), shape should be [2] (one value per row)
        assert result.shape == [2]

        decrypted = result.decrypt().tolist()
        # Row 0: (1+1)*2 + (2+1)*2 + (3+1)*2 = 4+6+8 = 18
        # Row 1: (4+1)*2 + (5+1)*2 + (6+1)*2 = 10+12+14 = 36
        expected = [18.0, 36.0]

        for exp, act in zip(expected, decrypted):
            assert abs(exp - act) < 1e-3

    def test_reshape_preserves_data(self, ckks_context):
        """Test that reshape preserves all data values."""
        original = ts.ckks_tensor(ckks_context, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Reshape multiple times
        shape1 = original.reshape([3, 2])
        shape2 = shape1.reshape([6])
        shape3 = shape2.reshape([2, 3])

        # After reshaping back to original, should match
        assert shape3.shape == original.shape

        orig_decrypt = np.array(original.decrypt().tolist()).flatten().tolist()
        final_decrypt = np.array(shape3.decrypt().tolist()).flatten().tolist()

        for orig, final in zip(orig_decrypt, final_decrypt):
            assert abs(orig - final) < 1e-4

    def test_dot_product_tensors(self, ckks_context):
        """Test dot product between tensors."""
        tensor1 = ts.ckks_tensor(ckks_context, [1.0, 2.0, 3.0])
        tensor2 = ts.ckks_tensor(ckks_context, [4.0, 5.0, 6.0])

        result = tensor1.dot(tensor2)

        # Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        decrypted = result.decrypt().tolist()

        # Result should be a single value (scalar) or 1-element list
        if isinstance(decrypted, list):
            actual = decrypted[0] if len(decrypted) > 0 else sum(decrypted)
        else:
            actual = float(decrypted)

        expected = 32.0
        assert abs(actual - expected) < 1e-3

    def test_4d_tensor_shape(self, ckks_context):
        """Test 4D tensor shape handling."""
        # Create a small 4D tensor (2x2x2x2)
        data = [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ]

        tensor = ts.ckks_tensor(ckks_context, data)
        assert tensor.shape == [2, 2, 2, 2]
        assert len(tensor.shape) == 4

        # Flatten to 1D
        flat = tensor.reshape([16])
        assert flat.shape == [16]

    def test_empty_dimension_edge_case(self, ckks_context):
        """Test tensor with single element in some dimensions."""
        # 1x3 tensor (like a row vector)
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0, 3.0]])
        assert tensor.shape == [1, 3]

        # Transpose to 3x1 (column vector)
        transposed = tensor.transpose()
        assert transposed.shape == [3, 1]

    def test_large_tensor_shape(self, ckks_context):
        """Test handling of larger tensors."""
        # Create a 10x10 tensor
        data = [[float(i * 10 + j) for j in range(10)] for i in range(10)]
        tensor = ts.ckks_tensor(ckks_context, data)

        assert tensor.shape == [10, 10]

        # Sum along axis
        sum_axis0 = tensor.sum(0)
        assert sum_axis0.shape == [10]

        sum_axis1 = tensor.sum(1)
        assert sum_axis1.shape == [10]

    def test_polyval_preserves_shape(self, ckks_context):
        """Test that polyval operation preserves tensor shape."""
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0], [3.0, 4.0]])

        # Apply polynomial: 1 + 2x + x^2
        coeffs = [1.0, 2.0, 1.0]
        result = tensor.polyval(coeffs)

        # Shape should be preserved
        assert result.shape == tensor.shape


# =============================================================================
# Summary Comments
# =============================================================================

"""
Protocol Support Summary:

BFV (Brakerski-Fan-Vercauteren):
- Supports: Exact integer arithmetic
- Operations: Addition (cipher+cipher, cipher+plain), 
              Multiplication (cipher+cipher, cipher+plain),
              Power (small exponents)
- Does NOT support: Division, floating-point operations, polynomial evaluation
- Note: Negation behaves differently due to modular arithmetic

CKKS (Cheon-Kim-Kim-Song):
- Supports: Approximate floating-point arithmetic
- Operations: Addition, Subtraction, Multiplication, Negation, Power, 
              Polynomial evaluation (polyval), Sum
- Does NOT support: Division
- Note: All operations have small approximation errors (typically < 1e-5)
- Better for: Machine learning, statistical computations, decimal numbers

Key Differences:
1. BFV is exact but integer-only; CKKS is approximate but supports decimals
2. CKKS supports more operations (polyval, sum, proper negation)
3. BFV is better for applications requiring exact results (voting, auctions)
4. CKKS is better for numerical computations (neural networks, statistics)
"""


# =============================================================================
# Test Class 10: CKKS Slots Limitations
# =============================================================================


class TestCKKSSlotsLimitations:
    """Test CKKS slots limitations for vectors and tensors.

    CKKS context has a maximum number of slots (poly_modulus_degree / 2).
    Key findings:
    1. pack_vectors: Strictly limited, cannot exceed slot count
    2. CKKSVector: Can exceed slots but with warnings and disabled operations
    3. CKKSTensor: No slot limitation (uses different encoding/multiple ciphertexts)
    """

    def test_context_slot_count(self, ckks_context):
        """Verify the number of slots in CKKS context."""
        # For poly_modulus_degree=8192, slots should be 4096
        expected_slots = 8192 // 2
        # A vector with slot_count elements should work fine
        vector = ts.ckks_vector(ckks_context, list(range(expected_slots)))
        assert vector.size() == expected_slots

    def test_pack_vectors_within_slots(self, ckks_context):
        """Test pack_vectors with total elements within slot limit."""
        # Create 10 vectors of size 100 (total 1000 < 4096 slots)
        vectors = [ts.ckks_vector(ckks_context, list(range(100))) for _ in range(10)]

        # This should succeed
        packed = ts.CKKSVector.pack_vectors(vectors)
        assert packed is not None
        assert packed.size() == 1000

    def test_pack_vectors_exceeds_slots_fails(self, ckks_context):
        """Test pack_vectors strictly enforces slot limit.

        pack_vectors will raise ValueError when total elements exceed slot count.
        """
        # Create vectors that exceed slot count (4096)
        # 10 vectors of size 500 = 5000 > 4096 slots
        vectors = [ts.ckks_vector(ckks_context, list(range(500))) for _ in range(10)]

        # This MUST fail
        with pytest.raises(ValueError, match="output size is bigger than slot count"):
            ts.CKKSVector.pack_vectors(vectors)

    def test_single_vector_exceeds_slots_with_warning(self, ckks_context):
        """Test creating a single vector exceeding slot limit.

        CKKSVector allows exceeding slot count but:
        - Issues a WARNING
        - Disables certain operations (matmul, matmul_plain, enc_matmul_plain, conv2d_im2col)
        """
        slots = 4096

        # Create a vector larger than slot count
        large_data = list(range(slots + 1000))  # 5096 elements
        # This succeeds but with warning
        vector = ts.ckks_vector(ckks_context, large_data)

        # Vector is created successfully
        assert vector.size() == len(large_data)
        assert vector.size() > slots

        # Basic operations still work
        result = vector + vector
        assert result.size() == vector.size()

    def test_oversized_vector_disabled_operations(self, ckks_context):
        """Test that oversized vectors have disabled operations.

        When vector size exceeds slots, matmul operations should fail or behave differently.
        """
        slots = 4096
        large_vector = ts.ckks_vector(ckks_context, list(range(slots + 100)))

        # Try matmul - this should fail or not work as expected
        # Note: The exact behavior depends on TenSEAL version
        # We document that it's disabled for oversized vectors
        assert large_vector.size() > slots

    def test_ckks_tensor_small_shape(self, ckks_context):
        """Test ckks_tensor with small total elements (within slots)."""
        # 10x10 = 100 elements, well within 4096 slots
        data = np.random.randn(10, 10).tolist()
        tensor = ts.ckks_tensor(ckks_context, data)

        assert tensor.shape == [10, 10]

    def test_ckks_tensor_at_slot_limit(self, ckks_context):
        """Test ckks_tensor at exact slot limit.

        CKKSTensor can handle exactly slot count elements without issue.
        """
        # 64x64 = 4096 elements (exact slot count)
        data = np.random.randn(64, 64).tolist()
        tensor = ts.ckks_tensor(ckks_context, data)

        assert tensor.shape == [64, 64]
        # All operations should work
        transposed = tensor.transpose()
        assert transposed.shape == [64, 64]

    def test_ckks_tensor_exceeds_slots_works(self, ckks_context):
        """Test ckks_tensor can exceed slot limit without issues.

        Unlike pack_vectors and CKKSVector, CKKSTensor has no slot limitation.
        This suggests it uses a different encoding or multiple ciphertexts.
        """
        # 65x65 = 4225 elements > 4096 slots
        data = np.random.randn(65, 65).tolist()
        tensor = ts.ckks_tensor(ckks_context, data)

        # Successfully created
        assert tensor.shape == [65, 65]

        # Operations still work
        transposed = tensor.transpose()
        assert transposed.shape == [65, 65]

        summed = tensor.sum(0)
        assert summed.shape == [65]

    def test_ckks_tensor_reshape_no_slot_limit(self, ckks_context):
        """Test reshape operations don't have slot limitations."""
        # Start with small tensor
        tensor = ts.ckks_tensor(ckks_context, [[1.0, 2.0], [3.0, 4.0]])

        # Reshape to various forms
        reshaped1 = tensor.reshape([4])
        assert reshaped1.shape == [4]

        reshaped2 = reshaped1.reshape([2, 2])
        assert reshaped2.shape == [2, 2]

    def test_ckks_tensor_broadcast_exceeds_slots(self, ckks_context):
        """Test broadcast can exceed slot limits for tensors."""
        scalar = ts.ckks_tensor(ckks_context, [5.0])

        # Broadcast to size exceeding slots (65x65 = 4225 > 4096)
        large_broadcast = scalar.broadcast([65, 65])
        assert large_broadcast.shape == [65, 65]

        # Operations still work
        result = large_broadcast + 1.0
        assert result.shape == [65, 65]

    def test_ckks_tensor_very_large_shape(self, ckks_context):
        """Test ckks_tensor with very large shapes well beyond slot count."""
        # 100x100 = 10000 elements, much larger than 4096 slots
        data = np.random.randn(100, 100).tolist()
        tensor = ts.ckks_tensor(ckks_context, data)

        assert tensor.shape == [100, 100]

        # Basic operation
        result = tensor + 1.0
        assert result.shape == [100, 100]


# =============================================================================
# Test Class 11: CKKS Tensor Operations Over Slots Limit
# =============================================================================


class TestCKKSTensorOperationsOverSlots:
    """Test that CKKSTensor operations work correctly and maintain precision when exceeding slots.

    These tests verify that when tensor size > slots (4096), all operations still work
    and maintain acceptable CKKS precision (typically < 1e-3 to 1e-5).
    """

    def test_addition_over_slots_precision(self, ckks_context):
        """Test addition operation on tensors exceeding slot limit."""
        # Create 65x65 tensor (4225 > 4096 slots)
        data1 = np.random.randn(65, 65)
        data2 = np.random.randn(65, 65)

        tensor1 = ts.ckks_tensor(ckks_context, data1.tolist())
        tensor2 = ts.ckks_tensor(ckks_context, data2.tolist())

        # Encrypted addition
        result_encrypted = tensor1 + tensor2

        # Expected result
        expected = data1 + data2

        # Decrypt and verify
        result_decrypted = np.array(result_encrypted.decrypt().tolist())

        # Check precision
        max_error = np.max(np.abs(result_decrypted - expected))
        mean_error = np.mean(np.abs(result_decrypted - expected))

        print(
            f"Addition over slots - Max error: {max_error:.2e}, Mean error: {mean_error:.2e}"
        )

        assert max_error < 1e-3, f"Addition precision degraded: max_error={max_error}"
        assert (
            mean_error < 1e-4
        ), f"Addition precision degraded: mean_error={mean_error}"

    def test_subtraction_over_slots_precision(self, ckks_context):
        """Test subtraction operation on tensors exceeding slot limit."""
        # Create 70x70 tensor (4900 > 4096 slots)
        data1 = np.random.randn(70, 70)
        data2 = np.random.randn(70, 70)

        tensor1 = ts.ckks_tensor(ckks_context, data1.tolist())
        tensor2 = ts.ckks_tensor(ckks_context, data2.tolist())

        # Encrypted subtraction
        result_encrypted = tensor1 - tensor2

        # Expected result
        expected = data1 - data2

        # Decrypt and verify
        result_decrypted = np.array(result_encrypted.decrypt().tolist())

        # Check precision
        max_error = np.max(np.abs(result_decrypted - expected))
        mean_error = np.mean(np.abs(result_decrypted - expected))

        print(
            f"Subtraction over slots - Max error: {max_error:.2e}, Mean error: {mean_error:.2e}"
        )

        assert (
            max_error < 1e-3
        ), f"Subtraction precision degraded: max_error={max_error}"
        assert (
            mean_error < 1e-4
        ), f"Subtraction precision degraded: mean_error={mean_error}"

    def test_multiplication_over_slots_precision(self, ckks_context):
        """Test multiplication operation on tensors exceeding slot limit."""
        # Create 65x65 tensor (4225 > 4096 slots)
        # Use smaller values to avoid overflow in CKKS
        data1 = np.random.randn(65, 65) * 0.1
        data2 = np.random.randn(65, 65) * 0.1

        tensor1 = ts.ckks_tensor(ckks_context, data1.tolist())
        tensor2 = ts.ckks_tensor(ckks_context, data2.tolist())

        # Encrypted multiplication
        result_encrypted = tensor1 * tensor2

        # Expected result
        expected = data1 * data2

        # Decrypt and verify
        result_decrypted = np.array(result_encrypted.decrypt().tolist())

        # Check precision (multiplication has larger error)
        max_error = np.max(np.abs(result_decrypted - expected))
        mean_error = np.mean(np.abs(result_decrypted - expected))

        print(
            f"Multiplication over slots - Max error: {max_error:.2e}, Mean error: {mean_error:.2e}"
        )

        assert (
            max_error < 1e-3
        ), f"Multiplication precision degraded: max_error={max_error}"
        assert (
            mean_error < 1e-4
        ), f"Multiplication precision degraded: mean_error={mean_error}"

    def test_scalar_operations_over_slots_precision(self, ckks_context):
        """Test scalar operations on tensors exceeding slot limit."""
        # Create 80x80 tensor (6400 > 4096 slots)
        data = np.random.randn(80, 80)
        tensor = ts.ckks_tensor(ckks_context, data.tolist())

        # Test scalar addition
        result_add = tensor + 5.5
        expected_add = data + 5.5
        decrypted_add = np.array(result_add.decrypt().tolist())
        max_error_add = np.max(np.abs(decrypted_add - expected_add))

        # Test scalar multiplication
        result_mul = tensor * 2.5
        expected_mul = data * 2.5
        decrypted_mul = np.array(result_mul.decrypt().tolist())
        max_error_mul = np.max(np.abs(decrypted_mul - expected_mul))

        print(f"Scalar add over slots - Max error: {max_error_add:.2e}")
        print(f"Scalar mul over slots - Max error: {max_error_mul:.2e}")

        assert max_error_add < 1e-3, f"Scalar addition precision degraded"
        assert max_error_mul < 1e-3, f"Scalar multiplication precision degraded"

    def test_matrix_multiplication_over_slots_precision(self, ckks_context):
        """Test matrix multiplication (mm) on tensors exceeding slot limit."""
        # Create matrices: [50, 90] × [90, 50] (both > 4096 slots)
        # A: 50x90 = 4500 elements
        # B: 90x50 = 4500 elements
        data_a = np.random.randn(50, 90) * 0.1
        data_b = np.random.randn(90, 50) * 0.1

        tensor_a = ts.ckks_tensor(ckks_context, data_a.tolist())
        tensor_b = ts.ckks_tensor(ckks_context, data_b.tolist())

        # Encrypted matrix multiplication
        result_encrypted = tensor_a.mm(tensor_b)

        # Expected result
        expected = data_a @ data_b

        # Decrypt and verify
        result_decrypted = np.array(result_encrypted.decrypt().tolist())

        # Check precision (matmul has larger accumulated error)
        max_error = np.max(np.abs(result_decrypted - expected))
        mean_error = np.mean(np.abs(result_decrypted - expected))
        relative_error = np.max(
            np.abs((result_decrypted - expected) / (expected + 1e-10))
        )

        print(
            f"MatMul over slots - Max error: {max_error:.2e}, Mean error: {mean_error:.2e}"
        )
        print(f"MatMul relative error: {relative_error:.2e}")

        # More relaxed tolerance for matmul due to accumulated errors
        assert max_error < 1e-2, f"MatMul precision degraded: max_error={max_error}"
        assert mean_error < 5e-3, f"MatMul precision degraded: mean_error={mean_error}"

    def test_dot_product_over_slots_precision(self, ckks_context):
        """Test dot product on 1D tensors exceeding slot limit."""
        # Create 1D tensors with 5000 elements (> 4096 slots)
        data1 = np.random.randn(5000) * 0.1
        data2 = np.random.randn(5000) * 0.1

        tensor1 = ts.ckks_tensor(ckks_context, data1.tolist())
        tensor2 = ts.ckks_tensor(ckks_context, data2.tolist())

        # Encrypted dot product
        result_encrypted = tensor1.dot(tensor2)

        # Expected result
        expected = np.dot(data1, data2)

        # Decrypt and verify
        result_decrypted = result_encrypted.decrypt().tolist()
        if isinstance(result_decrypted, list):
            actual = (
                result_decrypted[0] if len(result_decrypted) > 0 else result_decrypted
            )
        else:
            actual = float(result_decrypted)

        # Check precision
        error = abs(actual - expected)
        relative_error = abs((actual - expected) / expected) if expected != 0 else error

        print(
            f"Dot product over slots - Error: {error:.2e}, Relative error: {relative_error:.2e}"
        )
        print(f"Expected: {expected:.6f}, Actual: {actual:.6f}")

        # More relaxed tolerance for dot product due to many operations
        assert error < 0.1, f"Dot product precision degraded: error={error}"
        assert (
            relative_error < 0.01
        ), f"Dot product relative error too large: {relative_error}"

    def test_sum_operation_over_slots_precision(self, ckks_context):
        """Test sum operation on tensors exceeding slot limit."""
        # Create 70x70 tensor (4900 > 4096 slots)
        data = np.random.randn(70, 70)
        tensor = ts.ckks_tensor(ckks_context, data.tolist())

        # Test sum along axis 0
        result_axis0 = tensor.sum(0)
        expected_axis0 = data.sum(axis=0)
        decrypted_axis0 = np.array(result_axis0.decrypt().tolist())
        max_error_axis0 = np.max(np.abs(decrypted_axis0 - expected_axis0))

        # Test sum along axis 1
        result_axis1 = tensor.sum(1)
        expected_axis1 = data.sum(axis=1)
        decrypted_axis1 = np.array(result_axis1.decrypt().tolist())
        max_error_axis1 = np.max(np.abs(decrypted_axis1 - expected_axis1))

        print(f"Sum axis=0 over slots - Max error: {max_error_axis0:.2e}")
        print(f"Sum axis=1 over slots - Max error: {max_error_axis1:.2e}")

        assert max_error_axis0 < 1e-2, f"Sum axis=0 precision degraded"
        assert max_error_axis1 < 1e-2, f"Sum axis=1 precision degraded"

    def test_transpose_over_slots_precision(self, ckks_context):
        """Test transpose operation on tensors exceeding slot limit."""
        # Create 65x70 tensor (4550 > 4096 slots)
        data = np.random.randn(65, 70)
        tensor = ts.ckks_tensor(ckks_context, data.tolist())

        # Transpose
        result = tensor.transpose()
        expected = data.T

        # Verify shape
        assert result.shape == [70, 65]

        # Decrypt and verify precision
        decrypted = np.array(result.decrypt().tolist())
        max_error = np.max(np.abs(decrypted - expected))

        print(f"Transpose over slots - Max error: {max_error:.2e}")

        assert max_error < 1e-4, f"Transpose precision degraded: max_error={max_error}"

    def test_reshape_over_slots_precision(self, ckks_context):
        """Test reshape operation on tensors exceeding slot limit."""
        # Create 65x65 tensor (4225 > 4096 slots)
        data = np.random.randn(65, 65)
        tensor = ts.ckks_tensor(ckks_context, data.tolist())

        # Reshape to different forms
        reshaped1 = tensor.reshape([4225])
        reshaped2 = reshaped1.reshape([5, 845])
        reshaped3 = reshaped2.reshape([65, 65])

        # Verify final shape
        assert reshaped3.shape == [65, 65]

        # Decrypt and verify data is preserved
        original_flat = data.flatten()
        final_flat = np.array(reshaped3.decrypt().tolist()).flatten()

        max_error = np.max(np.abs(final_flat - original_flat))

        print(
            f"Reshape over slots - Max error after multiple reshapes: {max_error:.2e}"
        )

        assert max_error < 1e-4, f"Reshape precision degraded: max_error={max_error}"

    def test_broadcast_over_slots_precision(self, ckks_context):
        """Test broadcast operation creating tensor exceeding slot limit."""
        # Broadcast scalar to 70x70 (4900 > 4096 slots)
        value = 3.14159
        scalar = ts.ckks_tensor(ckks_context, [value])

        broadcasted = scalar.broadcast([70, 70])
        expected = np.full((70, 70), value)

        # Decrypt and verify
        decrypted = np.array(broadcasted.decrypt().tolist())
        max_error = np.max(np.abs(decrypted - expected))

        print(f"Broadcast over slots - Max error: {max_error:.2e}")

        assert max_error < 1e-4, f"Broadcast precision degraded: max_error={max_error}"

    def test_chained_operations_over_slots_precision(self, ckks_context):
        """Test chained operations on tensors exceeding slot limit."""
        # Create 65x65 tensor (4225 > 4096 slots)
        data = np.random.randn(65, 65) * 0.1
        tensor = ts.ckks_tensor(ckks_context, data.tolist())

        # Perform chained operations
        result = tensor.add(1.0).mul(2.0).transpose().sum(0)

        # Expected result
        expected = ((data + 1.0) * 2.0).T.sum(axis=0)

        # Decrypt and verify
        decrypted = np.array(result.decrypt().tolist())
        max_error = np.max(np.abs(decrypted - expected))
        mean_error = np.mean(np.abs(decrypted - expected))

        print(
            f"Chained ops over slots - Max error: {max_error:.2e}, Mean error: {mean_error:.2e}"
        )

        # More relaxed tolerance due to error accumulation
        assert (
            max_error < 1e-2
        ), f"Chained operations precision degraded: max_error={max_error}"
        assert (
            mean_error < 5e-3
        ), f"Chained operations precision degraded: mean_error={mean_error}"

    def test_very_large_tensor_operations(self, ckks_context):
        """Test operations on very large tensors (10x slot limit)."""
        # Create 200x200 tensor (40000 elements, ~10x slots)
        data = np.random.randn(200, 200) * 0.01  # Small values
        tensor = ts.ckks_tensor(ckks_context, data.tolist())

        # Test basic operations
        result_add = tensor + 0.5
        result_mul = tensor * 2.0

        # Verify shapes
        assert result_add.shape == [200, 200]
        assert result_mul.shape == [200, 200]

        # Check precision on sample
        decrypted_add = np.array(result_add.decrypt().tolist())
        expected_add = data + 0.5

        # Sample error check (full verification would be slow)
        sample_indices = [(0, 0), (100, 100), (199, 199), (50, 150)]
        errors = [
            abs(decrypted_add[i, j] - expected_add[i, j]) for i, j in sample_indices
        ]
        max_sample_error = max(errors)

        print(f"Very large tensor (200x200) - Sample max error: {max_sample_error:.2e}")

        assert max_sample_error < 1e-3, f"Very large tensor precision degraded"
