import numpy as np
import pytest

import mplang2.backends.bfv_impl as _bfv_impl  # noqa: F401
import mplang2.backends.tensor_impl as _tensor_impl  # noqa: F401
from mplang2.dialects import bfv, tensor


def test_bfv_e2e():
    """Test BFV basic workflow: Keygen -> Encrypt -> Add/Mul -> Decrypt."""

    def workload():
        # 1. Setup
        # Use smaller degree for faster test
        poly_modulus_degree = 4096
        pk, sk = bfv.keygen(poly_modulus_degree=poly_modulus_degree)
        relin_keys = bfv.make_relin_keys(sk)
        encoder = bfv.create_encoder(poly_modulus_degree=poly_modulus_degree)

        # 2. Data
        v1 = tensor.constant(np.array([1, 2, 3, 4], dtype=np.int64))
        v2 = tensor.constant(np.array([10, 20, 30, 40], dtype=np.int64))

        # 3. Encode & Encrypt
        pt1 = bfv.encode(v1, encoder)
        ct1 = bfv.encrypt(pt1, pk)

        pt2 = bfv.encode(v2, encoder)
        ct2 = bfv.encrypt(pt2, pk)

        # 4. Computation
        ct_sum = bfv.add(ct1, ct2)
        ct_prod = bfv.mul(ct1, ct2)
        ct_prod = bfv.relinearize(ct_prod, relin_keys)

        # 5. Decrypt
        pt_sum = bfv.decrypt(ct_sum, sk)
        res_sum = bfv.decode(pt_sum, encoder)

        pt_prod = bfv.decrypt(ct_prod, sk)
        res_prod = bfv.decode(pt_prod, encoder)

        return res_sum, res_prod

    res_sum, res_prod = workload()

    # Verify
    # BFV decode returns array of size poly_modulus_degree.
    # We only care about the first 4 elements.
    expected_sum = np.array([11, 22, 33, 44], dtype=np.int64)
    expected_prod = np.array([10, 40, 90, 160], dtype=np.int64)

    # Handle InterpObject wrapper if present
    val_sum = res_sum.runtime_obj if hasattr(res_sum, "runtime_obj") else res_sum
    val_prod = res_prod.runtime_obj if hasattr(res_prod, "runtime_obj") else res_prod

    np.testing.assert_array_equal(val_sum[:4], expected_sum)
    np.testing.assert_array_equal(val_prod[:4], expected_prod)


def test_bfv_rotate():
    """Test that BFV rotation works correctly."""

    def workload():
        poly_modulus_degree = 4096
        pk, sk = bfv.keygen(poly_modulus_degree=poly_modulus_degree)
        galois_keys = bfv.make_galois_keys(sk)
        encoder = bfv.create_encoder(poly_modulus_degree=poly_modulus_degree)

        v = tensor.constant(np.array([1, 2, 3, 4], dtype=np.int64))
        pt = bfv.encode(v, encoder)
        ct = bfv.encrypt(pt, pk)

        # Rotate by 1
        rotated_ct = bfv.rotate(ct, 1, galois_keys)

        # Decrypt
        decrypted = bfv.decrypt(rotated_ct, sk)
        return bfv.decode(decrypted, encoder)

    result = workload()
    # result is numpy array from decode_impl
    # If running in eager mode with Object wrappers, we might need .runtime_obj
    if hasattr(result, "runtime_obj"):
        res_data = result.runtime_obj
    else:
        res_data = result

    print(f"Rotated result: {res_data[:4]}")
    np.testing.assert_array_equal(res_data[:3], [2, 3, 4])
    # The last element might be 0 or 1 depending on slot count/behavior.
    # If it's 1, then it wrapped around.
    if res_data[3] == 1:
        np.testing.assert_array_equal(res_data[:4], [2, 3, 4, 1])
    else:
        np.testing.assert_array_equal(res_data[:4], [2, 3, 4, 0])


def test_bfv_arithmetic_mixed():
    """Test BFV mixed arithmetic (CT+CT, CT+PT, CT+Raw, Subtraction)."""

    def workload():
        poly_modulus_degree = 4096
        pk, sk = bfv.keygen(poly_modulus_degree=poly_modulus_degree)
        encoder = bfv.create_encoder(poly_modulus_degree=poly_modulus_degree)

        v1 = tensor.constant(np.array([10, 20, 30, 40], dtype=np.int64))
        v2 = tensor.constant(np.array([1, 2, 3, 4], dtype=np.int64))

        pt1 = bfv.encode(v1, encoder)
        ct1 = bfv.encrypt(pt1, pk)

        pt2 = bfv.encode(v2, encoder)
        ct2 = bfv.encrypt(pt2, pk)

        # 1. Subtraction (CT - CT)
        ct_sub = bfv.sub(ct1, ct2)

        # 2. Add Plain (CT + PT)
        ct_add_plain = bfv.add(ct1, pt2)

        # 3. Mul Plain (CT * PT)
        ct_mul_plain = bfv.mul(ct1, pt2)

        # 4. Sub Plain (CT - PT)
        ct_sub_plain = bfv.sub(ct1, pt2)

        # 5. Sub Plain Reverse (PT - CT) -> Not directly supported by bfv.sub usually?
        # But let's try if our backend handles it (it should negate CT)
        # Note: bfv.sub(pt2, ct1) might fail if dialect type checking enforces CT as first arg?
        # Dialect definition: _sub_ae(lhs, rhs) -> checks operands.
        # It allows mixed.
        ct_sub_plain_rev = bfv.sub(pt2, ct1)

        # Decrypt all
        res_sub = bfv.decode(bfv.decrypt(ct_sub, sk), encoder)
        res_add_plain = bfv.decode(bfv.decrypt(ct_add_plain, sk), encoder)
        res_mul_plain = bfv.decode(bfv.decrypt(ct_mul_plain, sk), encoder)
        res_sub_plain = bfv.decode(bfv.decrypt(ct_sub_plain, sk), encoder)
        res_sub_plain_rev = bfv.decode(bfv.decrypt(ct_sub_plain_rev, sk), encoder)

        return res_sub, res_add_plain, res_mul_plain, res_sub_plain, res_sub_plain_rev

    results = workload()

    # Unwrap if needed
    unwrapped = []
    for r in results:
        unwrapped.append(r.runtime_obj if hasattr(r, "runtime_obj") else r)

    res_sub, res_add_plain, res_mul_plain, res_sub_plain, res_sub_plain_rev = unwrapped

    # Expected
    # v1 = [10, 20, 30, 40]
    # v2 = [1, 2, 3, 4]

    # sub: [9, 18, 27, 36]
    np.testing.assert_array_equal(res_sub[:4], [9, 18, 27, 36])

    # add_plain: [11, 22, 33, 44]
    np.testing.assert_array_equal(res_add_plain[:4], [11, 22, 33, 44])

    # mul_plain: [10, 40, 90, 160]
    np.testing.assert_array_equal(res_mul_plain[:4], [10, 40, 90, 160])

    # sub_plain: [9, 18, 27, 36]
    np.testing.assert_array_equal(res_sub_plain[:4], [9, 18, 27, 36])

    # sub_plain_rev: v2 - v1 = [-9, -18, -27, -36]
    # Note: BFV is modular arithmetic. Negative numbers are represented as (Modulus - x).
    # But decode_int64 might handle signed integers if configured?
    # TenSEAL/SEAL BatchEncoder usually handles signed integers if plain_modulus is appropriate.
    # Our plain_modulus is 1032193.
    # Let's see what we get.
    print(f"Sub Plain Rev: {res_sub_plain_rev[:4]}")

    # If it returns large positive numbers, we can check against (modulus - x)
    # Or cast to int64?
    # Let's check if it matches expected negative values (if decoded correctly)
    # or check modular equivalence.

    # For now, let's just print and see.
    # If the encoder is configured for signed, it should return negative values.
    # 1032193 is small, so it fits in int64.

    # Assuming standard behavior:
    np.testing.assert_array_equal(res_sub_plain_rev[:4], [-9, -18, -27, -36])


def test_bfv_relinearize():
    """Test BFV relinearization (check size reduction)."""

    def workload():
        poly_modulus_degree = 4096
        pk, sk = bfv.keygen(poly_modulus_degree=poly_modulus_degree)
        relin_keys = bfv.make_relin_keys(sk)
        encoder = bfv.create_encoder(poly_modulus_degree=poly_modulus_degree)

        v = tensor.constant(np.array([2, 2, 2, 2], dtype=np.int64))
        pt = bfv.encode(v, encoder)
        ct = bfv.encrypt(pt, pk)

        # Mul increases size to 3
        ct2 = bfv.mul(ct, ct)

        # Relinearize reduces size to 2
        ct3 = bfv.relinearize(ct2, relin_keys)

        # Decrypt to verify correctness
        res = bfv.decode(bfv.decrypt(ct3, sk), encoder)

        # We can't easily check size in EDSL without exposing a size op.
        # But we can check correctness.
        return res

    result = workload()
    res_data = result.runtime_obj if hasattr(result, "runtime_obj") else result

    np.testing.assert_array_equal(res_data[:4], [4, 4, 4, 4])


def test_bfv_security_types():
    """Test that Public Context cannot be used for decryption."""
    poly_modulus_degree = 4096
    pk, sk = bfv.keygen(poly_modulus_degree=poly_modulus_degree)
    encoder = bfv.create_encoder(poly_modulus_degree=poly_modulus_degree)

    v = tensor.constant(np.array([1, 2, 3, 4], dtype=np.int64))
    pt = bfv.encode(v, encoder)
    ct = bfv.encrypt(pt, pk)

    # Try to decrypt with PK (should fail)
    # pk is BFVPublicContext, does not have decryptor
    # The dialect type checker raises TypeError because it expects PrivateKey
    with pytest.raises(TypeError):
        bfv.decrypt(ct, pk)

    # Decrypt with SK (should succeed)
    res = bfv.decode(bfv.decrypt(ct, sk), encoder)
    res_data = res.runtime_obj if hasattr(res, "runtime_obj") else res
    np.testing.assert_array_equal(res_data[:4], [1, 2, 3, 4])
