import numpy as np

import mplang2.backends.bfv_impl as _bfv_impl  # noqa: F401
import mplang2.backends.tensor_impl as _tensor_impl  # noqa: F401
from mplang2.dialects import bfv, tensor
from mplang2.edsl.jit import jit


def test_bfv_e2e():
    """Test BFV basic workflow: Keygen -> Encrypt -> Add/Mul -> Decrypt."""

    @jit
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

    np.testing.assert_array_equal(res_sum.runtime_obj[:4], expected_sum)
    np.testing.assert_array_equal(res_prod.runtime_obj[:4], expected_prod)
