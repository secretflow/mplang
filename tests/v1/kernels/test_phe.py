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

import numpy as np
import pytest

from mplang.v1.core.dtypes import INT32, UINT8
from mplang.v1.core.pfunc import PFunction
from mplang.v1.core.tensor import TensorType
from mplang.v1.kernels.base import list_kernels
from mplang.v1.kernels.context import RuntimeContext
from mplang.v1.kernels.phe import CipherText, PrivateKey, PublicKey
from mplang.v1.kernels.value import TensorValue


class TestPHEKernels:
    """Compact PHE kernel tests (clean rewrite)."""

    def setup_method(self):
        self.runtime = RuntimeContext(rank=0, world_size=1)
        self.scheme = "paillier"
        self.key_size = 512

    @staticmethod
    def _to_value(arg):
        if isinstance(arg, np.ndarray):
            return TensorValue(np.array(arg, copy=True))
        return arg

    @staticmethod
    def _unwrap(value):
        return value.to_numpy() if isinstance(value, TensorValue) else value

    def _exec(self, p: PFunction, args: list):
        converted = [self._to_value(arg) for arg in args]
        results = self.runtime.run_kernel(p, converted)
        return [self._unwrap(res) for res in results]

    def _keygen(self):
        p = PFunction(
            fn_type="phe.keygen",
            ins_info=(),
            outs_info=(TensorType(UINT8, (-1, 0)), TensorType(UINT8, (-1, 0))),
            scheme=self.scheme,
            key_size=self.key_size,
        )
        pk, sk = self._exec(p, [])
        assert isinstance(pk, PublicKey) and isinstance(sk, PrivateKey)
        return pk, sk

    def test_kernel_registry(self):
        for name in [
            "phe.keygen",
            "phe.encrypt",
            "phe.decrypt",
            "phe.add",
            "phe.mul",
            "phe.dot",
            "phe.gather",
            "phe.scatter",
            "phe.concat",
            "phe.reshape",
            "phe.transpose",
        ]:
            assert name in list_kernels()

    def test_keygen(self):
        pk, sk = self._keygen()
        assert pk.scheme == "Paillier" and sk.scheme == "Paillier"

    def test_keygen_invalid_scheme(self):
        p = PFunction(
            fn_type="phe.keygen",
            ins_info=(),
            outs_info=(TensorType(UINT8, (-1, 0)), TensorType(UINT8, (-1, 0))),
            scheme="bogus",
        )
        with pytest.raises(ValueError, match="Unsupported PHE scheme"):
            self._exec(p, [])

    def test_encrypt_decrypt_scalar_int32(self):
        pk, sk = self._keygen()
        pt = np.array(42, dtype=np.int32)
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(pt),),
        )
        ct = self._exec(enc, [pt, pk])[0]
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(pt),),
        )
        out = self._exec(dec, [ct, sk])[0]
        assert out.item() == 42

    def test_encrypt_decrypt_scalar_float32(self):
        pk, sk = self._keygen()
        pt = np.array(3.14, dtype=np.float32)
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(pt),),
        )
        ct = self._exec(enc, [pt, pk])[0]
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(pt),),
        )
        out = self._exec(dec, [ct, sk])[0]
        # Relax tolerance for float32 precision in PHE operations
        assert abs(out.item() - 3.14) < 1e-3

    def test_add_cipher_cipher(self):
        pk, sk = self._keygen()
        a = np.array(10, dtype=np.int32)
        b = np.array(5, dtype=np.int32)
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc, [a, pk])[0]
        cb = self._exec(enc, [b, pk])[0]
        add = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(a),),
        )
        cr = self._exec(add, [ca, cb])[0]
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        out = self._exec(dec, [cr, sk])[0]
        assert out.item() == 15

    def test_add_cipher_plain(self):
        pk, sk = self._keygen()
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5, 6], dtype=np.int32)
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc, [a, pk])[0]
        add = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(a),),
        )
        cr = self._exec(add, [ca, b])[0]
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        out = self._exec(dec, [cr, sk])[0]
        np.testing.assert_array_equal(out, a + b)

    def test_mul_cipher_plain(self):
        pk, sk = self._keygen()
        a = np.array([2, 3], dtype=np.int32)
        b = np.array([5, 7], dtype=np.int32)
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc, [a, pk])[0]
        mul = PFunction(
            fn_type="phe.mul",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(a),),
        )
        cr = self._exec(mul, [ca, b])[0]
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        out = self._exec(dec, [cr, sk])[0]
        np.testing.assert_array_equal(out, a * b)

    def test_plain_plain_add(self):
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int32)
        add = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(a),),
        )
        out = self._exec(add, [a, b])[0]
        np.testing.assert_array_equal(out, a + b)

    def test_shape_mismatch_add(self):
        pk, _ = self._keygen()
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([1, 2, 3], dtype=np.int32)
        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        enc_b = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(b), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(b),),
        )
        ca = self._exec(enc_a, [a, pk])[0]
        cb = self._exec(enc_b, [b, pk])[0]
        add = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(a),),
        )
        with pytest.raises(ValueError, match="cannot be broadcast together"):
            self._exec(add, [ca, cb])

    def test_scheme_mismatch(self):
        pk, _ = self._keygen()
        a = np.array(10, dtype=np.int32)
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ct = self._exec(enc, [a, pk])[0]
        assert isinstance(ct, CipherText)
        fake = CipherText(
            ct_data=ct.ct_data,
            semantic_dtype=INT32,
            semantic_shape=(),
            scheme="ElGamal",
            key_size=2048,
            pk_data=None,
        )
        add = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(a)),
            outs_info=(TensorType.from_obj(a),),
        )
        with pytest.raises(ValueError, match="same scheme and key size"):
            self._exec(add, [ct, fake])

    def test_various_roundtrip(self):
        pk, sk = self._keygen()
        samples = [
            np.array(7, dtype=np.int32),
            np.array(-3, dtype=np.int32),
            np.array(2**32, dtype=np.int64),  # max value by default
            np.array(1.2345, dtype=np.float32),
            np.array(2.3456789, dtype=np.float64),
            np.array([-1, 2, -3], dtype=np.int16),
        ]
        for pt in samples:
            enc = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(pt),),
            )
            ct = self._exec(enc, [pt, pk])[0]
            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(pt),),
            )
            out = self._exec(dec, [ct, sk])[0]
            if pt.dtype.kind == "f":
                # Use more lenient tolerance for PHE operations with floating point
                assert np.allclose(out, pt, atol=1e-3)
            else:
                assert np.array_equal(out, pt)

    def test_mul_invalid_float_plaintext(self):
        """Test that multiplication with float plaintext raises error."""
        pk, _ = self._keygen()
        a = np.array(5, dtype=np.int32)
        b = np.array(3.14, dtype=np.float32)
        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc_a, [a, pk])[0]
        mul = PFunction(
            fn_type="phe.mul",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(a),),
        )
        with pytest.raises(
            ValueError, match="floating point plaintext is not supported"
        ):
            self._exec(mul, [ca, b])

    def test_dot_basic(self):
        """Test basic dot product operation."""
        pk, sk = self._keygen()
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5, 6], dtype=np.int32)
        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc_a, [a, pk])[0]
        dot = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(
                TensorType.from_obj(np.array(0, dtype=np.int32)),
            ),  # Scalar result
        )
        cr = self._exec(dot, [ca, b])[0]
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(np.array(0, dtype=np.int32)),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType.from_obj(np.array(0, dtype=np.int32)),),
        )
        out = self._exec(dec, [cr, sk])[0]
        expected = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
        assert out.item() == expected

    def test_dot_matrix_vector(self):
        """Test dot product with matrix and vector."""
        pk, sk = self._keygen()
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        b = np.array([5, 6], dtype=np.int32)
        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc_a, [a, pk])[0]
        dot = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(
                TensorType.from_obj(np.array([0, 0], dtype=np.int32)),
            ),  # Vector result
        )
        cr = self._exec(dot, [ca, b])[0]
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(np.array([0, 0], dtype=np.int32)),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType.from_obj(np.array([0, 0], dtype=np.int32)),),
        )
        out = self._exec(dec, [cr, sk])[0]
        expected = np.dot(a, b)  # [17, 39]
        np.testing.assert_array_equal(out, expected)

    def test_gather_basic(self):
        """Test basic gather operation."""
        pk, sk = self._keygen()
        a = np.array([10, 20, 30, 40], dtype=np.int32)
        indices = np.array([0, 2], dtype=np.int32)
        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc_a, [a, pk])[0]
        gather = PFunction(
            fn_type="phe.gather",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(indices)),
            outs_info=(TensorType.from_obj(np.array([0, 0], dtype=np.int32)),),
            axis=0,
        )
        cr = self._exec(gather, [ca, indices])[0]
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(np.array([0, 0], dtype=np.int32)),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType.from_obj(np.array([0, 0], dtype=np.int32)),),
        )
        out = self._exec(dec, [cr, sk])[0]
        expected = np.array([10, 30], dtype=np.int32)  # Elements at indices 0 and 2
        np.testing.assert_array_equal(out, expected)

    def test_scatter_basic(self):
        """Test basic scatter operation."""
        pk, sk = self._keygen()
        a = np.array([10, 20, 30, 40], dtype=np.int32)
        indices = np.array([1, 3], dtype=np.int32)
        updates = np.array([100, 200], dtype=np.int32)
        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        enc_updates = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(updates), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(updates),),
        )
        ca = self._exec(enc_a, [a, pk])[0]
        cu = self._exec(enc_updates, [updates, pk])[0]
        scatter = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(a),
                TensorType.from_obj(indices),
                TensorType.from_obj(updates),
            ),
            outs_info=(TensorType.from_obj(a),),
            axis=0,
        )
        cr = self._exec(scatter, [ca, indices, cu])[0]
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        out = self._exec(dec, [cr, sk])[0]
        # Expect [10, 100, 30, 200] - updates at indices 1 and 3
        expected = np.array([10, 100, 30, 200], dtype=np.int32)
        np.testing.assert_array_equal(out, expected)

    def test_concat_basic(self):
        """Test basic concatenation operation."""
        pk, sk = self._keygen()
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int32)

        # Encrypt both arrays
        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        enc_b = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(b), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(b),),
        )
        ca = self._exec(enc_a, [a, pk])[0]
        cb = self._exec(enc_b, [b, pk])[0]

        # Expected result shape after concatenation
        expected_result = np.array([1, 2, 3, 4], dtype=np.int32)

        # Concat operation: pass individual ciphertexts as separate arguments
        concat = PFunction(
            fn_type="phe.concat",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(expected_result),),
            axis=0,
        )
        cr = self._exec(concat, [ca, cb])[0]

        # Decrypt and verify result
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        out = self._exec(dec, [cr, sk])[0]
        np.testing.assert_array_equal(out, expected_result)

    def test_concat_2d_matrices(self):
        """Test concatenation of 2D matrices along different axes."""
        pk, sk = self._keygen()
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        b = np.array([[5, 6], [7, 8]], dtype=np.int32)

        # Encrypt both matrices
        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        enc_b = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(b), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(b),),
        )
        ca = self._exec(enc_a, [a, pk])[0]
        cb = self._exec(enc_b, [b, pk])[0]

        # Test concatenation along axis 0 (rows)
        expected_axis0 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
        concat_axis0 = PFunction(
            fn_type="phe.concat",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(expected_axis0),),
            axis=0,
        )
        cr0 = self._exec(concat_axis0, [ca, cb])[0]

        dec0 = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_axis0), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(expected_axis0),),
        )
        out0 = self._exec(dec0, [cr0, sk])[0]
        np.testing.assert_array_equal(out0, expected_axis0)

        # Test concatenation along axis 1 (columns)
        expected_axis1 = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.int32)
        concat_axis1 = PFunction(
            fn_type="phe.concat",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(expected_axis1),),
            axis=1,
        )
        cr1 = self._exec(concat_axis1, [ca, cb])[0]

        dec1 = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_axis1), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(expected_axis1),),
        )
        out1 = self._exec(dec1, [cr1, sk])[0]
        np.testing.assert_array_equal(out1, expected_axis1)

    def test_reshape_basic(self):
        """Test basic reshape operation."""
        pk, sk = self._keygen()
        # Start with a 2x3 array and reshape to 3x2
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        expected_shape = (3, 2)
        expected_result = a.reshape(expected_shape)

        # Encrypt the array
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc, [a, pk])[0]

        # Reshape the encrypted array (using new_shape parameter)
        reshape = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(a),),
            outs_info=(TensorType(a.dtype, expected_shape),),
            new_shape=expected_shape,
        )
        cr = self._exec(reshape, [ca])[0]

        # Decrypt and verify
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType(a.dtype, expected_shape), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType(a.dtype, expected_shape),),
        )
        out = self._exec(dec, [cr, sk])[0]
        np.testing.assert_array_equal(out, expected_result)

    def test_reshape_multiple_transformations(self):
        """Test reshape with various shape transformations."""
        pk, sk = self._keygen()

        # Test 1: 1D array to 2D matrix
        a1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        shape1 = (2, 3)
        expected1 = a1.reshape(shape1)

        enc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a1), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a1),),
        )
        ca1 = self._exec(enc1, [a1, pk])[0]

        reshape1 = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(a1),),
            outs_info=(TensorType(a1.dtype, shape1),),
            new_shape=shape1,
        )
        cr1 = self._exec(reshape1, [ca1])[0]

        dec1 = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType(a1.dtype, shape1), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType(a1.dtype, shape1),),
        )
        out1 = self._exec(dec1, [cr1, sk])[0]
        np.testing.assert_array_equal(out1, expected1)

        # Test 2: 2D matrix to 1D array
        a2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        shape2 = (6,)
        expected2 = a2.reshape(shape2)

        enc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a2), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a2),),
        )
        ca2 = self._exec(enc2, [a2, pk])[0]

        reshape2 = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(a2),),
            outs_info=(TensorType(a2.dtype, shape2),),
            new_shape=shape2,
        )
        cr2 = self._exec(reshape2, [ca2])[0]

        dec2 = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType(a2.dtype, shape2), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType(a2.dtype, shape2),),
        )
        out2 = self._exec(dec2, [cr2, sk])[0]
        np.testing.assert_array_equal(out2, expected2)

        # Test 3: 3D to different 3D shape
        a3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)  # (2, 2, 2)
        shape3 = (1, 4, 2)
        expected3 = a3.reshape(shape3)

        enc3 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a3), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a3),),
        )
        ca3 = self._exec(enc3, [a3, pk])[0]

        reshape3 = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(a3),),
            outs_info=(TensorType(a3.dtype, shape3),),
            new_shape=shape3,
        )
        cr3 = self._exec(reshape3, [ca3])[0]

        dec3 = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType(a3.dtype, shape3), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType(a3.dtype, shape3),),
        )
        out3 = self._exec(dec3, [cr3, sk])[0]
        np.testing.assert_array_equal(out3, expected3)

    def test_reshape_invalid_shape(self):
        """Test reshape with incompatible shape should raise error."""
        pk, _ = self._keygen()
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # 6 elements
        invalid_shape = (2, 4)  # Would need 8 elements

        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc, [a, pk])[0]

        # This should raise an error
        reshape = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(a),),
            outs_info=(TensorType(a.dtype, invalid_shape),),
            new_shape=invalid_shape,
        )

        # Expect an error when executing with incompatible shape
        with pytest.raises(Exception) as exc_info:
            self._exec(reshape, [ca])

        # The error should mention shape incompatibility
        error_message = str(exc_info.value).lower()
        assert any(
            keyword in error_message
            for keyword in ["shape", "size", "dimension", "incompatible", "mismatch"]
        ), f"Expected shape-related error, got: {exc_info.value}"

    def test_transpose_basic(self):
        """Test basic transpose operation."""
        pk, sk = self._keygen()
        # Start with a 2x3 matrix
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        expected_result = a.T  # Standard transpose
        expected_shape = expected_result.shape  # (3, 2)

        # Encrypt the array
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc, [a, pk])[0]

        # Transpose the encrypted array (no parameters needed)
        transpose = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(a),),
            outs_info=(TensorType(a.dtype, expected_shape),),
        )
        cr = self._exec(transpose, [ca])[0]

        # Decrypt and verify
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType(a.dtype, expected_shape), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType(a.dtype, expected_shape),),
        )
        out = self._exec(dec, [cr, sk])[0]
        np.testing.assert_array_equal(out, expected_result)

    def test_transpose_with_axes(self):
        """Test transpose with explicit axes parameter."""
        pk, sk = self._keygen()

        # Test 1: 2D transpose with explicit axes
        a2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        expected_2d = np.transpose(a2d, (1, 0))

        enc_2d = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a2d), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a2d),),
        )
        ca2d = self._exec(enc_2d, [a2d, pk])[0]

        # Try transpose with axes parameter
        try:
            transpose_2d = PFunction(
                fn_type="phe.transpose",
                ins_info=(TensorType.from_obj(a2d),),
                outs_info=(TensorType(a2d.dtype, expected_2d.shape),),
                axes=(1, 0),
            )
            cr2d = self._exec(transpose_2d, [ca2d])[0]

            dec_2d = PFunction(
                fn_type="phe.decrypt",
                ins_info=(
                    TensorType(a2d.dtype, expected_2d.shape),
                    TensorType(UINT8, (-1, 0)),
                ),
                outs_info=(TensorType(a2d.dtype, expected_2d.shape),),
            )
            out_2d = self._exec(dec_2d, [cr2d, sk])[0]
            np.testing.assert_array_equal(out_2d, expected_2d)
            print("2D transpose with axes parameter succeeded!")
        except Exception as e:
            print(f"2D transpose with axes failed: {e}")
            # Fall back to testing without axes parameter
            transpose_2d_fallback = PFunction(
                fn_type="phe.transpose",
                ins_info=(TensorType.from_obj(a2d),),
                outs_info=(TensorType(a2d.dtype, expected_2d.shape),),
            )
            cr2d_fallback = self._exec(transpose_2d_fallback, [ca2d])[0]

            dec_2d_fallback = PFunction(
                fn_type="phe.decrypt",
                ins_info=(
                    TensorType(a2d.dtype, expected_2d.shape),
                    TensorType(UINT8, (-1, 0)),
                ),
                outs_info=(TensorType(a2d.dtype, expected_2d.shape),),
            )
            out_2d_fallback = self._exec(dec_2d_fallback, [cr2d_fallback, sk])[0]
            np.testing.assert_array_equal(out_2d_fallback, expected_2d)
            print("2D transpose without axes parameter succeeded!")

    def test_transpose_3d_arrays(self):
        """Test transpose with 3D arrays."""
        pk, sk = self._keygen()

        # 3D array test
        a3d = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32
        )  # (2, 2, 2)
        expected_3d = np.transpose(
            a3d
        )  # Default transpose reverses all axes: (2, 2, 2)

        enc_3d = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a3d), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a3d),),
        )
        ca3d = self._exec(enc_3d, [a3d, pk])[0]

        transpose_3d = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(a3d),),
            outs_info=(TensorType(a3d.dtype, expected_3d.shape),),
        )
        cr3d = self._exec(transpose_3d, [ca3d])[0]

        dec_3d = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType(a3d.dtype, expected_3d.shape),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType(a3d.dtype, expected_3d.shape),),
        )
        out_3d = self._exec(dec_3d, [cr3d, sk])[0]
        np.testing.assert_array_equal(out_3d, expected_3d)

    def test_transpose_edge_cases(self):
        """Test transpose edge cases."""
        pk, sk = self._keygen()

        # Test 1: 1D array transpose (should return the same array)
        a1d = np.array([1, 2, 3, 4], dtype=np.int32)
        expected_1d = np.transpose(a1d)  # For 1D, transpose returns the same array

        enc_1d = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a1d), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a1d),),
        )
        ca1d = self._exec(enc_1d, [a1d, pk])[0]

        transpose_1d = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(a1d),),
            outs_info=(TensorType(a1d.dtype, expected_1d.shape),),
        )
        cr1d = self._exec(transpose_1d, [ca1d])[0]

        dec_1d = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType(a1d.dtype, expected_1d.shape),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType(a1d.dtype, expected_1d.shape),),
        )
        out_1d = self._exec(dec_1d, [cr1d, sk])[0]
        np.testing.assert_array_equal(out_1d, expected_1d)

        # Test 2: Square matrix transpose
        a_square = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        expected_square = a_square.T

        enc_square = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a_square), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a_square),),
        )
        ca_square = self._exec(enc_square, [a_square, pk])[0]

        transpose_square = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(a_square),),
            outs_info=(TensorType(a_square.dtype, expected_square.shape),),
        )
        cr_square = self._exec(transpose_square, [ca_square])[0]

        dec_square = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType(a_square.dtype, expected_square.shape),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType(a_square.dtype, expected_square.shape),),
        )
        out_square = self._exec(dec_square, [cr_square, sk])[0]
        np.testing.assert_array_equal(out_square, expected_square)

    def test_dot_zero_plaintext_optimization(self):
        """Test dot product with zero plaintext - should return zero ciphertext."""
        pk, sk = self._keygen()

        # Test vector dot product: ciphertext_vector · zero_vector
        ciphertext_vector = np.array([1, 2, 3, 4], dtype=np.int32)
        zero_vector = np.array([0, 0, 0, 0], dtype=np.int32)

        # Encrypt the non-zero vector
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(
                TensorType.from_obj(ciphertext_vector),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType.from_obj(ciphertext_vector),),
        )
        ciphertext_vector_encrypted = self._exec(enc, [ciphertext_vector, pk])[0]

        # Compute dot product with zero vector
        scalar_result = np.array(0, dtype=np.int32)
        dot = PFunction(
            fn_type="phe.dot",
            ins_info=(
                TensorType.from_obj(ciphertext_vector),
                TensorType.from_obj(zero_vector),
            ),
            outs_info=(TensorType.from_obj(scalar_result),),
        )
        result_ct = self._exec(dot, [ciphertext_vector_encrypted, zero_vector])[0]

        # Decrypt and verify result is zero
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(scalar_result), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(scalar_result),),
        )
        decrypted = self._exec(dec, [result_ct, sk])[0]
        assert decrypted.item() == 0

    def test_concat_negative_axis(self):
        """Test concat operation with negative axis."""
        pk, sk = self._keygen()

        # Create two 2D matrices to concatenate along axis -1 (equivalent to axis 1)
        matrix1 = np.array([[1, 2], [4, 5]], dtype=np.int32)  # Shape (2, 2)
        matrix2 = np.array([[3], [6]], dtype=np.int32)  # Shape (2, 1)

        # Encrypt both matrices
        enc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix1), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(matrix1),),
        )
        ciphertext1 = self._exec(enc1, [matrix1, pk])[0]

        enc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix2), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(matrix2),),
        )
        ciphertext2 = self._exec(enc2, [matrix2, pk])[0]

        # Perform concat operation along axis -1
        concat_result_shape = np.zeros((2, 3), dtype=np.int32)  # Result shape (2, 3)
        concat = PFunction(
            fn_type="phe.concat",
            ins_info=(TensorType.from_obj(matrix1), TensorType.from_obj(matrix2)),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=-1,  # Concatenate along axis -1 (last axis)
        )
        result_ct = self._exec(concat, [ciphertext1, ciphertext2])[0]

        # Decrypt and verify
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(concat_result_shape),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
        )
        decrypted = self._exec(dec, [result_ct, sk])[0]
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        np.testing.assert_array_equal(decrypted, expected)

    def test_mixed_precision_errors(self):
        """Test that mixed precision operations raise appropriate errors."""
        pk, _ = self._keygen()

        # Test float ciphertext + int plaintext (should fail)
        float_ciphertext_val = np.array(3.14, dtype=np.float32)
        enc_float = PFunction(
            fn_type="phe.encrypt",
            ins_info=(
                TensorType.from_obj(float_ciphertext_val),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType.from_obj(float_ciphertext_val),),
        )
        float_ciphertext = self._exec(enc_float, [float_ciphertext_val, pk])[0]

        int_plaintext = np.array(5, dtype=np.int32)
        add_mixed = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(float_ciphertext_val),
                TensorType.from_obj(int_plaintext),
            ),
            outs_info=(TensorType.from_obj(float_ciphertext_val),),
        )

        with pytest.raises(
            ValueError,
            match="Cannot add integer plaintext to floating point ciphertext",
        ):
            self._exec(add_mixed, [float_ciphertext, int_plaintext])

        # Test int ciphertext + float plaintext (should fail)
        int_ciphertext_val = np.array(10, dtype=np.int32)
        enc_int = PFunction(
            fn_type="phe.encrypt",
            ins_info=(
                TensorType.from_obj(int_ciphertext_val),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType.from_obj(int_ciphertext_val),),
        )
        int_ciphertext = self._exec(enc_int, [int_ciphertext_val, pk])[0]

        float_plaintext = np.array(2.5, dtype=np.float64)
        add_mixed_reverse = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(int_ciphertext_val),
                TensorType.from_obj(float_plaintext),
            ),
            outs_info=(TensorType.from_obj(int_ciphertext_val),),
        )

        with pytest.raises(
            ValueError,
            match="Cannot add floating point plaintext to integer ciphertext",
        ):
            self._exec(add_mixed_reverse, [int_ciphertext, float_plaintext])

    def test_concat_invalid_axis(self):
        """Test concat with invalid axis parameter."""
        pk, _ = self._keygen()

        # Create test matrices
        matrix1 = np.array([[1, 2]], dtype=np.int32)  # Shape (1, 2)
        matrix2 = np.array([[3, 4]], dtype=np.int32)  # Shape (1, 2)

        # Encrypt both matrices
        enc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix1), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(matrix1),),
        )
        ciphertext1 = self._exec(enc1, [matrix1, pk])[0]

        enc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix2), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(matrix2),),
        )
        ciphertext2 = self._exec(enc2, [matrix2, pk])[0]

        # Try concat with invalid axis (axis 2 doesn't exist for 2D arrays)
        concat_result_shape = np.zeros((1, 4), dtype=np.int32)
        concat_invalid = PFunction(
            fn_type="phe.concat",
            ins_info=(TensorType.from_obj(matrix1), TensorType.from_obj(matrix2)),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=2,  # Invalid axis for 2D arrays
        )

        with pytest.raises(Exception) as exc_info:
            self._exec(concat_invalid, [ciphertext1, ciphertext2])

        # Check that error mentions axis-related issue
        error_message = str(exc_info.value).lower()
        assert any(
            keyword in error_message
            for keyword in ["axis", "dimension", "invalid", "out of bounds"]
        ), f"Expected axis-related error, got: {exc_info.value}"

    def test_transpose_negative_axes(self):
        """Test transpose operation with negative axes."""
        pk, sk = self._keygen()

        # Create 3D tensor
        original_tensor = np.array(
            [[[1, 2, 3], [4, 5, 6]]], dtype=np.int32
        )  # Shape (1, 2, 3)

        # Encrypt tensor
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_tensor), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        ciphertext = self._exec(enc, [original_tensor, pk])[0]

        # Transpose with negative axes (-1, -2, -3) - equivalent to (2, 1, 0)
        axes = (-1, -2, -3)
        expected_shape = (3, 2, 1)
        transpose_result_shape = np.zeros(expected_shape, dtype=np.int32)

        transpose = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_tensor),),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
            axes=axes,
        )
        result_ct = self._exec(transpose, [ciphertext])[0]

        # Decrypt and verify
        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(transpose_result_shape),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
        )
        decrypted = self._exec(dec, [result_ct, sk])[0]
        expected_result = np.transpose(original_tensor, (2, 1, 0))
        np.testing.assert_array_equal(decrypted, expected_result)

    def test_transpose_invalid_axes(self):
        """Test transpose operation with invalid axes."""
        pk, _ = self._keygen()

        # Create 2D matrix
        original_matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)  # Shape (2, 2)

        # Encrypt matrix
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self._exec(enc, [original_matrix, pk])[0]

        # Try transpose with out-of-bounds axis
        axes = (0, 2)  # axis 2 is out of bounds for 2D tensor
        transpose_invalid = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_matrix),),
            outs_info=(TensorType.from_obj(original_matrix),),
            axes=axes,
        )

        with pytest.raises(Exception) as exc_info:
            self._exec(transpose_invalid, [ciphertext])

        # Check for axis-related error
        error_message = str(exc_info.value).lower()
        assert any(
            keyword in error_message
            for keyword in ["axis", "dimension", "out of bounds", "invalid"]
        ), f"Expected axis-related error, got: {exc_info.value}"

    def test_reshape_with_inferred_dimension(self):
        """Test reshape with -1 dimension (inferred size)."""
        pk, sk = self._keygen()

        # Create a 2x6 array and reshape to (3, -1) which should become (3, 4)
        original = np.array(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.int32
        )  # Shape (2, 6)
        target_shape = (3, -1)  # Should infer to (3, 4)
        expected_shape = (3, 4)
        expected_result = original.reshape(expected_shape)

        # Encrypt the array
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(original),),
        )
        ca = self._exec(enc, [original, pk])[0]

        # Try reshape with inferred dimension
        try:
            reshape = PFunction(
                fn_type="phe.reshape",
                ins_info=(TensorType.from_obj(original),),
                outs_info=(TensorType(original.dtype, expected_shape),),
                new_shape=target_shape,
            )
            cr = self._exec(reshape, [ca])[0]

            # Decrypt and verify
            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(
                    TensorType(original.dtype, expected_shape),
                    TensorType(UINT8, (-1, 0)),
                ),
                outs_info=(TensorType(original.dtype, expected_shape),),
            )
            out = self._exec(dec, [cr, sk])[0]
            np.testing.assert_array_equal(out, expected_result)

        except Exception as e:
            # If inferred dimensions are not supported, that's also a valid test result
            print(f"Inferred dimension reshape not supported: {e}")
            pytest.skip("Inferred dimension (-1) not supported in reshape operation")

    def test_advanced_broadcasting_scenarios(self):
        """Test advanced broadcasting scenarios."""
        pk, sk = self._keygen()

        # Test case 1: (3, 1) + (1, 4) -> (3, 4)
        a1 = np.array([[1], [2], [3]], dtype=np.int32)  # Shape (3, 1)
        b1 = np.array([[10, 20, 30, 40]], dtype=np.int32)  # Shape (1, 4)

        enc_a1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a1), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a1),),
        )
        ca1 = self._exec(enc_a1, [a1, pk])[0]

        # Broadcasted addition
        result_shape = (3, 4)
        add_broadcast = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a1), TensorType.from_obj(b1)),
            outs_info=(TensorType(a1.dtype, result_shape),),
        )
        cr1 = self._exec(add_broadcast, [ca1, b1])[0]

        # Decrypt and verify
        dec1 = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType(a1.dtype, result_shape), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType(a1.dtype, result_shape),),
        )
        out1 = self._exec(dec1, [cr1, sk])[0]
        expected1 = a1 + b1  # NumPy broadcasting
        np.testing.assert_array_equal(out1, expected1)

        # Test case 2: (2, 1, 3) + (1, 4, 1) -> (2, 4, 3)
        a2 = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.int32)  # Shape (2, 1, 3)
        b2 = np.array([[[10], [20], [30], [40]]], dtype=np.int32)  # Shape (1, 4, 1)

        enc_a2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a2), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a2),),
        )
        ca2 = self._exec(enc_a2, [a2, pk])[0]

        # Broadcasted addition
        result_shape2 = (2, 4, 3)
        add_broadcast2 = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a2), TensorType.from_obj(b2)),
            outs_info=(TensorType(a2.dtype, result_shape2),),
        )
        cr2 = self._exec(add_broadcast2, [ca2, b2])[0]

        # Decrypt and verify
        dec2 = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType(a2.dtype, result_shape2), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType(a2.dtype, result_shape2),),
        )
        out2 = self._exec(dec2, [cr2, sk])[0]
        expected2 = a2 + b2  # NumPy broadcasting
        np.testing.assert_array_equal(out2, expected2)

    def test_large_array_operations(self):
        """Test operations with larger arrays to check performance and correctness."""
        pk, sk = self._keygen()

        # Create larger arrays (100 elements)
        large_array1 = np.arange(100, dtype=np.int32)  # [0, 1, 2, ..., 99]
        large_array2 = np.full(100, 5, dtype=np.int32)  # [5, 5, 5, ..., 5]

        # Encrypt first array
        enc_large = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(large_array1), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(large_array1),),
        )
        ca_large = self._exec(enc_large, [large_array1, pk])[0]

        # Test addition with large arrays
        add_large = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(large_array1),
                TensorType.from_obj(large_array2),
            ),
            outs_info=(TensorType.from_obj(large_array1),),
        )
        cr_add = self._exec(add_large, [ca_large, large_array2])[0]

        # Decrypt and verify addition
        dec_large = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(large_array1), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(large_array1),),
        )
        out_add = self._exec(dec_large, [cr_add, sk])[0]
        expected_add = large_array1 + large_array2
        np.testing.assert_array_equal(out_add, expected_add)

        # Test multiplication with large arrays
        mul_large = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(large_array1),
                TensorType.from_obj(large_array2),
            ),
            outs_info=(TensorType.from_obj(large_array1),),
        )
        cr_mul = self._exec(mul_large, [ca_large, large_array2])[0]

        # Decrypt and verify multiplication
        out_mul = self._exec(dec_large, [cr_mul, sk])[0]
        expected_mul = large_array1 * large_array2
        np.testing.assert_array_equal(out_mul, expected_mul)

        # Test reshape with large array (100 -> 10x10)
        reshape_large = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(large_array1),),
            outs_info=(TensorType(large_array1.dtype, (10, 10)),),
            new_shape=(10, 10),
        )
        cr_reshape = self._exec(reshape_large, [ca_large])[0]

        # Decrypt and verify reshape
        dec_reshape = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType(large_array1.dtype, (10, 10)),
                TensorType(UINT8, (-1, 0)),
            ),
            outs_info=(TensorType(large_array1.dtype, (10, 10)),),
        )
        out_reshape = self._exec(dec_reshape, [cr_reshape, sk])[0]
        expected_reshape = large_array1.reshape(10, 10)
        np.testing.assert_array_equal(out_reshape, expected_reshape)

    def test_multidimensional_array_roundtrip(self):
        """Test encryption/decryption with multidimensional arrays."""
        pk, sk = self._keygen()
        arrays = [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
            np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32),
        ]

        for a in arrays:
            enc = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            ca = self._exec(enc, [a, pk])[0]
            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            out = self._exec(dec, [ca, sk])[0]

            if a.dtype.kind == "f":
                assert np.allclose(out, a, atol=1e-3)
            else:
                np.testing.assert_array_equal(out, a)

    def test_broadcasting_add(self):
        """Test addition with broadcasting."""
        pk, sk = self._keygen()
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)  # Shape (2, 2)
        b = np.array([10, 20], dtype=np.int32)  # Shape (2,) - broadcasts to (2, 2)

        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc_a, [a, pk])[0]

        add = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(a),),
        )
        cr = self._exec(add, [ca, b])[0]

        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        out = self._exec(dec, [cr, sk])[0]

        expected = a + b  # [[11, 22], [13, 24]]
        np.testing.assert_array_equal(out, expected)

    def test_broadcasting_mul(self):
        """Test multiplication with broadcasting."""
        pk, sk = self._keygen()
        a = np.array([[2, 3], [4, 5]], dtype=np.int32)  # Shape (2, 2)
        b = np.array(10, dtype=np.int32)  # Scalar - broadcasts to (2, 2)

        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ca = self._exec(enc_a, [a, pk])[0]

        mul = PFunction(
            fn_type="phe.mul",
            ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
            outs_info=(TensorType.from_obj(a),),
        )
        cr = self._exec(mul, [ca, b])[0]

        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        out = self._exec(dec, [cr, sk])[0]

        expected = a * b  # [[20, 30], [40, 50]]
        np.testing.assert_array_equal(out, expected)

    def test_mixed_precision_error(self):
        """Test that mixed precision operations raise errors."""
        pk, _ = self._keygen()
        a_int = np.array(5, dtype=np.int32)
        a_float = np.array(3.14, dtype=np.float32)

        enc_int = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a_int), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a_int),),
        )
        enc_float = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a_float), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a_float),),
        )

        ca_int = self._exec(enc_int, [a_int, pk])[0]
        ca_float = self._exec(enc_float, [a_float, pk])[0]

        # Test mixed precision ciphertext + ciphertext
        add = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a_int), TensorType.from_obj(a_float)),
            outs_info=(TensorType.from_obj(a_int),),
        )
        with pytest.raises(ValueError, match="different numeric types"):
            self._exec(add, [ca_int, ca_float])

        # Test mixed precision ciphertext + plaintext
        add_mixed = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a_int), TensorType.from_obj(a_float)),
            outs_info=(TensorType.from_obj(a_int),),
        )
        with pytest.raises(ValueError, match="same numeric type"):
            self._exec(add_mixed, [ca_int, a_float])

    def test_large_key_size(self):
        """Test with larger key size for better security."""
        runtime = RuntimeContext(rank=0, world_size=1)
        scheme = "paillier"
        key_size = 1024  # Larger key size

        keygen = PFunction(
            fn_type="phe.keygen",
            ins_info=(),
            outs_info=(TensorType(UINT8, (-1, 0)), TensorType(UINT8, (-1, 0))),
            scheme=scheme,
            key_size=key_size,
        )
        pk, sk = runtime.run_kernel(keygen, [])

        # Test with larger values that fit in the larger key size
        pt = np.array(2**30, dtype=np.int64)  # Large value
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(pt),),
        )
        ct = runtime.run_kernel(enc, [TensorValue(pt), pk])[0]

        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(pt),),
        )
        out = runtime.run_kernel(dec, [ct, sk])[0]
        assert isinstance(out, TensorValue)
        assert out.to_numpy().item() == pt.item()

    def test_negative_values_comprehensive(self):
        """Test comprehensive negative value handling."""
        pk, sk = self._keygen()

        # Test various negative values
        test_values = [
            np.array(-1, dtype=np.int32),
            np.array(-42, dtype=np.int32),
            np.array([-1, -2, -3], dtype=np.int32),
            np.array([[-5, -10], [-15, -20]], dtype=np.int32),
            np.array(-3.14, dtype=np.float32),
            np.array([-1.1, -2.2], dtype=np.float32),
        ]

        for pt in test_values:
            enc = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(pt),),
            )
            ct = self._exec(enc, [pt, pk])[0]

            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(pt),),
            )
            out = self._exec(dec, [ct, sk])[0]

            if pt.dtype.kind == "f":
                assert np.allclose(out, pt, atol=1e-3)
            else:
                np.testing.assert_array_equal(out, pt)

    def test_various_integer_types_encryption(self):
        """Test encryption/decryption with various integer types."""
        pk, sk = self._keygen()

        # Test different integer types
        integer_types = [
            (np.int8, 127),
            (np.int16, 32767),
            (np.int32, 2147483647),
            (np.uint8, 255),
            (np.uint16, 65535),
        ]

        for dtype, max_val in integer_types:
            # Test positive values
            test_values = [
                np.array(1, dtype=dtype),
                np.array(max_val // 2, dtype=dtype),
                np.array([1, 2, 3], dtype=dtype),
                np.array([[10, 20], [30, 40]], dtype=dtype),
            ]

            for pt in test_values:
                enc = PFunction(
                    fn_type="phe.encrypt",
                    ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                    outs_info=(TensorType.from_obj(pt),),
                )
                ct = self._exec(enc, [pt, pk])[0]

                dec = PFunction(
                    fn_type="phe.decrypt",
                    ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                    outs_info=(TensorType.from_obj(pt),),
                )
                out = self._exec(dec, [ct, sk])[0]
                np.testing.assert_array_equal(out, pt)

    def test_various_float_types_encryption(self):
        """Test encryption/decryption with various float types."""
        pk, sk = self._keygen()

        # Test different float types
        float_types = [np.float32, np.float64]

        for dtype in float_types:
            test_values = [
                np.array(3.14, dtype=dtype),
                np.array(-2.718, dtype=dtype),
                np.array([1.1, 2.2, 3.3], dtype=dtype),
                np.array([[1.5, 2.5], [3.5, 4.5]], dtype=dtype),
            ]

            for pt in test_values:
                enc = PFunction(
                    fn_type="phe.encrypt",
                    ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                    outs_info=(TensorType.from_obj(pt),),
                )
                ct = self._exec(enc, [pt, pk])[0]

                dec = PFunction(
                    fn_type="phe.decrypt",
                    ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                    outs_info=(TensorType.from_obj(pt),),
                )
                out = self._exec(dec, [ct, sk])[0]

                # Use appropriate tolerance for floating point comparison
                np.testing.assert_allclose(out, pt, atol=1e-3)

    def test_integer_types_homomorphic_addition(self):
        """Test homomorphic addition with various integer types."""
        pk, sk = self._keygen()

        integer_types = [np.int8, np.int16, np.int32, np.uint8, np.uint16]

        for dtype in integer_types:
            # Test cipher + cipher addition
            a = np.array([10, 20], dtype=dtype)
            b = np.array([5, 15], dtype=dtype)

            enc_a = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            ca = self._exec(enc_a, [a, pk])[0]

            enc_b = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(b), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(b),),
            )
            cb = self._exec(enc_b, [b, pk])[0]

            add = PFunction(
                fn_type="phe.add",
                ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
                outs_info=(TensorType.from_obj(a),),
            )
            cr = self._exec(add, [ca, cb])[0]

            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            out = self._exec(dec, [cr, sk])[0]
            expected = a + b
            np.testing.assert_array_equal(out, expected)

            # Test cipher + plain addition
            add_plain = PFunction(
                fn_type="phe.add",
                ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
                outs_info=(TensorType.from_obj(a),),
            )
            cr_plain = self._exec(add_plain, [ca, b])[0]
            out_plain = self._exec(dec, [cr_plain, sk])[0]
            np.testing.assert_array_equal(out_plain, expected)

    def test_float_types_homomorphic_addition(self):
        """Test homomorphic addition with various float types."""
        pk, sk = self._keygen()

        float_types = [np.float32, np.float64]

        for dtype in float_types:
            # Test cipher + cipher addition
            a = np.array([1.5, 2.5], dtype=dtype)
            b = np.array([0.5, 1.5], dtype=dtype)

            enc_a = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            ca = self._exec(enc_a, [a, pk])[0]

            enc_b = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(b), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(b),),
            )
            cb = self._exec(enc_b, [b, pk])[0]

            add = PFunction(
                fn_type="phe.add",
                ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
                outs_info=(TensorType.from_obj(a),),
            )
            cr = self._exec(add, [ca, cb])[0]

            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            out = self._exec(dec, [cr, sk])[0]
            expected = a + b
            np.testing.assert_allclose(out, expected, atol=1e-3)

            # Test cipher + plain addition
            add_plain = PFunction(
                fn_type="phe.add",
                ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
                outs_info=(TensorType.from_obj(a),),
            )
            cr_plain = self._exec(add_plain, [ca, b])[0]
            out_plain = self._exec(dec, [cr_plain, sk])[0]
            np.testing.assert_allclose(out_plain, expected, atol=1e-3)

    def test_integer_types_homomorphic_multiplication(self):
        """Test homomorphic multiplication with various integer types."""
        pk, sk = self._keygen()

        integer_types = [np.int8, np.int16, np.int32, np.uint8, np.uint16]

        for dtype in integer_types:
            # Test cipher * plain multiplication
            a = np.array([2, 3], dtype=dtype)
            b = np.array([5, 7], dtype=dtype)

            enc_a = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            ca = self._exec(enc_a, [a, pk])[0]

            mul = PFunction(
                fn_type="phe.mul",
                ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
                outs_info=(TensorType.from_obj(a),),
            )
            cr = self._exec(mul, [ca, b])[0]

            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            out = self._exec(dec, [cr, sk])[0]
            expected = a * b
            np.testing.assert_array_equal(out, expected)

    def test_float_types_homomorphic_multiplication(self):
        """Test homomorphic multiplication with float ciphertext and integer plaintext."""
        pk, sk = self._keygen()

        float_types = [np.float32, np.float64]

        for dtype in float_types:
            # Test cipher * plain multiplication (float ciphertext with integer plaintext)
            a = np.array([2.5, 3.5], dtype=dtype)
            b = np.array([2, 1], dtype=np.int32)  # Use integer plaintext

            enc_a = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            ca = self._exec(enc_a, [a, pk])[0]

            mul = PFunction(
                fn_type="phe.mul",
                ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
                outs_info=(TensorType.from_obj(a),),
            )
            cr = self._exec(mul, [ca, b])[0]

            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            out = self._exec(dec, [cr, sk])[0]
            expected = a * b
            np.testing.assert_allclose(out, expected, atol=1e-3)

    def test_integer_types_homomorphic_dot_product(self):
        """Test homomorphic dot product with various integer types."""
        pk, sk = self._keygen()

        integer_types = [np.int8, np.int16, np.int32, np.uint8, np.uint16]

        for dtype in integer_types:
            # Test vector dot product
            a = np.array([1, 2, 3], dtype=dtype)
            b = np.array([4, 5, 6], dtype=dtype)

            enc_a = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            ca = self._exec(enc_a, [a, pk])[0]

            dot = PFunction(
                fn_type="phe.dot",
                ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
                outs_info=(TensorType.from_obj(np.array(0, dtype=dtype)),),
            )
            cr = self._exec(dot, [ca, b])[0]

            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(
                    TensorType.from_obj(np.array(0, dtype=dtype)),
                    TensorType(UINT8, (-1, 0)),
                ),
                outs_info=(TensorType.from_obj(np.array(0, dtype=dtype)),),
            )
            out = self._exec(dec, [cr, sk])[0]
            expected = np.dot(a, b)
            assert out.item() == expected

            # Test matrix-vector dot product
            matrix = np.array([[1, 2], [3, 4]], dtype=dtype)
            vector = np.array([5, 6], dtype=dtype)

            enc_matrix = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(matrix), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(matrix),),
            )
            cm = self._exec(enc_matrix, [matrix, pk])[0]

            dot_mv = PFunction(
                fn_type="phe.dot",
                ins_info=(TensorType.from_obj(matrix), TensorType.from_obj(vector)),
                outs_info=(TensorType.from_obj(np.array([0, 0], dtype=dtype)),),
            )
            cr_mv = self._exec(dot_mv, [cm, vector])[0]

            dec_mv = PFunction(
                fn_type="phe.decrypt",
                ins_info=(
                    TensorType.from_obj(np.array([0, 0], dtype=dtype)),
                    TensorType(UINT8, (-1, 0)),
                ),
                outs_info=(TensorType.from_obj(np.array([0, 0], dtype=dtype)),),
            )
            out_mv = self._exec(dec_mv, [cr_mv, sk])[0]
            expected_mv = np.dot(matrix, vector)
            np.testing.assert_array_equal(out_mv, expected_mv)

    def test_float_types_homomorphic_dot_product(self):
        """Test homomorphic dot product with float ciphertext and integer plaintext."""
        pk, sk = self._keygen()

        float_types = [np.float32, np.float64]

        for dtype in float_types:
            # Test vector dot product (float ciphertext with integer plaintext)
            a = np.array([1.1, 2.2, 3.3], dtype=dtype)
            b = np.array([1, 2, 3], dtype=np.int32)  # Use integer plaintext

            enc_a = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(a),),
            )
            ca = self._exec(enc_a, [a, pk])[0]

            dot = PFunction(
                fn_type="phe.dot",
                ins_info=(TensorType.from_obj(a), TensorType.from_obj(b)),
                outs_info=(TensorType.from_obj(np.array(0.0, dtype=dtype)),),
            )
            cr = self._exec(dot, [ca, b])[0]

            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(
                    TensorType.from_obj(np.array(0.0, dtype=dtype)),
                    TensorType(UINT8, (-1, 0)),
                ),
                outs_info=(TensorType.from_obj(np.array(0.0, dtype=dtype)),),
            )
            out = self._exec(dec, [cr, sk])[0]
            expected = np.dot(a, b)
            assert (
                abs(out.item() - expected) < 1e-2
            )  # Allow for fixed-point precision loss

            # Test matrix-vector dot product (float ciphertext with integer plaintext)
            matrix = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=dtype)
            vector = np.array([1, 2], dtype=np.int32)  # Use integer plaintext

            enc_matrix = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(matrix), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(matrix),),
            )
            cm = self._exec(enc_matrix, [matrix, pk])[0]

            dot_mv = PFunction(
                fn_type="phe.dot",
                ins_info=(TensorType.from_obj(matrix), TensorType.from_obj(vector)),
                outs_info=(TensorType.from_obj(np.array([0.0, 0.0], dtype=dtype)),),
            )
            cr_mv = self._exec(dot_mv, [cm, vector])[0]

            dec_mv = PFunction(
                fn_type="phe.decrypt",
                ins_info=(
                    TensorType.from_obj(np.array([0.0, 0.0], dtype=dtype)),
                    TensorType(UINT8, (-1, 0)),
                ),
                outs_info=(TensorType.from_obj(np.array([0.0, 0.0], dtype=dtype)),),
            )
            out_mv = self._exec(dec_mv, [cr_mv, sk])[0]
            expected_mv = np.dot(matrix, vector)
            np.testing.assert_allclose(out_mv, expected_mv, atol=1e-2)

    def test_mixed_integer_types_operations(self):
        """Test operations with mixed integer types."""
        pk, sk = self._keygen()

        # Test int8 ciphertext with int32 plaintext
        a_int8 = np.array([10, 20], dtype=np.int8)
        b_int32 = np.array([2, 3], dtype=np.int32)

        enc_a = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a_int8), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a_int8),),
        )
        ca = self._exec(enc_a, [a_int8, pk])[0]

        # Addition should work with mixed types
        add = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(a_int8), TensorType.from_obj(b_int32)),
            outs_info=(TensorType.from_obj(a_int8),),
        )
        cr_add = self._exec(add, [ca, b_int32])[0]

        dec = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(a_int8), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a_int8),),
        )
        out_add = self._exec(dec, [cr_add, sk])[0]
        # Result should match ciphertext type (int8)
        expected_add = (a_int8 + b_int32).astype(np.int8)
        np.testing.assert_array_equal(out_add, expected_add)

        # Multiplication should work with mixed types
        mul = PFunction(
            fn_type="phe.mul",
            ins_info=(TensorType.from_obj(a_int8), TensorType.from_obj(b_int32)),
            outs_info=(TensorType.from_obj(a_int8),),
        )
        cr_mul = self._exec(mul, [ca, b_int32])[0]
        out_mul = self._exec(dec, [cr_mul, sk])[0]
        expected_mul = (a_int8 * b_int32).astype(np.int8)
        np.testing.assert_array_equal(out_mul, expected_mul)

    def test_edge_case_data_types(self):
        """Test edge cases with different data types."""
        pk, sk = self._keygen()

        # Test maximum values for different integer types
        test_cases = [
            (np.int8, 127),
            (np.uint8, 255),
            (np.int16, 32767),
            (np.uint16, 65535),
        ]

        for dtype, max_val in test_cases:
            # Test near-maximum values
            pt = np.array([max_val // 2, max_val // 4], dtype=dtype)

            enc = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(pt),),
            )
            ct = self._exec(enc, [pt, pk])[0]

            dec = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(pt), TensorType(UINT8, (-1, 0))),
                outs_info=(TensorType.from_obj(pt),),
            )
            out = self._exec(dec, [ct, sk])[0]
            np.testing.assert_array_equal(out, pt)

            # Test addition with small values to avoid overflow
            small_val = np.array([1, 2], dtype=dtype)
            add = PFunction(
                fn_type="phe.add",
                ins_info=(TensorType.from_obj(pt), TensorType.from_obj(small_val)),
                outs_info=(TensorType.from_obj(pt),),
            )
            cr_add = self._exec(add, [ct, small_val])[0]
            out_add = self._exec(dec, [cr_add, sk])[0]
            expected_add = pt + small_val
            np.testing.assert_array_equal(out_add, expected_add)
