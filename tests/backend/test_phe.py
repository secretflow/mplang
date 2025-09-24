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

from mplang.backend.base import create_runtime, list_registered_kernels
from mplang.backend.phe import CipherText, PrivateKey, PublicKey
from mplang.core.dtype import INT32, UINT8
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType


class TestPHEKernels:
    """Compact PHE kernel tests (clean rewrite)."""

    def setup_method(self):
        self.runtime = create_runtime(0, 1)
        self.scheme = "paillier"
        self.key_size = 512

    def _exec(self, p: PFunction, args: list):
        return self.runtime.run_kernel(p, args)

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
        for name in ["phe.keygen", "phe.encrypt", "phe.decrypt", "phe.add", "phe.mul"]:
            assert name in list_registered_kernels()

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
        assert abs(out.item() - 3.14) < 1e-6

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
        with pytest.raises(ValueError, match="same shape"):
            self._exec(add, [ca, cb])

    def test_shape_mismatch_mul(self):
        pk, _ = self._keygen()
        a = np.array(5, dtype=np.int32)
        b = np.array([1, 2], dtype=np.int32)
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
        with pytest.raises(ValueError, match="shape mismatch"):
            self._exec(mul, [ca, b])

    def test_scheme_mismatch(self):
        pk, _ = self._keygen()
        a = np.array(10, dtype=np.int32)
        enc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(a), TensorType(UINT8, (-1, 0))),
            outs_info=(TensorType.from_obj(a),),
        )
        ct = self._exec(enc, [a, pk])[0]
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
        with pytest.raises(ValueError, match="same scheme/key size"):
            self._exec(add, [ct, fake])

    def test_various_roundtrip(self):
        pk, sk = self._keygen()
        samples = [
            np.array(7, dtype=np.int32),
            np.array(-3, dtype=np.int32),
            np.array(1234567890123, dtype=np.int64),
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
                assert np.allclose(out, pt, atol=1e-5)
            else:
                assert np.array_equal(out, pt)
