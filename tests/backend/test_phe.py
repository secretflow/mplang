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

from mplang.backend.phe import CipherText, PHEHandler, PrivateKey, PublicKey
from mplang.core.dtype import (
    BOOL,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
)
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType


class TestPHEHandler:
    """Test cases for PHEHandler with Paillier scheme."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = PHEHandler()
        self.scheme = "paillier"
        # Use smaller key size for faster testing, production should use 2048+ bits
        self.key_size = 1024

    def test_list_fn_names(self):
        """Test that handler lists correct function names."""
        fn_names = self.handler.list_fn_names()
        expected_functions = [
            "phe.keygen",
            "phe.encrypt",
            "phe.add",
            "phe.mul",
            "phe.decrypt",
        ]

        for expected_fn in expected_functions:
            assert expected_fn in fn_names
        assert len(fn_names) == len(expected_functions)

    def test_keygen_paillier(self):
        """Test key generation for Paillier scheme."""
        pfunc = PFunction(
            fn_type="phe.keygen",
            ins_info=(),
            outs_info=(
                TensorType(BOOL, ()),
                TensorType(BOOL, ()),
            ),  # Dummy types for keys
            scheme=self.scheme,
            key_size=self.key_size,
        )

        result = self.handler.execute(pfunc, [])
        assert len(result) == 2

        pk, sk = result
        assert isinstance(pk, PublicKey)
        assert isinstance(sk, PrivateKey)
        assert pk.scheme == "Paillier"  # lightPHE capitalizes the scheme name
        assert sk.scheme == "Paillier"
        assert pk.key_size == self.key_size
        assert sk.key_size == self.key_size

    def test_keygen_invalid_scheme(self):
        """Test key generation with invalid scheme."""
        pfunc = PFunction(
            fn_type="phe.keygen",
            ins_info=(),
            outs_info=(TensorType(BOOL, ()), TensorType(BOOL, ())),
            scheme="invalid_scheme",
            key_size=self.key_size,
        )

        with pytest.raises(ValueError, match="Unsupported PHE scheme"):
            self.handler.execute(pfunc, [])

    def test_keygen_with_args(self):
        """Test key generation with invalid arguments."""
        test_data = np.array(1)
        pfunc = PFunction(
            fn_type="phe.keygen",
            ins_info=(TensorType.from_obj(test_data),),
            outs_info=(TensorType(BOOL, ()), TensorType(BOOL, ())),
            scheme=self.scheme,
            key_size=self.key_size,
        )

        with pytest.raises(ValueError, match="Key generation expects no arguments"):
            self.handler.execute(pfunc, [test_data])

    def _generate_keypair(self):
        """Helper method to generate a keypair."""
        pfunc = PFunction(
            fn_type="phe.keygen",
            ins_info=(),
            outs_info=(TensorType(BOOL, ()), TensorType(BOOL, ())),
            scheme=self.scheme,
            key_size=self.key_size,
        )
        pk, sk = self.handler.execute(pfunc, [])
        return pk, sk

    def test_encrypt_decrypt_scalar_int32(self):
        """Test encrypt/decrypt cycle for scalar int32."""
        pk, sk = self._generate_keypair()

        # Test with int32 scalar
        plaintext = np.array(42, dtype=np.int32)

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext_result = self.handler.execute(encrypt_pfunc, [plaintext, pk])
        assert len(ciphertext_result) == 1

        ciphertext = ciphertext_result[0]
        assert isinstance(ciphertext, CipherText)
        assert ciphertext.semantic_dtype == INT32
        assert ciphertext.semantic_shape == ()

        # Decrypt
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        decrypted_result = self.handler.execute(decrypt_pfunc, [ciphertext, sk])
        assert len(decrypted_result) == 1

        decrypted = decrypted_result[0]
        assert isinstance(decrypted, np.ndarray)
        assert decrypted.dtype == np.int32
        assert decrypted.shape == ()
        assert decrypted.item() == 42

    def test_encrypt_decrypt_scalar_float32(self):
        """Test encrypt/decrypt cycle for scalar float32."""
        pk, sk = self._generate_keypair()

        # Test with float32 scalar
        plaintext = np.array(3.14, dtype=np.float32)

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext_result = self.handler.execute(encrypt_pfunc, [plaintext, pk])
        ciphertext = ciphertext_result[0]
        assert ciphertext.semantic_dtype == FLOAT32

        # Decrypt
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        decrypted_result = self.handler.execute(decrypt_pfunc, [ciphertext, sk])
        decrypted = decrypted_result[0]
        assert decrypted.dtype == np.float32
        assert abs(decrypted.item() - 3.14) < 1e-6

    def test_encrypt_decrypt_array_int64(self):
        """Test encrypt/decrypt cycle for array int64."""
        pk, sk = self._generate_keypair()

        # Test with int64 array
        plaintext = np.array([1, 2, 3, 4], dtype=np.int64)

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext_result = self.handler.execute(encrypt_pfunc, [plaintext, pk])
        ciphertext = ciphertext_result[0]
        assert ciphertext.semantic_dtype == INT64
        assert ciphertext.semantic_shape == (4,)

        # Decrypt
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        decrypted_result = self.handler.execute(decrypt_pfunc, [ciphertext, sk])
        decrypted = decrypted_result[0]
        assert decrypted.dtype == np.int64
        assert decrypted.shape == (4,)
        np.testing.assert_array_equal(decrypted, [1, 2, 3, 4])

    def test_encrypt_invalid_args(self):
        """Test encryption with invalid arguments."""
        _pk, _ = self._generate_keypair()

        # Test with wrong number of arguments
        plaintext = np.array(42)
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext),),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        with pytest.raises(
            ValueError, match="Encryption expects exactly two arguments"
        ):
            self.handler.execute(encrypt_pfunc, [plaintext])

    def test_decrypt_invalid_args(self):
        """Test decryption with invalid arguments."""
        _pk, _sk = self._generate_keypair()

        # Test with wrong number of arguments
        plaintext = np.array(42)
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        with pytest.raises(
            ValueError, match="Decryption expects exactly two arguments"
        ):
            self.handler.execute(decrypt_pfunc, [])

    def test_add_ciphertext_ciphertext_int32(self):
        """Test CipherText + CipherText addition with int32."""
        pk, sk = self._generate_keypair()

        # Encrypt two int32 scalars
        plaintext1 = np.array(10, dtype=np.int32)
        plaintext2 = np.array(20, dtype=np.int32)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc, [plaintext1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc, [plaintext2, pk])[0]

        # Add ciphertexts
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(plaintext1), TensorType.from_obj(plaintext1)),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == ()

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        assert decrypted.item() == 30

    def test_add_ciphertext_ciphertext_float64(self):
        """Test CipherText + CipherText addition with float64."""
        pk, sk = self._generate_keypair()

        # Encrypt two float64 scalars
        plaintext1 = np.array(1.5, dtype=np.float64)
        plaintext2 = np.array(2.5, dtype=np.float64)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc, [plaintext1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc, [plaintext2, pk])[0]

        # Add ciphertexts
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(plaintext1), TensorType.from_obj(plaintext1)),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext1, ciphertext2])[0]

        assert result_ct.semantic_dtype == FLOAT64

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        assert abs(decrypted.item() - 4.0) < 1e-10

    def test_add_ciphertext_ciphertext_array_int16(self):
        """Test CipherText + CipherText addition with int16 array."""
        pk, sk = self._generate_keypair()

        # Encrypt two int16 arrays
        plaintext1 = np.array([1, 2, 3], dtype=np.int16)
        plaintext2 = np.array([4, 5, 6], dtype=np.int16)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc, [plaintext1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc, [plaintext2, pk])[0]

        # Add ciphertexts
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(plaintext1), TensorType.from_obj(plaintext1)),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext1, ciphertext2])[0]

        assert result_ct.semantic_dtype == INT16
        assert result_ct.semantic_shape == (3,)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([5, 7, 9], dtype=np.int16)
        np.testing.assert_array_equal(decrypted, expected)

    def test_add_ciphertext_plaintext_int32(self):
        """Test CipherText + plaintext addition with int32."""
        pk, sk = self._generate_keypair()

        # Encrypt int32 scalar
        ciphertext_val = np.array(15, dtype=np.int32)
        plaintext_val = np.array(25, dtype=np.int32)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Add ciphertext + plaintext
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext, plaintext_val])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        assert decrypted.item() == 40

    def test_add_plaintext_ciphertext_float32(self):
        """Test plaintext + CipherText addition with float32."""
        pk, sk = self._generate_keypair()

        # Encrypt float32 scalar
        ciphertext_val = np.array(2.5, dtype=np.float32)
        plaintext_val = np.array(1.5, dtype=np.float32)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Add plaintext + ciphertext (commutative)
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(plaintext_val),
                TensorType.from_obj(ciphertext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(add_pfunc, [plaintext_val, ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == FLOAT32

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        assert abs(decrypted.item() - 4.0) < 1e-6

    def test_add_ciphertext_plaintext_array_int8(self):
        """Test CipherText + plaintext addition with int8 array."""
        pk, sk = self._generate_keypair()

        # Encrypt int8 array
        ciphertext_val = np.array([10, 20, 30], dtype=np.int8)
        plaintext_val = np.array([1, 2, 3], dtype=np.int8)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Add ciphertext + plaintext
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext, plaintext_val])[0]

        assert result_ct.semantic_dtype == INT8
        assert result_ct.semantic_shape == (3,)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([11, 22, 33], dtype=np.int8)
        np.testing.assert_array_equal(decrypted, expected)

    def test_add_plaintext_plaintext(self):
        """Test plaintext + plaintext addition (fallback to regular addition)."""
        # Regular numpy addition should work
        plaintext1 = np.array([1, 2, 3], dtype=np.int32)
        plaintext2 = np.array([4, 5, 6], dtype=np.int32)

        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(plaintext1), TensorType.from_obj(plaintext2)),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        result = self.handler.execute(add_pfunc, [plaintext1, plaintext2])[0]

        assert isinstance(result, np.ndarray)
        expected = np.array([5, 7, 9], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_add_shape_mismatch(self):
        """Test addition with shape mismatch."""
        pk, _ = self._generate_keypair()

        # Encrypt arrays with different shapes
        plaintext1 = np.array([1, 2], dtype=np.int32)
        plaintext2 = np.array([1, 2, 3], dtype=np.int32)

        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext2),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [plaintext1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [plaintext2, pk])[0]

        # Try to add - should fail
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(plaintext1), TensorType.from_obj(plaintext2)),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        with pytest.raises(
            ValueError, match="CipherText operands must have same shape"
        ):
            self.handler.execute(add_pfunc, [ciphertext1, ciphertext2])

    def test_add_scheme_mismatch(self):
        """Test addition with different schemes (simulate by manual creation)."""
        pk, _ = self._generate_keypair()

        # Encrypt one value normally
        plaintext = np.array(10, dtype=np.int32)
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc, [plaintext, pk])[0]

        # Create a fake ciphertext with different scheme
        fake_ciphertext = CipherText(
            ct_data=ciphertext1.ct_data,
            semantic_dtype=INT32,
            semantic_shape=(),
            scheme="ElGamal",  # Different scheme
            key_size=2048,
            pk_data=None,
        )

        # Try to add - should fail
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(plaintext), TensorType.from_obj(plaintext)),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        with pytest.raises(ValueError, match="must use same scheme and key size"):
            self.handler.execute(add_pfunc, [ciphertext1, fake_ciphertext])

    def test_add_ciphertext_plaintext_shape_mismatch(self):
        """Test CipherText + plaintext addition with shape mismatch."""
        pk, _ = self._generate_keypair()

        # Encrypt scalar
        ciphertext_val = np.array(10, dtype=np.int32)
        plaintext_val = np.array([1, 2], dtype=np.int32)  # Different shape

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Try to add - should fail
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        with pytest.raises(ValueError, match="operands must have same shape"):
            self.handler.execute(add_pfunc, [ciphertext, plaintext_val])

    def test_add_invalid_args(self):
        """Test addition with invalid number of arguments."""
        plaintext = np.array(1)
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(plaintext),),
            outs_info=(TensorType.from_obj(plaintext),),
        )

        with pytest.raises(ValueError, match="Addition expects exactly two arguments"):
            self.handler.execute(add_pfunc, [plaintext])

    def test_unsupported_function_type(self):
        """Test handler with unsupported function type."""
        plaintext = np.array(1)
        pfunc = PFunction(
            fn_type="phe.unsupported",
            ins_info=(),
            outs_info=(TensorType.from_obj(plaintext),),
        )

        with pytest.raises(ValueError, match="Unsupported PHE function type"):
            self.handler.execute(pfunc, [])

    def test_mul_ciphertext_plaintext_int32(self):
        """Test CipherText * plaintext multiplication with int32."""
        pk, sk = self._generate_keypair()

        # Encrypt int32 scalar
        ciphertext_val = np.array(5, dtype=np.int32)
        plaintext_multiplier = np.array(3, dtype=np.int32)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Multiply ciphertext * plaintext
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_multiplier),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plaintext_multiplier])[
            0
        ]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        assert decrypted.item() == 15  # 5 * 3

    def test_mul_ciphertext_plaintext_float64(self):
        """Test CipherText * plaintext multiplication with float64."""
        pk, sk = self._generate_keypair()

        # Encrypt float64 scalar
        ciphertext_val = np.array(2.5, dtype=np.float64)
        plaintext_multiplier = np.array(4.0, dtype=np.float64)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Multiply ciphertext * plaintext
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_multiplier),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plaintext_multiplier])[
            0
        ]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == FLOAT64

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        assert abs(decrypted.item() - 10.0) < 1e-10  # 2.5 * 4.0

    def test_mul_ciphertext_plaintext_array_int16(self):
        """Test CipherText * plaintext multiplication with int16 array."""
        pk, sk = self._generate_keypair()

        # Encrypt int16 array
        ciphertext_val = np.array([2, 4, 6], dtype=np.int16)
        plaintext_multiplier = np.array([3, 2, 1], dtype=np.int16)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Multiply ciphertext * plaintext
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_multiplier),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plaintext_multiplier])[
            0
        ]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT16
        assert result_ct.semantic_shape == (3,)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([6, 8, 6], dtype=np.int16)  # [2*3, 4*2, 6*1]
        np.testing.assert_array_equal(decrypted, expected)

    def test_mul_invalid_args(self):
        """Test multiplication with invalid arguments."""
        pk, _ = self._generate_keypair()

        # Encrypt a value
        plaintext = np.array(5, dtype=np.int32)
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [plaintext, pk])[0]

        # Test with wrong number of arguments
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(TensorType.from_obj(plaintext),),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        with pytest.raises(
            ValueError, match="Multiplication expects exactly two arguments"
        ):
            self.handler.execute(mul_pfunc, [ciphertext])

        # Test with plaintext as first argument
        plaintext_val = np.array(3, dtype=np.int32)
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(plaintext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(plaintext_val),),
        )
        with pytest.raises(
            ValueError, match="First argument must be a CipherText instance"
        ):
            self.handler.execute(mul_pfunc, [plaintext_val, plaintext_val])

    def test_mul_shape_mismatch(self):
        """Test multiplication with shape mismatch."""
        pk, _ = self._generate_keypair()

        # Encrypt scalar
        ciphertext_val = np.array(5, dtype=np.int32)
        plaintext_val = np.array([1, 2], dtype=np.int32)  # Different shape

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Try to multiply - should fail
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        with pytest.raises(ValueError, match="operands must have same shape"):
            self.handler.execute(mul_pfunc, [ciphertext, plaintext_val])

    def test_mul_float_not_supported(self):
        """Test that multiplication with floats raises an error in frontend."""
        import jax.numpy as jnp

        from mplang.frontend import phe

        # Test with float ciphertext
        float_ct = jnp.array(5.5, dtype=jnp.float32)
        int_pt = jnp.array(3, dtype=jnp.int32)

        with pytest.raises(
            ValueError,
            match="PHE multiplication does not support floating-point numbers",
        ):
            phe.mul(float_ct, int_pt)

        # Test with float plaintext
        int_ct = jnp.array(5, dtype=jnp.int32)
        float_pt = jnp.array(3.2, dtype=jnp.float32)

        with pytest.raises(
            ValueError,
            match="PHE multiplication does not support floating-point numbers",
        ):
            phe.mul(int_ct, float_pt)

        # Test with both floats
        with pytest.raises(
            ValueError,
            match="PHE multiplication does not support floating-point numbers",
        ):
            phe.mul(float_ct, float_pt)

    def test_various_numeric_types(self):
        """Test encryption/decryption with various numeric types."""
        pk, sk = self._generate_keypair()

        test_cases = [
            (np.array(100, dtype=np.uint8), UINT8),
            (np.array(-50, dtype=np.int8), INT8),
            (np.array(1000, dtype=np.uint16), UINT16),
            (np.array(-1000, dtype=np.int16), INT16),
            (np.array(100000, dtype=np.uint32), UINT32),
            (np.array(-100000, dtype=np.int32), INT32),
            (np.array(1000000000, dtype=np.uint64), UINT64),
            (np.array(-1000000000, dtype=np.int64), INT64),
            (np.array(3.14159, dtype=np.float32), FLOAT32),
            (np.array(2.718281828, dtype=np.float64), FLOAT64),
        ]

        for plaintext, expected_dtype in test_cases:
            # Encrypt
            encrypt_pfunc = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
                outs_info=(TensorType.from_obj(plaintext),),
            )
            ciphertext = self.handler.execute(encrypt_pfunc, [plaintext, pk])[0]
            assert ciphertext.semantic_dtype == expected_dtype

            # Decrypt
            decrypt_pfunc = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
                outs_info=(TensorType.from_obj(plaintext),),
            )
            decrypted = self.handler.execute(decrypt_pfunc, [ciphertext, sk])[0]
            assert decrypted.dtype == plaintext.dtype

            # Check value (with tolerance for floating point)
            if np.issubdtype(plaintext.dtype, np.floating):
                assert abs(decrypted.item() - plaintext.item()) < 1e-6
            else:
                assert decrypted.item() == plaintext.item()
