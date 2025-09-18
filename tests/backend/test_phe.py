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
            # our extensions
            "phe.dot",
            "phe.concat",
            "phe.gather",
            "phe.scatter",
            "phe.reshape",
            "phe.transpose",
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
        """Test addition with incompatible shapes for broadcasting."""
        pk, _ = self._generate_keypair()

        # Create tensors with truly incompatible shapes for broadcasting
        # Shape (3, 1) and (2, 1) cannot be broadcast together
        plaintext1 = np.array([[1], [2], [3]], dtype=np.int32)  # Shape (3, 1)
        plaintext2 = np.array([[4], [5]], dtype=np.int32)  # Shape (2, 1)

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

        # Try to add - should fail due to incompatible broadcasting
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(plaintext1), TensorType.from_obj(plaintext2)),
            outs_info=(TensorType.from_obj(plaintext1),),
        )
        with pytest.raises(
            ValueError, match="CipherText operands cannot be broadcast together"
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
        """Test CipherText + plaintext addition with incompatible shapes for broadcasting."""
        pk, _ = self._generate_keypair()

        # Create tensors with truly incompatible shapes for broadcasting
        # Shape (3, 1) and (2, 1) cannot be broadcast together
        ciphertext_val = np.array([[1], [2], [3]], dtype=np.int32)  # Shape (3, 1)
        plaintext_val = np.array([[4], [5]], dtype=np.int32)  # Shape (2, 1)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Try to add - should fail due to incompatible broadcasting
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        with pytest.raises(ValueError, match="Operands cannot be broadcast together"):
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
        """Test multiplication with incompatible shape for broadcasting."""
        pk, _ = self._generate_keypair()

        # Create tensors with truly incompatible shapes for broadcasting
        # Shape (3, 1) and (2, 1) cannot be broadcast together
        ciphertext_val = np.array([[1], [2], [3]], dtype=np.int32)  # Shape (3, 1)
        plaintext_val = np.array([[4], [5]], dtype=np.int32)  # Shape (2, 1)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Try to multiply - should fail due to incompatible broadcasting
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        with pytest.raises(ValueError, match="Operands cannot be broadcast together"):
            self.handler.execute(mul_pfunc, [ciphertext, plaintext_val])

    def test_mul_float_x_float_not_supported(self):
        """Test that float x float multiplication raises an error in frontend."""
        import jax.numpy as jnp

        from mplang.frontend import phe

        # Test that float x float is blocked
        float_ct = jnp.array(5.5, dtype=jnp.float32)
        float_pt = jnp.array(3.2, dtype=jnp.float32)

        with pytest.raises(
            ValueError,
            match="PHE multiplication does not support float x float operations",
        ):
            phe.mul(float_ct, float_pt)

        # Test that float x int is allowed (should not raise validation error)
        int_pt = jnp.array(3, dtype=jnp.int32)
        try:
            phe.mul(float_ct, int_pt)  # Should not raise float x float error
        except ValueError as e:
            if "float x float operations" in str(e):
                pytest.fail("float x int should be allowed")

    def test_mul_multidimensional_same_shape(self):
        """Test multiplication with multi-dimensional arrays of same shape."""
        pk, sk = self._generate_keypair()

        # Test with 2D arrays of same shape
        ciphertext_val = np.array([[1, 2], [3, 4]], dtype=np.int32)
        plaintext_val = np.array([[5, 6], [7, 8]], dtype=np.int32)

        # Encrypt 2D array
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Multiply with 2D plaintext
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plaintext_val])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array(
            [[5, 12], [21, 32]], dtype=np.int32
        )  # [[1*5, 2*6], [3*7, 4*8]]
        np.testing.assert_array_equal(decrypted, expected)

    def test_mul_multidimensional_broadcast_scalar(self):
        """Test multiplication with broadcasting: multi-dimensional ciphertext * scalar plaintext."""
        pk, sk = self._generate_keypair()

        # Test with 2D ciphertext and scalar plaintext
        ciphertext_val = np.array([[2, 4], [6, 8]], dtype=np.int32)
        plaintext_scalar = np.array(3, dtype=np.int32)

        # Encrypt 2D array
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Multiply with scalar plaintext
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_scalar),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plaintext_scalar])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2)  # Result should have ciphertext shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array(
            [[6, 12], [18, 24]], dtype=np.int32
        )  # [[2*3, 4*3], [6*3, 8*3]]
        np.testing.assert_array_equal(decrypted, expected)

    def test_mul_multidimensional_broadcast_vector(self):
        """Test multiplication with broadcasting: 2D ciphertext * 1D plaintext."""
        pk, sk = self._generate_keypair()

        # Test with 2D ciphertext and 1D plaintext that broadcasts
        ciphertext_val = np.array([[1, 2], [3, 4]], dtype=np.int32)
        plaintext_vec = np.array([10, 20], dtype=np.int32)  # Should broadcast to (2, 2)

        # Encrypt 2D array
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Multiply with 1D plaintext (should broadcast)
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_vec),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plaintext_vec])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (
            2,
            2,
        )  # Result should have broadcasted shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        # Broadcasting: [[1, 2], [3, 4]] * [[10, 20], [10, 20]] = [[10, 40], [30, 80]]
        expected = np.array([[10, 40], [30, 80]], dtype=np.int32)
        np.testing.assert_array_equal(decrypted, expected)

    def test_mul_multidimensional_broadcast_ciphertext(self):
        """Test multiplication with broadcasting where ciphertext needs to be broadcasted."""
        pk, sk = self._generate_keypair()

        # Test with 1D ciphertext and 2D plaintext
        ciphertext_val = np.array([2, 3], dtype=np.int32)  # Shape (2,)
        plaintext_val = np.array([[1, 4], [5, 6]], dtype=np.int32)  # Shape (2, 2)

        # Encrypt 1D array
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Multiply with 2D plaintext (ciphertext should broadcast)
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(
                TensorType.from_obj(plaintext_val),
            ),  # Result shape follows plaintext
        )
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plaintext_val])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (
            2,
            2,
        )  # Result should have broadcasted shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        # Broadcasting: [[2, 3], [2, 3]] * [[1, 4], [5, 6]] = [[2, 12], [10, 18]]
        expected = np.array([[2, 12], [10, 18]], dtype=np.int32)
        np.testing.assert_array_equal(decrypted, expected)

    def test_mul_multidimensional_broadcast_incompatible(self):
        """Test multiplication with incompatible shapes for broadcasting."""
        pk, _ = self._generate_keypair()

        # Test with incompatible shapes
        ciphertext_val = np.array([[1, 2, 3]], dtype=np.int32)  # Shape (1, 3)
        plaintext_val = np.array([[1], [2]], dtype=np.int32)  # Shape (2, 1)

        # Encrypt array
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Try to multiply with incompatible shape
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )

        # This should work with broadcasting, result shape should be (2, 3)
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plaintext_val])[0]
        assert result_ct.semantic_shape == (2, 3)

    def test_mul_multidimensional_3d_tensors(self):
        """Test multiplication with 3D tensors."""
        pk, sk = self._generate_keypair()

        # Test with 3D tensors
        ciphertext_val = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32
        )  # Shape (2, 2, 2)
        plaintext_val = np.array(
            [[[2, 1], [1, 2]], [[1, 2], [2, 1]]], dtype=np.int32
        )  # Shape (2, 2, 2)

        # Encrypt 3D array
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Multiply with 3D plaintext
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_val),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plaintext_val])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = ciphertext_val * plaintext_val  # Element-wise multiplication
        np.testing.assert_array_equal(decrypted, expected)

    def test_add_multidimensional_same_shape(self):
        """Test CipherText + CipherText addition with multi-dimensional arrays of same shape."""
        pk, sk = self._generate_keypair()

        # Test with 2D arrays of same shape
        ciphertext_val1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        ciphertext_val2 = np.array([[5, 6], [7, 8]], dtype=np.int32)

        # Encrypt both arrays
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc, [ciphertext_val1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc, [ciphertext_val2, pk])[0]

        # Add two 2D ciphertexts
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val1),
                TensorType.from_obj(ciphertext_val2),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array(
            [[6, 8], [10, 12]], dtype=np.int32
        )  # [[1+5, 2+6], [3+7, 4+8]]
        np.testing.assert_array_equal(decrypted, expected)

    def test_add_multidimensional_broadcast_scalar_ciphertext(self):
        """Test CipherText + CipherText addition with broadcasting: 2D + scalar."""
        pk, sk = self._generate_keypair()

        # Test with 2D ciphertext and scalar ciphertext
        ciphertext_val1 = np.array([[2, 4], [6, 8]], dtype=np.int32)
        ciphertext_val2 = np.array(10, dtype=np.int32)  # Scalar

        # Encrypt both arrays
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val2),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [ciphertext_val1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [ciphertext_val2, pk])[0]

        # Add 2D + scalar ciphertexts
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val1),
                TensorType.from_obj(ciphertext_val2),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2)  # Result should have 2D shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array(
            [[12, 14], [16, 18]], dtype=np.int32
        )  # [[2+10, 4+10], [6+10, 8+10]]
        np.testing.assert_array_equal(decrypted, expected)

    def test_add_multidimensional_broadcast_vector_ciphertext(self):
        """Test CipherText + CipherText addition with broadcasting: 2D + 1D."""
        pk, sk = self._generate_keypair()

        # Test with 2D ciphertext and 1D ciphertext that broadcasts
        ciphertext_val1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        ciphertext_val2 = np.array(
            [10, 20], dtype=np.int32
        )  # Should broadcast to (2, 2)

        # Encrypt both arrays
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val2),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [ciphertext_val1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [ciphertext_val2, pk])[0]

        # Add 2D + 1D ciphertexts (should broadcast)
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val1),
                TensorType.from_obj(ciphertext_val2),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (
            2,
            2,
        )  # Result should have broadcasted shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        # Broadcasting: [[1, 2], [3, 4]] + [[10, 20], [10, 20]] = [[11, 22], [13, 24]]
        expected = np.array([[11, 22], [13, 24]], dtype=np.int32)
        np.testing.assert_array_equal(decrypted, expected)

    def test_add_multidimensional_broadcast_plaintext_scalar(self):
        """Test CipherText + plaintext addition with broadcasting: 2D ciphertext + scalar plaintext."""
        pk, sk = self._generate_keypair()

        # Test with 2D ciphertext and scalar plaintext
        ciphertext_val = np.array([[2, 4], [6, 8]], dtype=np.int32)
        plaintext_scalar = np.array(5, dtype=np.int32)

        # Encrypt 2D array
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Add with scalar plaintext
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_scalar),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext, plaintext_scalar])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2)  # Result should have ciphertext shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array(
            [[7, 9], [11, 13]], dtype=np.int32
        )  # [[2+5, 4+5], [6+5, 8+5]]
        np.testing.assert_array_equal(decrypted, expected)

    def test_add_multidimensional_broadcast_plaintext_vector(self):
        """Test CipherText + plaintext addition with broadcasting: 2D ciphertext + 1D plaintext."""
        pk, sk = self._generate_keypair()

        # Test with 2D ciphertext and 1D plaintext that broadcasts
        ciphertext_val = np.array([[1, 2], [3, 4]], dtype=np.int32)
        plaintext_vec = np.array(
            [100, 200], dtype=np.int32
        )  # Should broadcast to (2, 2)

        # Encrypt 2D array
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_val, pk])[0]

        # Add with 1D plaintext (should broadcast)
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val),
                TensorType.from_obj(plaintext_vec),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext, plaintext_vec])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (
            2,
            2,
        )  # Result should have broadcasted shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        # Broadcasting: [[1, 2], [3, 4]] + [[100, 200], [100, 200]] = [[101, 202], [103, 204]]
        expected = np.array([[101, 202], [103, 204]], dtype=np.int32)
        np.testing.assert_array_equal(decrypted, expected)

    def test_add_multidimensional_3d_tensors(self):
        """Test addition with 3D tensors."""
        pk, sk = self._generate_keypair()

        # Test with 3D tensors
        ciphertext_val1 = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32
        )  # Shape (2, 2, 2)
        ciphertext_val2 = np.array(
            [[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.int32
        )  # Shape (2, 2, 2)

        # Encrypt both 3D arrays
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_val1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc, [ciphertext_val1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc, [ciphertext_val2, pk])[0]

        # Add 3D tensors
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(ciphertext_val1),
                TensorType.from_obj(ciphertext_val2),
            ),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_val1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_val1),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = ciphertext_val1 + ciphertext_val2  # Element-wise addition
        np.testing.assert_array_equal(decrypted, expected)

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

    def test_dot_ciphertext_plaintext_int32(self):
        """Test CipherText dot plaintext with int32 vectors."""
        pk, sk = self._generate_keypair()

        # Create int32 vectors
        ciphertext_vec = np.array([1, 2, 3, 4], dtype=np.int32)
        plaintext_vec = np.array([2, 1, 3, 2], dtype=np.int32)

        # Encrypt the first vector
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_vec, pk])[0]

        # Perform dot product
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(
                TensorType.from_obj(ciphertext_vec),
                TensorType.from_obj(plaintext_vec),
            ),
            outs_info=(
                TensorType.from_obj(np.array(0, dtype=np.int32)),
            ),  # Scalar result
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, plaintext_vec])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == ()  # Scalar result

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(np.array(0, dtype=np.int32)),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(np.array(0, dtype=np.int32)),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.dot(
            ciphertext_vec, plaintext_vec
        )  # 1*2 + 2*1 + 3*3 + 4*2 = 2 + 2 + 9 + 8 = 21
        assert decrypted.item() == expected

    def test_dot_ciphertext_plaintext_float64(self):
        """Test CipherText dot plaintext with float64 vectors."""
        pk, sk = self._generate_keypair()

        # Create float64 vectors
        ciphertext_vec = np.array([1.5, 2.0, 3.5], dtype=np.float64)
        plaintext_vec = np.array([2.0, 1.5, 1.0], dtype=np.float64)

        # Encrypt the first vector
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_vec, pk])[0]

        # Perform dot product
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(
                TensorType.from_obj(ciphertext_vec),
                TensorType.from_obj(plaintext_vec),
            ),
            outs_info=(
                TensorType.from_obj(np.array(0.0, dtype=np.float64)),
            ),  # Scalar result
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, plaintext_vec])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == FLOAT64
        assert result_ct.semantic_shape == ()  # Scalar result

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(np.array(0.0, dtype=np.float64)),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(np.array(0.0, dtype=np.float64)),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.dot(
            ciphertext_vec, plaintext_vec
        )  # 1.5*2.0 + 2.0*1.5 + 3.5*1.0 = 3.0 + 3.0 + 3.5 = 9.5
        assert abs(decrypted.item() - expected) < 1e-10

    def test_dot_ciphertext_plaintext_array_int16(self):
        """Test CipherText dot plaintext with int16 vectors."""
        pk, sk = self._generate_keypair()

        # Create int16 vectors
        ciphertext_vec = np.array([10, 20, 30], dtype=np.int16)
        plaintext_vec = np.array([1, 2, 3], dtype=np.int16)

        # Encrypt the first vector
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_vec, pk])[0]

        # Perform dot product
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(
                TensorType.from_obj(ciphertext_vec),
                TensorType.from_obj(plaintext_vec),
            ),
            outs_info=(
                TensorType.from_obj(np.array(0, dtype=np.int16)),
            ),  # Scalar result
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, plaintext_vec])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT16
        assert result_ct.semantic_shape == ()

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(np.array(0, dtype=np.int16)),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(np.array(0, dtype=np.int16)),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.dot(
            ciphertext_vec, plaintext_vec
        )  # 10*1 + 20*2 + 30*3 = 10 + 40 + 90 = 140
        assert decrypted.item() == expected

    def test_dot_invalid_args(self):
        """Test dot product with invalid arguments."""
        pk, _sk = self._generate_keypair()

        # Test with wrong number of arguments
        ciphertext_vec = np.array([1, 2], dtype=np.int32)
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_vec, pk])[0]

        # Test with only one argument
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ciphertext_vec),),
            outs_info=(TensorType.from_obj(np.array(0, dtype=np.int32)),),
        )
        with pytest.raises(
            ValueError, match="Dot product expects exactly two arguments"
        ):
            self.handler.execute(dot_pfunc, [ciphertext])

    def test_dot_shape_mismatch(self):
        """Test dot product with shape mismatch."""
        pk, _sk = self._generate_keypair()

        # Create vectors with different sizes
        ciphertext_vec = np.array([1, 2, 3], dtype=np.int32)
        plaintext_vec = np.array([1, 2], dtype=np.int32)  # Different size

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_vec, pk])[0]

        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(
                TensorType.from_obj(ciphertext_vec),
                TensorType.from_obj(plaintext_vec),
            ),
            outs_info=(TensorType.from_obj(np.array(0, dtype=np.int32)),),
        )
        with pytest.raises(
            ValueError, match="Shapes are not compatible for dot product"
        ):
            self.handler.execute(dot_pfunc, [ciphertext, plaintext_vec])

    def test_dot_scalar_operands(self):
        """Test dot product with scalar operands (now supported)."""
        pk, sk = self._generate_keypair()

        # Test with scalar (should work as our implementation supports scalar * scalar)
        ciphertext_scalar = np.array(5, dtype=np.int32)
        plaintext_scalar = np.array(3, dtype=np.int32)

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_scalar), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_scalar),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_scalar, pk])[0]

        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(
                TensorType.from_obj(ciphertext_scalar),
                TensorType.from_obj(plaintext_scalar),
            ),
            outs_info=(TensorType.from_obj(ciphertext_scalar),),
        )
        # This should work now (scalar * scalar)
        result = self.handler.execute(dot_pfunc, [ciphertext, plaintext_scalar])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ciphertext_scalar), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_scalar),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]

        # Expected: 5 * 3 = 15
        decrypted_value = np.asarray(decrypted)
        assert decrypted_value.item() == 15

    def test_dot_non_ciphertext_first_arg(self):
        """Test dot product with non-CipherText as first argument."""
        # Test with plaintext as first argument (should fail)
        plaintext_vec1 = np.array([1, 2, 3], dtype=np.int32)
        plaintext_vec2 = np.array([2, 1, 3], dtype=np.int32)

        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(
                TensorType.from_obj(plaintext_vec1),
                TensorType.from_obj(plaintext_vec2),
            ),
            outs_info=(TensorType.from_obj(np.array(0, dtype=np.int32)),),
        )
        with pytest.raises(
            ValueError, match="First argument must be a CipherText instance"
        ):
            self.handler.execute(dot_pfunc, [plaintext_vec1, plaintext_vec2])

    def test_gather_ciphertext_basic_int32(self):
        """Test gather operation with basic int32 array."""
        pk, sk = self._generate_keypair()

        # Create int32 array
        ciphertext_vec = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        indices = np.array([0, 2, 4], dtype=np.int32)

        # Encrypt the vector
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_vec, pk])[0]

        # Perform gather operation
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_vec),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType.from_obj(indices),),
        )
        result_ct = self.handler.execute(gather_pfunc, [ciphertext, indices])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (3,)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(indices), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(indices),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([10, 30, 50], dtype=np.int32)  # Values at indices [0, 2, 4]
        np.testing.assert_array_equal(decrypted, expected)

    def test_gather_ciphertext_scalar_index_float64(self):
        """Test gather operation with scalar index and float64 array."""
        pk, sk = self._generate_keypair()

        # Create float64 array
        ciphertext_vec = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        scalar_index = np.array(1, dtype=np.int32)  # Scalar index

        # Encrypt the vector
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_vec, pk])[0]

        # Perform gather operation
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_vec),
                TensorType.from_obj(scalar_index),
            ),
            outs_info=(TensorType.from_obj(scalar_index),),
        )
        result_ct = self.handler.execute(gather_pfunc, [ciphertext, scalar_index])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == FLOAT64
        assert result_ct.semantic_shape == ()  # Scalar result

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(scalar_index), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(scalar_index),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        assert abs(decrypted.item() - 2.5) < 1e-10  # Value at index 1

    def test_gather_invalid_args(self):
        """Test gather with invalid arguments."""
        pk, _sk = self._generate_keypair()

        ciphertext_vec = np.array([1, 2, 3], dtype=np.int32)
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_vec, pk])[0]

        # Test with wrong number of arguments
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(TensorType.from_obj(ciphertext_vec),),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        with pytest.raises(ValueError, match="Gather expects exactly two arguments"):
            self.handler.execute(gather_pfunc, [ciphertext])

    def test_gather_out_of_bounds_indices(self):
        """Test gather with out-of-bounds indices."""
        pk, _sk = self._generate_keypair()

        ciphertext_vec = np.array([1, 2, 3], dtype=np.int32)
        bad_indices = np.array([0, 1, 5], dtype=np.int32)  # Index 5 is out of bounds

        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_vec),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_vec, pk])[0]

        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_vec),
                TensorType.from_obj(bad_indices),
            ),
            outs_info=(TensorType.from_obj(bad_indices),),
        )
        with pytest.raises(ValueError, match="Indices are out of bounds"):
            self.handler.execute(gather_pfunc, [ciphertext, bad_indices])

    def test_gather_non_ciphertext_first_arg(self):
        """Test gather with non-CipherText as first argument."""
        plaintext_vec = np.array([1, 2, 3], dtype=np.int32)
        indices = np.array([0, 1], dtype=np.int32)

        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(plaintext_vec),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType.from_obj(indices),),
        )
        with pytest.raises(
            ValueError, match="First argument must be a CipherText instance"
        ):
            self.handler.execute(gather_pfunc, [plaintext_vec, indices])

    def test_gather_multidimensional_2d_matrix_indices_1d(self):
        """Test gather from 2D CipherText matrix using 1D indices."""
        pk, sk = self._generate_keypair()

        # Create 2D matrix: shape (3, 4)
        ciphertext_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32
        )

        # Indices to gather rows 0 and 2
        indices = np.array([0, 2], dtype=np.int32)

        # Encrypt the matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_matrix, pk])[0]

        # Expected result shape: (2, 4) - gathering 2 rows from the 3x4 matrix
        expected_result = np.array(
            [[1, 2, 3, 4], [9, 10, 11, 12]], dtype=np.int32  # Row 0  # Row 2
        )

        # Perform gather
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_matrix),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result = self.handler.execute(gather_pfunc, [ciphertext, indices])[0]

        # Verify result properties
        assert isinstance(result, CipherText)
        assert result.semantic_dtype == INT32
        assert result.semantic_shape == (2, 4)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_gather_multidimensional_3d_tensor_indices_2d(self):
        """Test gather from 3D CipherText tensor using 2D indices."""
        pk, sk = self._generate_keypair()

        # Create 3D tensor: shape (4, 2, 3)
        ciphertext_tensor = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],  # Slice 0
                [[7, 8, 9], [10, 11, 12]],  # Slice 1
                [[13, 14, 15], [16, 17, 18]],  # Slice 2
                [[19, 20, 21], [22, 23, 24]],  # Slice 3
            ],
            dtype=np.int32,
        )

        # 2D indices: shape (2, 3) - will result in shape (2, 3, 2, 3)
        indices = np.array(
            [
                [0, 1, 3],  # First row: gather slices 0, 1, 3
                [2, 0, 1],  # Second row: gather slices 2, 0, 1
            ],
            dtype=np.int32,
        )

        # Encrypt the tensor
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_tensor),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_tensor, pk])[0]

        # Expected result shape: (2, 3, 2, 3)
        expected_result = np.array(
            [
                [  # Row 0 of indices
                    [[1, 2, 3], [4, 5, 6]],  # Slice 0
                    [[7, 8, 9], [10, 11, 12]],  # Slice 1
                    [[19, 20, 21], [22, 23, 24]],  # Slice 3
                ],
                [  # Row 1 of indices
                    [[13, 14, 15], [16, 17, 18]],  # Slice 2
                    [[1, 2, 3], [4, 5, 6]],  # Slice 0
                    [[7, 8, 9], [10, 11, 12]],  # Slice 1
                ],
            ],
            dtype=np.int32,
        )

        # Perform gather
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_tensor),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result = self.handler.execute(gather_pfunc, [ciphertext, indices])[0]

        # Verify result properties
        assert isinstance(result, CipherText)
        assert result.semantic_dtype == INT32
        assert result.semantic_shape == (2, 3, 2, 3)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_gather_multidimensional_scalar_indices(self):
        """Test gather from multidimensional CipherText using scalar indices."""
        pk, sk = self._generate_keypair()

        # Create 3D tensor: shape (3, 2, 4)
        ciphertext_tensor = np.arange(24, dtype=np.int32).reshape(3, 2, 4)

        # Scalar index: select slice 1
        scalar_index = np.array(1, dtype=np.int32)

        # Encrypt the tensor
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_tensor),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_tensor, pk])[0]

        # Expected result: shape (2, 4) - slice 1 from the 3x2x4 tensor
        expected_result = ciphertext_tensor[1]  # Shape (2, 4)

        # Perform gather
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_tensor),
                TensorType.from_obj(scalar_index),
            ),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result = self.handler.execute(gather_pfunc, [ciphertext, scalar_index])[0]

        # Verify result properties
        assert isinstance(result, CipherText)
        assert result.semantic_dtype == INT32
        assert result.semantic_shape == (2, 4)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_gather_multidimensional_4d_tensor_indices_3d(self):
        """Test gather from 4D CipherText tensor using 3D indices."""
        pk, sk = self._generate_keypair()

        # Create 4D tensor: shape (2, 3, 2, 2)
        ciphertext_tensor = np.arange(24, dtype=np.int32).reshape(2, 3, 2, 2)

        # 3D indices: shape (2, 1, 2) - will result in shape (2, 1, 2, 3, 2, 2)
        indices = np.array(
            [
                [[0, 1]],  # First "page": gather slices 0, 1
                [[1, 0]],  # Second "page": gather slices 1, 0
            ],
            dtype=np.int32,
        )

        # Encrypt the tensor
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_tensor),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_tensor, pk])[0]

        # Expected result shape: (2, 1, 2, 3, 2, 2)
        expected_result = np.array(
            [
                [  # Page 0
                    [  # Row 0
                        ciphertext_tensor[0],  # Slice 0: shape (3, 2, 2)
                        ciphertext_tensor[1],  # Slice 1: shape (3, 2, 2)
                    ]
                ],
                [  # Page 1
                    [  # Row 0
                        ciphertext_tensor[1],  # Slice 1: shape (3, 2, 2)
                        ciphertext_tensor[0],  # Slice 0: shape (3, 2, 2)
                    ]
                ],
            ]
        )

        # Perform gather
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_tensor),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result = self.handler.execute(gather_pfunc, [ciphertext, indices])[0]

        # Verify result properties
        assert isinstance(result, CipherText)
        assert result.semantic_dtype == INT32
        assert result.semantic_shape == (2, 1, 2, 3, 2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_gather_multidimensional_float_types(self):
        """Test gather with floating point types."""
        pk, sk = self._generate_keypair()

        # Create 2D float tensor: shape (3, 2)
        ciphertext_matrix = np.array(
            [[1.5, 2.5], [3.7, 4.8], [5.1, 6.9]], dtype=np.float64
        )

        # Indices to gather rows 2 and 0
        indices = np.array([2, 0], dtype=np.int32)

        # Encrypt the matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_matrix, pk])[0]

        # Expected result: rows 2 and 0
        expected_result = np.array(
            [[5.1, 6.9], [1.5, 2.5]], dtype=np.float64  # Row 2  # Row 0
        )

        # Perform gather
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_matrix),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result = self.handler.execute(gather_pfunc, [ciphertext, indices])[0]

        # Verify result properties
        assert isinstance(result, CipherText)
        assert result.semantic_dtype == FLOAT64
        assert result.semantic_shape == (2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_allclose(decrypted_array, expected_result, rtol=1e-10)

    def test_gather_multidimensional_out_of_bounds(self):
        """Test gather with out of bounds indices in multidimensional context."""
        pk, _sk = self._generate_keypair()

        # Create 2D matrix: shape (3, 4)
        ciphertext_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32
        )

        # Encrypt the matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_matrix, pk])[0]

        # Out of bounds indices
        bad_indices = np.array(
            [0, 3, 1], dtype=np.int32
        )  # Index 3 is out of bounds for axis 0 (size 3)

        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_matrix),
                TensorType.from_obj(bad_indices),
            ),
            outs_info=(TensorType.from_obj(np.zeros((3, 4), dtype=np.int32)),),
        )
        with pytest.raises(ValueError, match="Indices are out of bounds for axis 0"):
            self.handler.execute(gather_pfunc, [ciphertext, bad_indices])

    def test_gather_multidimensional_scalar_ciphertext(self):
        """Test gather from scalar CipherText (should fail)."""
        pk, _sk = self._generate_keypair()

        # Create scalar ciphertext
        ciphertext_scalar = np.array(42, dtype=np.int32)

        # Encrypt the scalar
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ciphertext_scalar), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ciphertext_scalar),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ciphertext_scalar, pk])[0]

        # Attempt to gather from scalar (should fail)
        indices = np.array([0], dtype=np.int32)

        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(ciphertext_scalar),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType.from_obj(indices),),
        )
        with pytest.raises(ValueError, match="Cannot gather from scalar CipherText"):
            self.handler.execute(gather_pfunc, [ciphertext, indices])

    def test_scatter_ciphertext_basic_int32(self):
        """Test scatter operation with basic int32 array."""
        pk, sk = self._generate_keypair()

        # Create original and updated arrays
        original_vec = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        updated_values = np.array([100, 300], dtype=np.int32)
        indices = np.array([0, 2], dtype=np.int32)

        # Encrypt both arrays
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_vec),),
        )
        original_ciphertext = self.handler.execute(encrypt_pfunc, [original_vec, pk])[0]

        encrypt_pfunc_updated = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(updated_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(updated_values),),
        )
        updated_ciphertext = self.handler.execute(
            encrypt_pfunc_updated, [updated_values, pk]
        )[0]

        # Perform scatter operation
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_vec),
                TensorType.from_obj(indices),
                TensorType.from_obj(updated_values),
            ),
            outs_info=(TensorType.from_obj(original_vec),),
        )
        result_ct = self.handler.execute(
            scatter_pfunc, [original_ciphertext, indices, updated_ciphertext]
        )[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (5,)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(original_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_vec),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        # Note: scatter should update positions based on indices
        # indices=[0,2], updated_values=[100,300] -> positions 0 and 2 get new values
        expected = np.array(
            [100, 20, 300, 40, 50], dtype=np.int32
        )  # Updated at positions 0 and 2
        np.testing.assert_array_equal(decrypted, expected)

    def test_scatter_invalid_args(self):
        """Test scatter with invalid arguments."""
        pk, _sk = self._generate_keypair()

        original_vec = np.array([1, 2, 3], dtype=np.int32)
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_vec), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_vec),),
        )
        original_ciphertext = self.handler.execute(encrypt_pfunc, [original_vec, pk])[0]

        # Test with wrong number of arguments
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_vec),
                TensorType.from_obj(original_vec),
            ),
            outs_info=(TensorType.from_obj(original_vec),),
        )
        with pytest.raises(ValueError, match="Scatter expects exactly three arguments"):
            self.handler.execute(scatter_pfunc, [original_ciphertext, original_vec])

    def test_scatter_non_ciphertext_args(self):
        """Test scatter with non-CipherText arguments."""
        original_vec = np.array([1, 2, 3], dtype=np.int32)
        indices = np.array([0], dtype=np.int32)
        updated_values = np.array([10], dtype=np.int32)

        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_vec),
                TensorType.from_obj(indices),
                TensorType.from_obj(updated_values),
            ),
            outs_info=(TensorType.from_obj(original_vec),),
        )
        with pytest.raises(
            ValueError, match="First and third arguments must be CipherText instances"
        ):
            self.handler.execute(scatter_pfunc, [original_vec, indices, updated_values])

    def test_scatter_multidimensional_2d_matrix_indices_1d(self):
        """Test scatter into 2D CipherText matrix using 1D indices."""
        pk, sk = self._generate_keypair()

        # Create original 2D matrix: shape (4, 3)
        original_matrix = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32
        )

        # Indices to scatter into rows 0 and 2
        indices = np.array([0, 2], dtype=np.int32)

        # Updated values: shape (2, 3) - same as indices.shape + original.shape[1:]
        updated_values = np.array(
            [[100, 200, 300], [700, 800, 900]], dtype=np.int32  # New row 0  # New row 2
        )

        # Encrypt the original matrix and updated values
        encrypt_pfunc_orig = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        original_ciphertext = self.handler.execute(
            encrypt_pfunc_orig, [original_matrix, pk]
        )[0]

        encrypt_pfunc_upd = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(updated_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(updated_values),),
        )
        updated_ciphertext = self.handler.execute(
            encrypt_pfunc_upd, [updated_values, pk]
        )[0]

        # Expected result: original matrix with rows 0 and 2 replaced
        expected_result = np.array(
            [
                [100, 200, 300],  # Row 0 updated
                [4, 5, 6],  # Row 1 unchanged
                [700, 800, 900],  # Row 2 updated
                [10, 11, 12],  # Row 3 unchanged
            ],
            dtype=np.int32,
        )

        # Perform scatter
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(indices),
                TensorType.from_obj(updated_values),
            ),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        result = self.handler.execute(
            scatter_pfunc, [original_ciphertext, indices, updated_ciphertext]
        )[0]

        # Verify result properties
        assert isinstance(result, CipherText)
        assert result.semantic_dtype == INT32
        assert result.semantic_shape == (4, 3)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_scatter_multidimensional_3d_tensor_indices_2d(self):
        """Test scatter into 3D CipherText tensor using 2D indices."""
        pk, sk = self._generate_keypair()

        # Create original 3D tensor: shape (3, 2, 2)
        original_tensor = np.arange(12, dtype=np.int32).reshape(3, 2, 2)

        # 2D indices: shape (2, 2) - will update 4 slices total
        indices = np.array(
            [
                [0, 1],  # First row: update slices 0, 1
                [2, 0],  # Second row: update slices 2, 0 (0 gets updated twice)
            ],
            dtype=np.int32,
        )

        # Updated values: shape (2, 2, 2, 2) - indices.shape + original.shape[1:]
        updated_values = np.array(
            [
                [  # First row of indices
                    [[100, 101], [102, 103]],  # For slice 0
                    [[110, 111], [112, 113]],  # For slice 1
                ],
                [  # Second row of indices
                    [[200, 201], [202, 203]],  # For slice 2
                    [
                        [210, 211],
                        [212, 213],
                    ],  # For slice 0 (will overwrite first update)
                ],
            ],
            dtype=np.int32,
        )

        # Encrypt the original tensor and updated values
        encrypt_pfunc_orig = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        original_ciphertext = self.handler.execute(
            encrypt_pfunc_orig, [original_tensor, pk]
        )[0]

        encrypt_pfunc_upd = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(updated_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(updated_values),),
        )
        updated_ciphertext = self.handler.execute(
            encrypt_pfunc_upd, [updated_values, pk]
        )[0]

        # Expected result: original with slices updated
        expected_result = np.array(
            [
                [[210, 211], [212, 213]],  # Slice 0 (final update)
                [[110, 111], [112, 113]],  # Slice 1
                [[200, 201], [202, 203]],  # Slice 2
            ],
            dtype=np.int32,
        )

        # Perform scatter
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_tensor),
                TensorType.from_obj(indices),
                TensorType.from_obj(updated_values),
            ),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        result = self.handler.execute(
            scatter_pfunc, [original_ciphertext, indices, updated_ciphertext]
        )[0]

        # Verify result properties
        assert isinstance(result, CipherText)
        assert result.semantic_dtype == INT32
        assert result.semantic_shape == (3, 2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(original_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_scatter_multidimensional_scalar_indices(self):
        """Test scatter into multidimensional CipherText using scalar indices."""
        pk, sk = self._generate_keypair()

        # Create original 3D tensor: shape (4, 2, 3)
        original_tensor = np.arange(24, dtype=np.int32).reshape(4, 2, 3)

        # Scalar index: update slice 1
        scalar_index = np.array(1, dtype=np.int32)

        # Updated values: shape (2, 3) - scalar indices.shape + original.shape[1:]
        updated_values = np.array([[100, 101, 102], [103, 104, 105]], dtype=np.int32)

        # Encrypt the original tensor and updated values
        encrypt_pfunc_orig = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        original_ciphertext = self.handler.execute(
            encrypt_pfunc_orig, [original_tensor, pk]
        )[0]

        encrypt_pfunc_upd = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(updated_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(updated_values),),
        )
        updated_ciphertext = self.handler.execute(
            encrypt_pfunc_upd, [updated_values, pk]
        )[0]

        # Expected result: original with slice 1 updated
        expected_result = original_tensor.copy()
        expected_result[1] = updated_values

        # Perform scatter
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_tensor),
                TensorType.from_obj(scalar_index),
                TensorType.from_obj(updated_values),
            ),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        result = self.handler.execute(
            scatter_pfunc, [original_ciphertext, scalar_index, updated_ciphertext]
        )[0]

        # Verify result properties
        assert isinstance(result, CipherText)
        assert result.semantic_dtype == INT32
        assert result.semantic_shape == (4, 2, 3)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(original_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_scatter_multidimensional_float_types(self):
        """Test scatter with floating point types."""
        pk, sk = self._generate_keypair()

        # Create original 2D float matrix: shape (3, 2)
        original_matrix = np.array(
            [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float64
        )

        # Indices to scatter into rows 0 and 2
        indices = np.array([0, 2], dtype=np.int32)

        # Updated values: shape (2, 2)
        updated_values = np.array(
            [[10.1, 20.2], [50.5, 60.6]], dtype=np.float64  # New row 0  # New row 2
        )

        # Encrypt the original matrix and updated values
        encrypt_pfunc_orig = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        original_ciphertext = self.handler.execute(
            encrypt_pfunc_orig, [original_matrix, pk]
        )[0]

        encrypt_pfunc_upd = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(updated_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(updated_values),),
        )
        updated_ciphertext = self.handler.execute(
            encrypt_pfunc_upd, [updated_values, pk]
        )[0]

        # Expected result
        expected_result = np.array(
            [
                [10.1, 20.2],  # Row 0 updated
                [3.3, 4.4],  # Row 1 unchanged
                [50.5, 60.6],  # Row 2 updated
            ],
            dtype=np.float64,
        )

        # Perform scatter
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(indices),
                TensorType.from_obj(updated_values),
            ),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        result = self.handler.execute(
            scatter_pfunc, [original_ciphertext, indices, updated_ciphertext]
        )[0]

        # Verify result properties
        assert isinstance(result, CipherText)
        assert result.semantic_dtype == FLOAT64
        assert result.semantic_shape == (3, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_allclose(decrypted_array, expected_result, rtol=1e-10)

    def test_scatter_multidimensional_out_of_bounds(self):
        """Test scatter with out of bounds indices in multidimensional context."""
        pk, _sk = self._generate_keypair()

        # Create original 2D matrix: shape (3, 2)
        original_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)

        # Updated values: shape (2, 2)
        updated_values = np.array([[10, 20], [30, 40]], dtype=np.int32)

        # Encrypt the matrices
        encrypt_pfunc_orig = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        original_ciphertext = self.handler.execute(
            encrypt_pfunc_orig, [original_matrix, pk]
        )[0]

        encrypt_pfunc_upd = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(updated_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(updated_values),),
        )
        updated_ciphertext = self.handler.execute(
            encrypt_pfunc_upd, [updated_values, pk]
        )[0]

        # Out of bounds indices (index 3 is out of bounds for axis 0 with size 3)
        bad_indices = np.array([0, 3], dtype=np.int32)

        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(bad_indices),
                TensorType.from_obj(updated_values),
            ),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        with pytest.raises(ValueError, match="Indices are out of bounds for axis 0"):
            self.handler.execute(
                scatter_pfunc, [original_ciphertext, bad_indices, updated_ciphertext]
            )

    def test_scatter_multidimensional_shape_mismatch(self):
        """Test scatter with incompatible updated shape."""
        pk, _sk = self._generate_keypair()

        # Create original 2D matrix: shape (3, 4)
        original_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32
        )

        # Indices: shape (2,)
        indices = np.array([0, 2], dtype=np.int32)

        # Wrong updated values shape: (2, 3) instead of (2, 4)
        wrong_updated_values = np.array(
            [
                [100, 200, 300],  # Wrong: should be 4 elements
                [900, 1000, 1100],  # Wrong: should be 4 elements
            ],
            dtype=np.int32,
        )

        # Encrypt the matrices
        encrypt_pfunc_orig = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        original_ciphertext = self.handler.execute(
            encrypt_pfunc_orig, [original_matrix, pk]
        )[0]

        encrypt_pfunc_upd = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(wrong_updated_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(wrong_updated_values),),
        )
        updated_ciphertext = self.handler.execute(
            encrypt_pfunc_upd, [wrong_updated_values, pk]
        )[0]

        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(indices),
                TensorType.from_obj(wrong_updated_values),
            ),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        with pytest.raises(ValueError, match="Updated CipherText shape mismatch"):
            self.handler.execute(
                scatter_pfunc, [original_ciphertext, indices, updated_ciphertext]
            )

    def test_scatter_multidimensional_scalar_ciphertext(self):
        """Test scatter into scalar CipherText (should fail)."""
        pk, _sk = self._generate_keypair()

        # Create scalar ciphertext
        original_scalar = np.array(42, dtype=np.int32)
        updated_scalar = np.array(100, dtype=np.int32)

        # Encrypt the scalars
        encrypt_pfunc_orig = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_scalar), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_scalar),),
        )
        original_ciphertext = self.handler.execute(
            encrypt_pfunc_orig, [original_scalar, pk]
        )[0]

        encrypt_pfunc_upd = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(updated_scalar), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(updated_scalar),),
        )
        updated_ciphertext = self.handler.execute(
            encrypt_pfunc_upd, [updated_scalar, pk]
        )[0]

        # Attempt to scatter into scalar (should fail)
        indices = np.array([0], dtype=np.int32)

        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_scalar),
                TensorType.from_obj(indices),
                TensorType.from_obj(updated_scalar),
            ),
            outs_info=(TensorType.from_obj(original_scalar),),
        )
        with pytest.raises(ValueError, match="Cannot scatter into scalar CipherText"):
            self.handler.execute(
                scatter_pfunc, [original_ciphertext, indices, updated_ciphertext]
            )

    def test_concat_ciphertext_basic_int32(self):
        """Test concat operation with basic int32 arrays."""
        pk, sk = self._generate_keypair()

        # Create two arrays to concatenate
        vec1 = np.array([10, 20, 30], dtype=np.int32)
        vec2 = np.array([40, 50], dtype=np.int32)

        # Encrypt both arrays
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(vec1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(vec1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [vec1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(vec2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(vec2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [vec2, pk])[0]

        # Perform concat operation
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(vec1),
                TensorType.from_obj(vec2),
            ),
            outs_info=(TensorType.from_obj(np.array([1, 2, 3, 4, 5], dtype=np.int32)),),
        )
        result_ct = self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (5,)  # 3 + 2 elements

        # Decrypt and verify
        concat_result_shape = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(concat_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(concat_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        np.testing.assert_array_equal(decrypted, expected)

    def test_concat_ciphertext_float64(self):
        """Test concat operation with float64 arrays."""
        pk, sk = self._generate_keypair()

        # Create two float64 arrays to concatenate
        vec1 = np.array([1.5, 2.5], dtype=np.float64)
        vec2 = np.array([3.5, 4.5, 5.5], dtype=np.float64)

        # Encrypt both arrays
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(vec1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(vec1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [vec1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(vec2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(vec2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [vec2, pk])[0]

        # Perform concat operation
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(vec1),
                TensorType.from_obj(vec2),
            ),
            outs_info=(
                TensorType.from_obj(np.array([1, 2, 3, 4, 5], dtype=np.float64)),
            ),
        )
        result_ct = self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == FLOAT64
        assert result_ct.semantic_shape == (5,)  # 2 + 3 elements

        # Decrypt and verify
        concat_result_shape = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(concat_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(concat_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
        np.testing.assert_allclose(decrypted, expected, rtol=1e-10)

    def test_concat_invalid_args(self):
        """Test concat with invalid number of arguments."""
        pk, _sk = self._generate_keypair()

        vec1 = np.array([1, 2], dtype=np.int32)
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(vec1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(vec1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc, [vec1, pk])[0]

        # Test with wrong number of arguments
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(TensorType.from_obj(vec1),),
            outs_info=(TensorType.from_obj(vec1),),
        )
        with pytest.raises(ValueError, match="Concat expects exactly two arguments"):
            self.handler.execute(concat_pfunc, [ciphertext1])

    def test_concat_non_ciphertext_args(self):
        """Test concat with non-CipherText arguments."""
        vec1 = np.array([1, 2], dtype=np.int32)
        vec2 = np.array([3, 4], dtype=np.int32)

        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(vec1),
                TensorType.from_obj(vec2),
            ),
            outs_info=(TensorType.from_obj(np.array([1, 2, 3, 4], dtype=np.int32)),),
        )
        with pytest.raises(
            ValueError, match="All arguments must be CipherText instances"
        ):
            self.handler.execute(concat_pfunc, [vec1, vec2])

    def test_concat_dtype_mismatch(self):
        """Test concat with different dtypes."""
        pk, _sk = self._generate_keypair()

        vec1 = np.array([1, 2], dtype=np.int32)
        vec2 = np.array([3.0, 4.0], dtype=np.float32)

        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(vec1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(vec1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [vec1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(vec2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(vec2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [vec2, pk])[0]

        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(vec1),
                TensorType.from_obj(vec2),
            ),
            outs_info=(TensorType.from_obj(np.array([1, 2, 3, 4], dtype=np.int32)),),
        )
        with pytest.raises(
            ValueError, match="All CipherTexts must have same semantic dtype"
        ):
            self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])

    def test_concat_multidimensional_axis0_2d_matrices(self):
        """Test concat operation along axis 0 for 2D matrices."""
        pk, sk = self._generate_keypair()

        # Create two 2D matrices to concatenate along axis 0
        matrix1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # Shape (2, 3)
        matrix2 = np.array([[7, 8, 9]], dtype=np.int32)  # Shape (1, 3)

        # Encrypt both matrices
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [matrix1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [matrix2, pk])[0]

        # Perform concat operation along axis 0 (default)
        concat_result_shape = np.zeros((3, 3), dtype=np.int32)  # Result shape (3, 3)
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(matrix1),
                TensorType.from_obj(matrix2),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=0,  # Concatenate along axis 0
        )
        result_ct = self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (3, 3)  # (2+1, 3)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(concat_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(concat_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected)

    def test_concat_multidimensional_axis1_2d_matrices(self):
        """Test concat operation along axis 1 for 2D matrices."""
        pk, sk = self._generate_keypair()

        # Create two 2D matrices to concatenate along axis 1
        matrix1 = np.array([[1, 2], [4, 5]], dtype=np.int32)  # Shape (2, 2)
        matrix2 = np.array([[3], [6]], dtype=np.int32)  # Shape (2, 1)

        # Encrypt both matrices
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [matrix1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [matrix2, pk])[0]

        # Perform concat operation along axis 1
        concat_result_shape = np.zeros((2, 3), dtype=np.int32)  # Result shape (2, 3)
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(matrix1),
                TensorType.from_obj(matrix2),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=1,  # Concatenate along axis 1
        )
        result_ct = self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 3)  # (2, 2+1)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(concat_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(concat_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected)

    def test_concat_multidimensional_3d_tensors_axis0(self):
        """Test concat operation along axis 0 for 3D tensors."""
        pk, sk = self._generate_keypair()

        # Create two 3D tensors to concatenate along axis 0
        tensor1 = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32
        )  # Shape (2, 2, 2)
        tensor2 = np.array([[[9, 10], [11, 12]]], dtype=np.int32)  # Shape (1, 2, 2)

        # Encrypt both tensors
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(tensor1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(tensor1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [tensor1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(tensor2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(tensor2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [tensor2, pk])[0]

        # Perform concat operation along axis 0
        concat_result_shape = np.zeros(
            (3, 2, 2), dtype=np.int32
        )  # Result shape (3, 2, 2)
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(tensor1),
                TensorType.from_obj(tensor2),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=0,  # Concatenate along axis 0
        )
        result_ct = self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (3, 2, 2)  # (2+1, 2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(concat_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(concat_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32
        )
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected)

    def test_concat_multidimensional_3d_tensors_axis2(self):
        """Test concat operation along axis 2 for 3D tensors."""
        pk, sk = self._generate_keypair()

        # Create two 3D tensors to concatenate along axis 2
        tensor1 = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32
        )  # Shape (2, 2, 2)
        tensor2 = np.array(
            [[[10], [30]], [[50], [70]]], dtype=np.int32
        )  # Shape (2, 2, 1)

        # Encrypt both tensors
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(tensor1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(tensor1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [tensor1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(tensor2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(tensor2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [tensor2, pk])[0]

        # Perform concat operation along axis 2
        concat_result_shape = np.zeros(
            (2, 2, 3), dtype=np.int32
        )  # Result shape (2, 2, 3)
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(tensor1),
                TensorType.from_obj(tensor2),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=2,  # Concatenate along axis 2
        )
        result_ct = self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2, 3)  # (2, 2, 2+1)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(concat_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(concat_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array(
            [[[1, 2, 10], [3, 4, 30]], [[5, 6, 50], [7, 8, 70]]], dtype=np.int32
        )
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected)

    def test_concat_multidimensional_negative_axis(self):
        """Test concat operation with negative axis."""
        pk, sk = self._generate_keypair()

        # Create two 2D matrices to concatenate along axis -1 (equivalent to axis 1)
        matrix1 = np.array([[1, 2], [4, 5]], dtype=np.int32)  # Shape (2, 2)
        matrix2 = np.array([[3], [6]], dtype=np.int32)  # Shape (2, 1)

        # Encrypt both matrices
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [matrix1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [matrix2, pk])[0]

        # Perform concat operation along axis -1
        concat_result_shape = np.zeros((2, 3), dtype=np.int32)  # Result shape (2, 3)
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(matrix1),
                TensorType.from_obj(matrix2),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=-1,  # Concatenate along axis -1 (last axis)
        )
        result_ct = self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 3)  # (2, 2+1)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(concat_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(concat_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected)

    def test_concat_multidimensional_float_types(self):
        """Test concat operation with floating point types."""
        pk, sk = self._generate_keypair()

        # Create two 2D float matrices to concatenate
        matrix1 = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)  # Shape (2, 2)
        matrix2 = np.array([[5.5, 6.6]], dtype=np.float64)  # Shape (1, 2)

        # Encrypt both matrices
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [matrix1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [matrix2, pk])[0]

        # Perform concat operation along axis 0
        concat_result_shape = np.zeros((3, 2), dtype=np.float64)  # Result shape (3, 2)
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(matrix1),
                TensorType.from_obj(matrix2),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=0,  # Concatenate along axis 0
        )
        result_ct = self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == FLOAT64
        assert result_ct.semantic_shape == (3, 2)  # (2+1, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(concat_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(concat_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float64)
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_allclose(decrypted_array, expected, rtol=1e-10)

    def test_concat_multidimensional_invalid_axis(self):
        """Test concat operation with invalid axis."""
        pk, _sk = self._generate_keypair()

        # Create two 2D matrices
        matrix1 = np.array([[1, 2], [3, 4]], dtype=np.int32)  # Shape (2, 2)
        matrix2 = np.array([[5, 6], [7, 8]], dtype=np.int32)  # Shape (2, 2)

        # Encrypt both matrices
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [matrix1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [matrix2, pk])[0]

        # Try concat with invalid axis (axis 2 for 2D tensors)
        concat_result_shape = np.zeros((2, 2), dtype=np.int32)
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(matrix1),
                TensorType.from_obj(matrix2),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=2,  # Invalid axis for 2D tensors
        )
        with pytest.raises(
            ValueError, match="axis 2 is out of bounds for array of dimension 2"
        ):
            self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])

    def test_concat_multidimensional_dimension_mismatch(self):
        """Test concat operation with incompatible shapes."""
        pk, _sk = self._generate_keypair()

        # Create two matrices with incompatible shapes for concatenation
        matrix1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # Shape (2, 3)
        matrix2 = np.array(
            [[7, 8]], dtype=np.int32
        )  # Shape (1, 2) - incompatible for axis 0

        # Encrypt both matrices
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [matrix1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(matrix2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(matrix2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [matrix2, pk])[0]

        # Try concat along axis 0 - should fail due to dimension 1 mismatch (3 vs 2)
        concat_result_shape = np.zeros((3, 3), dtype=np.int32)
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(matrix1),
                TensorType.from_obj(matrix2),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=0,
        )
        with pytest.raises(
            ValueError,
            match="All CipherTexts must have same shape except along concatenation axis",
        ):
            self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])

    def test_concat_multidimensional_scalar_ciphertext(self):
        """Test concat operation with scalar CipherTexts (should fail)."""
        pk, _sk = self._generate_keypair()

        # Create scalar ciphertexts
        scalar1 = np.array(42, dtype=np.int32)
        scalar2 = np.array(100, dtype=np.int32)

        # Encrypt both scalars
        encrypt_pfunc1 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(scalar1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(scalar1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc1, [scalar1, pk])[0]

        encrypt_pfunc2 = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(scalar2), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(scalar2),),
        )
        ciphertext2 = self.handler.execute(encrypt_pfunc2, [scalar2, pk])[0]

        # Try concat scalars - should fail
        concat_result_shape = np.array([42, 100], dtype=np.int32)
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(
                TensorType.from_obj(scalar1),
                TensorType.from_obj(scalar2),
            ),
            outs_info=(TensorType.from_obj(concat_result_shape),),
            axis=0,
        )
        with pytest.raises(ValueError, match="Cannot concatenate scalar CipherTexts"):
            self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])

    def test_decrypt_multidimensional_2d_matrix_int32(self):
        """Test encrypt/decrypt cycle for 2D matrix int32."""
        pk, sk = self._generate_keypair()

        # Test with 2D int32 matrix
        plaintext = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

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
        assert ciphertext.semantic_shape == (2, 3)

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
        assert decrypted.shape == (2, 3)
        np.testing.assert_array_equal(decrypted, [[1, 2, 3], [4, 5, 6]])

    def test_decrypt_multidimensional_3d_tensor_float64(self):
        """Test encrypt/decrypt cycle for 3D tensor float64."""
        pk, sk = self._generate_keypair()

        # Test with 3D float64 tensor
        plaintext = np.array(
            [[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float64
        )

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext_result = self.handler.execute(encrypt_pfunc, [plaintext, pk])
        ciphertext = ciphertext_result[0]

        assert isinstance(ciphertext, CipherText)
        assert ciphertext.semantic_dtype == FLOAT64
        assert ciphertext.semantic_shape == (2, 2, 2)

        # Decrypt
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        decrypted_result = self.handler.execute(decrypt_pfunc, [ciphertext, sk])
        decrypted = decrypted_result[0]

        assert isinstance(decrypted, np.ndarray)
        assert decrypted.dtype == np.float64
        assert decrypted.shape == (2, 2, 2)
        np.testing.assert_allclose(decrypted, plaintext, rtol=1e-10)

    def test_decrypt_multidimensional_large_matrix_int16(self):
        """Test encrypt/decrypt cycle for large matrix int16."""
        pk, sk = self._generate_keypair()

        # Test with larger int16 matrix
        plaintext = np.arange(24, dtype=np.int16).reshape(4, 6)

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext_result = self.handler.execute(encrypt_pfunc, [plaintext, pk])
        ciphertext = ciphertext_result[0]

        assert isinstance(ciphertext, CipherText)
        assert ciphertext.semantic_dtype == INT16
        assert ciphertext.semantic_shape == (4, 6)

        # Decrypt
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        decrypted_result = self.handler.execute(decrypt_pfunc, [ciphertext, sk])
        decrypted = decrypted_result[0]

        assert isinstance(decrypted, np.ndarray)
        assert decrypted.dtype == np.int16
        assert decrypted.shape == (4, 6)
        np.testing.assert_array_equal(decrypted, plaintext)

    def test_decrypt_multidimensional_single_element_2d(self):
        """Test encrypt/decrypt cycle for single element 2D array."""
        pk, sk = self._generate_keypair()

        # Test with single element 2D array
        plaintext = np.array([[42]], dtype=np.int32)

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext_result = self.handler.execute(encrypt_pfunc, [plaintext, pk])
        ciphertext = ciphertext_result[0]

        assert isinstance(ciphertext, CipherText)
        assert ciphertext.semantic_shape == (1, 1)

        # Decrypt
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        decrypted_result = self.handler.execute(decrypt_pfunc, [ciphertext, sk])
        decrypted = decrypted_result[0]

        assert isinstance(decrypted, np.ndarray)
        assert decrypted.shape == (1, 1)
        assert decrypted[0, 0] == 42

    def test_decrypt_multidimensional_different_dtypes(self):
        """Test encrypt/decrypt cycle for multidimensional arrays with different dtypes."""
        pk, sk = self._generate_keypair()

        # Test various dtypes with 2D arrays
        test_cases = [
            (np.array([[1, 2], [3, 4]], dtype=np.int8), INT8),
            (np.array([[100, 200], [300, 400]], dtype=np.int64), INT64),
            (np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32), FLOAT32),
        ]

        for plaintext, expected_dtype in test_cases:
            # Encrypt
            encrypt_pfunc = PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
                outs_info=(TensorType.from_obj(plaintext),),
            )
            ciphertext_result = self.handler.execute(encrypt_pfunc, [plaintext, pk])
            ciphertext = ciphertext_result[0]

            assert isinstance(ciphertext, CipherText)
            assert ciphertext.semantic_dtype == expected_dtype
            assert ciphertext.semantic_shape == (2, 2)

            # Decrypt
            decrypt_pfunc = PFunction(
                fn_type="phe.decrypt",
                ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
                outs_info=(TensorType.from_obj(plaintext),),
            )
            decrypted_result = self.handler.execute(decrypt_pfunc, [ciphertext, sk])
            decrypted = decrypted_result[0]

            assert isinstance(decrypted, np.ndarray)
            assert decrypted.shape == (2, 2)
            if expected_dtype.is_floating:
                np.testing.assert_allclose(decrypted, plaintext, rtol=1e-6)
            else:
                np.testing.assert_array_equal(decrypted, plaintext)

    def test_decrypt_multidimensional_negative_values(self):
        """Test encrypt/decrypt cycle for multidimensional arrays with negative values."""
        pk, sk = self._generate_keypair()

        # Test with negative values in 2D array
        plaintext = np.array([[-1, -2, -3], [4, -5, 6]], dtype=np.int32)

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext_result = self.handler.execute(encrypt_pfunc, [plaintext, pk])
        ciphertext = ciphertext_result[0]

        assert isinstance(ciphertext, CipherText)
        assert ciphertext.semantic_shape == (2, 3)

        # Decrypt
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        decrypted_result = self.handler.execute(decrypt_pfunc, [ciphertext, sk])
        decrypted = decrypted_result[0]

        assert isinstance(decrypted, np.ndarray)
        assert decrypted.shape == (2, 3)
        np.testing.assert_array_equal(decrypted, [[-1, -2, -3], [4, -5, 6]])

    def test_dot_multidimensional_scalar_scalar(self):
        """Test dot product of scalar CipherText with scalar plaintext."""
        pk, sk = self._generate_keypair()

        # Test scalar * scalar
        ct_scalar = np.array(5, dtype=np.int32)
        pt_scalar = np.array(3, dtype=np.int32)

        # Encrypt scalar
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_scalar), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_scalar),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_scalar, pk])[0]

        # Dot product
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_scalar), TensorType.from_obj(pt_scalar)),
            outs_info=(TensorType.from_obj(ct_scalar),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, pt_scalar])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_shape == ()  # Scalar result

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(ct_scalar), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_scalar),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        assert isinstance(decrypted, np.ndarray)
        assert decrypted.item() == 15  # 5 * 3

    def test_dot_multidimensional_vector_vector(self):
        """Test dot product of vector CipherText with vector plaintext."""
        pk, sk = self._generate_keypair()

        # Test vector * vector (inner product)
        ct_vector = np.array([1, 2, 3], dtype=np.int32)
        pt_vector = np.array([4, 5, 6], dtype=np.int32)

        # Encrypt vector
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_vector), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_vector),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_vector, pk])[0]

        # Dot product
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_vector), TensorType.from_obj(pt_vector)),
            outs_info=(TensorType.from_obj(np.array(0, dtype=np.int32)),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, pt_vector])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_shape == ()  # Scalar result

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(np.array(0, dtype=np.int32)),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(np.array(0, dtype=np.int32)),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        assert isinstance(decrypted, np.ndarray)
        expected = np.dot(ct_vector, pt_vector)  # 1*4 + 2*5 + 3*6 = 32
        assert decrypted.item() == expected

    def test_dot_multidimensional_matrix_vector(self):
        """Test dot product of matrix CipherText with vector plaintext."""
        pk, sk = self._generate_keypair()

        # Test matrix * vector
        ct_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)  # 3x2 matrix
        pt_vector = np.array([7, 8], dtype=np.int32)  # 2-element vector

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_matrix, pk])[0]

        # Dot product
        expected_result = np.dot(ct_matrix, pt_vector)  # Should be 3-element vector
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType.from_obj(pt_vector)),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, pt_vector])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_shape == (3,)  # 3-element vector result

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_array_equal(decrypted, expected_result)

    def test_dot_multidimensional_vector_matrix(self):
        """Test dot product of vector CipherText with matrix plaintext."""
        pk, sk = self._generate_keypair()

        # Test vector * matrix
        ct_vector = np.array([1, 2, 3], dtype=np.int32)  # 3-element vector
        pt_matrix = np.array([[4, 5], [6, 7], [8, 9]], dtype=np.int32)  # 3x2 matrix

        # Encrypt vector
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_vector), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_vector),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_vector, pk])[0]

        # Dot product
        expected_result = np.dot(ct_vector, pt_matrix)  # Should be 2-element vector
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_vector), TensorType.from_obj(pt_matrix)),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, pt_matrix])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_shape == (2,)  # 2-element vector result

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_array_equal(decrypted, expected_result)

    def test_dot_multidimensional_matrix_matrix(self):
        """Test dot product of matrix CipherText with matrix plaintext."""
        pk, sk = self._generate_keypair()

        # Test matrix * matrix
        ct_matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)  # 2x2 matrix
        pt_matrix = np.array([[5, 6], [7, 8]], dtype=np.int32)  # 2x2 matrix

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_matrix, pk])[0]

        # Dot product
        expected_result = np.dot(ct_matrix, pt_matrix)  # Should be 2x2 matrix
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType.from_obj(pt_matrix)),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, pt_matrix])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_shape == (2, 2)  # 2x2 matrix result

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_array_equal(decrypted, expected_result)

    def test_dot_multidimensional_3d_tensor_matrix(self):
        """Test dot product of 3D tensor CipherText with matrix plaintext."""
        pk, sk = self._generate_keypair()

        # Test 3D tensor * matrix
        ct_tensor = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32
        )  # 2x2x2 tensor
        pt_matrix = np.array([[9, 10], [11, 12]], dtype=np.int32)  # 2x2 matrix

        # Encrypt tensor
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_tensor),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_tensor, pk])[0]

        # Dot product
        expected_result = np.dot(ct_tensor, pt_matrix)  # Should be 2x2x2 tensor
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_tensor), TensorType.from_obj(pt_matrix)),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, pt_matrix])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_shape == expected_result.shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_array_equal(decrypted, expected_result)

    def test_dot_multidimensional_4d_tensor_vector(self):
        """Test dot product of 4D tensor CipherText with vector plaintext."""
        pk, sk = self._generate_keypair()

        # Test 4D tensor * vector
        ct_tensor = np.arange(24, dtype=np.int32).reshape(2, 3, 2, 2)  # 2x3x2x2 tensor
        pt_vector = np.array([1, 2], dtype=np.int32)  # 2-element vector

        # Encrypt tensor
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_tensor),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_tensor, pk])[0]

        # Dot product
        expected_result = np.dot(ct_tensor, pt_vector)  # Should be 2x3x2 tensor
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_tensor), TensorType.from_obj(pt_vector)),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, pt_vector])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_shape == expected_result.shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_array_equal(decrypted, expected_result)

    def test_dot_multidimensional_float_types(self):
        """Test dot product with floating point types."""
        pk, sk = self._generate_keypair()

        # Test with float64
        ct_matrix = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        pt_vector = np.array([2.0, 3.0], dtype=np.float64)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_matrix, pk])[0]

        # Dot product
        expected_result = np.dot(ct_matrix, pt_vector)
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType.from_obj(pt_vector)),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, pt_vector])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_shape == expected_result.shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected_result, rtol=1e-10)

    def test_dot_multidimensional_shape_mismatch(self):
        """Test dot product with incompatible shapes."""
        pk, _ = self._generate_keypair()

        # Test incompatible shapes
        ct_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # 2x3 matrix
        pt_vector = np.array([7, 8], dtype=np.int32)  # 2-element vector (incompatible)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_matrix, pk])[0]

        # Try dot product - should fail
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType.from_obj(pt_vector)),
            outs_info=(TensorType.from_obj(np.array([0, 0], dtype=np.int32)),),
        )
        with pytest.raises(
            ValueError, match="Shapes are not compatible for dot product"
        ):
            self.handler.execute(dot_pfunc, [ciphertext, pt_vector])

    def test_dot_multidimensional_large_tensors(self):
        """Test dot product with larger tensors."""
        pk, sk = self._generate_keypair()

        # Test with larger matrices
        ct_matrix = np.random.randint(0, 10, size=(4, 5), dtype=np.int32)
        pt_matrix = np.random.randint(0, 10, size=(5, 3), dtype=np.int32)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(ct_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [ct_matrix, pk])[0]

        # Dot product
        expected_result = np.dot(ct_matrix, pt_matrix)  # Should be 4x3 matrix
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(TensorType.from_obj(ct_matrix), TensorType.from_obj(pt_matrix)),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, pt_matrix])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_shape == (4, 3)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected_result), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected_result),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_array_equal(decrypted, expected_result)

    def test_reshape_2d_to_1d(self):
        """Test reshape operation from 2D to 1D."""
        pk, sk = self._generate_keypair()

        # Create 2D matrix
        original_matrix = np.array(
            [[1, 2, 3], [4, 5, 6]], dtype=np.int32
        )  # Shape (2, 3)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Reshape to 1D
        new_shape = (6,)
        reshape_result_shape = np.zeros(new_shape, dtype=np.int32)
        reshape_pfunc = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(original_matrix),),
            outs_info=(TensorType.from_obj(reshape_result_shape),),
            new_shape=new_shape,
        )
        result_ct = self.handler.execute(reshape_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (6,)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(reshape_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(reshape_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected_result = original_matrix.flatten()  # [1, 2, 3, 4, 5, 6]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_reshape_1d_to_3d(self):
        """Test reshape operation from 1D to 3D."""
        pk, sk = self._generate_keypair()

        # Create 1D array
        original_array = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32
        )  # Shape (8,)

        # Encrypt array
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_array), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_array),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_array, pk])[0]

        # Reshape to 3D
        new_shape = (2, 2, 2)
        reshape_result_shape = np.zeros(new_shape, dtype=np.int32)
        reshape_pfunc = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(original_array),),
            outs_info=(TensorType.from_obj(reshape_result_shape),),
            new_shape=new_shape,
        )
        result_ct = self.handler.execute(reshape_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(reshape_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(reshape_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected_result = original_array.reshape(2, 2, 2)
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_reshape_with_inferred_dimension(self):
        """Test reshape operation with -1 (inferred dimension)."""
        pk, sk = self._generate_keypair()

        # Create 2D matrix
        original_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32
        )  # Shape (3, 4)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Reshape to (2, -1) - should infer dimension as 6
        new_shape = (2, -1)
        expected_shape = (2, 6)
        reshape_result_shape = np.zeros(expected_shape, dtype=np.int32)
        reshape_pfunc = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(original_matrix),),
            outs_info=(TensorType.from_obj(reshape_result_shape),),
            new_shape=new_shape,
        )
        result_ct = self.handler.execute(reshape_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 6)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(reshape_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(reshape_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected_result = original_matrix.reshape(2, 6)
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_reshape_invalid_size(self):
        """Test reshape operation with incompatible size."""
        pk, _sk = self._generate_keypair()

        # Create 2D matrix
        original_matrix = np.array(
            [[1, 2, 3], [4, 5, 6]], dtype=np.int32
        )  # Shape (2, 3), 6 elements

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Try to reshape to incompatible size (5 elements instead of 6)
        new_shape = (5,)
        reshape_result_shape = np.zeros(new_shape, dtype=np.int32)
        reshape_pfunc = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(original_matrix),),
            outs_info=(TensorType.from_obj(reshape_result_shape),),
            new_shape=new_shape,
        )
        with pytest.raises(
            ValueError, match="Cannot reshape CipherText with 6 elements to shape"
        ):
            self.handler.execute(reshape_pfunc, [ciphertext])

    def test_reshape_float_types(self):
        """Test reshape operation with floating point types."""
        pk, sk = self._generate_keypair()

        # Create 2D float matrix
        original_matrix = np.array(
            [[1.1, 2.2], [3.3, 4.4]], dtype=np.float64
        )  # Shape (2, 2)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Reshape to 1D
        new_shape = (4,)
        reshape_result_shape = np.zeros(new_shape, dtype=np.float64)
        reshape_pfunc = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(original_matrix),),
            outs_info=(TensorType.from_obj(reshape_result_shape),),
            new_shape=new_shape,
        )
        result_ct = self.handler.execute(reshape_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == FLOAT64
        assert result_ct.semantic_shape == (4,)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(reshape_result_shape), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(reshape_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected_result = original_matrix.flatten()
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_allclose(decrypted_array, expected_result, rtol=1e-10)

    def test_transpose_2d_matrix(self):
        """Test transpose operation on 2D matrix."""
        pk, sk = self._generate_keypair()

        # Create 2D matrix
        original_matrix = np.array(
            [[1, 2, 3], [4, 5, 6]], dtype=np.int32
        )  # Shape (2, 3)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Transpose (default behavior - reverse all axes)
        expected_shape = (3, 2)
        transpose_result_shape = np.zeros(expected_shape, dtype=np.int32)
        transpose_pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_matrix),),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
            # No axes specified - default transpose
        )
        result_ct = self.handler.execute(transpose_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (3, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(transpose_result_shape),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected_result = original_matrix.T  # [[1, 4], [2, 5], [3, 6]]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_transpose_3d_tensor_with_axes(self):
        """Test transpose operation on 3D tensor with specific axes."""
        pk, sk = self._generate_keypair()

        # Create 3D tensor
        original_tensor = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32
        )  # Shape (2, 2, 2)

        # Encrypt tensor
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_tensor, pk])[0]

        # Transpose with axes (2, 0, 1) - move last axis to first
        axes = (2, 0, 1)
        expected_shape = (2, 2, 2)  # Still (2, 2, 2) but rearranged
        transpose_result_shape = np.zeros(expected_shape, dtype=np.int32)
        transpose_pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_tensor),),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
            axes=axes,
        )
        result_ct = self.handler.execute(transpose_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(transpose_result_shape),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected_result = np.transpose(original_tensor, axes)
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_transpose_1d_vector(self):
        """Test transpose operation on 1D vector (should return same vector)."""
        pk, sk = self._generate_keypair()

        # Create 1D vector
        original_vector = np.array([1, 2, 3, 4], dtype=np.int32)  # Shape (4,)

        # Encrypt vector
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_vector), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_vector),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_vector, pk])[0]

        # Transpose 1D vector
        transpose_pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_vector),),
            outs_info=(TensorType.from_obj(original_vector),),
        )
        result_ct = self.handler.execute(transpose_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (4,)

        # Decrypt and verify (should be unchanged)
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(original_vector), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_vector),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, original_vector)

    def test_transpose_scalar(self):
        """Test transpose operation on scalar (should return same scalar)."""
        pk, sk = self._generate_keypair()

        # Create scalar
        original_scalar = np.array(42, dtype=np.int32)

        # Encrypt scalar
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_scalar), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_scalar),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_scalar, pk])[0]

        # Transpose scalar
        transpose_pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_scalar),),
            outs_info=(TensorType.from_obj(original_scalar),),
        )
        result_ct = self.handler.execute(transpose_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == ()

        # Decrypt and verify (should be unchanged)
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(original_scalar), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_scalar),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        decrypted_val = (
            np.asarray(decrypted).item()
            if hasattr(decrypted, "shape") and decrypted.shape == ()
            else np.asarray(decrypted)[()]
        )
        assert decrypted_val == 42

    def test_transpose_negative_axes(self):
        """Test transpose operation with negative axes."""
        pk, sk = self._generate_keypair()

        # Create 3D tensor
        original_tensor = np.array(
            [[[1, 2, 3], [4, 5, 6]]], dtype=np.int32
        )  # Shape (1, 2, 3)

        # Encrypt tensor
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_tensor, pk])[0]

        # Transpose with negative axes (-1, -2, -3) - equivalent to (2, 1, 0)
        axes = (-1, -2, -3)
        expected_shape = (3, 2, 1)
        transpose_result_shape = np.zeros(expected_shape, dtype=np.int32)
        transpose_pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_tensor),),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
            axes=axes,
        )
        result_ct = self.handler.execute(transpose_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (3, 2, 1)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(transpose_result_shape),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected_result = np.transpose(original_tensor, (2, 1, 0))
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_transpose_invalid_axes(self):
        """Test transpose operation with invalid axes."""
        pk, _sk = self._generate_keypair()

        # Create 2D matrix
        original_matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)  # Shape (2, 2)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Try transpose with out-of-bounds axis
        axes = (0, 2)  # axis 2 is out of bounds for 2D tensor
        transpose_pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_matrix),),
            outs_info=(TensorType.from_obj(original_matrix),),
            axes=axes,
        )
        with pytest.raises(
            ValueError, match="axis 2 is out of bounds for array of dimension 2"
        ):
            self.handler.execute(transpose_pfunc, [ciphertext])

    def test_transpose_duplicate_axes(self):
        """Test transpose operation with duplicate axes."""
        pk, _sk = self._generate_keypair()

        # Create 2D matrix
        original_matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)  # Shape (2, 2)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Try transpose with duplicate axes
        axes = (0, 0)  # Duplicate axes
        transpose_pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_matrix),),
            outs_info=(TensorType.from_obj(original_matrix),),
            axes=axes,
        )
        with pytest.raises(ValueError, match="axes cannot contain duplicate values"):
            self.handler.execute(transpose_pfunc, [ciphertext])

    def test_transpose_float_types(self):
        """Test transpose operation with floating point types."""
        pk, sk = self._generate_keypair()

        # Create 2D float matrix
        original_matrix = np.array(
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64
        )  # Shape (2, 3)

        # Encrypt matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Transpose
        expected_shape = (3, 2)
        transpose_result_shape = np.zeros(expected_shape, dtype=np.float64)
        transpose_pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_matrix),),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
        )
        result_ct = self.handler.execute(transpose_pfunc, [ciphertext])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == FLOAT64
        assert result_ct.semantic_shape == (3, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(transpose_result_shape),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(transpose_result_shape),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        expected_result = original_matrix.T
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_allclose(decrypted_array, expected_result, rtol=1e-10)

    def test_gather_axis_parameter_2d_matrix_axis_1(self):
        """Test gather along axis 1 for 2D matrix."""
        pk, sk = self._generate_keypair()

        # Test with 2x3 matrix
        original_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

        # Encrypt the matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Gather along axis 1 with indices [0, 2]
        indices = np.array([0, 2], dtype=np.int32)
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType(INT32, (2, 2)),),  # result shape: (2, 2)
            axis=1,
        )

        result_ct = self.handler.execute(gather_pfunc, [ciphertext, indices])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (
            2,
            2,
        )  # (2, 2) = original.shape[0], len(indices)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType(INT32, (2, 2)),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType(INT32, (2, 2)),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]

        # Expected result: gather columns 0 and 2
        expected_result = original_matrix[:, [0, 2]]  # [[1, 3], [4, 6]]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_gather_axis_parameter_3d_tensor_axis_2(self):
        """Test gather along axis 2 for 3D tensor."""
        pk, sk = self._generate_keypair()

        # Test with 2x2x3 tensor
        original_tensor = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32
        )

        # Encrypt the tensor
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_tensor, pk])[0]

        # Gather along axis 2 with indices [1, 0]
        indices = np.array([1, 0], dtype=np.int32)
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(original_tensor),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType(INT32, (2, 2, 2)),),  # result shape: (2, 2, 2)
            axis=2,
        )

        result_ct = self.handler.execute(gather_pfunc, [ciphertext, indices])[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == (2, 2, 2)

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType(INT32, (2, 2, 2)),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType(INT32, (2, 2, 2)),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]

        # Expected result: gather from last axis at indices [1, 0]
        expected_result = original_tensor[
            :, :, [1, 0]
        ]  # Get elements 1, 0 from last axis
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_gather_axis_parameter_negative_axis(self):
        """Test gather with negative axis parameter."""
        pk, sk = self._generate_keypair()

        # Test with 2x3 matrix
        original_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

        # Encrypt the matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Gather along axis -1 (equivalent to axis 1) with indices [0, 2]
        indices = np.array([0, 2], dtype=np.int32)
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType(INT32, (2, 2)),),
            axis=-1,  # negative axis
        )

        result_ct = self.handler.execute(gather_pfunc, [ciphertext, indices])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType(INT32, (2, 2)),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType(INT32, (2, 2)),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]

        # Expected result: same as gather along axis 1
        expected_result = original_matrix[:, [0, 2]]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_gather_axis_parameter_invalid_axis(self):
        """Test gather with invalid axis parameter."""
        pk, sk = self._generate_keypair()

        # Test with 2x3 matrix
        original_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

        # Encrypt the matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Try to gather along invalid axis 2 (matrix only has axes 0, 1)
        indices = np.array([0, 1], dtype=np.int32)
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(indices),
            ),
            outs_info=(TensorType(INT32, (2, 2)),),
            axis=2,  # invalid axis
        )

        with pytest.raises(ValueError, match="Axis 2 is out of bounds"):
            self.handler.execute(gather_pfunc, [ciphertext, indices])

    def test_scatter_axis_parameter_2d_matrix_axis_1(self):
        """Test scatter along axis 1 for 2D matrix."""
        pk, sk = self._generate_keypair()

        # Test with 2x3 matrix
        original_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

        # Encrypt the matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Create updates for scattering: 2x2 matrix (shape: indices.shape + remaining dims)
        updates_data = np.array([[10, 30], [40, 60]], dtype=np.int32)
        updates_ciphertext = self.handler.execute(
            PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(updates_data), TensorType(BOOL, ())),
                outs_info=(TensorType.from_obj(updates_data),),
            ),
            [updates_data, pk],
        )[0]

        # Scatter along axis 1 with indices [0, 2]
        indices = np.array([0, 2], dtype=np.int32)
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(indices),
                TensorType.from_obj(updates_data),
            ),
            outs_info=(TensorType.from_obj(original_matrix),),
            axis=1,
        )

        result_ct = self.handler.execute(
            scatter_pfunc, [ciphertext, indices, updates_ciphertext]
        )[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == original_matrix.shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]

        # Expected result: scatter updates into columns 0 and 2
        expected_result = original_matrix.copy()
        expected_result[:, [0, 2]] = updates_data  # [[10, 2, 30], [40, 5, 60]]
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_scatter_axis_parameter_3d_tensor_axis_2(self):
        """Test scatter along axis 2 for 3D tensor."""
        pk, sk = self._generate_keypair()

        # Test with 2x2x3 tensor
        original_tensor = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32
        )

        # Encrypt the tensor
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_tensor), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_tensor, pk])[0]

        # Create updates for scattering: 2x2x2 tensor (indices.shape + remaining dims)
        updates_data = np.array(
            [[[100, 300], [400, 600]], [[700, 900], [1000, 1200]]], dtype=np.int32
        )
        updates_ciphertext = self.handler.execute(
            PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(updates_data), TensorType(BOOL, ())),
                outs_info=(TensorType.from_obj(updates_data),),
            ),
            [updates_data, pk],
        )[0]

        # Scatter along axis 2 with indices [0, 2]
        indices = np.array([0, 2], dtype=np.int32)
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_tensor),
                TensorType.from_obj(indices),
                TensorType.from_obj(updates_data),
            ),
            outs_info=(TensorType.from_obj(original_tensor),),
            axis=2,
        )

        result_ct = self.handler.execute(
            scatter_pfunc, [ciphertext, indices, updates_ciphertext]
        )[0]

        assert isinstance(result_ct, CipherText)
        assert result_ct.semantic_dtype == INT32
        assert result_ct.semantic_shape == original_tensor.shape

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(original_tensor),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(original_tensor),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]

        # Expected result: scatter updates into positions [0, 2] along axis 2
        expected_result = original_tensor.copy()
        expected_result[:, :, [0, 2]] = updates_data
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_scatter_axis_parameter_negative_axis(self):
        """Test scatter with negative axis parameter."""
        pk, sk = self._generate_keypair()

        # Test with 2x3 matrix
        original_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

        # Encrypt the matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Create updates
        updates_data = np.array([[10, 30], [40, 60]], dtype=np.int32)
        updates_ciphertext = self.handler.execute(
            PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(updates_data), TensorType(BOOL, ())),
                outs_info=(TensorType.from_obj(updates_data),),
            ),
            [updates_data, pk],
        )[0]

        # Scatter along axis -1 (equivalent to axis 1) with indices [0, 2]
        indices = np.array([0, 2], dtype=np.int32)
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(indices),
                TensorType.from_obj(updates_data),
            ),
            outs_info=(TensorType.from_obj(original_matrix),),
            axis=-1,  # negative axis
        )

        result_ct = self.handler.execute(
            scatter_pfunc, [ciphertext, indices, updates_ciphertext]
        )[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType(BOOL, ()),
            ),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]

        # Expected result: same as scatter along axis 1
        expected_result = original_matrix.copy()
        expected_result[:, [0, 2]] = updates_data
        decrypted_array = np.asarray(decrypted)
        np.testing.assert_array_equal(decrypted_array, expected_result)

    def test_scatter_axis_parameter_invalid_axis(self):
        """Test scatter with invalid axis parameter."""
        pk, sk = self._generate_keypair()

        # Test with 2x3 matrix
        original_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

        # Encrypt the matrix
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_matrix), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_matrix),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_matrix, pk])[0]

        # Create updates
        updates_data = np.array([[10, 30], [40, 60]], dtype=np.int32)
        updates_ciphertext = self.handler.execute(
            PFunction(
                fn_type="phe.encrypt",
                ins_info=(TensorType.from_obj(updates_data), TensorType(BOOL, ())),
                outs_info=(TensorType.from_obj(updates_data),),
            ),
            [updates_data, pk],
        )[0]

        # Try to scatter along invalid axis 2
        indices = np.array([0, 1], dtype=np.int32)
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_matrix),
                TensorType.from_obj(indices),
                TensorType.from_obj(updates_data),
            ),
            outs_info=(TensorType.from_obj(original_matrix),),
            axis=2,  # invalid axis
        )

        with pytest.raises(ValueError, match="Axis 2 is out of bounds"):
            self.handler.execute(
                scatter_pfunc, [ciphertext, indices, updates_ciphertext]
            )


class TestPHEHandlerNegativeNumbers:
    """Test cases for PHEHandler with negative numbers, zero, and positive numbers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = PHEHandler()
        self.scheme = "paillier"
        # Use smaller key size for faster testing
        self.key_size = 1024

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

    def test_encrypt_decrypt_negative_float(self):
        """Test encrypt/decrypt with negative, zero, and positive floats."""
        pk, sk = self._generate_keypair()

        # Test with mixed float values: negative, zero, positive
        test_values = [-3.14, 0.0, 2.718]
        plaintext = np.array(test_values, dtype=np.float64)

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [plaintext, pk])[0]

        # Decrypt
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [ciphertext, sk])[0]

        # Verify results with tolerance
        np.testing.assert_allclose(decrypted, test_values, rtol=1e-6)

    def test_encrypt_decrypt_negative_int(self):
        """Test encrypt/decrypt with negative, zero, and positive integers."""
        pk, sk = self._generate_keypair()

        # Test with mixed int values: negative, zero, positive
        test_values = [-42, 0, 15]
        plaintext = np.array(test_values, dtype=np.int32)

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [plaintext, pk])[0]

        # Decrypt
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(plaintext), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(plaintext),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [ciphertext, sk])[0]

        # Verify results
        np.testing.assert_array_equal(decrypted, test_values)

    def test_add_ciphertext_ciphertext_negative(self):
        """Test CipherText + CipherText with negative numbers."""
        pk, sk = self._generate_keypair()

        # Test values with negatives, zero, positives
        values1 = np.array([-5.5, 0.0, 3.3], dtype=np.float64)
        values2 = np.array([2.2, -1.1, 0.0], dtype=np.float64)
        expected = values1 + values2  # [-3.3, -1.1, 3.3]

        # Encrypt both arrays
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(values1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(values1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc, [values1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc, [values2, pk])[0]

        # Add ciphertexts
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(TensorType.from_obj(values1), TensorType.from_obj(values1)),
            outs_info=(TensorType.from_obj(values1),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext1, ciphertext2])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(values1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(values1),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected, rtol=1e-6)

    def test_add_ciphertext_plaintext_negative(self):
        """Test CipherText + plaintext with negative numbers."""
        pk, sk = self._generate_keypair()

        # Test values with negatives, zero, positives
        cipher_values = np.array([-2.5, 0.0, 1.5], dtype=np.float64)
        plain_values = np.array([1.0, -3.0, 0.0], dtype=np.float64)
        expected = cipher_values + plain_values  # [-1.5, -3.0, 1.5]

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(cipher_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(cipher_values),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [cipher_values, pk])[0]

        # Add ciphertext + plaintext
        add_pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(
                TensorType.from_obj(cipher_values),
                TensorType.from_obj(plain_values),
            ),
            outs_info=(TensorType.from_obj(cipher_values),),
        )
        result_ct = self.handler.execute(add_pfunc, [ciphertext, plain_values])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(cipher_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(cipher_values),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected, rtol=1e-6)

    def test_mul_ciphertext_plaintext_negative(self):
        """Test CipherText * plaintext with negative numbers."""
        pk, sk = self._generate_keypair()

        # Test values with negatives, zero, positives
        cipher_values = np.array([-4.0, 0.0, 2.0], dtype=np.float64)
        plain_values = np.array([2.0, -3.0, 0.0], dtype=np.float64)
        expected = cipher_values * plain_values  # [-8.0, 0.0, 0.0]

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(cipher_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(cipher_values),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [cipher_values, pk])[0]

        # Multiply ciphertext * plaintext
        mul_pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(
                TensorType.from_obj(cipher_values),
                TensorType.from_obj(plain_values),
            ),
            outs_info=(TensorType.from_obj(cipher_values),),
        )
        result_ct = self.handler.execute(mul_pfunc, [ciphertext, plain_values])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(cipher_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(cipher_values),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected, rtol=1e-6)

    def test_dot_negative_values(self):
        """Test dot product with negative numbers."""
        pk, sk = self._generate_keypair()

        # Test values: ciphertext vector and plaintext vector with negatives, zero, positives
        cipher_vector = np.array([-2.0, 0.0, 3.0], dtype=np.float64)
        plain_vector = np.array([1.5, -2.0, 0.0], dtype=np.float64)
        expected = np.dot(
            cipher_vector, plain_vector
        )  # -2.0*1.5 + 0.0*(-2.0) + 3.0*0.0 = -3.0

        # Encrypt vector
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(cipher_vector), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(cipher_vector),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [cipher_vector, pk])[0]

        # Dot product
        dot_pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(
                TensorType.from_obj(cipher_vector),
                TensorType.from_obj(plain_vector),
            ),
            outs_info=(TensorType(FLOAT64, ()),),
        )
        result_ct = self.handler.execute(dot_pfunc, [ciphertext, plain_vector])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType(FLOAT64, ()), TensorType(BOOL, ())),
            outs_info=(TensorType(FLOAT64, ()),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected, rtol=1e-6)

    def test_gather_negative_values(self):
        """Test gather operation with negative numbers."""
        pk, sk = self._generate_keypair()

        # Test values with negatives, zero, positives
        test_values = np.array([-1.5, 0.0, 2.5, -3.0], dtype=np.float64)
        indices = np.array([0, 2, 1], dtype=np.int32)
        expected = test_values[indices]  # [-1.5, 2.5, 0.0]

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(test_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(test_values),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [test_values, pk])[0]

        # Gather
        gather_pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(TensorType.from_obj(test_values), TensorType.from_obj(indices)),
            outs_info=(TensorType.from_obj(expected),),
        )
        result_ct = self.handler.execute(gather_pfunc, [ciphertext, indices])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected, rtol=1e-6)

    def test_scatter_negative_values(self):
        """Test scatter operation with negative numbers."""
        pk, sk = self._generate_keypair()

        # Original array with negatives, zero, positives
        original_values = np.array([-1.0, 0.0, 2.0, -3.0], dtype=np.float64)
        # Values to scatter (also with negatives, zero, positives)
        scatter_values = np.array([5.0, -2.5], dtype=np.float64)
        indices = np.array([1, 3], dtype=np.int32)

        # Expected result
        expected = original_values.copy()
        expected[indices] = scatter_values  # [-1.0, 5.0, 2.0, -2.5]

        # Encrypt both arrays
        encrypt_pfunc_orig = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_values),),
        )
        original_ct = self.handler.execute(encrypt_pfunc_orig, [original_values, pk])[0]

        encrypt_pfunc_scatter = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(scatter_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(scatter_values),),
        )
        scatter_ct = self.handler.execute(encrypt_pfunc_scatter, [scatter_values, pk])[
            0
        ]

        # Scatter
        scatter_pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(
                TensorType.from_obj(original_values),
                TensorType.from_obj(indices),
                TensorType.from_obj(scatter_values),
            ),
            outs_info=(TensorType.from_obj(original_values),),
        )
        result_ct = self.handler.execute(
            scatter_pfunc, [original_ct, indices, scatter_ct]
        )[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(original_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_values),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected, rtol=1e-6)

    def test_concat_negative_values(self):
        """Test concat operation with negative numbers."""
        pk, sk = self._generate_keypair()

        # Test arrays with negatives, zero, positives
        values1 = np.array([-1.5, 0.0], dtype=np.float64)
        values2 = np.array([2.5, -3.0], dtype=np.float64)
        expected = np.concatenate([values1, values2])  # [-1.5, 0.0, 2.5, -3.0]

        # Encrypt both arrays
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(values1), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(values1),),
        )
        ciphertext1 = self.handler.execute(encrypt_pfunc, [values1, pk])[0]
        ciphertext2 = self.handler.execute(encrypt_pfunc, [values2, pk])[0]

        # Concat
        concat_pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=(TensorType.from_obj(values1), TensorType.from_obj(values2)),
            outs_info=(TensorType.from_obj(expected),),
        )
        result_ct = self.handler.execute(concat_pfunc, [ciphertext1, ciphertext2])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected, rtol=1e-6)

    def test_reshape_negative_values(self):
        """Test reshape operation with negative numbers."""
        pk, sk = self._generate_keypair()

        # Test array with negatives, zero, positives (2x2 -> 4x1)
        original_values = np.array([[-1.5, 0.0], [2.5, -3.0]], dtype=np.float64)
        expected = original_values.reshape(4)  # [-1.5, 0.0, 2.5, -3.0]

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_values),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_values, pk])[0]

        # Reshape
        reshape_pfunc = PFunction(
            fn_type="phe.reshape",
            ins_info=(TensorType.from_obj(original_values),),
            outs_info=(TensorType.from_obj(expected),),
            new_shape=(4,),
        )
        result_ct = self.handler.execute(reshape_pfunc, [ciphertext])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected, rtol=1e-6)

    def test_transpose_negative_values(self):
        """Test transpose operation with negative numbers."""
        pk, sk = self._generate_keypair()

        # Test array with negatives, zero, positives (2x2 matrix)
        original_values = np.array([[-1.5, 0.0], [2.5, -3.0]], dtype=np.float64)
        expected = original_values.T  # [[-1.5, 2.5], [0.0, -3.0]]

        # Encrypt
        encrypt_pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(TensorType.from_obj(original_values), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(original_values),),
        )
        ciphertext = self.handler.execute(encrypt_pfunc, [original_values, pk])[0]

        # Transpose
        transpose_pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(TensorType.from_obj(original_values),),
            outs_info=(TensorType.from_obj(expected),),
        )
        result_ct = self.handler.execute(transpose_pfunc, [ciphertext])[0]

        # Decrypt and verify
        decrypt_pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(TensorType.from_obj(expected), TensorType(BOOL, ())),
            outs_info=(TensorType.from_obj(expected),),
        )
        decrypted = self.handler.execute(decrypt_pfunc, [result_ct, sk])[0]
        np.testing.assert_allclose(decrypted, expected, rtol=1e-6)
