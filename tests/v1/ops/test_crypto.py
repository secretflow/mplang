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

from __future__ import annotations

import numpy as np
import pytest

from mplang.v1.core import UINT8
from mplang.v1.core.pfunc import PFunction
from mplang.v1.core.tensor import TensorType
from mplang.v1.ops import crypto
from mplang.v1.ops.base import get_registry, is_feop
from tests.v1.ops.dummy import DummyTensor


def test_crypto_module_registration():
    """Test that crypto module is properly registered."""
    reg = get_registry()
    assert reg.has_module("crypto")
    crypto_mod = reg.get_module("crypto")
    assert crypto_mod is not None


def test_enc_is_feoperation():
    """Test that enc is a proper FeOperation."""
    assert is_feop(crypto.enc)
    pfunc, args, _out_tree = crypto.enc(
        DummyTensor(UINT8, (10,)), DummyTensor(UINT8, (16,))
    )

    assert isinstance(pfunc, PFunction)
    assert pfunc.fn_type == "crypto.enc"
    assert len(args) == 2
    assert len(pfunc.ins_info) == 2
    assert len(pfunc.outs_info) == 1


def test_enc_default_algorithm():
    """Test enc with default algorithm (aes-ctr)."""
    plaintext = DummyTensor(UINT8, (8,))
    key = DummyTensor(UINT8, (16,))

    pfunc, _args, _out_tree = crypto.enc(plaintext, key)

    # Default should be aes-ctr
    assert pfunc.attrs["algo"] == "aes-ctr"
    # 16-byte overhead for aes-ctr
    expected_output = TensorType(UINT8, (24,))
    assert pfunc.outs_info[0] == expected_output


def test_enc_dynamic_length():
    """Test enc with dynamic length input."""
    plaintext = DummyTensor(UINT8, (-1,))  # Dynamic length
    key = DummyTensor(UINT8, (32,))

    pfunc, _args, _out_tree = crypto.enc(plaintext, key, algo="aes-gcm")

    # Should return dynamic length when input length is unknown
    expected_output = TensorType(UINT8, (-1,))
    assert pfunc.outs_info[0] == expected_output
    assert pfunc.attrs["algo"] == "aes-gcm"


def test_enc_unknown_algorithm():
    """Test enc with unknown algorithm."""
    plaintext = DummyTensor(UINT8, (38,))
    key = DummyTensor(UINT8, (32,))

    pfunc, _args, _out_tree = crypto.enc(plaintext, key, algo="unknown")

    # Unknown algorithm should result in dynamic length
    expected_output = TensorType(UINT8, (-1,))
    assert pfunc.outs_info[0] == expected_output
    assert pfunc.attrs["algo"] == "unknown"


def test_enc_type_validation():
    """Test that enc validates input types."""
    key = DummyTensor(UINT8, (16,))

    # Test non-UINT8 plaintext
    with pytest.raises(TypeError, match="enc expects UINT8 plaintext"):
        crypto.enc(DummyTensor(np.float32, (20,)), key)

    # Test multi-dimensional plaintext
    with pytest.raises(TypeError, match="enc expects 1-D plaintext"):
        crypto.enc(DummyTensor(UINT8, (20, 5)), key)


def test_enc_registration():
    """Test that enc operation is properly registered."""
    reg = get_registry()

    # Check that enc is registered under crypto module
    enc_op = reg.get_op("crypto", "enc")
    assert enc_op is crypto.enc

    # Check that it appears in module listings
    crypto_ops = reg.list_ops("crypto")
    assert ("crypto", "enc") in crypto_ops


@pytest.mark.parametrize(
    "algo,overhead,test_len,key_len",
    [
        ("aes-ctr", 16, 60, 32),  # General test case
        ("aes-gcm", 28, 60, 32),  # General test case
        ("sm4-gcm", 28, 60, 16),  # General test case for SM4
        ("unknown", -1, 60, 32),  # Unknown algorithm
    ],
)
def test_enc_algorithm_overhead(algo: str, overhead: int, test_len: int, key_len: int):
    """Test enc with different algorithms and their overheads."""
    plaintext = DummyTensor(UINT8, (test_len,))
    key = DummyTensor(UINT8, (key_len,))

    pfunc, args, _out_tree = crypto.enc(plaintext, key, algo=algo)

    if overhead >= 0:
        expected_output_len = test_len + overhead
        expected_output = TensorType(UINT8, (expected_output_len,))
        assert pfunc.outs_info[0] == expected_output
    else:
        # Unknown algorithm should produce dynamic length
        expected_output = TensorType(UINT8, (-1,))
        assert pfunc.outs_info[0] == expected_output

    # Verify algorithm is stored correctly
    assert pfunc.attrs["algo"] == algo
    # Verify arguments are passed correctly
    assert args == [plaintext, key]


def test_dec_is_feoperation():
    """Test that dec is a proper FeOperation."""
    assert is_feop(crypto.dec)
    pfunc, args, _out_tree = crypto.dec(
        DummyTensor(UINT8, (22,)), DummyTensor(UINT8, (16,))
    )

    assert isinstance(pfunc, PFunction)
    assert pfunc.fn_type == "crypto.dec"
    assert len(args) == 2
    assert len(pfunc.ins_info) == 2
    assert len(pfunc.outs_info) == 1


def test_dec_default_algorithm():
    """Test dec with default algorithm (aes-ctr)."""
    ciphertext = DummyTensor(UINT8, (24,))
    key = DummyTensor(UINT8, (16,))

    pfunc, _args, _out_tree = crypto.dec(ciphertext, key)

    # Default should be aes-ctr
    assert pfunc.attrs["algo"] == "aes-ctr"
    # 16-byte overhead for aes-ctr
    expected_output = TensorType(UINT8, (8,))
    assert pfunc.outs_info[0] == expected_output


def test_dec_dynamic_length():
    """Test dec with dynamic length input."""
    ciphertext = DummyTensor(UINT8, (-1,))  # Dynamic length
    key = DummyTensor(UINT8, (32,))

    pfunc, _args, _out_tree = crypto.dec(ciphertext, key, algo="aes-gcm")

    # Should return dynamic length when input length is unknown
    expected_output = TensorType(UINT8, (-1,))
    assert pfunc.outs_info[0] == expected_output
    assert pfunc.attrs["algo"] == "aes-gcm"


def test_dec_unknown_algorithm():
    """Test dec with unknown algorithm."""
    ciphertext = DummyTensor(UINT8, (50,))
    key = DummyTensor(UINT8, (32,))

    pfunc, _args, _out_tree = crypto.dec(ciphertext, key, algo="unknown")

    # Unknown algorithm should result in dynamic length
    expected_output = TensorType(UINT8, (-1,))
    assert pfunc.outs_info[0] == expected_output
    assert pfunc.attrs["algo"] == "unknown"


def test_dec_min_length_validation():
    """Test that dec validates minimum ciphertext length."""
    ciphertext = DummyTensor(UINT8, (8,))  # Too short for AES-GCM (needs at least 28)
    key = DummyTensor(UINT8, (32,))

    with pytest.raises(
        TypeError, match="dec expects ciphertext with at least 28 bytes"
    ):
        crypto.dec(ciphertext, key, algo="aes-gcm")


def test_dec_type_validation():
    """Test that dec validates input types."""
    key = DummyTensor(UINT8, (16,))

    # Test non-UINT8 ciphertext
    with pytest.raises(TypeError, match="dec expects UINT8 ciphertext"):
        crypto.dec(DummyTensor(np.float32, (20,)), key)

    # Test multi-dimensional ciphertext
    with pytest.raises(TypeError, match="dec expects 1-D ciphertext"):
        crypto.dec(DummyTensor(UINT8, (20, 5)), key)


def test_dec_roundtrip_compatibility():
    """Test that enc/dec roundtrip works with same parameters."""
    plaintext = DummyTensor(UINT8, (100,))

    # Test all algorithms with appropriate key sizes
    for algo in ["aes-ctr", "aes-gcm", "sm4-gcm"]:
        key_len = 16 if algo == "sm4-gcm" else 32
        key = DummyTensor(UINT8, (key_len,))
        # Encryption
        enc_pfunc, _enc_args, _enc_out_tree = crypto.enc(plaintext, key, algo=algo)

        # Decryption (using ciphertext length from enc output)
        ciphertext_len = enc_pfunc.outs_info[0].shape[0]
        ciphertext = DummyTensor(UINT8, (ciphertext_len,))

        dec_pfunc, _dec_args, _dec_out_tree = crypto.dec(ciphertext, key, algo=algo)

        # Check that decryption output matches plaintext length
        assert dec_pfunc.outs_info[0].shape[0] == plaintext.mptype._type.shape[0]

        # Check algorithm consistency
        assert enc_pfunc.attrs["algo"] == dec_pfunc.attrs["algo"] == algo


def test_enc_dec_symmetry():
    """Test symmetry between enc and dec operations."""
    plaintext_len = 64
    key_len = 32

    for algo in ["aes-ctr", "aes-gcm", "sm4-gcm"]:
        # Create test objects
        plaintext = DummyTensor(UINT8, (plaintext_len,))
        key = DummyTensor(UINT8, (key_len,))

        # Encryption
        enc_pfunc, enc_args, _enc_out_tree = crypto.enc(plaintext, key, algo=algo)

        # Create ciphertext with exact output size
        ciphertext_len = enc_pfunc.outs_info[0].shape[0]
        ciphertext = DummyTensor(UINT8, (ciphertext_len,))

        # Decryption
        dec_pfunc, dec_args, _dec_out_tree = crypto.dec(ciphertext, key, algo=algo)

        # Verify input/output symmetry
        # enc: plaintext_len + overhead = ciphertext_len
        # dec: ciphertext_len - overhead = plaintext_len
        overhead = ciphertext_len - plaintext_len
        assert overhead > 0

        # Verify enc inputs match dec inputs (same objects)
        assert enc_args[0] is plaintext  # enc plaintext
        assert enc_args[1] is key  # enc key
        assert dec_args[0] is ciphertext  # dec ciphertext
        assert dec_args[1] is key  # dec key (same key)

        # Verify function types
        assert enc_pfunc.fn_type == "crypto.enc"
        assert dec_pfunc.fn_type == "crypto.dec"

        # Verify algorithm consistency
        assert enc_pfunc.attrs["algo"] == dec_pfunc.attrs["algo"] == algo


def test_enc_dec_size_math():
    """Test that enc/dec size calculations are mathematically correct."""
    test_cases = [
        (16, 32),  # small plaintext, 32-byte key
        (64, 32),  # medium plaintext, 32-byte key
        (128, 64),  # larger plaintext, 64-byte key
    ]

    for plaintext_len, key_len in test_cases:
        for algo in ["aes-ctr", "aes-gcm", "sm4-gcm"]:
            plaintext = DummyTensor(UINT8, (plaintext_len,))
            key = DummyTensor(UINT8, (key_len,))

            # Encryption: plaintext -> ciphertext
            enc_pfunc, _, _ = crypto.enc(plaintext, key, algo=algo)
            ciphertext_len = enc_pfunc.outs_info[0].shape[0]

            # Decryption: ciphertext -> plaintext
            ciphertext = DummyTensor(UINT8, (ciphertext_len,))
            dec_pfunc, _, _ = crypto.dec(ciphertext, key, algo=algo)
            recovered_len = dec_pfunc.outs_info[0].shape[0]

            # Size math should be exact
            assert ciphertext_len > plaintext_len  # encryption adds overhead
            assert recovered_len == plaintext_len  # decryption recovers original

            # Calculate overhead
            overhead = ciphertext_len - plaintext_len
            expected_overhead = 16 if algo == "aes-ctr" else 28
            assert overhead == expected_overhead


def test_dec_registration():
    """Test that dec operation is properly registered."""
    reg = get_registry()

    # Check that dec is registered under crypto module
    dec_op = reg.get_op("crypto", "dec")
    assert dec_op is crypto.dec

    # Check that it appears in module listings
    crypto_ops = reg.list_ops("crypto")
    assert ("crypto", "dec") in crypto_ops


@pytest.mark.parametrize(
    "algo,overhead,ciphertext_len,key_len",
    [
        ("aes-ctr", 16, 72, 32),  # General test case: 60+16=76
        ("aes-gcm", 28, 88, 32),  # General test case: 60+28=88
        ("sm4-gcm", 28, 88, 16),  # General test case for SM4
        ("unknown", -1, 60, 32),  # Unknown algorithm
    ],
)
def test_dec_algorithm_overhead(
    algo: str, overhead: int, ciphertext_len: int, key_len: int
):
    """Test dec with different algorithms and their overheads."""
    ciphertext = DummyTensor(UINT8, (ciphertext_len,))
    key = DummyTensor(UINT8, (key_len,))

    pfunc, args, _out_tree = crypto.dec(ciphertext, key, algo=algo)

    if overhead >= 0:
        expected_output_len = ciphertext_len - overhead
        expected_output = TensorType(UINT8, (expected_output_len,))
        assert pfunc.outs_info[0] == expected_output
    else:
        # Unknown algorithm should produce dynamic length
        expected_output = TensorType(UINT8, (-1,))
        assert pfunc.outs_info[0] == expected_output

    # Verify algorithm is stored correctly
    assert pfunc.attrs["algo"] == algo
    # Verify arguments are passed correctly
    assert args == [ciphertext, key]
