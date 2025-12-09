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

"""
Comprehensive serialization/deserialization (roundtrip) tests for all Value types.

This ensures that all Value subclasses can be properly serialized and deserialized
without data loss, which is critical for distributed communication.
"""

from typing import cast

import numpy as np
import pytest

from mplang.v1.core.dtypes import FLOAT32, INT32
from mplang.v1.kernels.phe import CipherText, PrivateKey, PublicKey
from mplang.v1.kernels.value import (
    BytesBlob,
    TableValue,
    TensorValue,
    decode_value,
    encode_value,
)


class TestTensorValueSerde:
    """Test TensorValue serialization roundtrip."""

    def test_1d_float32(self):
        """Test 1D float32 array."""
        arr = np.array([1.0, 2.5, 3.7], dtype=np.float32)
        val = TensorValue(arr)
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, TensorValue)
        np.testing.assert_array_equal(decoded.to_numpy(), arr)
        assert decoded.to_numpy().dtype == arr.dtype

    def test_2d_int64(self):
        """Test 2D int64 array."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        val = TensorValue(arr)
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, TensorValue)
        np.testing.assert_array_equal(decoded.to_numpy(), arr)
        assert decoded.to_numpy().dtype == arr.dtype

    def test_3d_uint8(self):
        """Test 3D uint8 array."""
        arr = np.random.randint(0, 256, size=(2, 3, 4), dtype=np.uint8)
        val = TensorValue(arr)
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, TensorValue)
        np.testing.assert_array_equal(decoded.to_numpy(), arr)

    def test_scalar(self):
        """Test scalar tensor."""
        arr = np.array(42, dtype=np.int32)
        val = TensorValue(arr)
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, TensorValue)
        np.testing.assert_array_equal(decoded.to_numpy(), arr)

    def test_empty_array(self):
        """Test empty array."""
        arr = np.array([], dtype=np.float64)
        val = TensorValue(arr)
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, TensorValue)
        np.testing.assert_array_equal(decoded.to_numpy(), arr)

    def test_large_array(self):
        """Test large array (1MB)."""
        arr = np.random.randn(128, 1024).astype(np.float32)
        val = TensorValue(arr)
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, TensorValue)
        np.testing.assert_allclose(decoded.to_numpy(), arr)


class TestTableValueSerde:
    """Test TableValue serialization roundtrip."""

    @pytest.fixture
    def check_arrow(self):
        """Skip if pyarrow not available."""
        pytest.importorskip("pyarrow")

    def test_simple_dataframe(self, check_arrow):
        """Test simple pandas DataFrame."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
        val = TableValue(df)
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, TableValue)
        decoded_table = cast(TableValue, decoded)
        result_df = decoded_table.to_pandas()  # type: ignore[attr-defined]
        pd.testing.assert_frame_equal(result_df, df)

    def test_numeric_types(self, check_arrow):
        """Test various numeric types."""
        import pandas as pd

        df = pd.DataFrame({
            "int32": np.array([1, 2, 3], dtype=np.int32),
            "int64": np.array([10, 20, 30], dtype=np.int64),
            "float32": np.array([1.5, 2.5, 3.5], dtype=np.float32),
            "float64": np.array([10.5, 20.5, 30.5], dtype=np.float64),
        })
        val = TableValue(df)
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, TableValue)
        decoded_table = cast(TableValue, decoded)
        result_df = decoded_table.to_pandas()  # type: ignore[attr-defined]
        pd.testing.assert_frame_equal(result_df, df)

    def test_empty_dataframe(self, check_arrow):
        """Test empty DataFrame."""
        import pandas as pd

        df = pd.DataFrame()
        val = TableValue(df)
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, TableValue)
        decoded_table = cast(TableValue, decoded)
        result_df = decoded_table.to_pandas()  # type: ignore[attr-defined]
        assert len(result_df) == 0


class TestBytesBlobSerde:
    """Test BytesBlob serialization roundtrip."""

    def test_simple_bytes(self):
        """Test simple byte string."""
        data = b"hello world"
        val = BytesBlob(data)
        wire = encode_value(val)
        decoded = decode_value(wire)
        assert isinstance(decoded, BytesBlob)
        assert decoded.to_proto().payload == data

    def test_binary_data(self):
        """Test binary data with null bytes."""
        data = b"\x00\x01\x02\xff\xfe\xfd"
        val = BytesBlob(data)
        wire = encode_value(val)
        decoded = decode_value(wire)
        assert decoded.to_proto().payload == data

    def test_large_blob(self):
        """Test large binary blob (1MB)."""
        data = bytes(np.random.randint(0, 256, size=1024 * 1024, dtype=np.uint8))
        val = BytesBlob(data)
        wire = encode_value(val)
        decoded = decode_value(wire)
        assert decoded.to_proto().payload == data


class TestPHEValuesSerde:
    """Test PHE Value types serialization roundtrip."""

    def test_public_key(self):
        """Test PHE PublicKey serialization."""
        # Create a mock key_data (in real use, this comes from lightPHE)
        key_data = {"n": 12345, "g": 67890}
        val = PublicKey(
            key_data=key_data,
            scheme="Paillier",
            key_size=2048,
            max_value=2**32,
            fxp_bits=12,
            modulus=99999,
        )
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, PublicKey)
        assert decoded.scheme == "Paillier"
        assert decoded.key_size == 2048
        assert decoded.max_value == 2**32
        assert decoded.fxp_bits == 12
        assert decoded.modulus == 99999
        assert decoded.key_data == key_data

    def test_private_key(self):
        """Test PHE PrivateKey serialization."""
        sk_data = {"lambda": 111, "mu": 222}
        pk_data = {"n": 333, "g": 444}
        val = PrivateKey(
            sk_data=sk_data,
            pk_data=pk_data,
            scheme="Paillier",
            key_size=2048,
            max_value=2**32,
            fxp_bits=12,
            modulus=99999,
        )
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, PrivateKey)
        assert decoded.scheme == "Paillier"
        assert decoded.sk_data == sk_data
        assert decoded.pk_data == pk_data

    def test_ciphertext(self):
        """Test PHE CipherText serialization with real encryption."""
        from lightphe import LightPHE

        # Use real PHE encryption to create valid Ciphertext objects
        phe = LightPHE(algorithm_name="Paillier", key_size=512)  # Small key for speed

        # Encrypt some values to get real Ciphertext objects
        plaintext = np.array([10, 20], dtype=np.int32)
        ct_data = [phe.encrypt(int(val)) for val in plaintext]

        # Extract public key data
        pk_data = phe.cs.keys.get("public_key", {})
        modulus = phe.cs.plaintext_modulo

        val = CipherText(
            ct_data=ct_data,
            semantic_dtype=INT32,
            semantic_shape=(2,),
            scheme="Paillier",
            key_size=512,
            pk_data=pk_data,
            max_value=2**32,
            fxp_bits=12,
            modulus=modulus,
        )
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, CipherText)
        assert decoded.scheme == "Paillier"
        assert decoded.semantic_dtype == INT32
        assert decoded.semantic_shape == (2,)
        # Verify ct_data length and types
        assert len(decoded.ct_data) == 2
        from lightphe.models.Ciphertext import Ciphertext

        assert all(isinstance(ct, Ciphertext) for ct in decoded.ct_data)
        assert decoded.pk_data == pk_data

    def test_ciphertext_float(self):
        """Test CipherText with float semantic type using real encryption."""
        from lightphe import LightPHE

        # Use real PHE encryption to create valid Ciphertext objects
        phe = LightPHE(algorithm_name="Paillier", key_size=512)  # Small key for speed

        # Encrypt some values
        plaintext = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        # Convert to int for encryption (mimicking range encoding)
        ct_data = [phe.encrypt(int(val * 1000)) for val in plaintext]

        modulus = phe.cs.plaintext_modulo

        val = CipherText(
            ct_data=ct_data,
            semantic_dtype=FLOAT32,
            semantic_shape=(3,),
            scheme="Paillier",
            key_size=512,
            max_value=2**32,
            fxp_bits=12,
            modulus=modulus,
        )
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, CipherText)
        assert decoded.semantic_dtype == FLOAT32
        assert decoded.semantic_shape == (3,)
        # Verify ct_data
        assert len(decoded.ct_data) == 3
        from lightphe.models.Ciphertext import Ciphertext

        assert all(isinstance(ct, Ciphertext) for ct in decoded.ct_data)


class TestSPUValueSerde:
    """Test SPU SpuValue serialization roundtrip."""

    def test_spuvalue_basic(self):
        """Test SpuValue with mock share object."""
        pytest.importorskip("spu")
        import spu.libspu as libspu

        from mplang.v1.kernels.spu import SpuValue

        # Create a mock share (in real use, this comes from SPU makeshares)
        # SpuValue now expects share.meta (bytes) and share.share_chunks (list[bytes])
        class MockShare:
            def __init__(self, meta, chunks):
                self.meta = meta
                self.share_chunks = chunks

        share = MockShare(meta=b"mock_meta", chunks=[b"chunk1", b"chunk2"])
        typed_share = cast(libspu.Share, share)
        # Use DType instead of libspu.DataType
        from mplang.v1.core.dtypes import INT32

        val = SpuValue(
            shape=(2, 3),
            dtype=INT32,
            vtype=libspu.Visibility.VIS_SECRET,
            share=typed_share,
        )
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert isinstance(decoded, SpuValue)
        assert decoded.shape == (2, 3)
        assert decoded.dtype == INT32
        assert decoded.vtype == libspu.Visibility.VIS_SECRET
        # The decoded share is a real libspu.Share object now
        assert decoded.share.meta == b"mock_meta"
        assert decoded.share.share_chunks == [b"chunk1", b"chunk2"]


class TestValueEnvelopeProperties:
    """Test general Value envelope properties."""

    def test_kind_preserved(self):
        """Test that KIND is preserved in envelope."""
        val = BytesBlob(b"test")
        wire = encode_value(val)
        decoded = decode_value(wire)

        assert type(decoded).__name__ == "BytesBlob"
        assert decoded.KIND == "mplang.demo.BytesBlob"

    def test_wire_size_hint(self):
        """Test estimated_wire_size hint."""
        arr = np.zeros((100, 100), dtype=np.float32)
        val = TensorValue(arr)

        # estimated_wire_size may return None or an estimate
        size_hint = val.estimated_wire_size()
        if size_hint is not None:
            assert size_hint > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
