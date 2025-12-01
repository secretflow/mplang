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

"""Unit tests for mplang.kernels.value module."""

import numpy as np
import pytest

from mplang.v1.kernels.value import (
    BytesBlob,
    TableValue,
    TensorValue,
    Value,
    ValueDecodeError,
    ValueError,
    decode_value,
    encode_value,
    is_value_envelope,
    list_value_kinds,
    register_value,
)


class TestBytesBlob:
    """Test BytesBlob Value type."""

    def test_roundtrip(self):
        """Test encode/decode roundtrip for BytesBlob."""
        data = b"hello world"
        blob = BytesBlob(data)
        wire = encode_value(blob)

        assert is_value_envelope(wire)
        decoded = decode_value(wire)

        assert isinstance(decoded, BytesBlob)
        assert decoded.to_proto().payload == data

    def test_empty_bytes(self):
        """Test BytesBlob with empty bytes."""
        blob = BytesBlob(b"")
        wire = encode_value(blob)
        decoded = decode_value(wire)

        assert isinstance(decoded, BytesBlob)
        assert decoded.to_proto().payload == b""


class TestTensorValue:
    """Test TensorValue (ndarray) serialization."""

    def test_roundtrip_1d(self):
        """Test encode/decode roundtrip for 1D array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = TensorValue(arr)
        wire = encode_value(tensor)

        decoded = decode_value(wire)
        assert isinstance(decoded, TensorValue)
        np.testing.assert_array_equal(decoded.to_numpy(), arr)

    def test_roundtrip_2d(self):
        """Test encode/decode roundtrip for 2D array."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        tensor = TensorValue(arr)
        wire = encode_value(tensor)

        decoded = decode_value(wire)
        assert isinstance(decoded, TensorValue)
        np.testing.assert_array_equal(decoded.to_numpy(), arr)

    def test_dtype_preservation(self):
        """Test that dtype is preserved across serialization."""
        for dtype in [np.float32, np.float64, np.int32, np.uint8]:
            arr = np.array([1, 2, 3], dtype=dtype)
            tensor = TensorValue(arr)
            wire = encode_value(tensor)
            decoded = decode_value(wire)

            assert decoded.to_numpy().dtype == dtype

    def test_non_contiguous_array(self):
        """Test that non-contiguous arrays are handled."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])[:, ::2]  # non-contiguous
        assert not arr.flags.c_contiguous

        tensor = TensorValue(arr)
        wire = encode_value(tensor)
        decoded = decode_value(wire)

        np.testing.assert_array_equal(decoded.to_numpy(), arr)

    def test_invalid_input(self):
        """Test TensorValue rejects non-ndarray input."""
        with pytest.raises(TypeError, match=r"expects a numpy\.ndarray"):
            TensorValue([1, 2, 3])

    def test_truncated_payload(self):
        """Test decode error on truncated payload."""
        arr = np.array([1, 2, 3])
        tensor = TensorValue(arr)
        wire = encode_value(tensor)

        # Corrupt by truncating
        with pytest.raises(ValueDecodeError):
            decode_value(wire[:10])

    def test_repr(self):
        """Test __repr__ output."""
        arr = np.array([[1, 2], [3, 4]])
        tensor = TensorValue(arr)
        repr_str = repr(tensor)

        assert "TensorValue" in repr_str
        assert "shape=(2, 2)" in repr_str


class TestTableValue:
    """Test TableValue (DataFrame/Arrow) serialization."""

    @pytest.fixture
    def check_arrow_available(self):
        """Skip tests if pyarrow not available."""
        pytest.importorskip("pyarrow")

    def test_roundtrip_pandas(self, check_arrow_available):
        """Test encode/decode roundtrip for pandas DataFrame."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})

        table = TableValue(df)
        wire = encode_value(table)

        decoded = decode_value(wire)
        assert isinstance(decoded, TableValue)

        result_df = decoded.to_pandas()
        pd.testing.assert_frame_equal(result_df, df)

    def test_roundtrip_arrow_table(self, check_arrow_available):
        """Test encode/decode with pyarrow.Table directly."""
        import pyarrow as pa

        table_data = pa.table({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})

        table_value = TableValue(table_data)
        wire = encode_value(table_value)

        decoded = decode_value(wire)
        assert isinstance(decoded, TableValue)

        # Compare via pandas for convenience
        pd = pytest.importorskip("pandas")
        result_df = decoded.to_pandas()
        expected_df = table_data.to_pandas()
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_missing_pyarrow(self, monkeypatch):
        """Test graceful error when pyarrow is missing."""
        import sys

        # Hide pyarrow module
        monkeypatch.setitem(sys.modules, "pyarrow", None)

        # Should fail immediately when creating TableValue (now requires pyarrow)
        with pytest.raises(ValueError, match="pyarrow is required"):
            TableValue({"a": [1, 2]})

    def test_repr(self, check_arrow_available):
        """Test __repr__ output."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table = TableValue(df)

        # Force materialization
        _ = table.to_proto()

        repr_str = repr(table)
        assert "TableValue" in repr_str
        assert "cols=" in repr_str
        assert "rows=" in repr_str


class TestValueRegistry:
    """Test Value registration and dispatch."""

    def test_list_kinds(self):
        """Test listing registered value kinds."""
        kinds = list_value_kinds()

        assert "mplang.demo.BytesBlob" in kinds
        assert "mplang.ndarray" in kinds
        assert "mplang.dataframe.arrow" in kinds

    def test_unknown_kind_decode(self):
        """Test decode error for unknown kind."""
        from mplang.v1.protos.v1alpha1 import value_pb2

        # Create envelope with unknown kind
        env = value_pb2.ValueProto(
            kind="unknown.kind", value_version=1, payload=b"data"
        )
        wire = env.SerializeToString()

        with pytest.raises(ValueDecodeError, match="Unknown Value kind"):
            decode_value(wire)

    def test_register_duplicate_kind(self):
        """Test that duplicate KIND registration raises error."""
        with pytest.raises(ValueError, match="Duplicate Value KIND"):

            @register_value
            class DuplicateValue(Value):
                KIND = "mplang.demo.BytesBlob"  # Already registered
                WIRE_VERSION = 1

                def to_proto(self):
                    proto = self._new_proto()
                    proto.payload = b""
                    return proto

                @classmethod
                def from_proto(cls, proto):
                    return cls()

    def test_register_missing_kind(self):
        """Test that missing KIND raises error."""
        with pytest.raises(ValueError, match="missing KIND str"):

            @register_value
            class NoKindValue(Value):
                WIRE_VERSION = 1

                def to_proto(self):
                    proto = self._new_proto()
                    proto.payload = b""
                    return proto

                @classmethod
                def from_proto(cls, proto):
                    return cls()


class TestValueEnvelope:
    """Test ValueProto envelope encoding/decoding."""

    def test_is_value_envelope_valid(self):
        """Test is_value_envelope recognizes valid envelopes."""
        blob = BytesBlob(b"test")
        wire = encode_value(blob)

        assert is_value_envelope(wire)

    def test_is_value_envelope_invalid(self):
        """Test is_value_envelope rejects invalid data."""
        assert not is_value_envelope(b"not a protobuf")
        assert not is_value_envelope(b"")

    def test_corrupt_envelope(self):
        """Test decode error on corrupted envelope."""
        with pytest.raises(ValueDecodeError, match="Failed parsing ValueProto"):
            decode_value(b"corrupted data")

    def test_missing_kind(self):
        """Test decode error when kind field is missing."""
        from mplang.v1.protos.v1alpha1 import value_pb2

        env = value_pb2.ValueProto(
            kind="",  # Empty kind
            value_version=1,
            payload=b"data",
        )
        wire = env.SerializeToString()

        with pytest.raises(ValueDecodeError, match="Envelope missing kind"):
            decode_value(wire)


class TestVersioning:
    """Test wire version handling."""

    def test_version_mismatch_bytesblob(self):
        """Test BytesBlob version validation."""
        # Manually create envelope with unsupported version
        from mplang.v1.protos.v1alpha1 import value_pb2

        env = value_pb2.ValueProto(
            kind="mplang.demo.BytesBlob",
            value_version=999,  # Unsupported
            payload=b"data",
        )
        wire = env.SerializeToString()

        with pytest.raises(ValueDecodeError, match="Unsupported BytesBlob version"):
            decode_value(wire)

    def test_version_mismatch_tensor(self):
        """Test TensorValue version validation."""
        from mplang.v1.protos.v1alpha1 import value_pb2

        env = value_pb2.ValueProto(
            kind="mplang.ndarray", value_version=999, payload=b"invalid"
        )
        wire = env.SerializeToString()

        with pytest.raises(ValueDecodeError, match="Unsupported TensorValue version"):
            decode_value(wire)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
