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

"""Unit tests for SPU utilities."""

import pytest
import spu.libspu as libspu

from mplang.v1.utils.spu_utils import (
    list_fields,
    list_protocols,
    parse_field,
    parse_protocol,
)


class TestSPUProtocol:
    """Test SPU protocol utilities."""

    def test_list_method_works(self):
        """Test that list methods return expected protocols."""
        protocols = list_protocols()
        assert isinstance(protocols, list)
        assert len(protocols) == 5
        assert "SEMI2K" in protocols
        assert "ABY3" in protocols

    def test_case_sensitivity(self):
        """Test that protocol parsing is case sensitive."""
        # Valid uppercase should work
        result = parse_protocol("SEMI2K")
        assert result == libspu.ProtocolKind.SEMI2K

        # Invalid lowercase should fail
        with pytest.raises(ValueError):
            parse_protocol("semi2k")

        # Invalid mixed case should fail
        with pytest.raises(ValueError):
            parse_protocol("Semi2K")

    def test_integer_works(self):
        """Test that integer parsing works correctly."""
        # Test valid integers
        assert parse_protocol(1) == libspu.ProtocolKind.REF2K
        assert parse_protocol(2) == libspu.ProtocolKind.SEMI2K
        assert parse_protocol(3) == libspu.ProtocolKind.ABY3

    def test_invalid_raises(self):
        """Test that invalid inputs raise ValueError."""
        # Invalid string
        with pytest.raises(ValueError) as exc_info:
            parse_protocol("INVALID")
        assert "Invalid SPU protocol: INVALID" in str(exc_info.value)

        # Invalid integer
        with pytest.raises(ValueError) as exc_info:
            parse_protocol(999)
        assert "Invalid SPU protocol value: 999" in str(exc_info.value)


class TestSPUField:
    """Test SPU field type utilities."""

    def test_list_method_works(self):
        """Test that list methods return expected field types."""
        fields = list_fields()
        assert isinstance(fields, list)
        assert len(fields) == 3
        assert "FM64" in fields
        assert "FM32" in fields

    def test_case_sensitivity(self):
        """Test that field parsing is case sensitive."""
        # Valid uppercase should work
        result = parse_field("FM64")
        assert result == libspu.FieldType.FM64

        # Invalid lowercase should fail
        with pytest.raises(ValueError):
            parse_field("fm64")

        # Invalid mixed case should fail
        with pytest.raises(ValueError):
            parse_field("Fm64")

    def test_integer_works(self):
        """Test that integer parsing works correctly."""
        # Test valid integers
        assert parse_field(1) == libspu.FieldType.FM32
        assert parse_field(2) == libspu.FieldType.FM64
        assert parse_field(3) == libspu.FieldType.FM128

    def test_invalid_raises(self):
        """Test that invalid inputs raise ValueError."""
        # Invalid string
        with pytest.raises(ValueError) as exc_info:
            parse_field("INVALID")
        assert "Invalid SPU field type: INVALID" in str(exc_info.value)

        # Invalid integer
        with pytest.raises(ValueError) as exc_info:
            parse_field(999)
        assert "Invalid SPU field type value: 999" in str(exc_info.value)
