# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for serde module."""

from __future__ import annotations

import pytest

from mplang.v2.edsl import serde

# =============================================================================
# Test Fixtures: Sample registered classes
# =============================================================================


@serde.register_class
class Point:
    """Simple test class."""

    _serde_kind = "test.Point"

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def to_json(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_json(cls, data: dict) -> Point:
        return cls(data["x"], data["y"])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and self.x == other.x and self.y == other.y


@serde.register_class
class Box:
    """Nested test class containing another registered class."""

    _serde_kind = "test.Box"

    def __init__(self, origin: Point, size: tuple[int, int]):
        self.origin = origin
        self.size = size

    def to_json(self) -> dict:
        return {
            "origin": serde.to_json(self.origin),
            "size": serde.to_json(self.size),
        }

    @classmethod
    def from_json(cls, data: dict) -> Box:
        return cls(
            origin=serde.from_json(data["origin"]),
            size=serde.from_json(data["size"]),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Box)
            and self.origin == other.origin
            and self.size == other.size
        )


# =============================================================================
# Tests: Primitives
# =============================================================================


class TestPrimitives:
    """Test serialization of primitive types."""

    def test_none(self):
        assert serde.from_json(serde.to_json(None)) is None

    def test_bool(self):
        assert serde.from_json(serde.to_json(True)) is True
        assert serde.from_json(serde.to_json(False)) is False

    def test_int(self):
        assert serde.from_json(serde.to_json(42)) == 42
        assert serde.from_json(serde.to_json(-100)) == -100
        assert serde.from_json(serde.to_json(0)) == 0

    def test_float(self):
        assert serde.from_json(serde.to_json(3.14)) == 3.14
        assert serde.from_json(serde.to_json(-2.5)) == -2.5

    def test_str(self):
        assert serde.from_json(serde.to_json("hello")) == "hello"
        assert serde.from_json(serde.to_json("")) == ""
        assert serde.from_json(serde.to_json("中文")) == "中文"


# =============================================================================
# Tests: Collections
# =============================================================================


class TestCollections:
    """Test serialization of collection types."""

    def test_list(self):
        data = [1, 2, 3]
        result = serde.from_json(serde.to_json(data))
        assert result == data
        assert isinstance(result, list)

    def test_tuple(self):
        data = (1, 2, 3)
        result = serde.from_json(serde.to_json(data))
        assert result == data
        assert isinstance(result, tuple)

    def test_dict(self):
        data = {"a": 1, "b": 2}
        result = serde.from_json(serde.to_json(data))
        assert result == data

    def test_dict_with_int_keys(self):
        """Test dict with integer keys (like routing tables)."""
        data = {1: 0, 2: 1, 3: 2}
        result = serde.from_json(serde.to_json(data))
        assert result == data
        assert all(isinstance(k, int) for k in result.keys())

    def test_dict_with_tuple_keys(self):
        """Test dict with tuple keys."""
        data = {(0, 1): "ab", (2, 3): "cd"}
        result = serde.from_json(serde.to_json(data))
        assert result == data
        assert all(isinstance(k, tuple) for k in result.keys())

    def test_nested_collections(self):
        data = {"list": [1, 2], "tuple": (3, 4), "nested": {"x": [5, 6]}}
        result = serde.from_json(serde.to_json(data))
        assert result["list"] == [1, 2]
        assert result["tuple"] == (3, 4)
        assert result["nested"] == {"x": [5, 6]}

    def test_empty_collections(self):
        assert serde.from_json(serde.to_json([])) == []
        assert serde.from_json(serde.to_json(())) == ()
        assert serde.from_json(serde.to_json({})) == {}


# =============================================================================
# Tests: Registered Classes
# =============================================================================


class TestRegisteredClasses:
    """Test serialization of registered classes."""

    def test_simple_class(self):
        p = Point(10, 20)
        result = serde.from_json(serde.to_json(p))
        assert result == p
        assert isinstance(result, Point)

    def test_nested_class(self):
        box = Box(origin=Point(0, 0), size=(100, 200))
        result = serde.from_json(serde.to_json(box))
        assert result == box
        assert isinstance(result, Box)
        assert isinstance(result.origin, Point)

    def test_class_in_collection(self):
        points = [Point(1, 2), Point(3, 4)]
        result = serde.from_json(serde.to_json(points))
        assert result == points
        assert all(isinstance(p, Point) for p in result)


# =============================================================================
# Tests: Wire Format (dumps/loads)
# =============================================================================


class TestWireFormat:
    """Test bytes serialization functions."""

    def test_dumps_loads_compressed(self):
        data = {"key": [1, 2, 3], "nested": Point(5, 6)}
        serialized = serde.dumps(data, compress=True)
        result = serde.loads(serialized, compressed=True)
        assert result["key"] == [1, 2, 3]
        assert result["nested"] == Point(5, 6)

    def test_dumps_loads_uncompressed(self):
        data = [1, 2, 3]
        serialized = serde.dumps(data, compress=False)
        result = serde.loads(serialized, compressed=False)
        assert result == data

    def test_b64_roundtrip(self):
        p = Point(10, 20)
        data = {"point": p}
        b64_str = serde.dumps_b64(data)
        assert isinstance(b64_str, str)
        result = serde.loads_b64(b64_str)
        assert isinstance(result["point"], Point)
        assert result["point"] == p


# =============================================================================
# Tests: Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error cases."""

    def test_unregistered_class(self):
        class Unregistered:
            pass

        with pytest.raises(TypeError, match="Cannot serialize"):
            serde.to_json(Unregistered())

    def test_missing_kind(self):
        with pytest.raises(ValueError, match="Missing '_kind'"):
            serde.from_json({"x": 1})

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown type kind"):
            serde.from_json({"_kind": "nonexistent.Type"})

    def test_duplicate_registration(self):
        # First registration should succeed (already done above)
        # Re-registering the same class should be fine
        serde.register_class(Point)

        # But a different class with same kind should fail
        with pytest.raises(ValueError, match="Duplicate _serde_kind"):

            @serde.register_class
            class AnotherPoint:
                _serde_kind = "test.Point"  # Same as Point

                def to_json(self):
                    return {}

                @classmethod
                def from_json(cls, data):
                    return cls()


# =============================================================================
# Tests: Registry Functions
# =============================================================================


class TestRegistry:
    """Test registry utility functions."""

    def test_get_registered_class(self):
        assert serde.get_registered_class("test.Point") is Point
        assert serde.get_registered_class("nonexistent") is None

    def test_list_registered_kinds(self):
        kinds = serde.list_registered_kinds()
        assert "test.Point" in kinds
        assert "test.Box" in kinds
