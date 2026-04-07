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

"""
JSON-based serialization for MPLang types and graphs.

This module provides a secure, extensible serialization mechanism that replaces
cloudpickle. Each type is responsible for its own serialization via the
`@register_class` decorator pattern.

Usage:
    from mplang.edsl import serde

    @serde.register_class
    class MyType:
        _serde_kind = "mymodule.MyType"

        def to_json(self) -> dict:
            return {"field": self.field}

        @classmethod
        def from_json(cls, data: dict) -> "MyType":
            return cls(data["field"])

    # Serialize
    obj = MyType(...)
    json_data = serde.to_json(obj)

    # Deserialize
    obj2 = serde.from_json(json_data)

Security:
    Unlike pickle/cloudpickle, JSON deserialization only reconstructs data
    structures - it cannot execute arbitrary code. The `from_json` methods
    are explicitly defined by each registered class.
"""

from __future__ import annotations

import base64
import gzip
import json
import struct
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

import numpy as np

# =============================================================================
# Type Registry
# =============================================================================

# Global registry: kind string -> class
_CLASS_REGISTRY: dict[str, type] = {}

T = TypeVar("T")


def register_class(cls: type[T]) -> type[T]:
    """Decorator to register a class for JSON serialization.

    The class must define:
    - `_serde_kind: ClassVar[str]` - unique identifier for this type
    - `to_json(self) -> dict` - serialize instance to JSON-compatible dict
    - `from_json(cls, data: dict) -> Self` - deserialize from dict

    Example:
        @serde.register_class
        class MyType:
            _serde_kind = "mymodule.MyType"

            def to_json(self) -> dict:
                return {"value": self.value}

            @classmethod
            def from_json(cls, data: dict) -> "MyType":
                return cls(data["value"])
    """
    kind = getattr(cls, "_serde_kind", None)
    if kind is None:
        raise ValueError(
            f"{cls.__name__} must define `_serde_kind` class variable "
            "for serialization registration"
        )
    if kind in _CLASS_REGISTRY:
        existing = _CLASS_REGISTRY[kind]
        if existing is not cls:
            raise ValueError(
                f"Duplicate _serde_kind '{kind}': "
                f"already registered by {existing.__name__}"
            )
    _CLASS_REGISTRY[kind] = cls
    return cls


def get_registered_class(kind: str) -> type | None:
    """Get the class registered for a given kind string."""
    return _CLASS_REGISTRY.get(kind)


def list_registered_kinds() -> list[str]:
    """List all registered kind strings."""
    return list(_CLASS_REGISTRY.keys())


# =============================================================================
# Serialization Protocol
# =============================================================================


@runtime_checkable
class JsonSerializable(Protocol):
    """Protocol for types that can be serialized to JSON."""

    _serde_kind: ClassVar[str]

    def to_json(self) -> dict[str, Any]: ...

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> JsonSerializable: ...


# =============================================================================
# Core Serialization Functions
# =============================================================================


def to_json(obj: Any) -> dict[str, Any]:
    """Serialize an object to a JSON-compatible dict.

    The object must either:
    1. Be a registered class with `_serde_kind` and `to_json()` method
    2. Be a primitive type (int, float, str, bool, None)
    3. Be a list/tuple of serializable objects
    4. Be a dict with string keys and serializable values
    5. Be a numpy ndarray

    Args:
        obj: Object to serialize

    Returns:
        JSON-compatible dict with `_kind` field for type dispatch

    Raises:
        TypeError: If object cannot be serialized
    """
    # Registered classes
    if hasattr(obj, "_serde_kind") and hasattr(obj, "to_json"):
        data: dict[str, Any] = obj.to_json()
        data["_kind"] = obj._serde_kind
        return data

    # Primitives
    if obj is None:
        return {"_kind": "_null"}
    if isinstance(obj, bool):  # Must check before int (bool is subclass of int)
        return {"_kind": "_bool", "v": obj}

    if isinstance(obj, int):
        return {"_kind": "_int", "v": obj}
    if isinstance(obj, float):
        return {"_kind": "_float", "v": obj}
    if isinstance(obj, str):
        return {"_kind": "_str", "v": obj}

    # Numpy scalar types (int64, float32, etc.)
    if isinstance(obj, np.integer):
        return {"_kind": "_int", "v": int(obj)}
    if isinstance(obj, np.floating):
        return {"_kind": "_float", "v": float(obj)}

    # Numpy array - handle both numeric and object arrays
    if isinstance(obj, np.ndarray):
        # Object arrays need element-wise serialization
        if obj.dtype == np.object_:
            return {
                "_kind": "_ndarray_object",
                "shape": list(obj.shape),
                "items": [to_json(item) for item in obj.flat],
            }
        # Numeric arrays use efficient binary format
        return {
            "_kind": "_ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": base64.b64encode(obj.tobytes()).decode("ascii"),
        }

    # Array-like (JAX, etc.) - convert to numpy
    if hasattr(obj, "__array__"):
        return to_json(np.asarray(obj))

    # Collections
    if isinstance(obj, (list, tuple)):
        return {
            "_kind": "_list" if isinstance(obj, list) else "_tuple",
            "items": [to_json(item) for item in obj],
        }
    if isinstance(obj, dict):
        # Handle dicts with non-string keys by serializing as list of pairs
        # This preserves key types (int, tuple, etc.)
        has_non_string_keys = any(not isinstance(k, str) for k in obj.keys())
        if has_non_string_keys:
            return {
                "_kind": "_dict_pairs",
                "pairs": [[to_json(k), to_json(v)] for k, v in obj.items()],
            }
        return {
            "_kind": "_dict",
            "items": {k: to_json(v) for k, v in obj.items()},
        }

    # Bytes
    if isinstance(obj, bytes):
        return {
            "_kind": "_bytes",
            "data": base64.b64encode(obj).decode("ascii"),
        }

    raise TypeError(
        f"Cannot serialize object of type {type(obj).__name__}. "
        "Ensure the class is decorated with @serde.register_class "
        "and implements to_json()/from_json()."
    )


def from_json(data: dict[str, Any]) -> Any:
    """Deserialize an object from a JSON-compatible dict.

    Args:
        data: Dict with `_kind` field indicating the type

    Returns:
        Deserialized object

    Raises:
        ValueError: If `_kind` is missing or unknown
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    kind = data.get("_kind")
    if kind is None:
        raise ValueError("Missing '_kind' field in JSON data")

    # Built-in primitives
    if kind == "_null":
        return None
    if kind == "_bool":
        return bool(data["v"])
    if kind == "_int":
        return int(data["v"])
    if kind == "_float":
        return float(data["v"])
    if kind == "_str":
        return str(data["v"])

    # Collections
    if kind == "_list":
        return [from_json(item) for item in data["items"]]
    if kind == "_tuple":
        return tuple(from_json(item) for item in data["items"])
    if kind == "_dict":
        return {k: from_json(v) for k, v in data["items"].items()}
    if kind == "_dict_pairs":
        # Handle dicts with non-string keys
        return {from_json(pair[0]): from_json(pair[1]) for pair in data["pairs"]}

    # Bytes
    if kind == "_bytes":
        return base64.b64decode(data["data"])

    # Legacy numpy array formats - kept for backward compatibility
    # New serializations go through TensorValue (tensor_impl.TensorValue)
    if kind == "_ndarray":
        dtype_str = data["dtype"]
        shape = tuple(data["shape"])
        buffer = base64.b64decode(data["data"])
        dtype = np.dtype(dtype_str)
        return np.frombuffer(buffer, dtype=dtype).reshape(shape).copy()

    if kind == "_ndarray_object":
        shape = tuple(data["shape"])
        items = [from_json(item) for item in data["items"]]
        arr = np.empty(len(items), dtype=object)
        for i, item in enumerate(items):
            arr[i] = item
        # Always reshape - empty tuple () means scalar, which requires reshape
        return arr.reshape(shape)

    # Registered classes
    if kind in _CLASS_REGISTRY:
        cls = _CLASS_REGISTRY[kind]
        # Remove _kind before passing to from_json
        data_copy = {k: v for k, v in data.items() if k != "_kind"}
        return cls.from_json(data_copy)  # type: ignore[attr-defined]

    raise ValueError(
        f"Unknown type kind: '{kind}'. "
        "Ensure the class is registered with @serde.register_class "
        "and the module is imported."
    )


# =============================================================================
# Convenience Functions for Wire Format
# =============================================================================


def dumps(obj: Any, *, compress: bool = True) -> bytes:
    """Serialize object to bytes (JSON + optional gzip).

    Args:
        obj: Object to serialize
        compress: Whether to gzip compress the output (default: True)

    Returns:
        Serialized bytes
    """
    json_str = json.dumps(to_json(obj), separators=(",", ":"))
    data = json_str.encode("utf-8")
    if compress:
        data = gzip.compress(data)
    return data


def loads(data: bytes, *, compressed: bool = True) -> Any:
    """Deserialize object from bytes.

    Args:
        data: Serialized bytes
        compressed: Whether the data is gzip compressed (default: True)

    Returns:
        Deserialized object
    """
    if compressed:
        data = gzip.decompress(data)
    json_data = json.loads(data.decode("utf-8"))
    return from_json(json_data)


def dumps_b64(obj: Any, *, compress: bool = True) -> str:
    """Serialize object to base64 string (for JSON transport).

    Args:
        obj: Object to serialize
        compress: Whether to gzip compress (default: True)

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(dumps(obj, compress=compress)).decode("ascii")


def loads_b64(data: str, *, compressed: bool = True) -> Any:
    """Deserialize object from base64 string.

    Args:
        data: Base64-encoded string
        compressed: Whether the data is gzip compressed (default: True)

    Returns:
        Deserialized object
    """
    return loads(base64.b64decode(data), compressed=compressed)


# =============================================================================
# Binary Container Format
# =============================================================================
#
# Layout (all lengths are unsigned 32-bit little-endian integers):
#
#   [4 bytes] metadata_len
#   [metadata_len bytes] UTF-8 JSON metadata
#   for each binary segment i:
#       [4 bytes] segment_len
#       [segment_len bytes] raw binary data
#
# Inside the JSON metadata, large binary blobs are replaced by a reference
# placeholder:  {"_bref": <segment_index>, "len": <byte_length>}
#
# This eliminates the inner base64 encoding for types that opt in by
# implementing ``to_binary_json(ctx)`` / ``from_binary_json(data, segments)``.
# Types that do *not* opt in fall back to the standard ``to_json()`` path
# (their base64-encoded fields remain inside the JSON metadata unchanged).
#
# Wire efficiency comparison for a 100 MB float32 array (TensorValue):
#   dumps()         → JSON+base64 inner + gzip             ≈ 125 MB
#   dumps_binary()  → binary segment (no base64) + 4-byte header ≈ 100 MB
#   (gzip is intentionally NOT applied to binary segments because random /
#    already-compressed data compresses poorly; callers that need transport-
#    level compression can apply gzip around the whole blob.)
# =============================================================================


class BinaryContext:
    """Accumulates binary segments during serialization.

    Passed to ``to_binary_json()`` of registered classes.  Callers call
    :meth:`add_segment` to append a raw bytes object and receive the
    segment index to embed in the JSON metadata as a ``_bref`` placeholder.
    """

    def __init__(self) -> None:
        self._segments: list[bytes] = []

    def add_segment(self, data: bytes) -> int:
        """Append *data* as a new binary segment and return its index."""
        idx = len(self._segments)
        self._segments.append(data)
        return idx

    @property
    def segments(self) -> list[bytes]:
        return self._segments


def _to_binary_json(obj: Any, ctx: BinaryContext) -> dict[str, Any]:
    """Serialize *obj* to JSON-compatible data, placing binary blobs into *ctx*.

    This mirrors :func:`to_json` semantics but recursively applies binary-aware
    serialization to nested values so registered classes inside lists/tuples/
    dicts can also emit binary segment references.
    """
    # Registered classes with binary-aware serialization
    if hasattr(obj, "_serde_kind") and hasattr(obj, "to_binary_json"):
        data: dict[str, Any] = obj.to_binary_json(ctx)
        data["_kind"] = obj._serde_kind
        return data

    # Registered classes without binary-aware serialization (standard path)
    if hasattr(obj, "_serde_kind") and hasattr(obj, "to_json"):
        data = obj.to_json()
        data["_kind"] = obj._serde_kind
        return data

    # Primitive scalars (keep parity with to_json)
    if obj is None:
        return {"_kind": "_null"}
    if isinstance(obj, bool):
        return {"_kind": "_bool", "v": obj}
    if isinstance(obj, int):
        return {"_kind": "_int", "v": obj}
    if isinstance(obj, float):
        return {"_kind": "_float", "v": obj}
    if isinstance(obj, str):
        return {"_kind": "_str", "v": obj}
    if isinstance(obj, np.integer):
        return {"_kind": "_int", "v": int(obj)}
    if isinstance(obj, np.floating):
        return {"_kind": "_float", "v": float(obj)}

    # Numpy arrays: numeric arrays go to a raw segment; object arrays recurse.
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            return {
                "_kind": "_ndarray_object",
                "shape": list(obj.shape),
                "items": [_to_binary_json(item, ctx) for item in obj.flat],
            }
        seg_idx = ctx.add_segment(obj.tobytes())
        return {
            "_kind": "_ndarray_bref",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "bref": seg_idx,
        }

    # Raw bytes go to a segment instead of base64.
    if isinstance(obj, bytes):
        seg_idx = ctx.add_segment(obj)
        return {"_kind": "_bytes_bref", "bref": seg_idx}

    # Array-like (e.g., JAX arrays)
    if hasattr(obj, "__array__"):
        return _to_binary_json(np.asarray(obj), ctx)

    # Collections (recursive)
    if isinstance(obj, (list, tuple)):
        return {
            "_kind": "_list" if isinstance(obj, list) else "_tuple",
            "items": [_to_binary_json(item, ctx) for item in obj],
        }

    if isinstance(obj, dict):
        has_non_string_keys = any(not isinstance(k, str) for k in obj.keys())
        if has_non_string_keys:
            return {
                "_kind": "_dict_pairs",
                "pairs": [
                    [_to_binary_json(k, ctx), _to_binary_json(v, ctx)]
                    for k, v in obj.items()
                ],
            }

        return {
            "_kind": "_dict",
            "items": {k: _to_binary_json(v, ctx) for k, v in obj.items()},
        }

    # Fallback (keeps existing error behavior for unsupported types)
    return to_json(obj)


def _from_binary_json(data: dict[str, Any], segments: list[bytes]) -> Any:
    """Deserialize from a binary-aware JSON dict plus raw segments."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    kind = data.get("_kind")
    if kind is None:
        raise ValueError("Missing '_kind' field in binary JSON data")

    # Built-in primitives and collections.
    if kind in {"_null", "_bool", "_int", "_float", "_str", "_bytes", "_ndarray"}:
        return from_json(data)

    if kind == "_bytes_bref":
        return segments[data["bref"]]

    if kind == "_ndarray_bref":
        raw = segments[data["bref"]]
        arr = np.frombuffer(raw, dtype=np.dtype(data["dtype"]))
        return arr.reshape(tuple(data["shape"])).copy()

    if kind == "_list":
        return [_from_binary_json(item, segments) for item in data["items"]]

    if kind == "_tuple":
        return tuple(_from_binary_json(item, segments) for item in data["items"])

    if kind == "_dict":
        return {k: _from_binary_json(v, segments) for k, v in data["items"].items()}

    if kind == "_dict_pairs":
        return {
            _from_binary_json(pair[0], segments): _from_binary_json(pair[1], segments)
            for pair in data["pairs"]
        }

    if kind == "_ndarray_object":
        shape = tuple(data["shape"])
        items = [_from_binary_json(item, segments) for item in data["items"]]
        arr = np.empty(len(items), dtype=object)
        for i, item in enumerate(items):
            arr[i] = item
        return arr.reshape(shape)

    # Registered classes with binary-aware deserialization
    if kind in _CLASS_REGISTRY:
        cls = _CLASS_REGISTRY[kind]
        data_copy = {k: v for k, v in data.items() if k != "_kind"}
        if hasattr(cls, "from_binary_json"):
            return cls.from_binary_json(data_copy, segments)  # type: ignore[attr-defined]
        return cls.from_json(data_copy)  # type: ignore[attr-defined]

    # Fall back to standard from_json for built-in types
    return from_json(data)


def dumps_binary(obj: Any) -> bytes:
    """Serialize *obj* to the binary container format.

    Unlike :func:`dumps`, binary payloads inside registered classes (e.g.
    numpy array bytes in TensorValue, Arrow IPC bytes in TableValue) are
    stored as raw binary segments rather than being base64-encoded inside
    the JSON metadata.  This eliminates the inner base64 layer, reducing
    wire size by ~25 % for large tensors/tables and enabling better gzip
    compression when the caller applies it at the transport layer.

    Format::

        [4B LE] metadata_len
        [metadata_len B] UTF-8 JSON metadata
        for each segment:
            [4B LE] segment_len
            [segment_len B] raw bytes

    Args:
        obj: Object to serialize (must be supported by the serde registry).

    Returns:
        Binary blob ready for network transmission.
    """
    ctx = BinaryContext()
    meta_dict = _to_binary_json(obj, ctx)
    meta_bytes = json.dumps(meta_dict, separators=(",", ":")).encode("utf-8")

    parts: list[bytes] = []
    parts.append(struct.pack("<I", len(meta_bytes)))
    parts.append(meta_bytes)
    for seg in ctx.segments:
        parts.append(struct.pack("<I", len(seg)))
        parts.append(seg)

    return b"".join(parts)


def loads_binary(data: bytes) -> Any:
    """Deserialize an object from the binary container format.

    Args:
        data: Binary blob produced by :func:`dumps_binary`.

    Returns:
        Deserialized Python object.

    Raises:
        ValueError: If the data is malformed.
    """
    if len(data) < 4:
        raise ValueError("Binary data too short: missing metadata length header")

    offset = 0
    (meta_len,) = struct.unpack_from("<I", data, offset)
    offset += 4

    if offset + meta_len > len(data):
        raise ValueError("Binary data truncated: metadata extends beyond buffer")

    meta_bytes = data[offset : offset + meta_len]
    offset += meta_len

    meta_dict = json.loads(meta_bytes.decode("utf-8"))

    segments: list[bytes] = []
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("Binary data truncated: segment length header incomplete")
        (seg_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if offset + seg_len > len(data):
            raise ValueError("Binary data truncated: segment data incomplete")
        segments.append(data[offset : offset + seg_len])
        offset += seg_len

    return _from_binary_json(meta_dict, segments)
