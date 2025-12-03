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
    from mplang.v2.edsl import serde

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
