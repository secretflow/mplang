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

"""Base classes for runtime values in MPLang backends.

This module defines:
1. `Value`: The abstract base class for all backend runtime values. It provides
   a unified serialization interface via `to_json`/`from_json`.
2. `WrapValue`: A generic base class for values that simply wrap an external
   type (like numpy arrays, arrow tables, or cryptographic objects). It
   implements the "wrap/unwrap" pattern with automatic type conversion.

Usage:
    from mplang.v2.runtime.value import Value
    from mplang.v2.edsl import serde

    @serde.register_class
    class MyValue(Value):
        _serde_kind = "mymodule.MyValue"

        def __init__(self, data: bytes):
            self._data = data

        @property
        def data(self) -> bytes:
            return self._data

        def to_json(self) -> dict:
            return {"data": base64.b64encode(self._data).decode("ascii")}

        @classmethod
        def from_json(cls, data: dict) -> "MyValue":
            return cls(data=base64.b64decode(data["data"]))

        @classmethod
        def wrap(cls, val: bytes | MyValue) -> MyValue:
            if isinstance(val, cls):
                return val
            return cls(val)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from typing import Self


class Value(ABC):
    """Base class for all runtime values in MPLang backends.

    Subclasses must define:
    - `_serde_kind: ClassVar[str]` - unique identifier for serialization
    - `to_json(self) -> dict` - serialize to JSON-compatible dict
    - `from_json(cls, data: dict) -> Self` - deserialize from dict

    And should use the `@serde.register_class` decorator for registration.
    """

    # Class-level type identifier for serde dispatch.
    # Subclasses must define this.
    _serde_kind: ClassVar[str]

    # =========== Serialization Interface ===========

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        Note: Do NOT include `_kind` in the returned dict;
        `serde.to_json` adds it automatically based on `_serde_kind`.
        """
        ...

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """Deserialize from JSON-compatible dict."""
        ...


T = TypeVar("T")


class WrapValue(Value, Generic[T]):
    """Base class for values that wrap a specific native object.

    Provides standard implementation for:
    - `__init__` (calls `_convert` and stores data in `_data`)
    - `data` property
    - `unwrap` (returns `_data`)
    - `wrap` (factory method with idempotency check)
    """

    _data: T

    def __init__(self, data: Any):
        self._data = self._convert(data)

    def _convert(self, data: Any) -> T:
        """Convert input data to the underlying type T.

        Default implementation: assume data is already T.
        Subclasses should override this to handle type coercion.
        """
        if isinstance(data, WrapValue):
            return data.unwrap()  # type: ignore
        return data  # type: ignore

    @property
    def data(self) -> T:
        """Get the underlying raw data (read-only)."""
        return self._data

    @classmethod
    def wrap(cls, val: Any) -> Self:
        """Factory method: wrap a value into this Value type.

        Idempotent: if val is already this type, returns it as-is.
        Otherwise, calls constructor which triggers `_convert`.
        """
        if isinstance(val, cls):
            return val
        return cls(val)

    def unwrap(self) -> T:
        """Get the underlying raw data."""
        return self._data
