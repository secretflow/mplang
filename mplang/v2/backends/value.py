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

"""Base class for all runtime values in MPLang backends.

This module defines the `Value` abstract base class that all backend runtime
values should inherit from. It provides a unified serialization interface
via `to_json`/`from_json`.

Usage:
    from mplang.v2.backends.value import Value
    from mplang.v2.edsl import serde

    @serde.register_class
    class MyValue(Value):
        _serde_kind = "mymodule.MyValue"

        def __init__(self, data: bytes):
            self.data = data

        def to_json(self) -> dict:
            return {"data": base64.b64encode(self.data).decode("ascii")}

        @classmethod
        def from_json(cls, data: dict) -> "MyValue":
            return cls(data=base64.b64decode(data["data"]))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


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
    def from_json(cls, data: dict[str, Any]) -> Value:
        """Deserialize from JSON-compatible dict."""
        ...
