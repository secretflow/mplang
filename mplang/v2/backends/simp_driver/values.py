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

"""Simp Driver values (DriverVar)."""

from __future__ import annotations

from typing import Any, ClassVar, Self

from mplang.v2.edsl import serde
from mplang.v2.runtime.value import Value


@serde.register_class
class DriverVar(Value):
    """A value replicated (or sharded) on the Driver.

    A DriverVar holds a list of values, one for each party in the computation.
    """

    _serde_kind: ClassVar[str] = "simp.DriverVar"

    def __init__(self, values: list[Any]):
        self.values = values

    @property
    def world_size(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"DriverVar({self.values})"

    def __getitem__(self, idx: int) -> Any:
        return self.values[idx]

    def to_json(self) -> dict[str, Any]:
        return {"values": self.values}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        return cls(values=data["values"])
