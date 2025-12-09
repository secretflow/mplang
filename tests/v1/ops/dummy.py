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

"""Shared dummy frontend test helpers.

Provides minimal MPObject implementations for frontend typed_op tests
without touching core library code.
"""

from __future__ import annotations

from typing import Any

from mplang.v1.core.cluster import ClusterSpec
from mplang.v1.core.dtypes import DType
from mplang.v1.core.mpobject import MPContext, MPObject
from mplang.v1.core.mptype import MPType
from mplang.v1.core.table import TableType


class DummyContext(MPContext):
    def __init__(self) -> None:
        super().__init__(ClusterSpec.simple(world_size=1))


class DummyTensor(MPObject):
    """Minimal tensor MPObject used for type-only frontend operations."""

    def __init__(
        self, dtype: Any, shape: tuple[int, ...]
    ):  # dtype may be np.dtype, DType, etc.
        self._mptype = MPType.tensor(DType.from_any(dtype), shape)
        self._ctx = DummyContext()

    @property
    def mptype(self) -> MPType:  # type: ignore[override]
        return self._mptype

    @property
    def ctx(self) -> DummyContext:  # type: ignore[override]
        return self._ctx


class DummyTable(MPObject):
    """Minimal table MPObject for frontend tests that need table schemas."""

    def __init__(self, schema: TableType):
        self._mptype = MPType.table(schema)
        self._ctx = DummyContext()

    @property
    def mptype(self) -> MPType:  # type: ignore[override]
        return self._mptype

    @property
    def ctx(self) -> DummyContext:  # type: ignore[override]
        return self._ctx
