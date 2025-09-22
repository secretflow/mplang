"""Shared dummy frontend test helpers.

Provides minimal MPObject implementations for frontend typed_op tests
without touching core library code.
"""

from __future__ import annotations

from mplang.core.cluster import ClusterSpec
from mplang.core.dtype import DType
from mplang.core.mpobject import MPContext, MPObject
from mplang.core.mptype import MPType
from mplang.core.table import TableType


class DummyContext(MPContext):
    def __init__(self) -> None:
        super().__init__(ClusterSpec.simple(world_size=1))


class DummyTensor(MPObject):
    """Minimal tensor MPObject used for type-only frontend operations."""

    def __init__(
        self, dtype, shape: tuple[int, ...]
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
