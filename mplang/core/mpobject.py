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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mplang.core.dtype import DType
from mplang.core.mask import Mask
from mplang.core.mptype import MPType
from mplang.core.table import TableType
from mplang.core.tensor import Shape

if TYPE_CHECKING:
    from mplang.core.cluster import ClusterSpec


class MPContext:
    """The context of an MPObject.

    MPContext is the abstract base class for all execution contexts.
    It only holds the immutable cluster_spec plus lightweight parent/root
    helpers used to support stack-scoped extension state (attached lazily by
    external features on the root context).
    """

    def __init__(self, cluster_spec: ClusterSpec, *, parent: MPContext | None = None):
        if cluster_spec is None:
            raise ValueError("cluster_spec cannot be None")
        self.cluster_spec = cluster_spec
        # Parent link enables stack-scoped state sharing: ephemeral child contexts
        # (e.g. short-lived tracing) can delegate to a stable root without relying
        # on process-wide globals.
        self._parent: MPContext | None = parent

    # Basic topology helpers
    def world_size(self) -> int:
        return len(self.cluster_spec.nodes)

    @property
    def parent(self) -> MPContext | None:
        """Direct parent context or None if this is root."""
        return self._parent

    def root(self) -> MPContext:
        """Return the root context (follow parent chain)."""
        ctx: MPContext = self
        visited: set[int] = set()
        while ctx._parent is not None:
            if id(ctx) in visited:
                raise RuntimeError("Cycle detected in MPContext parent chain")
            visited.add(id(ctx))
            ctx = ctx._parent
        return ctx


class MPObject(ABC):
    """The base class for all objects in mp-system."""

    @property
    @abstractmethod
    def mptype(self) -> MPType:
        """The type information of the object.

        This property is readonly (mandatory) and will be used for JAX compilation
        to determine the appropriate data type during trace and compilation phases.
        MPType can be passed between different MPObjects as a value.
        """

    @property
    def dtype(self) -> DType:
        return self.mptype.dtype

    @property
    def shape(self) -> Shape:
        return self.mptype.shape

    @property
    def schema(self) -> TableType:
        """The table schema of the object.

        Only available for table types.
        """
        return self.mptype.schema

    @property
    def pmask(self) -> Mask | None:
        return self.mptype.pmask

    @property
    def attrs(self) -> dict[str, Any]:
        return self.mptype.attrs

    @property
    @abstractmethod
    def ctx(self) -> MPContext:
        """Return the context of the object."""


# Forward docstrings from MPType to MPObject
MPObject.dtype.__doc__ = MPType.dtype.__doc__
MPObject.shape.__doc__ = MPType.shape.__doc__
MPObject.schema.__doc__ = MPType.schema.__doc__
MPObject.pmask.__doc__ = MPType.pmask.__doc__
MPObject.attrs.__doc__ = MPType.attrs.__doc__
