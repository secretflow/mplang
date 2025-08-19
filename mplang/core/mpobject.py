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
from typing import Any

from mplang.core.dtype import DType
from mplang.core.mask import Mask
from mplang.core.mptype import MPType, Shape
from mplang.core.relation import RelationSchema


class MPContext(ABC):
    """The context of an MPObject."""

    @abstractmethod
    def psize(self) -> int:
        """Return the world size."""

    @abstractmethod
    def attrs(self) -> dict[str, Any]:
        """Return the attributes of the context."""

    def attr(self, key: str, default: Any = None) -> Any:
        """Return the attribute of the context by key."""
        return self.attrs().get(key, default)


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
    def schema(self) -> RelationSchema:
        """The relational schema of the object.

        Only available for relation types.
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
