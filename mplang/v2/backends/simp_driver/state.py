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

"""SimpDriver abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mplang.v2.runtime.dialect_state import DialectState

if TYPE_CHECKING:
    from concurrent.futures import Future

    from mplang.v2.edsl.graph import Graph


class SimpDriver(DialectState, ABC):
    """Abstract base class for Simp Host drivers.

    All simp drivers must implement submit/fetch/collect interface
    for dispatching work to workers.
    """

    dialect_name: str = "simp"

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Number of workers."""
        ...

    @abstractmethod
    def submit(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Future[Any]:
        """Submit graph execution to a worker."""
        ...

    @abstractmethod
    def fetch(self, rank: int, uri: str) -> Future[Any]:
        """Fetch data from a worker."""
        ...

    @abstractmethod
    def collect(self, futures: list[Future[Any]]) -> list[Any]:
        """Collect results from futures."""
        ...
