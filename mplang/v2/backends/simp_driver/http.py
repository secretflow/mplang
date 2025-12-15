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

"""Simp Driver HTTP IPC (SimpHttpDriver, make_driver)."""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
from typing import TYPE_CHECKING, Any

import httpx

from mplang.v2.backends.simp_driver.state import SimpDriver
from mplang.v2.edsl import serde
from mplang.v2.runtime.interpreter import Interpreter
from mplang.v2.runtime.object_store import ObjectStore

if TYPE_CHECKING:
    from concurrent.futures import Future

    from mplang.v2.edsl.graph import Graph
    from mplang.v2.edsl.spec import ClusterSpec


class SimpHttpDriver(SimpDriver):
    """Simp Driver for remote HTTP IPC.

    Implements submit/fetch/collect interface for dispatching work via HTTP.
    """

    dialect_name: str = "simp"

    def __init__(
        self,
        endpoints: list[str],
        *,
        cluster_spec: ClusterSpec | None = None,
    ) -> None:
        """Create remote simp driver.

        Args:
            endpoints: List of HTTP endpoints for workers.
            cluster_spec: Optional cluster specification for metadata.
        """
        # Normalize endpoints
        self._endpoints = []
        for ep in endpoints:
            if not ep.startswith("http://") and not ep.startswith("https://"):
                ep = f"http://{ep}"
            self._endpoints.append(ep)

        self._world_size = len(endpoints)
        self._cluster_spec = cluster_spec

        # Construct driver root
        data_root = pathlib.Path(os.environ.get("MPLANG_DATA_ROOT", ".mpl"))
        if cluster_spec:
            self.driver_root = data_root / cluster_spec.cluster_id / "__driver__"
        else:
            self.driver_root = data_root / "__remote__" / "__driver__"

        # HTTP client and executor
        self._client = httpx.Client(timeout=None)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._world_size
        )

    @property
    def world_size(self) -> int:
        return self._world_size

    def submit(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Future[Any]:
        """Submit execution to remote worker via HTTP."""
        return self._executor.submit(self._do_request, rank, graph, inputs, job_id)

    def collect(self, futures: list[Future[Any]]) -> list[Any]:
        """Collect results from futures."""
        return [f.result() for f in futures]

    def fetch(self, rank: int, uri: str) -> Future[Any]:
        """Fetch data from remote worker."""
        return self._executor.submit(self._do_fetch, rank, uri)

    def _do_request(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Execute HTTP request."""
        url = f"{self._endpoints[rank]}/exec"
        graph_b64 = serde.dumps_b64(graph)
        inputs_b64 = serde.dumps_b64(inputs)
        payload = {"graph": graph_b64, "inputs": inputs_b64}
        if job_id:
            payload["job_id"] = job_id

        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        return serde.loads_b64(resp.json()["result"])

    def _do_fetch(self, rank: int, uri: str) -> Any:
        """Fetch data from remote worker."""
        url = f"{self._endpoints[rank]}/fetch"
        resp = self._client.post(url, json={"uri": uri})
        resp.raise_for_status()
        return serde.loads_b64(resp.json()["result"])

    def shutdown(self) -> None:
        """Close HTTP client and executor."""
        self._client.close()
        self._executor.shutdown()


def make_driver(endpoints: list[str], *, cluster_spec: Any = None) -> Interpreter:
    """Create an Interpreter configured for remote SIMP execution.

    This factory creates a SimpHttpDriver and returns an Interpreter
    with the simp dialect state attached.

    Args:
        endpoints: List of HTTP endpoints for workers.
        cluster_spec: Optional ClusterSpec for metadata.

    Returns:
        Configured Interpreter with simp state attached.

    Example:
        >>> interp = make_driver(["http://worker1:8000", "http://worker2:8000"])
        >>> with interp:
        ...     result = my_func()
    """
    from mplang.v2.backends.simp_driver.ops import DRIVER_HANDLERS

    if cluster_spec is None:
        from mplang.v2.libs.device import ClusterSpec

        cluster_spec = ClusterSpec.simple(
            world_size=len(endpoints), endpoints=endpoints
        )

    state = SimpHttpDriver(endpoints, cluster_spec=cluster_spec)

    from collections.abc import Callable

    handlers: dict[str, Callable[..., Any]] = {**DRIVER_HANDLERS}  # type: ignore[dict-item]
    interp = Interpreter(
        name="DriverInterpreter",
        root_dir=state.driver_root,
        handlers=handlers,
        store=ObjectStore(fs_root=str(state.driver_root)),
    )
    interp.set_dialect_state("simp", state)
    interp._cluster_spec = cluster_spec  # type: ignore[attr-defined]

    return interp
