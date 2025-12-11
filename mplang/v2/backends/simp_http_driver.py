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

"""SIMP HTTP Driver module.

Provides the HTTP-based driver for distributed deployment.
This driver coordinates remote HTTP workers.

Usage:
    from mplang.v2.backends.simp_http_driver import SimpHttpDriver

    endpoints = ["http://host1:8000", "http://host2:8000"]
    driver = SimpHttpDriver(world_size=2, endpoints=endpoints)
    result = driver.evaluate_graph(graph, inputs)
"""

from __future__ import annotations

import concurrent.futures
import logging
import pathlib
from typing import Any

import httpx

from mplang.v2.backends.simp_host import SimpHost
from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Graph

logger = logging.getLogger(__name__)


class SimpHttpDriver(SimpHost):
    """SIMP driver that coordinates remote HTTP workers.

    This driver sends Graph IR and inputs to remote worker endpoints
    via HTTP, then collects and assembles the results.

    Attributes:
        endpoints: List of HTTP endpoints for each worker
        executor: Thread pool for parallel HTTP requests
    """

    def __init__(
        self,
        world_size: int,
        endpoints: list[str],
        *,
        root_dir: pathlib.Path | None = None,
    ):
        """Initialize the HTTP driver.

        Args:
            world_size: Number of workers
            endpoints: HTTP endpoints for each worker (must match world_size)
            root_dir: Driver root directory. If None, generates default path.
        """
        if len(endpoints) != world_size:
            raise ValueError(
                f"endpoints length ({len(endpoints)}) must match world_size ({world_size})"
            )
        # Generate default root_dir if not provided
        if root_dir is None:
            import os

            data_root = pathlib.Path(os.environ.get("MPLANG_DATA_ROOT", ".mpl"))
            root_dir = data_root / f"__http_{world_size}" / "__host__"

        super().__init__(world_size, root_dir=root_dir)
        self.cluster_root = root_dir.parent
        self.endpoints = endpoints
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=world_size)
        self.client = httpx.Client(timeout=None)

    def _submit(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Submit graph execution to a remote worker."""
        assert self.executor is not None
        return self.executor.submit(
            self._execute_on_worker, rank, graph, inputs, job_id
        )

    def _collect(self, futures: list[Any]) -> list[Any]:
        """Collect results from all futures."""
        return [f.result() for f in futures]

    def _execute_on_worker(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Execute graph on a remote worker via HTTP."""
        url = f"{self.endpoints[rank]}/exec"
        logger.debug(f"Driver submitting to rank {rank} url={url}")

        # Use secure JSON serialization instead of pickle
        graph_b64 = serde.dumps_b64(graph)
        inputs_b64 = serde.dumps_b64(inputs)

        payload = {"graph": graph_b64, "inputs": inputs_b64}
        if job_id:
            payload["job_id"] = job_id

        try:
            resp = self.client.post(
                url,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"Driver received result from rank {rank}")
            return serde.loads_b64(data["result"])
        except Exception as e:
            logger.error(f"Driver failed to execute on rank {rank}: {e}")
            raise RuntimeError(f"Failed to execute on rank {rank}: {e}") from e

    def _fetch(self, rank: int, uri: str) -> Any:
        """Fetch data from a remote worker via HTTP."""
        assert self.executor is not None
        return self.executor.submit(self._do_fetch, rank, uri)

    def _do_fetch(self, rank: int, uri: str) -> Any:
        url = f"{self.endpoints[rank]}/fetch"
        logger.debug(f"Driver fetching from rank {rank} uri={uri}")

        payload = {"uri": uri}
        try:
            resp = self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return serde.loads_b64(data["result"])
        except Exception as e:
            logger.error(f"Driver failed to fetch from rank {rank}: {e}")
            raise RuntimeError(f"Failed to fetch from rank {rank}: {e}") from e

    def shutdown(self) -> None:
        """Shutdown the driver."""
        if self.executor:
            self.executor.shutdown()
        self.client.close()
