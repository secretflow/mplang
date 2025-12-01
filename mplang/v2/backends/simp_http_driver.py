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

import base64
import concurrent.futures
import logging
from typing import Any

import cloudpickle as pickle
import httpx

from mplang.v2.backends.simp_host import SimpHost
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

    def __init__(self, world_size: int, endpoints: list[str]):
        """Initialize the HTTP driver.

        Args:
            world_size: Number of workers
            endpoints: HTTP endpoints for each worker (must match world_size)
        """
        if len(endpoints) != world_size:
            raise ValueError(
                f"endpoints length ({len(endpoints)}) must match world_size ({world_size})"
            )
        super().__init__(world_size)
        self.endpoints = endpoints
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=world_size)

    def _submit(self, rank: int, graph: Graph, inputs: list[Any]) -> Any:
        """Submit graph execution to a remote worker."""
        return self.executor.submit(self._execute_on_worker, rank, graph, inputs)

    def _collect(self, futures: list[Any]) -> list[Any]:
        """Collect results from all futures."""
        return [f.result() for f in futures]

    def _execute_on_worker(self, rank: int, graph: Graph, inputs: list[Any]) -> Any:
        """Execute graph on a remote worker via HTTP."""
        url = f"{self.endpoints[rank]}/exec"
        logger.debug(f"Driver submitting to rank {rank} url={url}")

        graph_pkl = base64.b64encode(pickle.dumps(graph)).decode("utf-8")
        inputs_pkl = base64.b64encode(pickle.dumps(inputs)).decode("utf-8")

        try:
            resp = httpx.post(
                url,
                json={"graph_pkl": graph_pkl, "inputs_pkl": inputs_pkl},
                timeout=None,  # Execution might take long
            )
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"Driver received result from rank {rank}")
            return pickle.loads(base64.b64decode(data["result"]))
        except Exception as e:
            logger.error(f"Driver failed to execute on rank {rank}: {e}")
            raise RuntimeError(f"Failed to execute on rank {rank}: {e}") from e

    def shutdown(self) -> None:
        """Shutdown the driver."""
        self.executor.shutdown()
