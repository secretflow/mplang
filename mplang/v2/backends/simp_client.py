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

"""SIMP Client Contexts for Host Interpreter.

This module defines the client-side capability contexts that are passed to
the Host Interpreter. These contexts handle:
1. Submitting jobs to workers (SPMD dispatch).
2. Collecting results.
3. Managing I/O endpoints.
"""

from __future__ import annotations

import concurrent.futures
import logging
import pathlib
from typing import Any, Protocol, runtime_checkable

import httpx

from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Graph

logger = logging.getLogger(__name__)


@runtime_checkable
class SimpClient(Protocol):
    """Protocol for SIMP Host Clients."""

    world_size: int

    def submit(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Submit a graph execution to a specific rank."""
        ...

    def collect(self, futures: list[Any]) -> list[Any]:
        """Collect results from futures."""
        ...

    def fetch(self, rank: int, uri: str) -> Any:
        """Fetch data from a specific rank."""
        ...


class ThreadingClient:
    """Invokes workers via threading (shared process).

    This client manages local worker threads for simulation.
    """

    def __init__(
        self,
        world_size: int,
        context: Any,  # Simulation Context (orchestrator state)
        root_dir: pathlib.Path,
        use_serde: bool = False,
    ):
        self.world_size = world_size
        self.context = context
        self.root_dir = root_dir
        self.use_serde = use_serde
        self.workers: list[Any] = []  # List of worker Interpreters, set by Factory

    def submit(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Submit execution to local worker thread."""
        return self.context.executor.submit(
            self._run_worker, rank, graph, inputs, job_id=job_id
        )

    def collect(self, futures: list[Any]) -> list[Any]:
        """Wait for threads."""
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_EXCEPTION
        )
        for f in done:
            exc = f.exception()
            if exc:
                # Cancel others and fail
                for nf in futures:
                    nf.cancel()
                self.context.shutdown(wait=False)
                raise exc
        return [f.result() for f in futures]

    def fetch(self, rank: int, uri: str) -> Any:
        """Fetch directly from worker store."""
        worker = self.workers[rank]
        # Access store directly from worker context
        # Worker interpreter has 'context' attribute which is SimpWorkerContext
        worker_ctx = worker.context
        return self.context.executor.submit(lambda: worker_ctx.store.get(uri))

    def _run_worker(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Execute on pure worker interpreter."""
        worker_interp = self.workers[rank]
        # Resolve inputs (URI -> Value) locally for simulation if needed
        # But wait, SimpWorkerContext logic was "resolve inputs from store".
        # We need to call the standard interpreter.evaluate_graph method.
        # But 'inputs' here might be URIs if they came from HostVar.

        # In the old SimpWorker.execute_job, we did resolution manually.
        # Here we should rely on the worker interpreter to handle it?
        # Standard evaluate_graph doesn't auto-resolve URIs (it expects Values).
        # We need a small helper or wrapper for "Remote Execution Logic"
        # which is exactly what the Worker Definition provides.

        # Let's assume the worker interpreter is configured to handle this,
        # or we do the resolution here (which is cleaner for "Simulation").
        # Actually, in real RPC, the Worker Runtime receives the request and resolves it.
        # So we should delegate to a helper function on the Worker Context?

        # Let's verify what WorkerContext does. It has 'store'.
        worker_ctx = worker_interp.context

        # Resolve inputs
        resolved_inputs = []
        for inp in inputs:
            if isinstance(inp, str) and "://" in inp:
                resolved_inputs.append(worker_ctx.store.get(inp))
            else:
                resolved_inputs.append(inp)

        # Execute
        results = worker_interp.evaluate_graph(graph, resolved_inputs, job_id)

        # Store results
        if not graph.outputs:
            return None
        if len(graph.outputs) == 1:
            val = results
            if val is None:
                return None
            return worker_ctx.store.put(val)
        else:
            return [worker_ctx.store.put(res) for res in results]


class HttpClient:
    """HTTP-based client."""

    def __init__(self, world_size: int, endpoints: list[str]):
        if len(endpoints) != world_size:
            raise ValueError("Endpoint count mismatch")
        self.world_size = world_size
        self.endpoints = endpoints
        self.client = httpx.Client(timeout=None)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=world_size)

    def submit(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        return self.executor.submit(
            self._do_request, rank, graph, inputs, job_id
        )

    def collect(self, futures: list[Any]) -> list[Any]:
        return [f.result() for f in futures]

    def fetch(self, rank: int, uri: str) -> Any:
        return self.executor.submit(self._do_fetch, rank, uri)

    def _do_request(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        url = f"{self.endpoints[rank]}/exec"
        graph_b64 = serde.dumps_b64(graph)
        inputs_b64 = serde.dumps_b64(inputs)
        payload = {"graph": graph_b64, "inputs": inputs_b64}
        if job_id:
            payload["job_id"] = job_id

        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        return serde.loads_b64(resp.json()["result"])

    def _do_fetch(self, rank: int, uri: str) -> Any:
        url = f"{self.endpoints[rank]}/fetch"
        resp = self.client.post(url, json={"uri": uri})
        resp.raise_for_status()
        return serde.loads_b64(resp.json()["result"])

    def shutdown(self):
        self.client.close()
        self.executor.shutdown()
