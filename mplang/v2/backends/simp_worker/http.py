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

"""SIMP HTTP Worker module.

Provides the HTTP-based worker entry point for distributed deployment.
This module contains:
- HttpCommunicator: HTTP-based inter-worker communication
- create_worker_app: Factory for FastAPI application

Usage:
    # Start a worker server
    from mplang.v2.backends.simp_http_worker import create_worker_app
    import uvicorn

    app = create_worker_app(rank=0, world_size=3, endpoints=[...])
    uvicorn.run(app, host="0.0.0.0", port=8000)

Security:
    This module uses secure JSON-based serialization (serde) for computation
    graphs and data between workers. Unlike pickle, JSON deserialization
    cannot execute arbitrary code, making it safe for network communication.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import pathlib
import threading
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mplang.v2.backends import spu_impl as _spu_impl  # noqa: F401
from mplang.v2.backends import tensor_impl as _tensor_impl  # noqa: F401

# Register operation implementations (side-effect imports)
from mplang.v2.backends.simp_worker import SimpWorker
from mplang.v2.backends.simp_worker import ops as _simp_worker_ops  # noqa: F401
from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Graph
from mplang.v2.runtime.interpreter import ExecutionTracer, Interpreter
from mplang.v2.runtime.object_store import ObjectStore

logger = logging.getLogger(__name__)


class HttpCommunicator:
    """Communicator using HTTP requests for inter-worker communication.

    Uses a background thread pool for sending to avoid blocking the main execution.

    Attributes:
        rank: This worker's rank
        world_size: Total number of workers
        endpoints: HTTP endpoints for all workers
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        endpoints: list[str],
        tracer: ExecutionTracer | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.endpoints = endpoints
        self.tracer = tracer
        self._mailbox: dict[str, Any] = {}
        self._cond = threading.Condition()
        self._send_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=world_size, thread_name_prefix=f"comm_send_{rank}"
        )
        self._pending_sends: list[concurrent.futures.Future[None]] = []
        self.client = httpx.Client(timeout=None)

    def send(self, to: int, key: str, data: Any) -> None:
        """Send data to another rank asynchronously."""
        future = self._send_executor.submit(self._do_send, to, key, data)
        self._pending_sends.append(future)

    def _do_send(self, to: int, key: str, data: Any) -> None:
        """Perform the HTTP send."""
        url = f"{self.endpoints[to]}/comm/{key}"
        logger.debug(f"Rank {self.rank} sending to {to} key={key}")
        # Use secure JSON serialization
        payload = serde.dumps_b64(data)
        size_bytes = len(payload)

        # Log to profiler
        if self.tracer:
            self.tracer.log_custom_event(
                name="comm.send",
                start_ts=time.time(),
                end_ts=time.time(),  # Instant event for size? Or measure duration?
                cat="comm",
                args={"to": to, "key": key, "bytes": size_bytes},
            )

        try:
            t0 = time.time()
            resp = self.client.put(url, json={"data": payload, "from_rank": self.rank})
            resp.raise_for_status()
            duration = time.time() - t0
            if self.tracer:
                self.tracer.log_custom_event(
                    name="comm.send_req",
                    start_ts=t0,
                    end_ts=t0 + duration,
                    cat="comm",
                    args={"to": to, "key": key, "bytes": size_bytes},
                )
        except Exception as e:
            logger.error(f"Rank {self.rank} failed to send to {to}: {e}")
            raise RuntimeError(f"Failed to send to {to} ({url}): {e}") from e

    def recv(self, frm: int, key: str) -> Any:
        """Receive data from another rank (blocking)."""
        logger.debug(f"Rank {self.rank} waiting recv from {frm} key={key}")
        with self._cond:
            while key not in self._mailbox:
                self._cond.wait(timeout=1.0)
            return self._mailbox.pop(key)

    def on_receive(self, key: str, data: Any) -> None:
        """Called when data is received from the HTTP endpoint."""
        with self._cond:
            if key in self._mailbox:
                logger.warning(f"Rank {self.rank} overwriting key={key}")
            self._mailbox[key] = data
            self._cond.notify_all()

    def wait_pending_sends(self) -> None:
        """Wait for all pending sends to complete."""
        for future in self._pending_sends:
            try:
                future.result(timeout=60.0)
            except Exception as e:
                logger.error(f"Rank {self.rank} send failed: {e}")
        self._pending_sends.clear()

    def shutdown(self) -> None:
        """Shutdown the send executor."""
        self.wait_pending_sends()
        self._send_executor.shutdown(wait=True)
        self.client.close()


class ExecRequest(BaseModel):
    """Request model for /exec endpoint."""

    graph: str
    inputs: str
    job_id: str | None = None


class CommRequest(BaseModel):
    """Request model for /comm endpoint."""

    data: str
    from_rank: int


class FetchRequest(BaseModel):
    """Request model for /fetch endpoint."""

    uri: str


def create_worker_app(
    rank: int,
    world_size: int,
    endpoints: list[str],
    spu_endpoints: dict[int, str] | None = None,
) -> FastAPI:
    """Create a FastAPI app for the worker.

    The app uses async endpoints with run_in_executor to allow concurrent
    handling of /exec and /comm requests. This is essential for cross-party
    communication where one party sends while another receives.

    Args:
        rank: The global rank of this worker.
        world_size: Total number of workers.
        endpoints: HTTP endpoints for all workers (for shuffle communication).
        spu_endpoints: Optional dict mapping global_rank -> BRPC endpoint for SPU.

    Returns:
        FastAPI application instance
    """
    import asyncio

    app = FastAPI(title=f"SIMP Worker {rank}")

    # Persistence root: ${MPLANG_DATA_ROOT}/<cluster_id>/node<rank>/
    data_root = pathlib.Path(os.environ.get("MPLANG_DATA_ROOT", ".mpl"))
    cluster_id = os.environ.get("MPLANG_CLUSTER_ID", f"__http_{world_size}")
    root_dir = data_root / cluster_id / f"node{rank}"
    trace_dir = root_dir / "trace"

    # Enable tracing by default for now (or make it configurable via env)
    tracer = ExecutionTracer(enabled=True, trace_dir=trace_dir)
    tracer.start()

    comm = HttpCommunicator(rank, world_size, endpoints, tracer=tracer)
    store = ObjectStore(fs_root=str(root_dir))
    ctx = SimpWorker(rank, world_size, comm, store, spu_endpoints)

    # Register handlers
    from collections.abc import Callable
    from typing import cast

    from mplang.v2.backends.simp_worker.ops import WORKER_HANDLERS

    # func_impl is already imported at module level for side-effects
    handlers: dict[str, Callable[..., Any]] = {**WORKER_HANDLERS}  # type: ignore[dict-item]

    worker = Interpreter(
        tracer=tracer, root_dir=root_dir, handlers=handlers, store=store
    )
    # Register SimpWorker context as 'simp' dialect state
    worker.set_dialect_state("simp", ctx)

    exec_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix=f"exec_{rank}"
    )

    def _do_execute(graph: Graph, inputs: list[Any], job_id: str | None = None) -> Any:
        """Execute graph in worker thread."""
        # Resolve URI inputs (None means rank has no data)
        resolved_inputs = [
            store.get(inp) if inp is not None else None for inp in inputs
        ]

        result = worker.evaluate_graph(graph, resolved_inputs)
        comm.wait_pending_sends()

        # Store results and return URIs (result is always a list)
        if not graph.outputs:
            return None
        return [store.put(res) if res is not None else None for res in result]

    @app.post("/exec")
    async def execute(req: ExecRequest) -> dict[str, str]:
        """Execute a graph on this worker."""
        logger.debug(f"Worker {rank} received exec request")
        try:
            # Use secure JSON deserialization
            graph = serde.loads_b64(req.graph)
            inputs = serde.loads_b64(req.inputs)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                exec_pool, _do_execute, graph, inputs, req.job_id
            )
            return {"result": serde.dumps_b64(result)}
        except Exception as e:
            logger.error(f"Worker {rank} exec failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.put("/comm/{key}")
    async def receive_comm(key: str, req: CommRequest) -> dict[str, str]:
        """Receive communication data from another worker."""
        logger.debug(f"Worker {rank} received comm key={key} from {req.from_rank}")
        try:
            # Use secure JSON deserialization
            data = serde.loads_b64(req.data)
            comm.on_receive(key, data)
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Worker {rank} comm failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/fetch")
    async def fetch(req: FetchRequest) -> dict[str, str]:
        """Fetch data by URI."""
        logger.debug(f"Worker {rank} received fetch request for {req.uri}")
        try:
            state = cast(SimpWorker, worker.get_dialect_state("simp"))
            val = state.store.get(req.uri)
            return {"result": serde.dumps_b64(val)}
        except Exception as e:
            logger.error(f"Worker {rank} fetch failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/objects")
    async def list_objects() -> dict[str, list[str]]:
        """List all objects in the worker's store."""
        try:
            state = cast(SimpWorker, worker.get_dialect_state("simp"))
            return {"objects": state.store.list_objects()}
        except Exception as e:
            logger.error(f"Worker {rank} list_objects failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "rank": str(rank), "world_size": str(world_size)}

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        """Cleanup on shutdown."""
        comm.shutdown()
        exec_pool.shutdown(wait=True)

    return app
