"""SIMP HTTP Worker module.

Provides the HTTP-based worker entry point for distributed deployment.
This module contains:
- HttpCommunicator: HTTP-based inter-worker communication
- create_worker_app: Factory for FastAPI application

Usage:
    # Start a worker server
    from mplang2.backends.simp_http_worker import create_worker_app
    import uvicorn

    app = create_worker_app(rank=0, world_size=3, endpoints=[...])
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import base64
import concurrent.futures
import logging
import threading
from typing import Any

import cloudpickle as pickle
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Register operation implementations (side-effect imports)
from mplang2.backends import simp_impl as _simp_impl  # noqa: F401
from mplang2.backends import tensor_impl as _tensor_impl  # noqa: F401
from mplang2.backends.simp_worker import WorkerInterpreter
from mplang2.edsl.graph import Graph

logger = logging.getLogger(__name__)


class HttpCommunicator:
    """Communicator using HTTP requests for inter-worker communication.

    Uses a background thread pool for sending to avoid blocking the main execution.

    Attributes:
        rank: This worker's rank
        world_size: Total number of workers
        endpoints: HTTP endpoints for all workers
    """

    def __init__(self, rank: int, world_size: int, endpoints: list[str]):
        self.rank = rank
        self.world_size = world_size
        self.endpoints = endpoints
        self._mailbox: dict[str, Any] = {}
        self._cond = threading.Condition()
        self._send_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=world_size, thread_name_prefix=f"comm_send_{rank}"
        )
        self._pending_sends: list[concurrent.futures.Future[None]] = []

    def send(self, to: int, key: str, data: Any) -> None:
        """Send data to another rank asynchronously."""
        future = self._send_executor.submit(self._do_send, to, key, data)
        self._pending_sends.append(future)

    def _do_send(self, to: int, key: str, data: Any) -> None:
        """Perform the HTTP send."""
        url = f"{self.endpoints[to]}/comm/{key}"
        logger.debug(f"Rank {self.rank} sending to {to} key={key}")
        payload = base64.b64encode(pickle.dumps(data)).decode("utf-8")
        try:
            resp = httpx.put(
                url, json={"data": payload, "from_rank": self.rank}, timeout=60.0
            )
            resp.raise_for_status()
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


class ExecRequest(BaseModel):
    """Request model for /exec endpoint."""

    graph_pkl: str
    inputs_pkl: str


class CommRequest(BaseModel):
    """Request model for /comm endpoint."""

    data: str
    from_rank: int


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
    comm = HttpCommunicator(rank, world_size, endpoints)
    worker = WorkerInterpreter(rank, world_size, comm, spu_endpoints)
    exec_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix=f"exec_{rank}"
    )

    def _do_execute(graph: Graph, inputs: list[Any]) -> Any:
        """Execute graph in worker thread."""
        result = worker.evaluate_graph(graph, inputs)
        comm.wait_pending_sends()
        return result

    @app.post("/exec")
    async def execute(req: ExecRequest) -> dict[str, str]:
        """Execute a graph on this worker."""
        logger.debug(f"Worker {rank} received exec request")
        try:
            graph = pickle.loads(base64.b64decode(req.graph_pkl))
            inputs = pickle.loads(base64.b64decode(req.inputs_pkl))
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(exec_pool, _do_execute, graph, inputs)
            return {"result": base64.b64encode(pickle.dumps(result)).decode("utf-8")}
        except Exception as e:
            logger.error(f"Worker {rank} exec failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.put("/comm/{key}")
    async def receive_comm(key: str, req: CommRequest) -> dict[str, str]:
        """Receive communication data from another worker."""
        logger.debug(f"Worker {rank} received comm key={key} from {req.from_rank}")
        try:
            data = pickle.loads(base64.b64decode(req.data))
            comm.on_receive(key, data)
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Worker {rank} comm failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        """Cleanup on shutdown."""
        comm.shutdown()
        exec_pool.shutdown(wait=True)

    return app
