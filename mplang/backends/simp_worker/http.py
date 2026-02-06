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
    from mplang.backends.simp_http_worker import create_worker_app
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
import os
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mplang.backends.simp_worker.state import SimpWorker
from mplang.edsl import serde
from mplang.edsl.graph import Graph
from mplang.runtime.interpreter import ExecutionTracer, Interpreter
from mplang.runtime.object_store import ObjectStore
from mplang.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RecvTimeoutError(TimeoutError):
    """Raised when recv() times out waiting for a message."""

    def __init__(self, frm: int, key: str, timeout: float):
        self.frm = frm
        self.key = key
        self.timeout = timeout
        super().__init__(
            f"Timeout after {timeout}s waiting for message from rank {frm} key={key}"
        )


class SendTimeoutError(TimeoutError):
    """Raised when send_sync() times out."""

    def __init__(self, to: int, key: str, timeout: float):
        self.to = to
        self.key = key
        self.timeout = timeout
        super().__init__(f"Timeout after {timeout}s sending to rank {to} key={key}")


# ---------------------------------------------------------------------------
# Configuration and Statistics
# ---------------------------------------------------------------------------


@dataclass
class CommConfig:
    """Configuration for HttpCommunicator.

    Attributes:
        default_send_timeout: Default timeout (seconds) for send_sync operations.
        default_recv_timeout: Default timeout (seconds) for recv operations.
            None means wait indefinitely (legacy behavior).
        http_timeout: Timeout for individual HTTP requests (None = no timeout).
        max_retries: Maximum number of retries for failed sends.
        retry_backoff: Base delay (seconds) for exponential backoff between retries.
    """

    default_send_timeout: float = 60.0
    default_recv_timeout: float | None = None  # None = no timeout (backward compatible)
    http_timeout: float | None = None
    max_retries: int = 0
    retry_backoff: float = 1.0


@dataclass
class CommStats:
    """Communication statistics for observability.

    All fields are updated atomically using a lock.
    """

    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    send_errors: int = 0
    recv_timeouts: int = 0
    total_send_time_ms: float = 0.0
    total_recv_wait_time_ms: float = 0.0

    # Use RLock (reentrant lock) because snapshot() calls avg_* properties
    # which also acquire the lock
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record_send(self, size_bytes: int, duration_ms: float) -> None:
        """Record a successful send."""
        with self._lock:
            self.messages_sent += 1
            self.bytes_sent += size_bytes
            self.total_send_time_ms += duration_ms

    def record_send_error(self) -> None:
        """Record a send error."""
        with self._lock:
            self.send_errors += 1

    def record_recv(self, size_bytes: int, wait_time_ms: float) -> None:
        """Record a successful receive."""
        with self._lock:
            self.messages_received += 1
            self.bytes_received += size_bytes
            self.total_recv_wait_time_ms += wait_time_ms

    def record_recv_timeout(self) -> None:
        """Record a receive timeout."""
        with self._lock:
            self.recv_timeouts += 1

    @property
    def avg_send_latency_ms(self) -> float:
        """Average send latency in milliseconds."""
        with self._lock:
            if self.messages_sent == 0:
                return 0.0
            return self.total_send_time_ms / self.messages_sent

    @property
    def avg_recv_wait_time_ms(self) -> float:
        """Average receive wait time in milliseconds."""
        with self._lock:
            if self.messages_received == 0:
                return 0.0
            return self.total_recv_wait_time_ms / self.messages_received

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of current stats as a dict."""
        with self._lock:
            return {
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
                "send_errors": self.send_errors,
                "recv_timeouts": self.recv_timeouts,
                "avg_send_latency_ms": self.avg_send_latency_ms,
                "avg_recv_wait_time_ms": self.avg_recv_wait_time_ms,
            }

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.messages_sent = 0
            self.messages_received = 0
            self.bytes_sent = 0
            self.bytes_received = 0
            self.send_errors = 0
            self.recv_timeouts = 0
            self.total_send_time_ms = 0.0
            self.total_recv_wait_time_ms = 0.0


# ---------------------------------------------------------------------------
# HttpCommunicator
# ---------------------------------------------------------------------------


class HttpCommunicator:
    """Communicator using HTTP requests for inter-worker communication.

    Uses a background thread pool for sending to avoid blocking the main execution.

    Attributes:
        rank: This worker's rank
        world_size: Total number of workers
        endpoints: HTTP endpoints for all workers
        config: Communication configuration
        stats: Communication statistics
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        endpoints: list[str],
        tracer: ExecutionTracer | None = None,
        config: CommConfig | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.endpoints = endpoints
        self.tracer = tracer
        self.config = config or CommConfig()
        self.stats = CommStats()
        self._mailbox: dict[tuple[int, str], Any] = {}
        self._cond = threading.Condition()
        self._send_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=world_size, thread_name_prefix=f"comm_send_{rank}"
        )
        self._pending_sends: list[concurrent.futures.Future[None]] = []
        self.client = httpx.Client(timeout=self.config.http_timeout)

    def send(self, to: int, key: str, data: Any, *, is_raw_bytes: bool = False) -> None:
        """Send data to another rank asynchronously.

        The send is queued and executed in a background thread. Exceptions are
        captured and will be raised when wait_pending_sends() is called.

        Args:
            to: Target rank.
            key: Message key.
            data: Payload.
            is_raw_bytes: If True, treat `data` as raw bytes and transmit as
                base64-encoded bytes (no serde). If False, the transport may still
                treat `bytes` payloads as raw bytes.
        """
        future = self._send_executor.submit(self._do_send, to, key, data, is_raw_bytes)
        self._pending_sends.append(future)

    def send_sync(
        self,
        to: int,
        key: str,
        data: Any,
        *,
        is_raw_bytes: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Send data to another rank synchronously (blocking).

        Waits for the send to complete before returning. Raises immediately
        if the send fails.

        Args:
            to: Target rank.
            key: Message key.
            data: Payload.
            is_raw_bytes: If True, treat `data` as raw bytes.
            timeout: Timeout in seconds. If None, uses config.default_send_timeout.

        Raises:
            SendTimeoutError: If send times out.
            RuntimeError: If send fails for other reasons.
        """
        effective_timeout = (
            timeout if timeout is not None else self.config.default_send_timeout
        )
        future = self._send_executor.submit(self._do_send, to, key, data, is_raw_bytes)
        try:
            future.result(timeout=effective_timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            self.stats.record_send_error()
            raise SendTimeoutError(to, key, effective_timeout) from e

    def _do_send(self, to: int, key: str, data: Any, is_raw_bytes: bool) -> None:
        """Perform the HTTP send."""
        url = f"{self.endpoints[to]}/comm/{key}"
        logger.debug(f"Rank {self.rank} sending to {to} key={key}, url={url}")

        # Raw-bytes transport rule:
        # - If caller explicitly marks raw bytes, always use raw path.
        # - Otherwise, if payload is `bytes`, use raw path.
        # This avoids coupling encoding format to key naming conventions.
        send_raw_bytes = is_raw_bytes or isinstance(data, bytes)

        if send_raw_bytes:
            import base64

            payload = base64.b64encode(data).decode("ascii")
        else:
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
            resp = self.client.put(
                url,
                json={
                    "data": payload,
                    "from_rank": self.rank,
                    "is_raw_bytes": send_raw_bytes,
                },
            )
            resp.raise_for_status()
            duration = time.time() - t0
            duration_ms = duration * 1000

            # Record stats
            self.stats.record_send(size_bytes, duration_ms)

            if self.tracer:
                self.tracer.log_custom_event(
                    name="comm.send_req",
                    start_ts=t0,
                    end_ts=t0 + duration,
                    cat="comm",
                    args={"to": to, "key": key, "bytes": size_bytes},
                )
        except Exception as e:
            self.stats.record_send_error()
            logger.error(f"Rank {self.rank} failed to send to {to}: {e}")
            raise RuntimeError(f"Failed to send to {to} ({url}): {e}") from e

    def recv(self, frm: int, key: str, *, timeout: float | None = None) -> Any:
        """Receive data from another rank (blocking).

        Args:
            frm: Source rank.
            key: Message key.
            timeout: Timeout in seconds. If None, uses config.default_recv_timeout.
                If config.default_recv_timeout is also None, waits indefinitely.

        Returns:
            The received data.

        Raises:
            RecvTimeoutError: If timeout is reached before message arrives.
        """
        logger.debug(f"Rank {self.rank} waiting recv from {frm} key={key}")
        mailbox_key = (frm, key)

        # Determine effective timeout
        effective_timeout = (
            timeout if timeout is not None else self.config.default_recv_timeout
        )

        t0 = time.time()
        with self._cond:
            while mailbox_key not in self._mailbox:
                # Calculate remaining time if timeout is set
                if effective_timeout is not None:
                    elapsed = time.time() - t0
                    remaining = effective_timeout - elapsed
                    if remaining <= 0:
                        self.stats.record_recv_timeout()
                        raise RecvTimeoutError(frm, key, effective_timeout)
                    wait_time = min(1.0, remaining)
                else:
                    wait_time = 1.0

                self._cond.wait(timeout=wait_time)

            data = self._mailbox.pop(mailbox_key)

        # Record stats (estimate size from data)
        wait_time_ms = (time.time() - t0) * 1000
        # Estimate bytes - for bytes data use len, otherwise use a nominal value
        size_bytes = len(data) if isinstance(data, bytes) else 0
        self.stats.record_recv(size_bytes, wait_time_ms)

        return data

    def on_receive(self, from_rank: int, key: str, data: Any) -> None:
        """Called when data is received from the HTTP endpoint."""
        mailbox_key = (from_rank, key)
        with self._cond:
            if mailbox_key in self._mailbox:
                raise RuntimeError(
                    f"Mailbox overflow: key {mailbox_key} already exists"
                )
            self._mailbox[mailbox_key] = data
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
    is_raw_bytes: bool = False  # NEW: indicates raw bytes (not serde)


class FetchRequest(BaseModel):
    """Request model for /fetch endpoint."""

    uri: str


def create_worker_app(
    rank: int,
    world_size: int,
    endpoints: list[str],
    spu_endpoints: dict[int, str] | None = None,
    enable_tracing: bool = False,
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
        enable_tracing: Whether to enable execution tracing for profiler.

    Returns:
        FastAPI application instance
    """
    import asyncio

    # Register operation implementations (lazy import to avoid slow module-level loading)
    from mplang.backends import spu_impl as _spu_impl  # noqa: F401
    from mplang.backends import tensor_impl as _tensor_impl  # noqa: F401
    from mplang.backends.simp_worker import ops as _simp_worker_ops  # noqa: F401

    app = FastAPI(title=f"SIMP Worker {rank}")

    # Persistence root: ${MPLANG_DATA_ROOT}/<cluster_id>/node<rank>/
    data_root = pathlib.Path(os.environ.get("MPLANG_DATA_ROOT", ".mpl"))
    cluster_id = os.environ.get("MPLANG_CLUSTER_ID", f"__http_{world_size}")
    root_dir = data_root / cluster_id / f"node{rank}"

    tracer = None
    if enable_tracing:
        trace_dir = root_dir / "trace"
        tracer = ExecutionTracer(enabled=True, trace_dir=trace_dir)
        tracer.start()

    comm = HttpCommunicator(rank, world_size, endpoints, tracer=tracer)
    store = ObjectStore(fs_root=str(root_dir))
    ctx = SimpWorker(rank, world_size, comm, store, spu_endpoints)

    # Register handlers
    from collections.abc import Callable
    from typing import cast

    from mplang.backends.simp_worker.ops import WORKER_HANDLERS

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
            # Handle raw bytes (SPU channels) vs serde data
            if req.is_raw_bytes:
                # Decode base64 to raw bytes
                import base64

                data = base64.b64decode(req.data)
            else:
                # Use secure JSON deserialization
                data = serde.loads_b64(req.data)

            comm.on_receive(req.from_rank, key, data)
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
