"""SIMP HTTP Backend.

Provides HttpCommunicator, HttpWorker (FastAPI), and HttpHost.
"""

import base64
import concurrent.futures
import logging
import threading
from typing import Any

import cloudpickle as pickle
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mplang2.backends.tensor_impl  # noqa: F401
from mplang2.backends.simp_host import SimpHost
from mplang2.backends.simp_worker import WorkerInterpreter
from mplang2.edsl.graph import Graph

logger = logging.getLogger(__name__)


class HttpCommunicator:
    """Communicator using HTTP requests."""

    def __init__(self, rank: int, world_size: int, endpoints: list[str]):
        self.rank = rank
        self.world_size = world_size
        self.endpoints = endpoints
        self._mailbox: dict[str, Any] = {}
        self._cond = threading.Condition()

    def send(self, to: int, key: str, data: Any) -> None:
        url = f"{self.endpoints[to]}/comm/{key}"
        logger.debug(f"Rank {self.rank} sending to {to} key={key} url={url}")
        # Serialize data
        payload = base64.b64encode(pickle.dumps(data)).decode("utf-8")
        try:
            # Use a short timeout for connection, longer for read if needed
            httpx.put(url, json={"data": payload, "from_rank": self.rank}, timeout=30.0)
            logger.debug(f"Rank {self.rank} sent to {to} key={key}")
        except Exception as e:
            logger.error(f"Rank {self.rank} failed to send to {to}: {e}")
            raise RuntimeError(f"Failed to send to {to} ({url}): {e}") from e

    def recv(self, frm: int, key: str) -> Any:
        logger.debug(f"Rank {self.rank} waiting recv from {frm} key={key}")
        with self._cond:
            while key not in self._mailbox:
                self._cond.wait()
            logger.debug(f"Rank {self.rank} received from {frm} key={key}")
            return self._mailbox.pop(key)

    def on_receive(self, key: str, data: Any) -> None:
        logger.debug(f"Rank {self.rank} on_receive key={key}")
        with self._cond:
            if key in self._mailbox:
                # This might happen if multiple sends occur with same key (shouldn't in SIMP)
                logger.warning(f"Rank {self.rank} overwriting key={key}")
            self._mailbox[key] = data
            self._cond.notify_all()


class ExecRequest(BaseModel):
    graph_pkl: str
    inputs_pkl: str


class CommRequest(BaseModel):
    data: str
    from_rank: int


def create_worker_app(rank: int, world_size: int, endpoints: list[str]) -> FastAPI:
    """Create a FastAPI app for the worker."""
    app = FastAPI()
    comm = HttpCommunicator(rank, world_size, endpoints)
    worker = WorkerInterpreter(rank, world_size, comm)

    @app.post("/exec")
    def execute(req: ExecRequest) -> dict[str, str]:
        logger.debug(f"Worker {rank} received exec request")
        try:
            graph = pickle.loads(base64.b64decode(req.graph_pkl))
            inputs = pickle.loads(base64.b64decode(req.inputs_pkl))
            result = worker.evaluate_graph(graph, inputs)
            logger.debug(f"Worker {rank} finished exec request")
            return {"result": base64.b64encode(pickle.dumps(result)).decode("utf-8")}
        except Exception as e:
            logger.error(f"Worker {rank} exec failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.put("/comm/{key}")
    async def receive_comm(key: str, req: CommRequest) -> dict[str, str]:
        logger.debug(f"Worker {rank} received comm key={key} from {req.from_rank}")
        try:
            data = pickle.loads(base64.b64decode(req.data))
            comm.on_receive(key, data)
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Worker {rank} comm failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


class HttpHost(SimpHost):
    """Host that coordinates networked workers."""

    def __init__(self, world_size: int, endpoints: list[str]):
        super().__init__(world_size)
        self.endpoints = endpoints
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=world_size)

    def _submit(self, rank: int, graph: Graph, inputs: dict[Any, Any]) -> Any:
        return self.executor.submit(self._run_party, rank, graph, inputs)

    def _collect(self, futures: list[Any]) -> list[Any]:
        return [f.result() for f in futures]

    def _run_party(self, rank: int, graph: Graph, inputs: dict[Any, Any]) -> Any:
        url = f"{self.endpoints[rank]}/exec"
        logger.debug(f"Host submitting to rank {rank} url={url}")
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
            logger.debug(f"Host received result from rank {rank}")
            return pickle.loads(base64.b64decode(data["result"]))
        except Exception as e:
            logger.error(f"Host failed to execute on rank {rank}: {e}")
            raise RuntimeError(f"Failed to execute on rank {rank}: {e}") from e

    def shutdown(self) -> None:
        self.executor.shutdown()
