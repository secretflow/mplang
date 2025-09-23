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

"""Unified HTTP server spawning utilities for tests.

Centralizes dynamic port allocation, process management, and health checking
to remove duplication across runtime/integration test modules.
"""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
from dataclasses import dataclass

import httpx

DEFAULT_HOST = "localhost"
DEFAULT_HEALTH_PATH = "/health"


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((DEFAULT_HOST, 0))
        return s.getsockname()[1]


def get_free_ports(n: int) -> list[int]:
    """Return a list of n unique free ports.

    Note: This function is not safe from race conditions. For spawning servers,
    prefer a reservation strategy like the one in `spawn_http_servers`.
    This is suitable for client-side ephemeral port suggestions where no
    binding occurs.
    """
    ports = set()
    for _ in range(n * 2):  # Try a few more times in case of collision
        if len(ports) >= n:
            break
        ports.add(get_free_port())
    if len(ports) < n:
        raise RuntimeError(f"Could not get {n} unique free ports")
    return list(ports)


@dataclass
class SpawnResult:
    ports: list[int]
    addresses: list[str]
    processes: list[multiprocessing.Process]

    def stop(self) -> None:
        for proc in self.processes:
            if not proc.is_alive():
                continue
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()


def _verbose() -> bool:
    return os.environ.get("MPLANG_TEST_HTTP_VERBOSE", "0") not in {
        "",
        "0",
        "false",
        "False",
    }


def _run_server_process(host, write_conn, log_level):
    """The target function for the server process using spawn context."""
    import uvicorn

    from mplang.runtime.server import app

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]

    write_conn.send(port)
    write_conn.close()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ws="none",
    )
    server = uvicorn.Server(config)
    if _verbose():
        print(f"[spawn_http_servers] starting server pid={os.getpid()} port={port}")

    server.run(sockets=[sock])


def spawn_http_servers(
    n: int,
    *,
    host: str = DEFAULT_HOST,
    health_path: str = DEFAULT_HEALTH_PATH,
    max_wait: float = 10.0,
    poll_interval: float = 0.1,
    request_timeout: float = 0.25,
    log_level: str = "critical",
) -> SpawnResult:
    """Spawn n uvicorn servers using spawn context (safe for JAX/multithreaded environments).

    This approach involves:
    1. Creating and binding sockets to ephemeral ports (port=0) in the child process.
    2. Passing the port number to the parent process via pipe.
    3. Uvicorn in the child process then uses the existing socket.
    This avoids port conflicts and race conditions.
    """

    ports = []
    processes: list[multiprocessing.Process] = []

    # Use 'spawn' context to avoid deadlocks with JAX and other multithreaded libraries
    ctx = multiprocessing.get_context("spawn")

    try:
        for _ in range(n):
            read_conn, write_conn = ctx.Pipe(duplex=False)
            p = ctx.Process(
                target=_run_server_process,
                args=(host, write_conn, log_level),
            )
            p.daemon = True
            p.start()
            processes.append(p)  # type: ignore[arg-type]

            write_conn.close()  # Close the write end in the parent process
            try:
                port = read_conn.recv()  # Expect the child to send back the port
                ports.append(port)
            except EOFError:
                p.terminate()
                p.join(timeout=5)
                raise RuntimeError(
                    "Failed to start server process, no port received"
                ) from None
            finally:
                read_conn.close()
    except Exception:
        # If any part of the loop fails, clean up all processes started so far
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
                if p.is_alive():
                    p.kill()
        raise

    addresses = [f"http://{host}:{p}" for p in ports]
    # Health check and cleanup
    return _wait_for_servers(
        ports,
        processes,
        host,
        health_path,
        max_wait,
        poll_interval,
        request_timeout,
        addresses,
    )


def _wait_for_servers(
    ports,
    processes,
    host,
    health_path,
    max_wait,
    poll_interval,
    request_timeout,
    addresses,
):
    """Common logic to wait for servers to become healthy."""
    # Health check loop
    attempts = int(max_wait / poll_interval)
    for port in ports:
        ready = False
        last_ex = None
        for attempt in range(attempts):
            try:
                if _verbose():
                    print(
                        f"[spawn_http_servers] health check for port {port} (attempt {attempt + 1}/{attempts})"
                    )
                r = httpx.get(
                    f"http://{host}:{port}{health_path}", timeout=request_timeout
                )
                # The flexible health check logic from test_communicator is now centralized here.
                if r.status_code == 200:
                    payload = r.json()
                    if payload == {"status": "ok"} or (
                        isinstance(payload.get("status"), dict)
                        and payload["status"].get("code") == 404
                    ):
                        ready = True
                        if _verbose() and payload != {"status": "ok"}:
                            print(
                                f"[spawn_http_servers] WARNING: health returned nested status form: {payload}"
                            )
                        break
            except Exception as e:
                last_ex = e
            time.sleep(poll_interval)

        if not ready:
            # Stop already started processes before raising
            for p in processes:
                if p.is_alive():
                    p.terminate()
            msg = f"Server on port {port} failed to become healthy in {max_wait}s."
            if last_ex:
                msg += f" Last exception: {last_ex}"
            raise RuntimeError(msg)
    if _verbose():
        print(
            f"[spawn_http_servers] All {len(ports)} servers are healthy on ports: {ports}"
        )

    return SpawnResult(ports=ports, addresses=addresses, processes=processes)


# Pytest integration (optional)
try:  # pragma: no cover - import guard for non-pytest contexts
    import pytest

    @pytest.fixture(scope="module")
    def http_servers(request):  # type: ignore
        """Generic parametrized fixture.

        Usage: @pytest.mark.parametrize('http_servers', [3], indirect=True)
        """
        n = getattr(request, "param", 1)
        result = spawn_http_servers(n)
        try:
            yield result
        finally:
            result.stop()

except Exception:  # pragma: no cover
    pass
