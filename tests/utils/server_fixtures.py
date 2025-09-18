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
import sys
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


def _run_server_process(app, host, port, sock, log_level):
    """The target function for the server process using fork context (with socket)."""
    import uvicorn

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
    # The child process inherits the socket file descriptor. Uvicorn is
    # instructed to use this existing socket instead of creating a new one.
    server.run(sockets=[sock])


def _run_server_process_spawn(app_module, app_name, host, write_conn, log_level):
    """The target function for the server process using spawn context (without socket)."""
    import importlib

    import uvicorn

    # Import the app dynamically to avoid pickling issues
    module = importlib.import_module(app_module)
    app = getattr(module, app_name)

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
    # Child process binds to the port directly (spawn doesn't inherit sockets)
    server.run(sockets=[sock])


def spawn_http_servers(
    app,
    n: int,
    *,
    host: str = DEFAULT_HOST,
    health_path: str = DEFAULT_HEALTH_PATH,
    max_wait: float = 10.0,
    poll_interval: float = 0.1,
    request_timeout: float = 0.25,
    log_level: str = "critical",
) -> SpawnResult:
    """Spawn n uvicorn servers, choosing between fork and spawn contexts based on safety.

    This function automatically detects whether JAX or other multithreaded libraries
    are in use and chooses the appropriate multiprocessing context:
    - fork: More efficient, works with file descriptors, but incompatible with multithreaded code
    - spawn: Safer for multithreaded environments like JAX, but requires different approach
    """
    # Check if JAX is imported (indicating potential multithreading issues)
    # Also check if we're dealing with the runtime server app
    use_spawn = (
        "jax" in sys.modules or "jaxlib" in sys.modules
    )  # Use spawn if JAX or JAXlib is imported (multithreaded context)

    if use_spawn:
        return _spawn_http_servers_spawn(
            app,
            n,
            host=host,
            health_path=health_path,
            max_wait=max_wait,
            poll_interval=poll_interval,
            request_timeout=request_timeout,
            log_level=log_level,
        )
    else:
        return _spawn_http_servers_fork(
            app,
            n,
            host=host,
            health_path=health_path,
            max_wait=max_wait,
            poll_interval=poll_interval,
            request_timeout=request_timeout,
            log_level=log_level,
        )


def _spawn_http_servers_fork(
    app,
    n: int,
    *,
    host: str = DEFAULT_HOST,
    health_path: str = DEFAULT_HEALTH_PATH,
    max_wait: float = 10.0,
    poll_interval: float = 0.1,
    request_timeout: float = 0.25,
    log_level: str = "critical",
) -> SpawnResult:
    """Spawn n uvicorn servers using fork context (original implementation)."""
    sockets = []
    ports = []
    for _ in range(n):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, 0))
        sockets.append(sock)
        ports.append(sock.getsockname()[1])

    addresses = [f"http://{host}:{p}" for p in ports]
    processes: list[multiprocessing.Process] = []

    # Use 'fork' context for efficiency when multithreading isn't a concern
    ctx = multiprocessing.get_context("fork")

    for port, sock in zip(ports, sockets, strict=True):
        p = ctx.Process(
            target=_run_server_process, args=(app, host, port, sock, log_level)
        )
        p.daemon = True
        p.start()
        processes.append(p)
        # The socket can be closed in the parent process after the child has inherited it.
        sock.close()

    # Health check and cleanup logic remains the same
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


def _spawn_http_servers_spawn(
    app,
    n: int,
    *,
    host: str = DEFAULT_HOST,
    health_path: str = DEFAULT_HEALTH_PATH,
    max_wait: float = 10.0,
    poll_interval: float = 0.1,
    request_timeout: float = 0.25,
    log_level: str = "critical",
) -> SpawnResult:
    """Spawn n uvicorn servers using spawn context (safe for JAX/multithreaded environments)."""
    # For spawn context with the runtime server, we use a hardcoded approach
    # since the FastAPI app can't be pickled and socket inheritance doesn't work
    # Check if this is the runtime server app by examining where it was imported from
    import mplang.runtime.server

    if app is mplang.runtime.server.app:
        # Reserve ports by binding and releasing them
        ports = []
        processes: list[multiprocessing.Process] = []

        # Use 'spawn' context to avoid deadlocks with JAX and other multithreaded libraries
        ctx = multiprocessing.get_context("spawn")

        # Hardcoded approach for mplang runtime server
        for _ in range(n):
            read_conn, write_conn = ctx.Pipe(duplex=False)
            p = ctx.Process(
                target=_run_server_process_spawn,
                args=("mplang.runtime.server", "app", host, write_conn, log_level),
            )
            p.daemon = True
            p.start()
            processes.append(p)

            write_conn.close()  # Close the write end in the parent process
            try:
                port = read_conn.recv()  # Expect the child to send back the port
                ports.append(port)
            except EOFError:
                p.terminate()
                p.join()
                raise RuntimeError(
                    "Failed to start server process, no port received"
                ) from None
            finally:
                read_conn.close()

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
    else:
        # For other apps, fall back to fork context with a warning
        import warnings

        warnings.warn(
            f"Spawn context not implemented for app module {app.__module__}, falling back to fork",
            stacklevel=2,
        )
        return _spawn_http_servers_fork(
            app,
            n,
            host=host,
            health_path=health_path,
            max_wait=max_wait,
            poll_interval=poll_interval,
            request_timeout=request_timeout,
            log_level=log_level,
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

    @pytest.fixture
    def http_servers(request):  # type: ignore
        """Generic parametrized fixture.

        Usage: @pytest.mark.parametrize('http_servers', [3], indirect=True)
        """
        n = getattr(request, "param", 1)
        from mplang.runtime.server import app as runtime_app

        result = spawn_http_servers(runtime_app, n)
        try:
            yield result
        finally:
            result.stop()

except Exception:  # pragma: no cover
    pass
