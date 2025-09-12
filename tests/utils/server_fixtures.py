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
    return [get_free_port() for _ in range(n)]


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
    """Spawn n uvicorn servers for the provided ASGI app.

    Returns a SpawnResult that should be explicitly stopped (or wrapped by a fixture).
    """
    import uvicorn  # local import to avoid test collection side-effects

    ports = get_free_ports(n)
    addresses = [f"http://{host}:{p}" for p in ports]
    processes: list[multiprocessing.Process] = []

    def make_process(port: int) -> multiprocessing.Process:
        def run():
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level=log_level,
                ws="none",
            )
            server = uvicorn.Server(config)
            if _verbose():
                print(
                    f"[spawn_http_servers] starting server pid={os.getpid()} port={port}"
                )
            server.run()

        p = multiprocessing.Process(target=run, daemon=True)
        return p

    for port in ports:
        p = make_process(port)
        p.start()
        processes.append(p)

    # Health check loop
    attempts = int(max_wait / poll_interval)
    for port in ports:
        ready = False
        for _ in range(attempts):
            try:
                r = httpx.get(
                    f"http://{host}:{port}{health_path}", timeout=request_timeout
                )
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(poll_interval)
        if not ready:
            # Stop already started processes before raising
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
            raise RuntimeError(f"Server on port {port} failed to become healthy")

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
