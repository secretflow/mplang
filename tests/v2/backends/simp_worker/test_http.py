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

"""Tests for simp_worker/http.py and simp_driver/http.py (HTTP IPC)."""

import logging
import multiprocessing
import time

import httpx
import numpy as np
import pytest
import uvicorn

import mplang.v2 as mp
import mplang.v2.edsl as el
from mplang.v2.backends.simp_worker.http import create_worker_app
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import simp, tensor
from mplang.v2.edsl.context import pop_context, push_context

logging.basicConfig(level=logging.DEBUG)


def wait_for_server_ready(endpoint, timeout=30, check_interval=0.5):
    """Wait for a server to be ready by checking health endpoint."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with httpx.Client(timeout=2.0) as client:
                # Try to make a simple request to check if server is responsive
                response = client.get(f"{endpoint}/health", timeout=2.0)
                if response.status_code == 200:
                    logging.debug(f"Server at {endpoint} is ready")
                    return True
                else:
                    logging.debug(
                        f"Server at {endpoint} returned status {response.status_code}"
                    )
        except (
            httpx.RequestError,
            httpx.TimeoutException,
            httpx.ConnectError,
            ConnectionError,
        ) as e:
            # Server not ready yet, continue waiting
            logging.debug(f"Server at {endpoint} not ready: {e}")
        except Exception as e:
            # Unexpected error, log it but continue waiting
            logging.warning(f"Unexpected error checking server at {endpoint}: {e}")

        time.sleep(check_interval)

    logging.error(f"Server at {endpoint} failed to become ready within {timeout}s")
    return False


def run_worker(rank, world_size, endpoints, port):
    # Configure logging for the worker process
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"%(asctime)s [Rank {rank}] %(levelname)s: %(message)s",
        force=True,
    )
    app = create_worker_app(rank, world_size, endpoints)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


@pytest.fixture(scope="module")
def http_cluster():
    world_size = 2
    base_port = 19300  # Changed to avoid port conflicts
    endpoints = [f"http://127.0.0.1:{base_port + i}" for i in range(world_size)]

    ctx = multiprocessing.get_context("spawn")
    processes = []
    for i in range(world_size):
        p = ctx.Process(
            target=run_worker, args=(i, world_size, endpoints, base_port + i)
        )
        p.start()
        processes.append(p)

    # Wait for all servers to be ready
    all_ready = True
    for endpoint in endpoints:
        logging.info(f"Waiting for server at {endpoint} to be ready...")
        if not wait_for_server_ready(endpoint, timeout=30):
            logging.error(f"Server at {endpoint} failed to become ready")
            all_ready = False

    if not all_ready:
        # Clean up processes if servers failed to start
        for p in processes:
            try:
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    logging.warning(
                        f"Process {p.pid} did not terminate gracefully, killing it."
                    )
                    p.kill()
                    p.join(timeout=2)
            except Exception as e:
                logging.error(f"Error cleaning up process {p.pid}: {e}")
        pytest.fail("Failed to start all HTTP worker servers")

    # Give servers a moment to fully initialize after health checks pass
    time.sleep(0.5)
    logging.info("All HTTP worker servers are ready")

    # Create cluster spec for device API
    cluster_spec = mp.ClusterSpec.from_dict({
        "nodes": [
            {"name": f"node_{i}", "endpoint": endpoints[i]} for i in range(world_size)
        ],
        "devices": {
            "P0": {"kind": "PPU", "members": ["node_0"]},
            "P1": {"kind": "PPU", "members": ["node_1"]},
        },
    })
    # REMOVED: set_global_cluster(cluster_spec)

    # Create driver using factory function
    driver = simp.make_driver(endpoints, cluster_spec=cluster_spec)
    push_context(driver)

    yield driver

    pop_context()

    # Shutdown driver
    state = driver.get_dialect_state("simp")
    if hasattr(state, "shutdown"):
        state.shutdown()

    for p in processes:
        try:
            p.terminate()
            p.join(timeout=5)  # Increased timeout for graceful shutdown
            if p.is_alive():
                logging.warning(
                    f"Process {p.pid} did not terminate gracefully, killing it."
                )
                p.kill()
                p.join(timeout=2)
                if p.is_alive():
                    logging.error(f"Process {p.pid} could not be killed")
        except Exception as e:
            logging.error(f"Error cleaning up process {p.pid}: {e}")


def _add_fn(a, b):
    return a + b


def _add_one(val):
    return tensor.run_jax(_add_fn, val, tensor.constant(1.0))


def test_http_e2e(http_cluster):
    driver = http_cluster

    # Define computation
    def workflow():
        # Party 0 creates data
        x = simp.constant((0,), np.array([1.0, 2.0]))

        # Shuffle to Party 1
        y = simp.shuffle_static(x, {1: 0})

        z = simp.pcall_static((1,), _add_one, y)
        return z

    traced = el.trace(workflow)
    graph = traced.graph

    # Execute with retry for robustness
    max_retries = 3
    retry_delay = 1.0

    with driver:
        for attempt in range(max_retries):
            try:
                # Execute
                # Inputs are empty since we use constants
                results = driver.evaluate_graph(graph, {})

                # Fetch results
                values = mp.fetch(results)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise the exception
                    raise
                logging.warning(
                    f"Attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)

        # Verify
        # evaluate_graph returns list of outputs; for single output, extract it
        output_values = values[0]

        # Party 0 result is None (not involved in final output)
        # Party 1 result should be [2.0, 3.0]
        assert output_values[0] is None, (
            f"Party 0 should be None, got {output_values[0]}"
        )

        val1 = output_values[1]
        result_1 = val1.data if isinstance(val1, TensorValue) else val1
        np.testing.assert_allclose(result_1, [2.0, 3.0])
