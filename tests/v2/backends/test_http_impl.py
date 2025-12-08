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

"""Tests for HTTP backend implementation."""

import logging
import multiprocessing
import time

import numpy as np
import pytest
import uvicorn

import mplang.v2.edsl as el
from mplang.v2.backends.simp_http_driver import SimpHttpDriver
from mplang.v2.backends.simp_http_worker import create_worker_app
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import simp, tensor

logging.basicConfig(level=logging.DEBUG)


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
    base_port = 19000
    endpoints = [f"http://127.0.0.1:{base_port + i}" for i in range(world_size)]

    ctx = multiprocessing.get_context("spawn")
    processes = []
    for i in range(world_size):
        p = ctx.Process(
            target=run_worker, args=(i, world_size, endpoints, base_port + i)
        )
        p.start()
        processes.append(p)

    # Wait for servers to start
    time.sleep(2)

    yield SimpHttpDriver(world_size, endpoints)

    for p in processes:
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            logging.warning(f"Process {p.pid} did not terminate, killing it.")
            p.kill()
            p.join()


def _add_fn(a, b):
    return a + b


def _add_one(val):
    return tensor.run_jax(_add_fn, val, tensor.constant(1.0))


def test_http_e2e(http_cluster):
    host = http_cluster

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

    # Execute
    # Inputs are empty since we use constants
    results = host.evaluate_graph(graph, {})

    # Fetch results
    values = host.fetch(results)

    # Verify
    # Party 0 result is None (not involved in final output)
    # Party 1 result should be [2.0, 3.0]

    # Note: evaluate_graph returns a list of results (one per output).
    # Since graph has 1 output, each party returns a list of 1 element.
    # However, for Party 0 (which returns None), it seems to be unwrapped or handled differently.
    if isinstance(values[0], list):
        assert values[0] == [None]
    else:
        assert values[0] is None

    val1 = values[1]
    if isinstance(val1, list):
        val1 = val1[0]

    result_1 = val1.data if isinstance(val1, TensorValue) else val1
    np.testing.assert_allclose(result_1, [2.0, 3.0])
