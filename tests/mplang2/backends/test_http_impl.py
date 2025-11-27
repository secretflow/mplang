"""Tests for HTTP backend implementation."""

import logging
import multiprocessing
import time

import numpy as np
import pytest
import uvicorn

import mplang2.edsl as el
from mplang2.backends.simp_http import HttpHost, create_worker_app
from mplang2.dialects import simp, tensor

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

    yield HttpHost(world_size, endpoints)

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
    with el.Tracer() as tracer:
        # Party 0 creates data
        x = simp.constant((0,), np.array([1.0, 2.0]))

        # Shuffle to Party 1
        y = simp.shuffle_static(x, {1: 0})

        z = simp.pcall_static((1,), _add_one, y)

        # Return result from Party 1
        tracer.finalize(z)

    graph = tracer.graph

    # Execute
    # Inputs are empty since we use constants
    results = host.evaluate_graph(graph, {})

    # Verify
    # Party 0 result is None (not involved in final output)
    # Party 1 result should be [2.0, 3.0]

    assert results[0] is None
    np.testing.assert_allclose(results[1], [2.0, 3.0])
