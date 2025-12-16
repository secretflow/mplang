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

"""Tests for simp_driver/http.py (SimpHttpDriver)."""

import multiprocessing
import time

import pytest

import mplang.v2 as mp
from mplang.v2.dialects import simp
from mplang.v2.edsl.context import pop_context, push_context


def run_worker(rank: int, world_size: int, port: int, endpoints: list[str]) -> None:
    """Run a single worker server."""
    import uvicorn

    from mplang.v2.backends.simp_worker.http import create_worker_app

    app = create_worker_app(rank, world_size, endpoints)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


@pytest.fixture(scope="module")
def driver_cluster():
    """Start worker servers and return a driver Interpreter."""
    world_size = 2
    base_port = 18200  # Use high port to avoid conflicts
    ports = [base_port + i for i in range(world_size)]
    endpoints = [f"http://127.0.0.1:{p}" for p in ports]

    # Start workers in separate processes
    processes: list[multiprocessing.Process] = []
    for rank in range(world_size):
        p = multiprocessing.Process(
            target=run_worker,
            args=(rank, world_size, ports[rank], endpoints),
        )
        p.start()
        processes.append(p)

    # Wait for servers to start
    time.sleep(1.0)

    # Check if processes are alive
    dead_procs = [p for p in processes if not p.is_alive()]
    if dead_procs:
        for p in processes:
            p.terminate()
        raise RuntimeError(
            f"Failed to start workers. Dead processes: {len(dead_procs)}"
        )

    # Create cluster spec
    cluster_spec = mp.ClusterSpec.from_dict({
        "nodes": [
            {"name": f"node_{i}", "endpoint": f"127.0.0.1:{ports[i]}"}
            for i in range(world_size)
        ],
        "devices": {
            "SP0": {
                "kind": "SPU",
                "members": [f"node_{i}" for i in range(world_size)],
                "config": {"protocol": "SEMI2K", "field": "FM128"},
            },
            "P0": {"kind": "PPU", "members": ["node_0"]},
            "P1": {"kind": "PPU", "members": ["node_1"]},
        },
    })

    # Set global cluster for device API
    # REMOVED: set_global_cluster(cluster_spec)

    # Create driver using factory function
    driver = simp.make_driver(endpoints, cluster_spec=cluster_spec)
    push_context(driver)

    yield driver

    # Cleanup
    pop_context()
    state = driver.get_dialect_state("simp")
    if hasattr(state, "shutdown"):
        state.shutdown()
    for p in processes:
        p.terminate()
    for p in processes:
        p.join(timeout=2)


class TestDriverBasic:
    """Basic SimpHttpDriver tests."""

    def test_driver_creation(self, driver_cluster):
        """Test driver Interpreter can be created."""
        assert driver_cluster is not None
        assert isinstance(driver_cluster, mp.Interpreter)

    def test_driver_has_simp_state(self, driver_cluster):
        """Test driver has simp dialect state."""
        state = driver_cluster.get_dialect_state("simp")
        assert state is not None
        assert hasattr(state, "world_size")

    def test_driver_context_manager(self, driver_cluster):
        """Test Interpreter can be used as context manager."""
        with driver_cluster:
            pass  # Just verify it doesn't raise


class TestDriverExecution:
    """Test executing computations via Driver."""

    def test_simple_ppu_computation(self, driver_cluster):
        """Test simple computation on PPU device."""
        with driver_cluster:

            def add_one():
                x = mp.device("P0")(lambda: 42)()
                return x

            result = mp.evaluate(add_one)
            value = mp.fetch(result)
            assert value == 42

    def test_ppu_computation_with_jax(self, driver_cluster):
        """Test JAX computation on PPU device."""
        import jax.numpy as jnp

        with driver_cluster:

            def jax_sum():
                arr = mp.device("P0")(lambda: jnp.array([1, 2, 3]))()
                result = mp.device("P0")(lambda x: jnp.sum(x))(arr)
                return result

            result = mp.evaluate(jax_sum)
            value = mp.fetch(result)
            assert value == 6

    def test_cross_party_transfer(self, driver_cluster):
        """Test data transfer between parties."""
        with driver_cluster:

            def transfer():
                x = mp.device("P0")(lambda: 100)()
                y = mp.put("P1", x)
                return y

            result = mp.evaluate(transfer)
            # y is on P1, so fetch will use device attribute to get from P1
            value = mp.fetch(result)
            assert value == 100

    @pytest.mark.skip(
        reason="SPU requires BRPC link setup in HTTP worker, not yet implemented"
    )
    def test_spu_computation(self, driver_cluster):
        """Test secure computation on SPU."""

        def secure_add():
            x = mp.device("P0")(lambda: 10)()
            y = mp.device("P1")(lambda: 20)()
            z = mp.device("SP0")(lambda a, b: a + b)(x, y)
            return mp.put("P0", z)

        result = mp.evaluate(secure_add)
        value = mp.fetch(result)
        assert value == 30


class TestDriverFetch:
    """Test fetch functionality with Driver."""

    def test_fetch_by_party_name(self, driver_cluster):
        """Test fetching result by party name."""
        with driver_cluster:

            def create_on_p0():
                return mp.device("P0")(lambda: 999)()

            result = mp.evaluate(create_on_p0)
            # fetch uses device attribute to get from P0
            value = mp.fetch(result)
            assert value == 999

    def test_fetch_all_parties(self, driver_cluster):
        """Test fetching from all parties returns list."""
        with driver_cluster:

            def create_on_p0():
                return mp.device("P0")(lambda: 123)()

            result = mp.evaluate(create_on_p0)
            # When no party specified, returns all parties' values as list
            # But for SPMD replicated execution, each party gets the same result
            values = mp.fetch(result)
            # Result should contain 123 (either as scalar or in list)
            if isinstance(values, list):
                assert 123 in values
            else:
                assert values == 123
