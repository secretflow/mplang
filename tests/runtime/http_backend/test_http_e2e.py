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

"""
End-to-end functional tests for HttpDriver evaluate and fetch operations.
"""

import multiprocessing
import time

import jax.numpy as jnp
import numpy as np
import pytest
import uvicorn

import mplang
import mplang.simp as simp
import mplang.smpc as smpc
from mplang.runtime.http_backend.driver import HttpDriver
from mplang.runtime.http_backend.server import app

# Global state for test servers
e2e_server_processes: dict[int, multiprocessing.Process] = {}


def run_e2e_server(port: int):
    """Function to run a uvicorn server on a specific port for e2e testing."""
    config = uvicorn.Config(
        app,
        host="localhost",
        port=port,
        log_level="critical",
        ws="none",  # Disable websockets to avoid deprecation warnings
    )
    server = uvicorn.Server(config)
    server.run()


@pytest.fixture(scope="module", autouse=True)
def start_e2e_servers():
    """Fixture to start servers for HttpDriver e2e testing."""
    # Start servers for e2e tests - 5 parties: P0, P1, P2, P3, P4
    # SPU mask = 01110, meaning P1, P2, P3 form the SPU
    ports = [15001, 15002, 15003, 15004, 15005]

    # Start servers in separate processes
    for port in ports:
        process = multiprocessing.Process(target=run_e2e_server, args=(port,))
        process.daemon = True
        e2e_server_processes[port] = process
        process.start()

    # Wait for servers to be ready via health check
    import httpx

    for port in ports:
        ready = False
        for _ in range(100):  # up to ~10s
            try:
                r = httpx.get(f"http://localhost:{port}/health", timeout=0.2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.1)
        if not ready:
            raise RuntimeError(f"Server on port {port} failed to start in time")

    yield

    # Teardown: stop all server processes
    for port in ports:
        if port in e2e_server_processes:
            process = e2e_server_processes[port]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)  # Wait up to 5 seconds
                if process.is_alive():
                    process.kill()  # Force kill if still alive


@pytest.fixture
def http_driver():
    """Fixture to create HttpDriver for testing with 5 parties."""
    # 5 parties: P0, P1, P2, P3, P4
    # SPU mask = 01110, meaning P1, P2, P3 form the SPU
    node_addrs = {
        0: "http://localhost:15001",  # P0 - plaintext party
        1: "http://localhost:15002",  # P1 - SPU party
        2: "http://localhost:15003",  # P2 - SPU party
        3: "http://localhost:15004",  # P3 - SPU party
        4: "http://localhost:15005",  # P4 - plaintext party
    }
    return HttpDriver(node_addrs)


@pytest.mark.skip(
    reason="Works when run individually, but hangs when run with 'uv run pytest tests/runtime/'"
)
def test_simple_addition_e2e(http_driver):
    """Test simple addition computation using HttpDriver with 5 parties."""
    # Create test data
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    def constant_add_fn(x, y):
        x_const = simp.constant(x)
        y_const = simp.constant(y)
        # Use JAX function for addition
        return simp.run(jnp.add)(x_const, y_const)

    # Evaluate the computation
    result = mplang.evaluate(http_driver, constant_add_fn, x, y)

    # Fetch the result
    fetched = mplang.fetch(http_driver, result)

    # Verify result - all 5 parties should get the same result
    expected = x + y
    for i, actual in enumerate(fetched):
        assert np.allclose(
            actual, expected
        ), f"Mismatch at rank {i}: {actual} vs {expected}"


def test_secure_comparison_e2e(http_driver):
    """Test secure comparison computation using HttpDriver with 5 parties."""
    # Create test data - data comes from different parties
    x = np.array([5.0])  # From P0
    y = np.array([3.0])  # From P4

    @mplang.function
    def secure_compare():
        # Create constants
        x_const = simp.constant(x)
        y_const = simp.constant(y)

        # Seal them for secure computation - data from P0 and P4
        x_sealed = smpc.sealFrom(x_const, 0)  # P0 provides data
        y_sealed = smpc.sealFrom(y_const, 4)  # P4 provides data

        # Perform secure comparison
        import jax.numpy as jnp

        result = smpc.srun(lambda a, b: jnp.where(a > b, True, False))(
            x_sealed, y_sealed
        )

        # Reveal the result
        revealed = smpc.reveal(result)
        return revealed

    # Evaluate the computation
    result = mplang.evaluate(http_driver, secure_compare)

    # Fetch the result
    fetched = mplang.fetch(http_driver, result)

    # Verify result - all 5 parties should get the same result
    expected = x > y  # Should be True since 5.0 > 3.0
    for i in range(5):
        assert (
            fetched[i] == expected
        ), f"Mismatch at rank {i}: {fetched[i]} vs {expected}"


def test_three_way_comparison_e2e(http_driver):
    """Test multi-party comparison (millionaire problem) using HttpDriver with 5 parties."""
    # Create test data - wealth from different parties
    wealth_a = np.array([50.0])  # P0's wealth
    wealth_b = np.array([30.0])  # P2's wealth
    wealth_c = np.array([70.0])  # P4's wealth

    @mplang.function
    def millionaire_problem():
        # Create constants for each party's wealth
        wealth_a_const = simp.constant(wealth_a)
        wealth_b_const = simp.constant(wealth_b)
        wealth_c_const = simp.constant(wealth_c)

        # Seal the wealth values for secure computation
        wealth_a_sealed = smpc.sealFrom(wealth_a_const, 0)  # P0 provides wealth
        wealth_b_sealed = smpc.sealFrom(wealth_b_const, 2)  # P2 provides wealth
        wealth_c_sealed = smpc.sealFrom(wealth_c_const, 4)  # P4 provides wealth

        # Perform secure comparison to find the richest
        import jax.numpy as jnp

        # Find who has the maximum wealth
        max_ab = smpc.srun(jnp.maximum)(wealth_a_sealed, wealth_b_sealed)
        max_wealth = smpc.srun(jnp.maximum)(max_ab, wealth_c_sealed)

        # Check if each party is the richest
        a_is_richest = smpc.srun(lambda a, max_w: a >= max_w)(
            wealth_a_sealed, max_wealth
        )
        b_is_richest = smpc.srun(lambda b, max_w: b >= max_w)(
            wealth_b_sealed, max_wealth
        )
        c_is_richest = smpc.srun(lambda c, max_w: c >= max_w)(
            wealth_c_sealed, max_wealth
        )

        # Reveal the results
        a_result = smpc.reveal(a_is_richest)
        b_result = smpc.reveal(b_is_richest)
        c_result = smpc.reveal(c_is_richest)

        return a_result, b_result, c_result

    # Evaluate the computation
    a_result, b_result, c_result = mplang.evaluate(http_driver, millionaire_problem)

    # Fetch the results
    fetched_a = mplang.fetch(http_driver, a_result)
    fetched_b = mplang.fetch(http_driver, b_result)
    fetched_c = mplang.fetch(http_driver, c_result)

    # Verify results - P4 (wealth_c = 70.0) should be the richest
    for i in range(5):
        assert not fetched_a[i], f"P0 should not be richest at rank {i}: {fetched_a[i]}"
        assert not fetched_b[i], f"P2 should not be richest at rank {i}: {fetched_b[i]}"
        assert fetched_c[i], f"P4 should be richest at rank {i}: {fetched_c[i]}"


def test_multiple_operations_e2e(http_driver):
    """Test multiple operations in sequence using HttpDriver with 5 parties."""
    # Test data - from different parties
    a = np.array([10.0, 20.0])  # From P0
    b = np.array([5.0, 15.0])  # From P3

    @mplang.function
    def multi_operations():
        # Create constants
        a_const = simp.constant(a)
        b_const = simp.constant(b)

        # Seal for secure computation - data from P0 and P3
        a_sealed = smpc.sealFrom(a_const, 0)  # P0 provides data
        b_sealed = smpc.sealFrom(b_const, 3)  # P3 provides data

        # Multiple operations
        import jax.numpy as jnp

        # Addition
        sum_result = smpc.srun(jnp.add)(a_sealed, b_sealed)

        # Multiplication
        mul_result = smpc.srun(jnp.multiply)(a_sealed, b_sealed)

        # Comparison
        cmp_result = smpc.srun(lambda x, y: x > y)(a_sealed, b_sealed)

        # Reveal all results
        sum_revealed = smpc.reveal(sum_result)
        mul_revealed = smpc.reveal(mul_result)
        cmp_revealed = smpc.reveal(cmp_result)

        return sum_revealed, mul_revealed, cmp_revealed

    # Evaluate the computation
    sum_result, mul_result, cmp_result = mplang.evaluate(http_driver, multi_operations)

    # Fetch the results
    fetched_sum = mplang.fetch(http_driver, sum_result)
    fetched_mul = mplang.fetch(http_driver, mul_result)
    fetched_cmp = mplang.fetch(http_driver, cmp_result)

    # Verify results
    expected_sum = a + b
    expected_mul = a * b
    expected_cmp = a > b

    # All 5 parties should get the same results
    for i in range(5):
        assert np.allclose(fetched_sum[i], expected_sum), f"Sum mismatch at rank {i}"
        assert np.allclose(fetched_mul[i], expected_mul), f"Mul mismatch at rank {i}"
        assert np.array_equal(fetched_cmp[i], expected_cmp), f"Cmp mismatch at rank {i}"
