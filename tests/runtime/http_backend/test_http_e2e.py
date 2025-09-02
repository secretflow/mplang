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
    # Start servers for e2e tests on different ports than unit tests
    ports = [15001, 15002]

    # Start servers in separate processes
    for port in ports:
        process = multiprocessing.Process(target=run_e2e_server, args=(port,))
        process.daemon = True
        e2e_server_processes[port] = process
        process.start()

    # Give servers time to start up
    time.sleep(0.2)  # Increased timeout for process startup

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
    """Fixture to create HttpDriver for testing."""
    node_addrs = {
        0: "http://localhost:15001",
        1: "http://localhost:15002",
    }
    return HttpDriver(node_addrs)


def test_simple_addition_e2e(http_driver):
    """Test simple addition computation using HttpDriver."""
    # Create test data
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    def constant_add_fn(x, y):
        x_const = simp.constant(x)
        y_const = simp.constant(y)
        # Use JAX function for addition
        import jax.numpy as jnp

        return simp.run(jnp.add)(x_const, y_const)

    # Evaluate the computation
    result = mplang.evaluate(http_driver, constant_add_fn, x, y)

    # Fetch the result
    fetched = mplang.fetch(http_driver, result)

    # Verify result
    expected = x + y
    for i, actual in enumerate(fetched):
        assert np.allclose(
            actual, expected
        ), f"Mismatch at rank {i}: {actual} vs {expected}"


def test_secure_comparison_e2e(http_driver):
    """Test secure comparison computation using HttpDriver."""
    # Create test data
    x = np.array([5.0])
    y = np.array([3.0])

    @mplang.function
    def secure_compare():
        # Create constants
        x_const = simp.constant(x)
        y_const = simp.constant(y)

        # Seal them for secure computation
        x_sealed = smpc.sealFrom(x_const, 0)
        y_sealed = smpc.sealFrom(y_const, 1)

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

    # Verify result
    expected = x > y  # Should be True since 5.0 > 3.0
    assert fetched[0] == expected, f"Mismatch at rank 0: {fetched[0]} vs {expected}"
    assert fetched[1] == expected, f"Mismatch at rank 1: {fetched[1]} vs {expected}"


def test_three_way_comparison_e2e(http_driver):
    """Test three-way comparison (millionaire problem) using HttpDriver."""
    # Create test data
    wealth_a = np.array([50.0])
    wealth_b = np.array([30.0])

    @mplang.function
    def millionaire_problem():
        # Create constants for each party's wealth
        wealth_a_const = simp.constant(wealth_a)
        wealth_b_const = simp.constant(wealth_b)

        # Seal the wealth values for secure computation
        wealth_a_sealed = smpc.sealFrom(wealth_a_const, 0)
        wealth_b_sealed = smpc.sealFrom(wealth_b_const, 1)

        # Perform secure three-way comparison
        import jax.numpy as jnp

        result = smpc.srun(lambda x, y: jnp.where(x > y, 1, jnp.where(x < y, 0, -1)))(
            wealth_a_sealed, wealth_b_sealed
        )

        # Reveal the result
        revealed_result = smpc.reveal(result)
        return revealed_result

    # Evaluate the computation
    result = mplang.evaluate(http_driver, millionaire_problem)

    # Fetch the result
    fetched = mplang.fetch(http_driver, result)

    # Verify result (1 means first party richer, 0 means second party richer, -1 means equal)
    expected = (
        1 if wealth_a[0] > wealth_b[0] else (0 if wealth_a[0] < wealth_b[0] else -1)
    )
    assert fetched[0] == expected, f"Mismatch at rank 0: {fetched[0]} vs {expected}"
    assert fetched[1] == expected, f"Mismatch at rank 1: {fetched[1]} vs {expected}"


def test_multiple_operations_e2e(http_driver):
    """Test multiple operations in sequence using HttpDriver."""
    # Test data
    a = np.array([10.0, 20.0])
    b = np.array([5.0, 15.0])

    @mplang.function
    def multi_operations():
        # Create constants
        a_const = simp.constant(a)
        b_const = simp.constant(b)

        # Seal for secure computation
        a_sealed = smpc.sealFrom(a_const, 0)
        b_sealed = smpc.sealFrom(b_const, 1)

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

    for i in range(len(fetched_sum)):
        assert np.allclose(fetched_sum[i], expected_sum), f"Sum mismatch at rank {i}"
        assert np.allclose(fetched_mul[i], expected_mul), f"Mul mismatch at rank {i}"
        assert np.array_equal(fetched_cmp[i], expected_cmp), f"Cmp mismatch at rank {i}"
