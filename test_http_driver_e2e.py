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

#!/usr/bin/env python3
"""
End-to-end test for HttpDriver to verify evaluate and fetch functionality.
"""

import logging
import multiprocessing
import sys
import threading
import time
import traceback

import numpy as np
import uvicorn

import mplang
import mplang.simp as simp

# Configure logging to see detailed error information
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
import mplang.smpc as smpc
from mplang.runtime.http_backend.driver import HttpDriver
from mplang.runtime.http_backend.server import app


def _run_simple_addition_test(driver: HttpDriver):
    """Internal logic for simple addition test."""
    try:
        print("--- Running Simple Addition Test ---")
        # Create a simple constant computation
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        def constant_add_fn(x, y):
            x_const = simp.constant(x)
            y_const = simp.constant(y)
            # simp doesn't have add, use a jax function
            import jax.numpy as jnp

            return simp.run(jnp.add)(x_const, y_const)

        # Evaluate the computation
        print("Evaluating constant addition...")
        result = mplang.evaluate(driver, constant_add_fn, x, y)
        print(f"Evaluation completed, result type: {type(result)}")

        # Fetch the result
        print("Fetching result...")
        fetched = mplang.fetch(driver, result)
        print(f"Fetched result: {fetched}")

        # Verify result
        expected = x + y
        for i, actual in enumerate(fetched):
            assert np.allclose(
                actual, expected
            ), f"Mismatch at rank {i}: {actual} vs {expected}"

        print("✅ Simple Addition Test passed!")
        return True

    except Exception as e:
        print(f"❌ Simple Addition Test failed: {e}")
        traceback.print_exc()
        return False


def _run_millionaire_test(driver: HttpDriver):
    """Internal logic for millionaire problem test."""
    try:
        print("\n--- Running Millionaire Problem Test ---")
        # For HttpDriver, use a simpler test first
        x = np.array([5.0])
        y = np.array([3.0])

        @mplang.function
        def simple_compare():
            # Create constants
            x_const = simp.constant(x)
            y_const = simp.constant(y)

            # Seal them
            x_sealed = smpc.sealFrom(x_const, 0)
            y_sealed = smpc.sealFrom(y_const, 1)

            # Compare
            import jax.numpy as jnp

            result = smpc.srun(lambda a, b: jnp.where(a > b, True, False))(
                x_sealed, y_sealed
            )

            # Reveal
            revealed = smpc.reveal(result)

            return revealed

        # Evaluate the computation
        print("Evaluating simple comparison...")
        result = mplang.evaluate(driver, simple_compare)
        print(f"Evaluation completed, result type: {type(result)}")

        # Fetch the result
        print("Fetching result...")
        fetched = mplang.fetch(driver, result)
        print(f"Fetched result: {fetched}")

        # Verify result
        expected = x > y  # Should be True since 5.0 > 3.0
        assert fetched[0] == expected, f"Mismatch at rank 0: {fetched[0]} vs {expected}"
        assert fetched[1] == expected, f"Mismatch at rank 1: {fetched[1]} vs {expected}"

        print("✅ Millionaire Problem Test passed!")
        return True

    except Exception as e:
        print(f"❌ Millionaire Problem Test failed: {e}")
        traceback.print_exc()
        return False


def run_server(port: int, shutdown_event: threading.Event):
    """Run a uvicorn server in a separate process."""
    config = uvicorn.Config(app, host="localhost", port=port, log_level="info")
    server = uvicorn.Server(config)

    # Create a thread to run the server
    server_thread = threading.Thread(target=server.run)
    server_thread.start()

    # Wait for the shutdown event
    shutdown_event.wait()

    # Gracefully shut down the server
    server.should_exit = True
    server_thread.join()


def run_all_e2e_tests():
    """Run all end-to-end tests with a single server lifecycle."""
    ports = [15001, 15002]
    processes = []
    shutdown_event = multiprocessing.Event()

    # Start servers in background processes
    for port in ports:
        process = multiprocessing.Process(
            target=run_server, args=(port, shutdown_event)
        )
        process.start()
        processes.append(process)

    # Wait for servers to start
    print("Waiting for servers to start...")
    time.sleep(2)  # Give servers a moment to initialize
    print("Servers should be up.")

    all_success = True
    try:
        # Create HttpDriver
        node_addrs = {
            0: "http://localhost:15001",
            1: "http://localhost:15002",
        }
        driver = HttpDriver(node_addrs)
        print(f"HttpDriver created with world_size: {driver.world_size}")

        if not _run_simple_addition_test(driver):
            all_success = False

        if not _run_millionaire_test(driver):
            all_success = False

    finally:
        print("\nShutting down servers...")
        shutdown_event.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        print("Servers shut down.")

    return all_success


if __name__ == "__main__":
    # Set start method for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    # First, run simulator tests to ensure logic is correct
    print("--- Running Simulator Tests ---")
    sim2 = mplang.Simulator(2)

    def millionaire_sim():
        wealth = simp.run(lambda: np.random.randint(0, 101))()
        sealed_wealth = smpc.seal(wealth)
        import jax.numpy as jnp

        result = smpc.srun(lambda x, y: jnp.where(x > y, 1, jnp.where(x < y, 0, -1)))(
            *sealed_wealth
        )
        revealed_result = smpc.reveal(result)
        return wealth, revealed_result

    try:
        wealth, result = mplang.evaluate(sim2, millionaire_sim)
        fetched_wealth = mplang.fetch(sim2, wealth)
        fetched_result = mplang.fetch(sim2, result)
        wealth_p0, wealth_p1 = fetched_wealth
        expected_result = (
            1 if wealth_p0 > wealth_p1 else (0 if wealth_p0 < wealth_p1 else -1)
        )
        assert fetched_result[0] == expected_result
        print("✅ Simulator tests passed!")
    except Exception as e:
        print(f"❌ Simulator tests failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Running E2E HTTP Tests ---")
    e2e_success = run_all_e2e_tests()

    sys.exit(0 if e2e_success else 1)
