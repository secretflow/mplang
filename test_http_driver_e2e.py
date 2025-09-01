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
import sys
import threading
import time
import traceback

import numpy as np
import uvicorn

import mplang
import mplang.simp as simp
from mplang.runtime.http_backend.driver import HttpDriver
from mplang.runtime.http_backend.server import app

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def run_server(port: int):
    """Run HTTP server on the given port."""
    config = uvicorn.Config(app, host="localhost", port=port, log_level="debug")
    server = uvicorn.Server(config)
    server.run()


def test_simple_addition():
    """Test simple addition with HttpDriver."""
    # Start servers in background threads
    ports = [15001, 15002]
    server_threads = []

    for port in ports:
        thread = threading.Thread(target=run_server, args=(port,), daemon=True)
        thread.start()
        server_threads.append(thread)

    # Wait for servers to start
    time.sleep(3)

    try:
        # Create HttpDriver
        node_addrs = {
            0: "http://localhost:15001",
            1: "http://localhost:15002",
        }

        driver = HttpDriver(node_addrs)
        print(f"HttpDriver created with world_size: {driver.world_size}")

        # Create a simple constant computation
        x = np.array([1.0, 2.0, 3.0])

        def constant_fn(x):
            return simp.constant(x)

        # Evaluate the computation
        print("Evaluating constant...")
        result = mplang.evaluate(driver, constant_fn, x)
        print(f"Evaluation completed, result type: {type(result)}")

        # Fetch the result
        print("Fetching result...")
        fetched = mplang.fetch(driver, result)
        print(f"Fetched result: {fetched}")

        # Verify result - each rank should have the same constant
        for i, actual in enumerate(fetched):
            assert np.allclose(actual, x), f"Mismatch at rank {i}: {actual} vs {x}"

        print("✅ Test passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_addition()
    sys.exit(0 if success else 1)
