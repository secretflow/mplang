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

"""Device: Simulator vs Driver

Learning objectives:
1. Use Simulator for local multi-threaded testing
2. Use Driver for distributed HTTP-based execution
3. Understand the differences between Simulator and Driver

Key concepts:
- Simulator: single-process, all parties in threads, fast iteration
- Driver: multi-process, HTTP-based, for real distributed deployment

Migrated from mplang v1 to mplang2.
"""

import random

import httpx

import mplang.v2 as mp


def millionaire():
    """Simple millionaire using device API (from 00_device_basics)."""
    # Generate data on specific parties using mp.put for constants
    x = mp.put("P0", random.randint(0, 10))
    y = mp.put("P1", random.randint(0, 10))

    # Secure comparison on SPU
    result = mp.device("SP0")(lambda a, b: a < b)(x, y)

    result = mp.put("P0", result)

    return x, y, result


# ============================================================================
# Simulator: Local Multi-Threaded Execution
# ============================================================================


def run_with_simulator():
    """Simulator: best for rapid prototyping and testing."""
    print("\n" + "=" * 70)
    print("Running with Simulator (local, multi-threaded)")
    print("=" * 70)

    cluster_spec = mp.ClusterSpec.simple(2)
    sim = mp.make_simulator(2, cluster_spec=cluster_spec)
    mp.set_root_context(sim)

    print("\n--- Millionaire problem (device API) ---")
    x, y, result = mp.evaluate(millionaire)

    # Fetch results from all parties
    x_vals = mp.fetch(x)
    y_vals = mp.fetch(y)
    result_vals = mp.fetch(result)

    # Values are HostVar holding per-party results
    print(f"P0 value: {x_vals}")
    print(f"P1 value: {y_vals}")
    print(f"x < y (SPU): {result_vals}")


# ============================================================================
# Driver: Distributed HTTP-based Execution
# ============================================================================


def run_with_driver():
    """Driver: for real distributed deployment.

    Same API as Simulator, just different backend (HTTP instead of threads).

    Usage:
        Terminal 1: python -m mplang.v2.cli up --world-size 2 --base-port 8100
        Terminal 2: python tutorials/v2/02_simulation_and_driver.py
    """
    endpoints = ["http://127.0.0.1:8100", "http://127.0.0.1:8101"]

    print("\n" + "=" * 70)
    print("Running with Driver (distributed, HTTP-based)")
    print("=" * 70)

    # Check if workers are running
    print("\nChecking workers...")
    all_healthy = True
    for i, ep in enumerate(endpoints):
        try:
            resp = httpx.get(f"{ep}/health", timeout=2)
            if resp.status_code == 200:
                print(f"  Worker {i} ({ep}): ✓ healthy")
            else:
                print(f"  Worker {i} ({ep}): ✗ error (status {resp.status_code})")
                all_healthy = False
        except Exception:
            print(f"  Worker {i} ({ep}): ✗ not running")
            all_healthy = False

    if not all_healthy:
        # Workers not running - show help
        print("\n--- Workers not running. To start: ---")
        print("""
    # Quick start: 2 workers on localhost
    python -m mplang.v2.cli up --world-size 2 --base-port 8100

    # Or from config file:
    python -m mplang.v2.cli up -c examples/conf/3pc.yaml

    # Then re-run this script to execute on distributed workers.
        """)
        return

    # Workers running - execute!
    print("\n--- Executing on distributed workers ---")

    # Create driver - same interface as Simulator!
    cluster = mp.ClusterSpec.simple(world_size=2, endpoints=endpoints)
    driver = mp.make_driver(endpoints, cluster_spec=cluster)

    # Define computation - same as Simulator!
    @mp.function
    def secure_add(x, y):
        return x + y

    # Run with driver context - same as Simulator!
    import jax.numpy as jnp

    with driver:
        x = jnp.array([1, 2, 3])
        y = jnp.array([4, 5, 6])
        result = secure_add(x, y)
        print("  Input x: [1, 2, 3]")
        print("  Input y: [4, 5, 6]")
        print(f"  Result:  {result.tolist()}")

    driver.shutdown()
    print("\n✓ Distributed execution completed!")


def main():
    """Run simulator and driver demos."""
    print("=" * 70)
    print("Device Tutorial: Simulator vs Driver")
    print("=" * 70)

    run_with_simulator()
    run_with_driver()

    print("\n" + "=" * 70)
    print("Summary:")
    print("- Simulator: local threads, fast testing")
    print("- Driver: HTTP workers, real distributed")
    print("- Same API: with sim/driver: + @mp.function")
    print("=" * 70)


if __name__ == "__main__":
    """
    Usage:
       uv run tutorials/v2/02_simulation_and_driver.py

    To run distributed:
       Terminal 1: python -m mplang.v2.cli up --world-size 2 --base-port 8100
       Terminal 2: uv run tutorials/v2/02_simulation_and_driver.py
    """
    main()
