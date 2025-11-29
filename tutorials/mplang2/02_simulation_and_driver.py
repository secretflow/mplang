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
2. Understand Simulator behavior in mplang2

Key concepts:
- Simulator: single-process, all parties in threads, fast iteration

Note: Driver (distributed execution) is not yet available in mplang2.
This tutorial focuses on Simulator functionality.

Migrated from mplang v1 to mplang2.
"""

import random

import mplang2 as mp


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

    # Create simulator with simple 2-party setup
    sim = mp.Simulator.simple(2)

    print("\n--- Millionaire problem (device API) ---")
    x, y, result = mp.evaluate(sim, millionaire)

    # Fetch results from all parties
    x_vals = mp.fetch(sim, x)
    y_vals = mp.fetch(sim, y)
    result_vals = mp.fetch(sim, result)

    # Values are HostVar holding per-party results
    print(f"P0 value: {x_vals}")
    print(f"P1 value: {y_vals}")
    print(f"x < y (SPU): {result_vals}")


def main():
    """Run simulator demo."""
    print("=" * 70)
    print("Device Tutorial: Simulator Execution")
    print("=" * 70)

    run_with_simulator()

    print("\n" + "=" * 70)
    print("Note: Driver (distributed execution) not yet in mplang2.")
    print("=" * 70)


if __name__ == "__main__":
    """
    Usage:
       uv run tutorials/mplang2/02_simulation_and_driver.py
    """
    main()
