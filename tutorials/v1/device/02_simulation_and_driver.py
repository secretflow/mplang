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
2. Use Driver for distributed multi-party execution
3. Understand the differences and when to use each

Key differences:
- Simulator: single-process, all parties in threads, fast iteration
- Driver: multi-process/multi-host, real distributed setup, production-ready
"""

import random

import yaml

import mplang.v1 as mp


def millionaire():
    """Simple millionaire using device API (from 00_device_basics)."""
    # Generate data on specific parties
    x = mp.device("P0")(lambda: random.randint(0, 10))()
    y = mp.device("P1")(lambda: random.randint(0, 10))()

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

    sim = mp.Simulator.simple(2)

    print("\n--- Millionaire problem (device API) ---")
    x, y, result = mp.evaluate(sim, millionaire)
    print(f"P0 value: {mp.fetch(sim, x)}")
    print(f"P1 value: {mp.fetch(sim, y)}")
    print(f"x < y (SPU): {mp.fetch(sim, result)}")


# ============================================================================
# Driver: Distributed Multi-Party Execution
# ============================================================================


def run_with_driver(cluster_spec):
    """Driver: for real distributed execution across hosts."""
    print("\n" + "=" * 70)
    print("Running with Driver (distributed, multi-process)")
    print("=" * 70)

    # Driver connects to running cluster
    driver = mp.Driver(cluster_spec)

    print("\n--- Millionaire problem (device API) ---")
    x, y, result = mp.evaluate(driver, millionaire)
    print(f"P0 value: {mp.fetch(driver, x)}")
    print(f"P1 value: {mp.fetch(driver, y)}")
    print(f"x < y (SPU): {mp.fetch(driver, result)}")


# ============================================================================
# CLI: Support Both Modes
# ============================================================================


def cmd_main():
    """Command-line interface to choose simulator or driver."""
    import argparse

    parser = argparse.ArgumentParser(description="MPLang: Simulator vs Driver")
    parser.add_argument(
        "-c",
        "--config",
        default="examples/v1/conf/3pc.yaml",
        help="Cluster config YAML",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("sim", help="Run with Simulator (local)")
    subparsers.add_parser("run", help="Run with Driver (distributed)")
    args = parser.parse_args()

    # Load cluster spec
    with open(args.config) as file:
        conf = yaml.safe_load(file)
    cluster_spec = mp.ClusterSpec.from_dict(conf)

    if args.command == "sim":
        run_with_simulator()
    elif args.command == "run":
        run_with_driver(cluster_spec)
    else:
        parser.print_help()
        print("\nNo command specified. Running with Simulator by default.")
        run_with_simulator()


if __name__ == "__main__":
    """
    Usage:

    1. Simulator (local, no setup needed):
       uv run tutorials/device/02_simulation_and_driver.py sim

    2. Driver (distributed):
       Step 1: Start cluster in separate terminal:
         uv run python -m mplang.runtime.cli up -c examples/v1/conf/3pc.yaml

       Step 2: Run computation:
         uv run tutorials/device/02_simulation_and_driver.py run
    """
    cmd_main()
