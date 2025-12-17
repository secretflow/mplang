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
3. Same code runs on both - just swap context

Key concepts:
- Simulator: single-process, all parties in threads, fast iteration
- Driver: multi-process, HTTP-based, for real distributed deployment
- Same cluster_spec works for both sim and driver
"""

import httpx
import jax
import jax.numpy as jnp

import mplang.v2 as mp

# Modify JAX config to avoid "Error reading persistent compilation cache" warning
# appearing in some environments where the cache is not properly initialized.
jax.config.update("jax_enable_compilation_cache", False)

# ============================================================================
# Shared cluster definition - works for both Simulator and Driver
# ============================================================================

ENDPOINTS = ["http://127.0.0.1:8100", "http://127.0.0.1:8101"]

cluster_spec = mp.ClusterSpec.simple(world_size=2, endpoints=ENDPOINTS)


# ============================================================================
# Shared computation - runs on both Simulator and Driver
# ============================================================================


@mp.function
def millionaire(alice_wealth, bob_wealth):
    """Classic millionaire problem: who is richer without revealing amounts."""
    return alice_wealth > bob_wealth


def run_computation(ctx_name: str):
    """Run millionaire problem and print results."""
    alice = jnp.array(5_000_000)  # Alice: $5M
    bob = jnp.array(3_000_000)  # Bob: $3M
    alice_is_richer = millionaire(alice, bob)
    print(f"  [{ctx_name}] Alice($5M) > Bob($3M)? {bool(alice_is_richer)}")


# ============================================================================
# Health probe - simple inline check
# ============================================================================


def probe(endpoints: list[str]) -> bool:
    """Check if all workers are healthy. Returns True if all OK."""
    all_ok = True
    for i, ep in enumerate(endpoints):
        try:
            resp = httpx.get(f"{ep}/health", timeout=2)
            status = "✓" if resp.status_code == 200 else f"✗ ({resp.status_code})"
            if resp.status_code != 200:
                all_ok = False
        except Exception:
            status = "✗ (not running)"
            all_ok = False
        print(f"  Worker {i}: {status}")
    return all_ok


# ============================================================================
# Main: Run same code on Simulator, then Driver
# ============================================================================


def main():
    print("=" * 70)
    print("Simulator vs Driver: Same Code, Different Backend")
    print("=" * 70)

    # --- Pattern 1: Simulator (always works, local threads) ---
    print("\n--- Simulator (local, multi-threaded) ---")
    sim = mp.make_simulator(2, cluster_spec=cluster_spec)
    with sim:
        run_computation("Simulator")

    # --- Pattern 2: Driver (requires workers to be running) ---
    print("\n--- Driver (distributed, HTTP-based) ---")
    print("Probing workers...")
    if not probe(ENDPOINTS):
        print("\n  Workers not running. To start:")
        print("    python -m mplang.v2.cli up -w 2 -p 8100")
        print("  Then re-run this script.")
    else:
        driver = mp.make_driver(ENDPOINTS, cluster_spec=cluster_spec)
        with driver:
            run_computation("Driver")
        driver.shutdown()
        print("  ✓ Distributed execution completed!")

    print("\n" + "=" * 70)
    print("Key insight: Same cluster_spec, same @mp.function, same with-block.")
    print("Only difference: make_simulator() vs make_driver()")
    print("=" * 70)


if __name__ == "__main__":
    main()
