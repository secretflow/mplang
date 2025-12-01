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

"""Function Decorator: Compilation, Auditability, and Performance

Learning objectives:
1. Understand the role of @mp.function for graph compilation
2. Learn how compilation enables auditability and verification
3. See performance benefits from reducing cross-party round trips
4. Master when to use @function vs inline device() calls

Key concepts:
- @mp.function: Compiles Python code into MPIR (Multi-Party IR)
- Compilation: Traces all operations once, executes the graph many times
- Auditability: IR can be inspected, verified, and optimized before execution
- Performance: Eliminates Python interpreter overhead and batches communication

Prerequisites: Complete 00_device_basics.py
"""

import time

import mplang.v1 as mp

# Define a simple 2-party cluster
cluster_spec = mp.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61920"},
        {"name": "node_1", "endpoint": "127.0.0.1:61921"},
    ],
    "devices": {
        "SP0": {
            "kind": "SPU",
            "members": ["node_0", "node_1"],
            "config": {"protocol": "SEMI2K", "field": "FM128"},
        },
        "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
        "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
    },
})


def millionaire():
    """Explicitly specify device for each operation."""
    # Generate data on specific parties
    x = mp.device("P0")(lambda: 100)()
    y = mp.device("P1")(lambda: 200)()

    # Secure comparison on SPU
    result = mp.device("SP0")(lambda a, b: a < b)(x, y)
    result = mp.put("P0", result)

    return x, y, result


def heavy_computation():
    """Multiple operations benefit from compilation."""
    x = mp.device("P0")(lambda: 0)()

    # Compile captures the loop structure
    for i in range(100):
        x = mp.device("P0")(lambda a: a + i)(x)

    return x


# ============================================================================
# Main: Demonstrate Compilation Benefits
# ============================================================================


def main():
    sim = mp.Simulator(cluster_spec)

    print("=" * 70)
    print("Function Decorator: Compilation, Auditability, and Performance")
    print("=" * 70)

    print("\n--- Pattern 1: Compiled Workflow Execution ---")
    result0 = mp.evaluate(sim, millionaire)
    millionaire_jitted = mp.function(millionaire)
    result1 = mp.evaluate(sim, millionaire_jitted)

    # assert results are the same
    assert mp.fetch(sim, result0) == mp.fetch(sim, result1)

    print("\n--- Pattern 2: Performance---")
    # perform heavy computation without compilation
    start = time.time()
    _ = mp.evaluate(sim, heavy_computation)
    elapsed = time.time() - start
    print(f"Computed without @mp.function in {elapsed:.4f}s")

    heavy_computation_jitted = mp.function(heavy_computation)
    start = time.time()
    _ = mp.evaluate(sim, heavy_computation_jitted)
    elapsed = time.time() - start
    print(f"Computed with @mp.function in {elapsed:.4f}s")
    # Note: on simulator, the driver call overhead is trivial compared to real network latency.
    # In real deployments, the performance gain from compilation will be more pronounced.

    print("\n--- Pattern 3: Auditability - Inspect IR ---")
    # Compile to TracedFunction (does not execute yet)
    traced = mp.compile(sim, millionaire)

    # IR can be inspected programmatically
    print("Compiled MPIR (first 15 lines):")
    ir_text = traced.compiler_ir()
    lines = ir_text.split("\n")[:15]
    for line in lines:
        print(f"  {line}")
    print("  ...")


if __name__ == "__main__":
    main()
