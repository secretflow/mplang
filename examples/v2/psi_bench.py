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

"""Benchmark for End-to-End RR22 PSI Protocol.

Measures the full protocol throughput using SimpSimulator (2-party local simulation).

Usage:
    # Via CLI (simulator mode with profiling)
    python -m mplang.v2.cli sim -f examples/v2/psi_bench.py --profile

    # Via CLI (distributed HTTP cluster)
    python -m mplang.v2.cli run -c cluster.yaml -f examples/v2/psi_bench.py

    # Direct execution (standalone)
    python examples/v2/psi_bench.py
"""

from __future__ import annotations

import time
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# MPLang Entry Point (for CLI: `mplang sim -f psi_bench.py`)
# ---------------------------------------------------------------------------


def __mp_main__(*args: str) -> dict[str, Any]:
    """MPLang workload entry point.

    Args:
        *args: CLI arguments (first arg is n_items, default 100000)

    Returns:
        Benchmark results dict
    """
    import mplang.v2 as mp
    from mplang.v2.dialects import simp
    from mplang.v2.libs.mpc.psi import rr22

    # Parse n_items from CLI args
    n_items = int(args[0]) if args else 100000

    print(f"\n--- PSI Benchmark N={n_items} ---")

    # Use Optimized OKVS Kernels
    from mplang.v2.dialects import field

    field.solve_okvs = field.solve_okvs_opt
    field.decode_okvs = field.decode_okvs_opt

    SENDER = 0
    RECEIVER = 1

    # Generate Data
    sender_items = np.arange(n_items, dtype=np.uint64)
    np.random.shuffle(sender_items)
    receiver_items = sender_items.copy()

    # Define Protocol
    def job() -> Any:
        s_handle = simp.constant((SENDER,), sender_items)
        r_handle = simp.constant((RECEIVER,), receiver_items)
        return rr22.psi_intersect(SENDER, RECEIVER, n_items, s_handle, r_handle)

    # Compile
    t0 = time.time()
    # Uses implicit context from CLI or caller
    traced = mp.compile(job)
    compile_time = time.time() - t0
    print(f"Compile Time: {compile_time:.4f}s")

    # Op Distribution
    graph = traced.graph
    op_counts = Counter(op.opcode for op in graph.operations)
    print(f"Total Ops: {len(graph.operations)}")
    for op_name, count in sorted(op_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {op_name}: {count}")

    # Execute
    t1 = time.time()
    result = mp.evaluate(traced)
    _ = mp.fetch(result)
    exec_time = time.time() - t1

    throughput = n_items / exec_time
    print(f"Exec Time: {exec_time:.4f}s")
    print(f"Throughput: {throughput:,.0f} items/sec")

    return {"n_items": n_items, "exec_time": exec_time, "throughput": throughput}


# ---------------------------------------------------------------------------
# Standalone Execution (for direct `python psi_bench.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import mplang.v2 as mp

    # Create simulator (2-party, with profiling enabled)
    sim = mp.make_simulator(2, enable_profiling=True)

    try:
        # Use 'with sim:' to set implicit context for standalone run
        with sim:
            # Warmup
            print("Warming up...")
            __mp_main__("1000")

            # Benchmark
            sizes = ["10000", "100000", "1000000"]
            results = {}

            for n in sizes:
                res = __mp_main__(n)
                results[int(n)] = res

        # Summary
        print("\n" + "=" * 60)
        print("PSI BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"{'N Items':<15} {'Exec (s)':<12} {'Throughput':<20}")
        print("-" * 60)
        for n, res in results.items():
            print(f"{n:<15} {res['exec_time']:<12.4f} {res['throughput']:,.0f}")
        print("=" * 60)

        # Profiler Summary
        mp.get_profiler().print_summary()

    finally:
        # Stop tracer and save
        backend = sim.backend
        if hasattr(backend, "tracer") and backend.tracer:
            backend.tracer.stop(filename_prefix="psi_bench")
        backend.shutdown()
