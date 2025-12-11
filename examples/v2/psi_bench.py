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
This includes:
1. Python Scheduler overhead (Tracing + Execution)
2. Data Transfer overhead (Simulation)
3. C++ Kernel Computation (OKVS, OKVS Decode, AES)
"""

import os
import sys
import time
from collections import Counter
from typing import Any

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np

import mplang.v2 as mp
import mplang.v2.dialects.simp as simp
from mplang.v2.libs.mpc.psi import rr22


def benchmark_e2e_psi(n_items: int):
    print(f"\n--- Benchmarking E2E PSI N={n_items} ---")

    # 1. Setup Simulator
    # Enable primitive profiling globally
    from mplang.v2.edsl import registry

    registry.enable_profiling()

    # Enable backend profiling (Perfetto)
    sim = mp.Simulator.simple(2, enable_profiler=True)

    # --- MONKEY PATCH: Use Optimized OKVS Kernels ---
    from mplang.v2.dialects import field

    print("[INFO] Monkey-patching OKVS to use 'okvs_opt' (Mega-Binning)...")

    # Override with Opt primitives
    field.solve_okvs = field.solve_okvs_opt
    field.decode_okvs = field.decode_okvs_opt
    # -----------------------------------------------

    SENDER = 0
    RECEIVER = 1

    # 2. Generate Data
    # Use identical items to ensure correctness check passes implicitly
    # (Though we don't strictly verify correctness in bench for speed, we could)
    start_gen = time.time()
    # Use 64-bit random integers
    # Note: Generating 1M random numbers in Python/NumPy is fast enough
    shared_items = np.arange(n_items, dtype=np.uint64)
    np.random.shuffle(shared_items)

    sender_items = shared_items
    receiver_items = shared_items

    time.time() - start_gen
    # print(f"Data Gen: {gen_time:.4f}s")

    # 3. Define Protocol Job
    def job() -> Any:
        # PCall to place data (Simulated IO)
        s_handle = simp.constant((SENDER,), sender_items)
        r_handle = simp.constant((RECEIVER,), receiver_items)

        # Run Protocol
        # Returns intersection_mask on Sender
        mask = rr22.psi_intersect(SENDER, RECEIVER, n_items, s_handle, r_handle)
        return mask

    # 4. Compile (Tracing)
    start_compile = time.time()
    traced = mp.compile(sim, job)
    end_compile = time.time()
    compile_time = end_compile - start_compile
    print(f"Compile Time: {compile_time:.4f}s")

    # 4.1. Analyze Operator Distribution
    graph = traced.graph
    n_ops = len(graph.operations)
    print("\n" + "=" * 50)
    print("OPERATION DISTRIBUTION")
    print("=" * 50)

    op_counts = Counter(op.opcode for op in graph.operations)
    for op_name, count in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
        pct = count / n_ops * 100
        bar = "█" * int(pct / 2)
        print(f"  {op_name:<30} {count:>5}  ({pct:>5.1f}%) {bar}")
    print("=" * 50 + "\n")

    # 5. Execution (Runtime)
    start_exec = time.time()
    result_obj = mp.evaluate(sim, traced)

    # Force alignment/computation by fetching result
    # In SimpSimulator, evaluate is eager for standard ops, but fetching ensures
    # all futures are resolved.
    _ = mp.fetch(sim, result_obj)

    end_exec = time.time()
    exec_time = end_exec - start_exec

    throughput = n_items / exec_time

    print(f"Execution Time: {exec_time:.4f}s")
    print(f"Throughput: {throughput:,.0f} items/sec")

    # 6. Stop Profiler and Export Trace
    if hasattr(sim, "backend") and getattr(sim.backend, "profiler", None) is not None:
        trace_file = sim.backend.profiler.stop(filename_prefix=f"psi_trace_n{n_items}")
        print(f"Profiler trace exported to: {trace_file}")

    print("\nBENCHMARK PROFILING SUMMARY:")
    registry.get_profiler().print_summary()

    return exec_time, throughput


if __name__ == "__main__":
    # Warmup JAX/Simulator
    print("Warming up...")
    benchmark_e2e_psi(1000)

    sizes = [10000, 100000, 1000000]
    results = {}

    for n in sizes:
        t_exec, t_ops = benchmark_e2e_psi(n)
        results[n] = (t_exec, t_ops)

    # Summary
    print("\n" + "=" * 70)
    print("PSI END-TO-END BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'N Items':<15} {'Exec Time (s)':<15} {'Throughput (ops/s)':<20}")
    print("-" * 70)

    for n in sizes:
        t_exec, t_ops = results[n]
        print(f"{n:<15} {t_exec:<15.4f} {t_ops:,.0f}")
    print("=" * 70)
