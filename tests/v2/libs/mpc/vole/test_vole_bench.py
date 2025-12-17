#!/usr/bin/env python3
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

"""Benchmark Silver VOLE vs Gilboa VOLE performance."""

import time

import numpy as np

import mplang.v2 as mp
from mplang.v2.dialects import crypto, tensor
from mplang.v2.libs.mpc.vole import vole as gilboa_vole
from mplang.v2.libs.mpc.vole.silver import estimate_silver_communication, silver_vole


def benchmark_silver(sim, n: int, runs: int = 3):
    """Benchmark Silver VOLE."""
    sender, receiver = 0, 1

    def job():
        return silver_vole(sender, receiver, n)

    # Warmup
    traced = mp.compile(job, context=sim)
    mp.evaluate(traced, context=sim)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        mp.evaluate(traced, context=sim)
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


def benchmark_gilboa(sim, n: int, runs: int = 3):
    """Benchmark Gilboa VOLE."""
    sender, receiver = 0, 1

    def job():
        # Gilboa requires providers
        def u_provider():
            u_bytes = crypto.random_bytes(n * 16)

            def _reshape(b):
                return b.reshape(n, 2).view(np.uint64)

            return tensor.run_jax(_reshape, u_bytes)

        def delta_provider():
            d_bytes = crypto.random_bytes(16)

            def _view(b):
                return b.view(np.uint64)

            return tensor.run_jax(_view, d_bytes)

        return gilboa_vole(sender, receiver, n, u_provider, delta_provider)

    # Warmup
    traced = mp.compile(job, context=sim)
    mp.evaluate(traced, context=sim)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        mp.evaluate(traced, context=sim)
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


def main():
    print("=" * 60)
    print("Silver VOLE vs Gilboa VOLE Benchmark")
    print("=" * 60)

    sim = mp.make_simulator(2)

    test_sizes = [1000, 10000, 100000]

    results = []

    for n in test_sizes:
        print(f"\nN = {n:,}")
        print("-" * 40)

        # Benchmark Silver
        try:
            silver_time, silver_std = benchmark_silver(sim, n, runs=3)
            print(f"  Silver: {silver_time * 1000:.1f} ms ± {silver_std * 1000:.1f} ms")
        except Exception as e:
            print(f"  Silver: FAILED - {e}")
            silver_time = float("inf")

        # Benchmark Gilboa
        try:
            gilboa_time, gilboa_std = benchmark_gilboa(sim, n, runs=3)
            print(f"  Gilboa: {gilboa_time * 1000:.1f} ms ± {gilboa_std * 1000:.1f} ms")
        except Exception as e:
            print(f"  Gilboa: FAILED - {e}")
            gilboa_time = float("inf")

        # Compare
        if silver_time != float("inf") and gilboa_time != float("inf"):
            ratio = gilboa_time / silver_time if silver_time > 0 else 0
            print(f"  Ratio: Gilboa is {ratio:.2f}x of Silver")

        # Communication
        est = estimate_silver_communication(n)
        print(f"  Silver comm: {est['silver_bytes'] / 1024:.1f} KB")
        print(f"  Gilboa comm: {est['gilboa_bytes'] / 1024:.1f} KB")
        print(f"  Compression: {est['compression_ratio']:.1f}x")

        results.append({
            "n": n,
            "silver_ms": silver_time * 1000,
            "gilboa_ms": gilboa_time * 1000,
            "compression": est["compression_ratio"],
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"{'N':>10} | {'Silver':>10} | {'Gilboa':>10} | {'Time Ratio':>10} | {'Comm Comp':>10}"
    )
    print("-" * 60)
    for r in results:
        ratio = r["gilboa_ms"] / r["silver_ms"] if r["silver_ms"] > 0 else 0
        print(
            f"{r['n']:>10,} | {r['silver_ms']:>8.1f}ms | {r['gilboa_ms']:>8.1f}ms | {ratio:>9.2f}x | {r['compression']:>9.1f}x"
        )

    print("\nRECOMMENDATION:")
    avg_time_ratio = np.mean([
        r["gilboa_ms"] / r["silver_ms"] for r in results if r["silver_ms"] > 0
    ])
    if avg_time_ratio > 0.8:
        print("  Silver is NOT significantly slower than Gilboa.")
        print(
            "  C++ LDPC kernel is LOW PRIORITY - communication savings are the main benefit."
        )
    else:
        print("  Silver is significantly slower than Gilboa.")
        print("  C++ LDPC kernel is RECOMMENDED for performance parity.")


if __name__ == "__main__":
    main()
