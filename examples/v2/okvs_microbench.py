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

import time

import numpy as np

import mplang.v2.backends.field_impl as field_impl


def benchmark_okvs_direct(n=1_000_000):
    print(f"--- Benchmarking OKVS Optimization (Direct Kernel Call, N={n}) ---")

    # Production Check: M=1.4N (Matches GCT logic for N=1000 bins).
    # Increased to 1.5N to clear edge cases in microbench.
    m = int(n * 1.5)

    print(f"Generating data (N={n}, M={m})...")
    keys = np.random.randint(0, 2**64 - 1, size=(n,), dtype=np.uint64)
    values = np.random.randint(0, 2**64 - 1, size=(n, 2), dtype=np.uint64)
    seed = np.array([123, 456], dtype=np.uint64)

    # 1. Measure Naive
    print("Running Naive...")
    start = time.time()
    for _ in range(5):
        _ = field_impl._okvs_solve_impl(keys, values, m, seed)
    t_naive = (time.time() - start) / 5.0

    # 2. Measure Optimized
    print("Running Optimized...")
    start = time.time()
    for _ in range(5):
        _ = field_impl._okvs_solve_opt_impl(keys, values, m, seed)
    t_opt = (time.time() - start) / 5.0

    print("\nResults:")
    print(f"Naive Time: {t_naive * 1000:.2f} ms")
    print(f"Opt Time:   {t_opt * 1000:.2f} ms")
    print(f"Speedup:    {t_naive / t_opt:.2f}x")


if __name__ == "__main__":
    benchmark_okvs_direct()
