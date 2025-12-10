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

"""Benchmark for OKVS-PSI (Single Threaded).

Measures the computational throughput of the OKVS-based PSI protocol.
bypasses SimpSimulator threading overhead to focus on Kernel performance.
"""

import os
import sys
import time

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import jax.numpy as jnp
import numpy as np

import mplang.v2.dialects.field as field
import mplang.v2.dialects.tensor as tensor
import mplang.v2.edsl.typing as elt
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.runtime.interpreter import InterpObject, Interpreter


def benchmark_okvs_psi(n_items):
    print(f"\n--- Benchmarking N={n_items} ---")
    interp = Interpreter()

    # helper
    def to_obj(arr):
        return InterpObject(
            TensorValue(jnp.array(arr)), elt.Tensor[elt.u64, arr.shape], interp
        )

    # 1. Data Gen (Unique Keys)
    start_time = time.time()
    # Use full 64-bit range to avoid collisions and simulate real hashes
    keys = np.arange(n_items, dtype=np.uint64)
    # randomize
    np.random.shuffle(keys)

    key_obj = to_obj(keys)
    time.time() - start_time

    # 0. Legacy Baseline (DH-PSI Simulation)
    # Measure 2048-bit modular exponentiation throughput on 1 core.
    # We sample a small batch to estimate.
    if n_items <= 100000:
        base_batch = 100
        # Random 2048-bit integers
        import secrets

        modulus = secrets.randbits(2048)
        exponent = secrets.randbits(256)
        inputs = [secrets.randbits(2048) for _ in range(base_batch)]

        start_dh = time.time()
        for x in inputs:
            pow(x, exponent, modulus)
        end_dh = time.time()

        dh_time_per_item = (end_dh - start_dh) / base_batch
        dh_throughput = 1.0 / dh_time_per_item
        print(
            f"Legacy DH-PSI (Baseline): ~{dh_throughput:,.0f} items/sec (Est. from {base_batch} ops)"
        )
    else:
        print("Legacy DH-PSI (Baseline): Skipped (Too slow for large N)")

    # 2. VOLE Simulation (Random generation cost)
    # M = 1.35 * N
    M = int(n_items * 1.35)
    if M % 128 != 0:
        M = ((M // 128) + 1) * 128

    start_vole = time.time()
    # Simulate VOLE generation (Offline Phase cost)
    # Sender: U, V. Recv: W, Delta.
    # In practice, this uses AES-NI PRG (fast).

    with interp:
        # Generate U, V, Delta
        # We simulate cost by generating Approx amount of random data via AES expand

        # Seed
        seed = jnp.array([[123, 456]], dtype=jnp.uint64)  # (1, 2)
        seed_obj = to_obj(seed)

        # Expand to M*2 (for V) + M*2 (for U) + M*2 (for W)
        # AES Expand is the core cost.

        # Warmup / Run
        _ = field.aes_expand(seed_obj, M)  # U
        _ = field.aes_expand(seed_obj, M)  # V
        _ = field.aes_expand(seed_obj, M)  # W (Receiver side)

    end_vole = time.time()
    vole_time = end_vole - start_vole
    print(f"VOLE (Simulated): {vole_time:.4f}s")

    # 3. OKVS Encode (Receiver)
    # Step 1: Hash Input to Value (Target)
    # Step 2: Solve System
    start_okvs = time.time()
    with interp:
        # Hash
        def _prep_seeds(items):
            lo = items
            hi = jnp.zeros_like(items)
            return jnp.stack([lo, hi], axis=1)

        seeds = tensor.run_jax(_prep_seeds, key_obj)
        h_y_expanded = field.aes_expand(seeds, 1)

        def _reshape(exp):
            return exp.reshape(exp.shape[0], 2)

        h_y = tensor.run_jax(_reshape, h_y_expanded)

        # Solve
        p_storage = field.solve_okvs(key_obj, h_y, m=M, seed=seed_obj)

        # Mask (P ^ W) - Simulating W access
        # Create dummy W
        w_obj = to_obj(np.zeros((M, 2), dtype=np.uint64))
        q = field.add(p_storage, w_obj)

        # Force execution
        _ = p_storage.runtime_obj.unwrap()
        _ = q.runtime_obj.unwrap()

    end_okvs = time.time()
    okvs_time = end_okvs - start_okvs
    print(f"Receiver (Hash+Encode): {okvs_time:.4f}s")

    # 4. Sender Decode
    # Step 1: Decode Q -> S
    # Step 2: Decode V -> V_dec
    # Step 3: Hash X -> H_x
    # Step 4: T = S ^ V_dec ^ H_x

    # 4. Sender Decode
    start_decode = time.time()
    q_obj = to_obj(np.zeros((M, 2), dtype=np.uint64))  # Dummy Q
    v_obj = to_obj(np.zeros((M, 2), dtype=np.uint64))  # Dummy V

    with interp:
        s_dec = field.decode_okvs(key_obj, q_obj, seed=seed_obj)
        v_dec = field.decode_okvs(key_obj, v_obj, seed=seed_obj)

        # Hash X (Same data for bench)
        seeds_x = tensor.run_jax(_prep_seeds, key_obj)
        h_x_exp = field.aes_expand(seeds_x, 1)
        h_x = tensor.run_jax(_reshape, h_x_exp)

        t = field.add(s_dec, v_dec)
        t = field.add(t, h_x)

        _ = t.runtime_obj.unwrap()

    end_decode = time.time()
    decode_time = end_decode - start_decode
    print(f"Sender (Decode+Calc): {decode_time:.4f}s")

    total_time = vole_time + okvs_time + decode_time
    throughput = n_items / total_time
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {throughput:,.0f} items/sec")

    return total_time, throughput


if __name__ == "__main__":
    sizes = [1000, 10000, 100000, 1000000]  # Scale up
    results = {}
    for n in sizes:
        total_time, throughput = benchmark_okvs_psi(n)
        results[n] = (total_time, throughput)

    # Summary
    print("\n" + "=" * 70)
    print("PSI BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'N Items':<15} {'Total Time (s)':<15} {'Throughput (ops/s)':<20}")
    print("-" * 70)

    for n in sizes:
        t_total, t_throughput = results[n]
        print(f"{n:<15} {t_total:<15.4f} {t_throughput:,.0f}")
    print("=" * 70)
