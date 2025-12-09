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

"""Single-file debug: Verify the complete Gilboa calculation step by step."""

import numpy as np

from mplang.v2.backends.field_impl import _gf128_mul_impl


def prg_expand(seeds_u64, n):
    """Simple PRG using numpy random."""
    # seeds_u64: (K, 2) u64
    k = seeds_u64.shape[0]
    output = np.zeros((k, n, 2), dtype=np.uint64)
    for i in range(k):
        seed = [int(seeds_u64[i, 0]), int(seeds_u64[i, 1])]
        rng = np.random.default_rng(seed)
        output[i] = rng.integers(0, 2**64, size=(n, 2), dtype=np.uint64)
    return output


def test_gilboa_manual():
    """Manual Gilboa calculation to verify the algorithm."""
    K = 128
    N = 2

    # 1. Create seeds - T for Sender, Q for Receiver
    # Q[i] = T[i] when d[i]=0, Q[i] = T[i]^S when d[i]=1
    np.random.seed(42)
    t_seeds = np.random.randint(0, 256, (K, 16), dtype=np.uint8)  # (128, 16)
    s_bytes = np.random.randint(0, 256, (16,), dtype=np.uint8)  # (16,)

    # Delta with known bits: delta = 3 (bits 0 and 1 set)
    delta = np.array([3, 0], dtype=np.uint64)
    delta_bits = np.unpackbits(delta.view(np.uint8), bitorder="little")[:K]  # (128,)

    print(f"delta: {delta}")
    print(f"delta_bits[:10]: {delta_bits[:10]}")
    print(f"Number of 1 bits: {np.sum(delta_bits)}")

    # Q seeds
    q_seeds = t_seeds.copy()
    for i in range(K):
        if delta_bits[i]:
            q_seeds[i] = t_seeds[i] ^ s_bytes

    # 2. Expand seeds to (K, N) u128 values
    # Convert 16-byte seeds to (2,) u64 for AES expand
    t_seeds_u64 = t_seeds.view(np.uint64).reshape(K, 2)
    q_seeds_u64 = q_seeds.view(np.uint64).reshape(K, 2)
    ts_seeds_u64 = (t_seeds ^ s_bytes).view(np.uint64).reshape(K, 2)

    v0 = prg_expand(t_seeds_u64, N)  # (K, N, 2)
    v1 = prg_expand(ts_seeds_u64, N)  # (K, N, 2)
    w_raw = prg_expand(q_seeds_u64, N)  # (K, N, 2)

    print(f"\nv0.shape: {v0.shape}")
    print(f"v0[0,0]: {v0[0, 0]}")
    print(f"w_raw[0,0]: {w_raw[0, 0]}")

    # Verify: w_raw[i] = v0[i] if delta[i]=0, else v1[i]
    for i in range(K):
        expected = v0[i] if delta_bits[i] == 0 else v1[i]
        assert np.array_equal(w_raw[i], expected), f"Mismatch at i={i}"
    print("w_raw matches expected (v0 or v1)")

    # 3. U values
    u = np.array([[1, 0], [2, 0]], dtype=np.uint64)  # (N, 2)

    # 4. Compute terms: term[i] = U * x^i
    powers = []
    for i in range(K):
        lo, hi = 0, 0
        if i < 64:
            lo = 1 << i
        else:
            hi = 1 << (i - 64)
        powers.append(np.array([[lo, hi]], dtype=np.uint64))

    term = np.zeros((K, N, 2), dtype=np.uint64)
    for i in range(K):
        p_broad = np.tile(powers[i], (N, 1))  # (N, 2)
        term[i] = _gf128_mul_impl(u, p_broad)

    print(f"\nterm[0,0]: {term[0, 0]} (should be U[0]*x^0 = U[0] = [1,0])")
    print(f"term[1,0]: {term[1, 0]} (should be U[0]*x^1 = [2,0])")

    # 5. Compute M: M[i] = V0[i] ^ V1[i] ^ term[i]
    m = v0 ^ v1 ^ term  # (K, N, 2)

    # 6. Compute V (Sender output): V = XOR_sum(V0[i])
    v_sender = v0[0].copy()
    for i in range(1, K):
        v_sender ^= v0[i]

    print(f"\nv_sender[0]: {v_sender[0]}")

    # 7. Compute W (Receiver output): W = XOR_sum(W_raw[i] ^ (delta[i] * M[i]))
    w_receiver = np.zeros((N, 2), dtype=np.uint64)
    for i in range(K):
        if delta_bits[i]:
            w_receiver ^= w_raw[i] ^ m[i]
        else:
            w_receiver ^= w_raw[i]

    print(f"w_receiver[0]: {w_receiver[0]}")

    # 8. Verify: W = V + U * delta
    delta_broad = np.tile(delta, (N, 1))
    u_times_delta = _gf128_mul_impl(u, delta_broad)
    expected_w = v_sender ^ u_times_delta

    print(f"\nu_times_delta[0]: {u_times_delta[0]}")
    print(f"expected_w[0]: {expected_w[0]}")
    print(f"actual_w[0]: {w_receiver[0]}")
    print(f"diff[0]: {expected_w[0] ^ w_receiver[0]}")

    np.testing.assert_array_equal(
        w_receiver, expected_w, err_msg="Gilboa Manual FAILED!"
    )
    print("\nGilboa Manual PASSED!")


if __name__ == "__main__":
    test_gilboa_manual()
