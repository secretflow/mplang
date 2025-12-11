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

"""Minimal test to verify GF(2^128) powers reconstruction."""

import numpy as np

from mplang.v2.backends.field_impl import _gf128_mul_impl


def test_powers_reconstruction():
    """Verify sum(delta_i * x^i) = delta in GF(2^128)."""

    # Random delta
    delta = np.array([0x123456789ABCDEF0, 0xFEDCBA9876543210], dtype=np.uint64)
    print(f"Original delta: [{delta[0]:016x}, {delta[1]:016x}]")

    # Decompose to bits (little endian)
    delta_bytes = delta.view(np.uint8)  # 16 bytes
    delta_bits = np.unpackbits(delta_bytes, bitorder="little")  # 128 bits
    print(f"delta_bits[:10]: {delta_bits[:10]}")
    print(f"delta_bits sum: {np.sum(delta_bits)}")  # Number of 1s

    # Generate powers of x
    powers = []
    for i in range(128):
        lo, hi = 0, 0
        if i < 64:
            lo = 1 << i
        else:
            hi = 1 << (i - 64)
        powers.append(np.array([[lo, hi]], dtype=np.uint64))  # (1, 2)

    powers_arr = np.concatenate(powers, axis=0)  # (128, 2)
    print(
        f"powers[0]: [{powers_arr[0, 0]:016x}, {powers_arr[0, 1]:016x}]"
    )  # Should be [1, 0]
    print(
        f"powers[63]: [{powers_arr[63, 0]:016x}, {powers_arr[63, 1]:016x}]"
    )  # Should be [0x8000..., 0]
    print(
        f"powers[64]: [{powers_arr[64, 0]:016x}, {powers_arr[64, 1]:016x}]"
    )  # Should be [0, 1]

    # Reconstruct delta = sum(delta_bits[i] * x^i)
    reconstructed = np.zeros((1, 2), dtype=np.uint64)
    for i in range(128):
        if delta_bits[i]:
            reconstructed ^= powers_arr[i : i + 1]

    reconstructed = reconstructed[0]
    print(f"Reconstructed delta: [{reconstructed[0]:016x}, {reconstructed[1]:016x}]")

    # Verify
    np.testing.assert_array_equal(
        reconstructed, delta, err_msg="Powers reconstruction failed!"
    )
    print("Powers reconstruction PASSED!")

    # Now verify: sum(delta_i * U * x^i) = U * delta
    U = np.array([[0xAAAAAAAAAAAAAAAA, 0x5555555555555555]], dtype=np.uint64)  # (1, 2)
    print(f"\nU: [{U[0, 0]:016x}, {U[0, 1]:016x}]")

    # Method 1: Direct multiplication
    delta_broad = np.tile(delta, (1, 1))  # (1, 2)
    expected = _gf128_mul_impl(U, delta_broad)[0]
    print(f"U * delta (direct): [{expected[0]:016x}, {expected[1]:016x}]")

    # Method 2: Sum of U * x^i for each delta_bit
    sum_result = np.zeros(2, dtype=np.uint64)
    for i in range(128):
        if delta_bits[i]:
            term = _gf128_mul_impl(U, powers_arr[i : i + 1])[0]
            sum_result ^= term

    print(f"sum(delta_i * U * x^i): [{sum_result[0]:016x}, {sum_result[1]:016x}]")

    # Verify
    np.testing.assert_array_equal(
        sum_result, expected, err_msg="Sum decomposition failed!"
    )
    print("Sum decomposition PASSED!")


if __name__ == "__main__":
    test_powers_reconstruction()
