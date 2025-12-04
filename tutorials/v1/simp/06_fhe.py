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

"""Tutorial 11: Fully Homomorphic Encryption (FHE) with SIMP API

This tutorial demonstrates a three-party computation using FHE (CKKS scheme):
1. All three parties generate random floating-point numbers
2. Party 0 generates FHE context pair (private and public)
3. Party 0 broadcasts the public context to all parties
4. Each party encrypts their data using the public context
5. Each party sends their encrypted data to Party 0
6. Party 0 computes operations (sum, product, dot product, polynomial evaluation)
7. Party 0 decrypts the results

FHE vs PHE:
- PHE (Partially Homomorphic): Supports either addition OR multiplication (not both)
- FHE (Fully Homomorphic): Supports BOTH addition AND multiplication on encrypted data
- CKKS scheme: Optimized for approximate floating-point arithmetic
- BFV scheme: Optimized for exact integer arithmetic
"""

import numpy as np

import mplang.v1 as mp
from mplang.v1.ops import fhe


@mp.function
def three_party_fhe_sum():
    """Perform a three-party FHE computation to sum private values (basic example)."""

    # Step 1: All parties generate random numbers
    data = mp.prank()
    data = mp.run_jax(lambda x: x.astype(np.float32), data)

    # Step 2: Party 0 generates FHE context pair (using CKKS for floating-point)
    private_ctx, public_ctx, _ = mp.run_at(0, fhe.keygen, scheme="CKKS")

    # Step 3: Party 0 broadcasts public context to all parties
    world_mask = mp.Mask.all(3)
    public_ctx_bcasted = mp.bcast_m(world_mask, 0, public_ctx)

    # Step 4: Each party encrypts their data with public context
    encrypted = mp.run(None, fhe.encrypt, data, public_ctx_bcasted)

    # Step 5: All parties send encrypted data to Party 0
    e0, e1, e2 = mp.gather_m(world_mask, 0, encrypted)

    # Step 6: Party 0 computes homomorphic sum
    sum_e0_e1 = mp.run_at(0, fhe.add, e0, e1)
    encrypted_sum = mp.run_at(0, fhe.add, sum_e0_e1, e2)

    # Step 7: Party 0 decrypts the final result
    final_result = mp.run_at(0, fhe.decrypt, encrypted_sum, private_ctx)

    return final_result


def run_simulation():
    """Run FHE simulations locally."""
    print("=" * 70)
    print("FHE Tutorial - Fully Homomorphic Encryption Demo")
    print("=" * 70)

    # === Example 1: Basic three-party sum ===
    print("\n--- Example 1: Three-Party FHE Sum (Basic) ---")
    sim1 = mp.Simulator.simple(3)
    result1 = mp.evaluate(sim1, three_party_fhe_sum)
    sum_value = mp.fetch(sim1, result1)
    # Extract the actual value from party 0's result
    if isinstance(sum_value, (list, tuple)):
        sum_value = sum_value[0]
    print(f"Three-party encrypted sum result: {sum_value}")
    print("✓ Successfully computed sum on encrypted data!")

    # === Show compilation IR ===
    print("\n--- Compilation IR (Basic Example) ---")
    compiled = mp.compile(sim1, three_party_fhe_sum)
    print("Function compiled successfully:")
    print(compiled.compiler_ir())


if __name__ == "__main__":
    run_simulation()
