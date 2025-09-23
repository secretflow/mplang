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

"""Tutorial 8: Partially Homomorphic Encryption (PHE) with SIMP API

This tutorial demonstrates a three-party computation using PHE:
1. All three parties generate random numbers
2. Party 0 generates a key pair
3. Party 0 broadcasts the public key to all parties
4. Each party encrypts their data using the public key
5. Each party sends their encrypted data to Party 0
6. Party 0 computes the sum and decrypts the result
"""

import mplang
import mplang.simp as simp
from mplang.frontend import phe


@mplang.function
def three_party_phe_sum():
    """Perform a three-party PHE computation to sum private values."""

    # Step 1: All parties generate random numbers
    data = simp.prank()

    # Step 2: Party 0 generates PHE key pair
    pkey, skey = simp.runAt(0, phe.keygen)()

    # Step 3: Party 0 broadcasts public key to all parties
    world_mask = mplang.Mask.all(3)
    pkey_bcasted = simp.bcast_m(world_mask, 0, pkey)

    # Step 4: Each party encrypts their data
    encrypted = simp.run(phe.encrypt)(data, pkey_bcasted)

    # Step 5: All parties send encrypted data to Party 0
    # Gather all encrypted data at Party 0
    e0, e1, e2 = simp.gather_m(world_mask, 0, encrypted)

    # Step 6: Party 0 computes sum and decrypts
    sum_e0_e1 = simp.runAt(0, phe.add)(e0, e1)

    # Add the third encrypted value
    encrypted_sum = simp.runAt(0, phe.add)(sum_e0_e1, e2)

    # Decrypt the final result
    final_result = simp.runAt(0, phe.decrypt)(encrypted_sum, skey)

    return final_result


def run_simulation():
    """Run the PHE simulation locally."""
    # Set up 3-party simulation with PHE support
    sim = mplang.Simulator.simple(3)
    result = mplang.evaluate(sim, three_party_phe_sum)
    print(f"Simulation completed. Final sum: {mplang.fetch(sim, result)}")

    compiled = mplang.compile(sim, three_party_phe_sum)
    print("compiled:", compiled.compiler_ir())


if __name__ == "__main__":
    run_simulation()
