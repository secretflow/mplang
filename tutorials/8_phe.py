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

import jax.numpy as jnp
import mplang
import mplang.mpi as mpi
import mplang.simp as simp
from mplang.frontend import phe
import numpy as np


@mplang.function
def three_party_phe_sum():
    """Perform a three-party PHE computation to sum private values."""

    # Step 1: All parties generate random numbers
    data = simp.prank()

    # Step 2: Party 0 generates PHE key pair
    pkey, skey = simp.runAt(0, phe.keygen)()

    # Step 3: Party 0 broadcasts public key to all parties
    world_mask = mplang.Mask.all(3)
    pkey_bcasted = mpi.bcast_m(world_mask, 0, pkey)

    # Step 4: Each party encrypts their data
    encrypted = simp.run(phe.encrypt)(data, pkey_bcasted)

    # Step 5: All parties send encrypted data to Party 0
    # Gather all encrypted data at Party 0
    e0, e1, e2 = mpi.gather_m(world_mask, 0, encrypted)

    # Step 6: Party 0 computes sum and decrypts
    sum_e0_e1 = simp.runAt(0, phe.add)(e0, e1)

    # Add the third encrypted value
    encrypted_sum = simp.runAt(0, phe.add)(sum_e0_e1, e2)

    # Decrypt the final result
    final_result = simp.runAt(0, phe.decrypt)(encrypted_sum, skey)

    return final_result


@mplang.function
def test_2d_matrix_operations():
    """Test PHE operations with 2D matrices using new tensor operators."""

    # Step 1: Create 2D matrices at each party
    # Party 0: 2x3 matrix
    # Party 1: 3x2 matrix (for dot product)
    # Party 2: 2x3 matrix (same shape as Party 0)

    def create_2d_data():
        """Create 2D data based on party rank."""
        rank = simp.prank()
        # Use different shapes for different operations
        # We'll create fixed data for demonstration
        return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # 2x3 for all parties

    data = simp.run(create_2d_data)()

    # Step 2: Generate PHE keys at Party 0
    pkey, skey = simp.runAt(0, phe.keygen)()

    # Step 3: Broadcast public key
    world_mask = mplang.Mask.all(3)
    pkey_bcasted = mpi.bcast_m(world_mask, 0, pkey)

    # Step 4: Encrypt data at each party
    encrypted = simp.run(phe.encrypt)(data, pkey_bcasted)

    # Gather encrypted data at Party 0
    e0, e1, e2 = mpi.gather_m(world_mask, 0, encrypted)

    # Step 5: Perform various tensor operations at Party 0

    # Test RESHAPE: reshape e0 from (2,3) to (6,)
    reshaped_e0 = simp.runAt(0, phe.reshape)(e0, (6,))

    # Test TRANSPOSE: transpose e0 from (2,3) to (3,2)
    transposed_e0 = simp.runAt(0, phe.transpose)(e0)

    # Test DOT: e0 (2x3) dot transposed_e0 (3x2) -> (2x2)
    # Create plaintext matrix for dot product
    def create_dot_matrix():
        return np.array([[1, 2], [3, 1], [2, 1]], dtype=np.int32)  # 3x2

    dot_matrix = simp.runAt(0, create_dot_matrix)()
    dot_result = simp.runAt(0, phe.dot)(e0, dot_matrix)

    # Test GATHER: gather specific elements from reshaped_e0
    def create_gather_indices():
        return np.array([0, 2, 4], dtype=np.int32)

    gather_indices = simp.runAt(0, create_gather_indices)()
    gathered = simp.runAt(0, phe.gather)(reshaped_e0, gather_indices)

    # Test CONCAT: concatenate e0 and e2 along axis 0 -> (4,3)
    concat_result = simp.runAt(0, phe.concat)([e0, e2], axis=0)

    # Test SCATTER: create updates and scatter into reshaped_e0
    def create_scatter_indices():
        return np.array([1, 3, 5], dtype=np.int32)

    scatter_indices = simp.runAt(0, create_scatter_indices)()
    # Create encrypted updates (use gathered values)
    scattered = simp.runAt(0, phe.scatter)(reshaped_e0, scatter_indices, gathered)

    # Decrypt results for verification
    dot_decrypted = simp.runAt(0, phe.decrypt)(dot_result, skey)
    gathered_decrypted = simp.runAt(0, phe.decrypt)(gathered, skey)
    concat_decrypted = simp.runAt(0, phe.decrypt)(concat_result, skey)

    return dot_decrypted, gathered_decrypted, concat_decrypted


@mplang.function
def test_3d_tensor_operations():
    """Test PHE operations with 3D tensors using advanced tensor operators."""

    # Step 1: Create 3D tensors at each party - use simple approach
    def create_3d_data():
        """Create 3D data - same for all parties for simplicity."""
        # Use same data for all parties to avoid TraceVar issues
        return np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32
        )  # 2x2x3

    data = simp.run(create_3d_data)()

    # Step 2: Generate PHE keys at Party 0
    pkey, skey = simp.runAt(0, phe.keygen)()

    # Step 3: Broadcast public key
    world_mask = mplang.Mask.all(3)
    pkey_bcasted = mpi.bcast_m(world_mask, 0, pkey)

    # Step 4: Encrypt data at each party
    encrypted = simp.run(phe.encrypt)(data, pkey_bcasted)

    # Gather encrypted data at Party 0
    e0, e1, e2 = mpi.gather_m(world_mask, 0, encrypted)

    # Step 5: Perform various 3D tensor operations at Party 0

    # Test RESHAPE: reshape e0 from (2,2,3) to (12,)
    reshaped_e0 = simp.runAt(0, phe.reshape)(e0, (12,))

    # Test TRANSPOSE: transpose e0 -> (3,2,2)
    transposed_e0 = simp.runAt(0, phe.transpose)(e0, axes=(2, 0, 1))

    # Test DOT: use plaintext matrix for dot product with reshaped encrypted tensor
    # Reshape e0 to (3,4) for valid matrix multiplication with (4,3) plaintext
    reshaped_for_dot_e0 = simp.runAt(0, phe.reshape)(e0, (3, 4))

    # Create plaintext matrix (4,3) for dot product
    def create_3d_dot_matrix():
        return np.array(
            [[1, 2, 1], [3, 1, 2], [2, 4, 1], [1, 3, 2]], dtype=np.int32
        )  # 4x3

    dot_matrix_3d = simp.runAt(0, create_3d_dot_matrix)()
    dot_result = simp.runAt(0, phe.dot)(reshaped_for_dot_e0, dot_matrix_3d)

    # Test GATHER: gather specific elements from reshaped_e0
    def create_3d_gather_indices():
        return np.array([0, 3, 6, 9], dtype=np.int32)

    gather_indices = simp.runAt(0, create_3d_gather_indices)()
    gathered = simp.runAt(0, phe.gather)(reshaped_e0, gather_indices)

    # Test multi-axis GATHER: demonstrate gathering along different axes
    # For 3D tensor e0 with shape (2,2,3):

    # Gather along axis=0 (default) - gather slices [0,1] -> shape (2,2,3)
    indices_axis0 = simp.runAt(0, lambda: np.array([0, 1], dtype=np.int32))()
    gathered_axis0 = simp.runAt(0, phe.gather)(e0, indices_axis0, axis=0)

    # Gather along axis=1 - gather rows [0,1] from each slice -> shape (2,2,3)
    gathered_axis1 = simp.runAt(0, phe.gather)(e0, indices_axis0, axis=1)

    # Gather along axis=2 - gather columns [0,2] from each position -> shape (2,2,2)
    indices_axis2 = simp.runAt(0, lambda: np.array([0, 2], dtype=np.int32))()
    gathered_axis2 = simp.runAt(0, phe.gather)(e0, indices_axis2, axis=2)

    # Test CONCAT: concatenate e0 and e2 along axis 0 -> (4,2,3)
    concat_result = simp.runAt(0, phe.concat)([e0, e2], axis=0)

    # Test SCATTER: create updates and scatter into reshaped_e0
    def create_3d_scatter_indices():
        return np.array([1, 4, 7, 10], dtype=np.int32)

    scatter_indices = simp.runAt(0, create_3d_scatter_indices)()
    scattered = simp.runAt(0, phe.scatter)(reshaped_e0, scatter_indices, gathered)

    # Test multi-axis SCATTER: demonstrate scattering along different axes
    # For 3D tensor e0 with shape (2,2,3):

    # Create update values for scatter operations
    # For axis=1 scatter: update shape should match the indices shape plus remaining dims
    def create_axis1_updates():
        # Updates for scattering at positions [0,1] along axis=1: shape should be (2,2,3)
        return np.array(
            [[[100, 101, 102], [103, 104, 105]], [[106, 107, 108], [109, 110, 111]]],
            dtype=np.int32,
        )

    axis1_updates = simp.runAt(0, create_axis1_updates)()
    axis1_updates_encrypted = simp.runAt(0, phe.encrypt)(axis1_updates, pkey_bcasted)

    # Scatter along axis=1 - scatter updates at rows [0,1] for each slice
    scattered_axis1 = simp.runAt(0, phe.scatter)(
        e0, indices_axis0, axis1_updates_encrypted, axis=1
    )

    # For axis=2 scatter: create smaller updates that match the gathered shape
    def create_axis2_updates():
        # Updates for scattering at positions [0,2] along axis=2: shape should be (2,2,2)
        return np.array(
            [[[200, 201], [202, 203]], [[204, 205], [206, 207]]], dtype=np.int32
        )

    axis2_updates = simp.runAt(0, create_axis2_updates)()
    axis2_updates_encrypted = simp.runAt(0, phe.encrypt)(axis2_updates, pkey_bcasted)

    # Scatter along axis=2 - scatter updates at columns [0,2] for each position
    scattered_axis2 = simp.runAt(0, phe.scatter)(
        e0, indices_axis2, axis2_updates_encrypted, axis=2
    )

    # Decrypt results for verification
    dot_decrypted = simp.runAt(0, phe.decrypt)(dot_result, skey)
    gathered_decrypted = simp.runAt(0, phe.decrypt)(gathered, skey)
    concat_decrypted = simp.runAt(0, phe.decrypt)(concat_result, skey)

    # Decrypt some multi-axis results for demonstration
    gathered_axis0_decrypted = simp.runAt(0, phe.decrypt)(gathered_axis0, skey)
    gathered_axis2_decrypted = simp.runAt(0, phe.decrypt)(gathered_axis2, skey)
    scattered_axis1_decrypted = simp.runAt(0, phe.decrypt)(scattered_axis1, skey)

    return (
        dot_decrypted,
        gathered_decrypted,
        concat_decrypted,
        gathered_axis0_decrypted,
        gathered_axis2_decrypted,
        scattered_axis1_decrypted,
    )


def run_simulation():
    """Run the PHE simulation locally."""
    # Set up 3-party simulation with PHE support
    sim = mplang.Simulator(3)

    # Test original 3-party PHE sum
    result = mplang.evaluate(sim, three_party_phe_sum)
    print(f"Original PHE sum completed. Final sum: {mplang.fetch(sim, result)}")

    # Test 2D matrix operations
    print("\n=== Testing 2D Matrix Operations ===")
    result_2d = mplang.evaluate(sim, test_2d_matrix_operations)
    fetched_2d = mplang.fetch(sim, result_2d)
    # fetched_2d is a tuple of (dot_results, gather_results, concat_results)
    # Each result is [party0_result, None, None]
    dot_results, gather_results, concat_results = fetched_2d
    dot_result = dot_results[0]  # Get Party 0's result
    gathered_result = gather_results[0]
    concat_result = concat_results[0]

    print(f"2D DOT result shape: {dot_result.shape}, values: \n{dot_result}")
    print(f"2D GATHER result shape: {gathered_result.shape}, values: {gathered_result}")
    print(f"2D CONCAT result shape: {concat_result.shape}")

    # Test 3D tensor operations
    print("\n=== Testing 3D Tensor Operations ===")
    result_3d = mplang.evaluate(sim, test_3d_tensor_operations)
    fetched_3d = mplang.fetch(sim, result_3d)
    # New format with multi-axis results
    (
        dot_results_3d,
        gather_results_3d,
        concat_results_3d,
        gathered_axis0_results,
        gathered_axis2_results,
        scattered_axis1_results,
    ) = fetched_3d

    dot_3d = dot_results_3d[0]
    gathered_3d = gather_results_3d[0]
    concat_3d = concat_results_3d[0]
    gathered_axis0_3d = gathered_axis0_results[0]
    gathered_axis2_3d = gathered_axis2_results[0]
    scattered_axis1_3d = scattered_axis1_results[0]

    print(f"3D DOT result shape: {dot_3d.shape}, values: \n{dot_3d}")
    print(f"3D GATHER result shape: {gathered_3d.shape}, values: {gathered_3d}")
    print(f"3D CONCAT result shape: {concat_3d.shape}")

    # Display multi-axis results
    print(f"\n=== Multi-axis Operations Results ===")
    print(f"3D GATHER axis=0 result shape: {gathered_axis0_3d.shape}")
    print(
        f"3D GATHER axis=2 result shape: {gathered_axis2_3d.shape}, values: \n{gathered_axis2_3d}"
    )
    print(f"3D SCATTER axis=1 result shape: {scattered_axis1_3d.shape}")

    # Show compilation results
    compiled = mplang.compile(sim, three_party_phe_sum)
    print("\n=== Compilation IR ===")
    print("Original function compiled:", compiled.compiler_ir())

    compiled_2d = mplang.compile(sim, test_2d_matrix_operations)
    print("\n=== Compilation IR ===")
    print("2D matrix operations function compiled:", compiled_2d.compiler_ir())

    compiled_3d = mplang.compile(sim, test_3d_tensor_operations)
    print("\n=== Compilation IR ===")
    print("3D tensor operations function compiled:", compiled_3d.compiler_ir())


if __name__ == "__main__":
    run_simulation()
