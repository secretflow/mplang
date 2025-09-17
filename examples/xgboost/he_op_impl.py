from typing import List
import jax.numpy as jnp
import numpy as np
import mplang
import mplang.mpi as mpi
import mplang.simp as simp
from mplang.frontend import phe


def batch_feature_wise_bucket_sum_np(
    arr: np.ndarray,
    subgroup_map: np.ndarray,
    order_map: np.ndarray,
    bucket_num: int,
    group_size: int,
) -> np.ndarray:
    """
    Compute batch feature-wise bucket cumulative sums for XGBoost gradient histogram.

    This function calculates cumulative gradient and hessian sums for each subgroup,
    feature, and bucket combination in XGBoost tree splitting.

    Args:
        arr: Gradient and Hessian values, shape (sample_size, gh_size)
        subgroup_map: Subgroup masks, shape (group_size, sample_size)
        order_map: Feature bucket mapping, shape (sample_size, feature_size)
        bucket_num: Number of buckets per feature
        group_size: Number of subgroups

    Returns:
        Cumulative sums, shape (group_size, feature_size * bucket_num, gh_size)
    """
    sample_size, gh_size = arr.shape
    feature_size = order_map.shape[1]

    # Initialize result array
    result = np.zeros((group_size, feature_size * bucket_num, gh_size))

    # Process each subgroup
    for group_idx in range(group_size):
        # Get samples belonging to this subgroup
        group_mask = subgroup_map[group_idx].astype(bool)
        group_samples = arr[group_mask]  # shape: (valid_samples, gh_size)
        group_order_map = order_map[group_mask]  # shape: (valid_samples, feature_size)

        # Process each feature
        for feature_idx in range(feature_size):
            feature_buckets = group_order_map[
                :, feature_idx
            ]  # bucket indices for this feature

            # Calculate cumulative sum for each bucket
            for bucket_idx in range(bucket_num):
                # Find samples that belong to buckets <= current bucket
                bucket_mask = feature_buckets <= bucket_idx

                if np.any(bucket_mask):
                    # Sum all samples in buckets from 0 to bucket_idx
                    bucket_sum = np.sum(group_samples[bucket_mask], axis=0)
                    result[group_idx, feature_idx * bucket_num + bucket_idx] = (
                        bucket_sum
                    )
                else:
                    # If no samples in this bucket range, use previous bucket's sum
                    if bucket_idx > 0:
                        result[group_idx, feature_idx * bucket_num + bucket_idx] = (
                            result[group_idx, feature_idx * bucket_num + bucket_idx - 1]
                        )

    return result


def test_batch_feature_wise_bucket_sum_np():
    sample_size = 6
    feature_size = 2
    gh_size = 2
    bucket_num = 3
    group_size = 2

    # shape: (sample_size, gh_size)
    m1_np = np.array(
        [
            [1, 10],  # sample 0
            [2, 20],  # sample 1
            [3, 30],  # sample 2
            [4, 40],  # sample 3
            [5, 50],  # sample 4
            [6, 60],  # sample 5
        ]
    )
    # Subgroup 0: samples 0, 1, 2
    # Subgroup 1: samples 3, 4, 5
    subgroup0_mask = np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)
    subgroup1_mask = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    subgroup_map = np.array([subgroup0_mask, subgroup1_mask])

    # shape: (sample_size, feature_size)
    order_map = np.array(
        [
            # f0, f1
            [0, 1],  # sample 0
            [1, 2],  # sample 1
            [0, 0],  # sample 2
            [2, 1],  # sample 3
            [1, 0],  # sample 4
            [0, 2],  # sample 5
        ],
        dtype=np.int8,
    )

    cumsum_sums = batch_feature_wise_bucket_sum_np(
        m1_np, subgroup_map, order_map, bucket_num, group_size
    )

    # shape: (feature_size * bucket_num, gh_size)
    expected0_cumsum = np.array(
        [
            [4, 40],
            [6, 60],
            [6, 60],
            [3, 30],
            [4, 40],
            [6, 60],
        ]
    )

    # shape: (feature_size * bucket_num, gh_size)
    expected1_cumsum = np.array(
        [
            [6, 60],
            [11, 110],
            [15, 150],
            [5, 50],
            [9, 90],
            [15, 150],
        ]
    )

    np.testing.assert_array_equal(cumsum_sums[0, ...], expected0_cumsum)
    np.testing.assert_array_equal(cumsum_sums[1, ...], expected1_cumsum)


@mplang.function
def batch_feature_wise_bucket_sum_mplang(
    arr: mplang.MPObject,  # encrypted
    subgroup_map: mplang.MPObject,  # plaintext
    order_map: mplang.MPObject,  # plaintext
    bucket_num: int,
    group_size: int,
    rank: int,
) -> List[mplang.MPObject]:
    """
    Compute batch feature-wise bucket cumulative sums for XGBoost gradient histogram using PHE.

    This is the encrypted version of batch_feature_wise_bucket_sum_np using mplang PHE operations.

    Args:
        arr: Encrypted gradient and Hessian values, shape (sample_size, gh_size)
        subgroup_map: Plaintext subgroup masks, shape (group_size, sample_size)
        order_map: Plaintext feature bucket mapping, shape (sample_size, feature_size)
        bucket_num: Number of buckets per feature
        group_size: Number of subgroups
        rank: Execution rank for simp.runAt operations

    Returns:
        List of encrypted cumulative sums, each element shape (feature_size * bucket_num, gh_size)
    """

    # Get dimensions
    sample_size = arr.shape[0]
    gh_size = arr.shape[1]
    feature_size = order_map.shape[1]

    def compute_using_mask_multiplication():
        """
        Use mask multiplication and inner product to avoid dynamic shape operations.

        Key idea:
        1. For each group, multiply order_map with group mask to get order_map_new where
           mask=0 positions become -1 (invalid)
        2. For each bucket, create 0/1 vectors for bucket membership
        3. Use inner product to sum samples belonging to each bucket
        """

        def extract_group_mask(group_idx):
            def slice_group(sg_map):
                return sg_map[group_idx]  # Extract group_idx-th row

            return simp.runAt(rank, slice_group)(subgroup_map)

        # Create modified order_map for each group where mask=0 positions become -1
        def create_masked_order_map(mask, om):
            """Multiply order_map with mask, setting invalid positions to -1"""

            def apply_mask(m, order_m):
                # Expand mask to match order_map shape: (sample_size,) -> (sample_size, feature_size)
                mask_expanded = jnp.expand_dims(m, axis=1)  # (sample_size, 1)
                mask_full = jnp.broadcast_to(
                    mask_expanded, order_m.shape
                )  # (sample_size, feature_size)

                # Where mask=0, set order values to -1 (invalid bucket)
                # Where mask=1, keep original order values
                return jnp.where(mask_full == 1, order_m, -1)

            return simp.runAt(rank, apply_mask)(mask, om)

        # Extract group masks and create masked order maps for all groups
        group_masks = []
        group_order_maps = []
        for group_idx in range(group_size):
            group_mask = extract_group_mask(group_idx)  # shape: (sample_size,)
            group_masks.append(group_mask)

            group_order_map = create_masked_order_map(
                group_mask, order_map
            )  # (sample_size, feature_size)
            group_order_maps.append(group_order_map)

        # Now compute bucket sums for each group using inner products
        def compute_group_bucket_sums(group_order_map):
            """Compute bucket sums for one group using inner product approach"""

            # Initialize result list to store all bucket results
            bucket_results = []

            # Process each feature
            for feature_idx in range(feature_size):
                # Process each bucket for this feature
                for bucket_idx in range(bucket_num):

                    # Create bucket membership vector: 1 if sample belongs to bucket <= bucket_idx, 0 otherwise
                    def create_bucket_mask(gom, f_idx, b_idx):
                        """Create mask for samples in buckets <= b_idx for feature f_idx"""

                        feature_buckets = gom[
                            :, f_idx
                        ]  # bucket values for this feature

                        # Create cumulative bucket mask: 1 if bucket_value <= b_idx AND bucket_value >= 0
                        # (bucket_value >= 0 ensures we only include valid samples from this group)
                        valid_and_in_bucket = (feature_buckets >= 0) & (
                            feature_buckets <= b_idx
                        )
                        return valid_and_in_bucket.astype(jnp.float32)

                    bucket_mask = simp.runAt(rank, create_bucket_mask)(
                        group_order_map, feature_idx, bucket_idx
                    )

                    # Use inner product to sum encrypted values for this bucket
                    # bucket_mask: (sample_size,) -> bucket_mask_col: (sample_size, 1) for matrix multiplication
                    def reshape_bucket_mask_to_col(mask):

                        return mask.reshape(-1, 1)  # (sample_size, 1)

                    bucket_mask_col = simp.runAt(rank, reshape_bucket_mask_to_col)(
                        bucket_mask
                    )

                    # Compute weighted sum: arr.T @ bucket_mask_col -> (gh_size, 1)
                    # arr: (sample_size, gh_size) -> arr.T: (gh_size, sample_size)
                    # bucket_mask_col: (sample_size, 1)
                    # Result: (gh_size, 1)

                    arr_transposed = simp.runAt(rank, phe.transpose)(
                        arr
                    )  # (gh_size, sample_size)

                    # Now we can do: encrypted_matrix @ plaintext_matrix
                    bucket_sum_col = simp.runAt(rank, phe.dot)(
                        arr_transposed, bucket_mask_col
                    )  # (gh_size, 1)

                    # Use PHE transpose to convert (gh_size, 1) to (1, gh_size)
                    bucket_sum = simp.runAt(rank, phe.transpose)(
                        bucket_sum_col
                    )  # (1, gh_size)

                    # Reshape to (1, gh_size) and add to results
                    bucket_results.append(bucket_sum)

            # Concatenate all bucket results: list of (1, gh_size) -> (feature_size * bucket_num, gh_size)
            # Since we need to concatenate multiple tensors, do it step by step
            first_result = bucket_results[0]
            for i in range(1, len(bucket_results)):
                first_result = simp.runAt(rank, phe.concat)(
                    [first_result, bucket_results[i]], axis=0
                )

            return first_result

        # Compute bucket sums for all groups and return as list
        group_bucket_results = []
        for group_idx in range(group_size):
            group_buckets = compute_group_bucket_sums(
                group_order_maps[group_idx]
            )  # (feature_size * bucket_num, gh_size)
            group_bucket_results.append(group_buckets)

        return group_bucket_results

    return compute_using_mask_multiplication()


def run_buick_sum():
    sample_size = 6
    feature_size = 2
    gh_size = 2
    bucket_num = 3
    group_size = 2

    # shape: (sample_size, gh_size)
    m1_np = np.array(
        [
            [1, 10],  # sample 0
            [2, 20],  # sample 1
            [3, 30],  # sample 2
            [4, 40],  # sample 3
            [5, 50],  # sample 4
            [6, 60],  # sample 5
        ]
    )
    # Subgroup 0: samples 0, 1, 2
    # Subgroup 1: samples 3, 4, 5
    subgroup0_mask = np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)
    subgroup1_mask = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    subgroup_map = np.array([subgroup0_mask, subgroup1_mask])

    # shape: (sample_size, feature_size)
    order_map = np.array(
        [
            # f0, f1
            [0, 1],  # sample 0
            [1, 2],  # sample 1
            [0, 0],  # sample 2
            [2, 1],  # sample 3
            [1, 0],  # sample 4
            [0, 2],  # sample 5
        ],
        dtype=np.int8,
    )

    pkey, skey = simp.runAt(0, phe.keygen)()
    world_mask = mplang.Mask.all(2)
    pkey_bcasted = mpi.bcast_m(world_mask, 0, pkey)

    m1 = simp.runAt(0, lambda x: x)(m1_np)

    encrypted_arr = simp.runAt(0, phe.encrypt)(m1, pkey_bcasted)
    encrypted_arr = mpi.p2p(0, 1, encrypted_arr)

    subgroup_map = simp.runAt(1, lambda x: x)(subgroup_map)
    order_map = simp.runAt(1, lambda x: x)(order_map)

    bucket_sum_list = batch_feature_wise_bucket_sum_mplang(
        encrypted_arr, subgroup_map, order_map, bucket_num, group_size, rank=1
    )

    # Decrypt each group result separately
    decrypted_results = []
    for group_idx in range(group_size):
        decrypted_group = simp.runAt(0, phe.decrypt)(
            mpi.p2p(1, 0, bucket_sum_list[group_idx]), skey
        )
        decrypted_results.append(decrypted_group)

    return decrypted_results


def run_buick_sum_3_groups():
    """Test with 3 groups to verify dynamic group support"""
    sample_size = 9
    feature_size = 2
    gh_size = 2
    bucket_num = 3
    group_size = 3

    # shape: (sample_size, gh_size)
    m1_np = np.array(
        [
            [1, 10],  # sample 0 - group 0
            [2, 20],  # sample 1 - group 1
            [3, 30],  # sample 2 - group 0
            [4, 40],  # sample 3 - group 1
            [5, 50],  # sample 4 - group 2
            [6, 60],  # sample 5 - group 0
            [7, 70],  # sample 6 - group 1
            [8, 80],  # sample 7 - group 2
            [9, 90],  # sample 8 - group 2
        ]
    )
    # Subgroup 0: samples 0, 2, 5
    # Subgroup 1: samples 1, 3, 6
    # Subgroup 2: samples 4, 7, 8
    subgroup0_mask = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    subgroup1_mask = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
    subgroup2_mask = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1], dtype=np.int8)
    subgroup_map = np.array([subgroup0_mask, subgroup1_mask, subgroup2_mask])

    # shape: (sample_size, feature_size)
    order_map = np.array(
        [
            # f0, f1
            [0, 1],  # sample 0 - group 0
            [1, 2],  # sample 1 - group 1
            [0, 0],  # sample 2 - group 0
            [2, 1],  # sample 3 - group 1
            [1, 0],  # sample 4 - group 2
            [0, 2],  # sample 5 - group 0
            [1, 0],  # sample 6 - group 1
            [2, 1],  # sample 7 - group 2
            [0, 2],  # sample 8 - group 2
        ],
        dtype=np.int8,
    )

    pkey, skey = simp.runAt(0, phe.keygen)()
    world_mask = mplang.Mask.all(2)
    pkey_bcasted = mpi.bcast_m(world_mask, 0, pkey)

    m1 = simp.runAt(0, lambda x: x)(m1_np)

    encrypted_arr = simp.runAt(0, phe.encrypt)(m1, pkey_bcasted)
    encrypted_arr = mpi.p2p(0, 1, encrypted_arr)

    subgroup_map = simp.runAt(1, lambda x: x)(subgroup_map)
    order_map = simp.runAt(1, lambda x: x)(order_map)

    bucket_sum_list = batch_feature_wise_bucket_sum_mplang(
        encrypted_arr, subgroup_map, order_map, bucket_num, group_size, rank=1
    )

    # Decrypt each group result separately
    decrypted_results = []
    for group_idx in range(group_size):
        decrypted_group = simp.runAt(0, phe.decrypt)(
            mpi.p2p(1, 0, bucket_sum_list[group_idx]), skey
        )
        decrypted_results.append(decrypted_group)

    return decrypted_results


def main(print_output=True, print_ir=False, run_2_groups=True, run_3_groups=True):
    """
    Main test function for batch feature-wise bucket sum implementation.

    Args:
        print_output: Whether to print detailed output results
        print_ir: Whether to print compilation IR
        run_2_groups: Whether to run 2-group test
        run_3_groups: Whether to run 3-group test
    """
    sim = mplang.Simulator(2)

    # Test with 2 groups
    if run_2_groups:
        print("=== Testing with 2 groups ===")
        result_2_groups = mplang.evaluate(sim, run_buick_sum)
        fetched_2_groups = mplang.fetch(sim, result_2_groups)
        # fetched_2_groups is [[group0_from_rank0, None], [group1_from_rank0, None]]
        # We need to extract the first element from each inner list

        if print_output:
            print(
                f"2-group PHE sum completed. Number of groups: {len(fetched_2_groups)}"
            )
            for i, group_item in enumerate(fetched_2_groups):
                group_result = (
                    group_item[0] if group_item[0] is not None else group_item[1]
                )
                if group_result is not None:
                    print(f"Group {i} shape: {group_result.shape}")
                else:
                    raise ValueError(f"Group {i} result is None")
            print()

        # Extract group results
        # fetched returns [[group0_result, None], [group1_result, None]]
        out_2_0 = fetched_2_groups[0][0]  # First group's result
        out_2_1 = fetched_2_groups[1][0]  # Second group's result

        if print_output:
            print(f"group 0 sum: {out_2_0}")
            print(f"group 1 sum: {out_2_1}")
            print()

        # Verify 2-group test correctness
        expected_2_0 = np.array(
            [
                [4, 40],  # bucket 0 for feature 0: samples 0,2 (buckets <=0)
                [6, 60],  # bucket 1 for feature 0: samples 0,1,2 (buckets <=1)
                [6, 60],  # bucket 2 for feature 0: samples 0,1,2 (buckets <=2)
                [3, 30],  # bucket 0 for feature 1: sample 2 (bucket <=0)
                [4, 40],  # bucket 1 for feature 1: samples 0,2 (buckets <=1)
                [6, 60],  # bucket 2 for feature 1: samples 0,1,2 (buckets <=2)
            ]
        )

        expected_2_1 = np.array(
            [
                [6, 60],  # bucket 0 for feature 0: sample 5 (bucket <=0)
                [11, 110],  # bucket 1 for feature 0: samples 4,5 (buckets <=1)
                [15, 150],  # bucket 2 for feature 0: samples 3,4,5 (buckets <=2)
                [5, 50],  # bucket 0 for feature 1: sample 4 (bucket <=0)
                [9, 90],  # bucket 1 for feature 1: samples 3,4 (buckets <=1)
                [15, 150],  # bucket 2 for feature 1: samples 3,4,5 (buckets <=2)
            ]
        )

        np.testing.assert_array_equal(out_2_0, expected_2_0)
        np.testing.assert_array_equal(out_2_1, expected_2_1)
        print("✓ 2-group test passed!")
        print()

    # Test with 3 groups
    if run_3_groups:
        print("=== Testing with 3 groups ===")
        result_3_groups = mplang.evaluate(sim, run_buick_sum_3_groups)
        fetched_3_groups = mplang.fetch(sim, result_3_groups)
        # fetched_3_groups is [[group0_from_rank0, None], [group1_from_rank0, None], [group2_from_rank0, None]]

        if print_output:
            print(
                f"3-group PHE sum completed. Number of groups: {len(fetched_3_groups)}"
            )
            for i, group_item in enumerate(fetched_3_groups):
                group_result = (
                    group_item[0] if group_item[0] is not None else group_item[1]
                )
                if group_result is not None:
                    print(f"Group {i} shape: {group_result.shape}")
                else:
                    raise ValueError(f"Group {i} result is None")
            print()

        # Extract group results
        # fetched returns [[group0_result, None], [group1_result, None], [group2_result, None]]
        out_3_0 = fetched_3_groups[0][0]  # First group's result
        out_3_1 = fetched_3_groups[1][0]  # Second group's result
        out_3_2 = fetched_3_groups[2][0]  # Third group's result

        if print_output:
            print(f"group 0 sum: {out_3_0}")
            print(f"group 1 sum: {out_3_1}")
            print(f"group 2 sum: {out_3_2}")
            print()

        # Verify 3-group test correctness
        # Group 0: samples 0,2,5 with values [1,10], [3,30], [6,60]
        # order_map: [0,1], [0,0], [0,2]
        expected_3_0 = np.array(
            [
                [10, 100],  # bucket 0 for feature 0: samples 0,2,5 (buckets <=0)
                [10, 100],  # bucket 1 for feature 0: samples 0,2,5 (buckets <=1)
                [10, 100],  # bucket 2 for feature 0: samples 0,2,5 (buckets <=2)
                [3, 30],  # bucket 0 for feature 1: sample 2 (bucket <=0)
                [4, 40],  # bucket 1 for feature 1: samples 0,2 (buckets <=1)
                [10, 100],  # bucket 2 for feature 1: samples 0,2,5 (buckets <=2)
            ]
        )

        # Group 1: samples 1,3,6 with values [2,20], [4,40], [7,70]
        # order_map: [1,2], [2,1], [1,0]
        expected_3_1 = np.array(
            [
                [0, 0],  # bucket 0 for feature 0: no samples (no buckets <=0)
                [9, 90],  # bucket 1 for feature 0: samples 1,6 (buckets <=1)
                [13, 130],  # bucket 2 for feature 0: samples 1,3,6 (buckets <=2)
                [7, 70],  # bucket 0 for feature 1: sample 6 (bucket <=0)
                [11, 110],  # bucket 1 for feature 1: samples 3,6 (buckets <=1)
                [13, 130],  # bucket 2 for feature 1: samples 1,3,6 (buckets <=2)
            ]
        )

        # Group 2: samples 4,7,8 with values [5,50], [8,80], [9,90]
        # order_map: [1,0], [2,1], [0,2]
        expected_3_2 = np.array(
            [
                [9, 90],  # bucket 0 for feature 0: sample 8 (bucket <=0)
                [14, 140],  # bucket 1 for feature 0: samples 4,8 (buckets <=1)
                [22, 220],  # bucket 2 for feature 0: samples 4,7,8 (buckets <=2)
                [5, 50],  # bucket 0 for feature 1: sample 4 (bucket <=0)
                [13, 130],  # bucket 1 for feature 1: samples 4,7 (buckets <=1)
                [22, 220],  # bucket 2 for feature 1: samples 4,7,8 (buckets <=2)
            ]
        )

        np.testing.assert_array_equal(out_3_0, expected_3_0)
        np.testing.assert_array_equal(out_3_1, expected_3_1)
        np.testing.assert_array_equal(out_3_2, expected_3_2)
        print("✓ 3-group test passed!")
        print()

    # Print compilation IR if requested
    if print_ir:
        if run_2_groups:
            compiled_2_groups = mplang.compile(sim, run_buick_sum)
            print("=== 2-Group Compilation IR ===")
            print("2-group function compiled:", compiled_2_groups.compiler_ir())
            print()

        if run_3_groups:
            compiled_3_groups = mplang.compile(sim, run_buick_sum_3_groups)
            print("=== 3-Group Compilation IR ===")
            print("3-group function compiled:", compiled_3_groups.compiler_ir())
            print()

    print("All tests completed successfully! ✨")


if __name__ == "__main__":
    print_output = False
    print_ir = True
    run_2_groups = False
    run_3_groups = True

    main(print_output, print_ir, run_2_groups, run_3_groups)
