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

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.ops import segment_sum

import mplang as mp
from mplang.ops import phe

# ==============================================================================
# Naming Conventions for Shapes & Variables
# ==============================================================================
# m: number of samples
# n: number of features
# k: number of bins for discretization
# t: number of nodes at the current tree level
#
# g: gradient of the loss function w.r.t. predictions, shape (m,)
# h: hessian (second derivative) of the loss function, shape (m,)
# bt: "branching of tree", an array where bt[i] stores the node index for sample i, shape (m,)
#
# ap: active party
# pp: passive party
#
# In the codes, we will collect some data in a list:
#   all_xxx (List[mp.MPObject]) :  index 0 for ap, others for pp

# ==============================================================================
# Part 0: Some Helper Functions
# ==============================================================================


def p2p_list(frm: mp.Rank, to: mp.Rank, objs: list[mp.MPObject]) -> list[mp.MPObject]:
    return [mp.p2p(frm, to, obj) for obj in objs]


def batch_feature_wise_bucket_sum_mplang(
    arr: mp.MPObject,  # encrypted
    subgroup_map: mp.MPObject,  # plaintext
    order_map: mp.MPObject,  # plaintext
    bucket_num: int,
    group_size: int,
    rank: int,
) -> list[mp.MPObject]:
    """
    Compute batch feature-wise bucket cumulative sums for XGBoost gradient histogram using PHE.

    This is the encrypted version of batch_feature_wise_bucket_sum_np using mplang PHE operations.

    Args:
        arr: Encrypted gradient and Hessian values, shape (sample_size, gh_size)
        subgroup_map: Plaintext subgroup masks, shape (group_size, sample_size)
        order_map: Plaintext feature bucket mapping, shape (sample_size, feature_size)
        bucket_num: Number of buckets per feature
        group_size: Number of subgroups
        rank: Execution rank for mp.rat operations

    Returns:
        List of encrypted cumulative sums, each element shape (feature_size * bucket_num, gh_size)
    """

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
                return sg_map[group_idx]

            return mp.run_jax_at(rank, slice_group, subgroup_map)

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

            return mp.run_jax_at(rank, apply_mask, mask, om)

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
                        return valid_and_in_bucket.astype(jnp.int32)

                    bucket_mask = mp.run_jax_at(
                        rank,
                        create_bucket_mask,
                        group_order_map,
                        feature_idx,
                        bucket_idx,
                    )

                    # Use inner product to sum encrypted values for this bucket
                    # bucket_mask: (sample_size,) -> bucket_mask_col: (sample_size, 1) for matrix multiplication
                    def reshape_bucket_mask_to_col(mask):
                        return mask.reshape(-1, 1)  # (sample_size, 1)

                    bucket_mask_col = mp.run_jax_at(
                        rank, reshape_bucket_mask_to_col, bucket_mask
                    )

                    # Compute weighted sum: arr.T @ bucket_mask_col -> (gh_size, 1)
                    # arr: (sample_size, gh_size) -> arr.T: (gh_size, sample_size)
                    # bucket_mask_col: (sample_size, 1)
                    # Result: (gh_size, 1)

                    arr_transposed = mp.run_at(
                        rank, phe.transpose, arr
                    )  # (gh_size, sample_size)

                    # Now we can do: encrypted_matrix @ plaintext_matrix
                    bucket_sum_col = mp.run_at(
                        rank, phe.dot, arr_transposed, bucket_mask_col
                    )  # (gh_size, 1)

                    # Use PHE transpose to convert (gh_size, 1) to (1, gh_size)
                    bucket_sum = mp.run_at(
                        rank, phe.transpose, bucket_sum_col
                    )  # (1, gh_size)

                    # Reshape to (1, gh_size) and add to results
                    bucket_results.append(bucket_sum)

            # Concatenate all bucket results: list of (1, gh_size) -> (feature_size * bucket_num, gh_size)
            # Since we need to concatenate multiple tensors, do it step by step
            first_result = bucket_results[0]
            for i in range(1, len(bucket_results)):
                first_result = mp.run_at(
                    rank, phe.concat, first_result, bucket_results[i], axis=0
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


# ==============================================================================
# Part 1: Core Data Structures
# ==============================================================================


class Tree(NamedTuple):
    """
    Represents a single, pre-allocated decision tree stored in flat arrays.
    The tree structure is implicit: node `i` has children `2*i+1` (left) and `2*i+2` (right).

    Attributes:
        feature (jnp.ndarray): Shape (n_nodes,). Feature index used for splitting at each node.
        threshold (jnp.ndarray): Shape (n_nodes,). Threshold value for the split at each node.
        value (jnp.ndarray): Shape (n_nodes,). Prediction value if the node is a leaf.
        is_leaf (jnp.ndarray): Shape (n_nodes,). Boolean mask, True if the node is a leaf.
        owned_party_id (jnp.ndarray): Shape (n_nodes,). Party ID of the node owner.
    """

    feature: list[mp.MPObject]  # each party has its own feature
    threshold: list[mp.MPObject]  # each party has its own threshold

    value: mp.MPObject  # owned by ap only

    is_leaf: mp.MPObject  # TODO: all parties have the same is_leaf now

    # SGB specific
    owned_party_id: mp.MPObject  # TODO: all parties have the same owned_party_id now


class TreeEnsemble(NamedTuple):
    """
    Represents an entire XGBoost model.

    Attributes:
        max_depth (int): Maximum depth of the trees in the ensemble.
        trees (List[Tree]): A list of Tree NamedTuples.
        initial_prediction (jnp.ndarray): Shape (m,). The base prediction (logit) for all samples.
        bins (jnp.ndarray): Shape (n, n_bins-1). The bin boundaries for each feature.
    """

    max_depth: int
    trees: list[Tree]
    initial_prediction: mp.MPObject  # owned by ap

    # bins: jnp.ndarray


# ==============================================================================
# Part 2: Core Local Mathematical & Binning Functions
# ==============================================================================


@jax.jit
def compute_init_pred(y: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the initial prediction for the given target.
    """
    p_base = jnp.mean(y)
    p_base = jnp.clip(p_base, 1e-15, 1 - 1e-15)
    return jnp.log(p_base / (1 - p_base))


@jax.jit
def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + jnp.exp(-x))


@jax.jit
def gradient(y_true: jnp.ndarray, y_pred_logits: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the gradient of the log loss for binary classification.
    Shape: (m,) -> (m,).
    """
    p = sigmoid(y_pred_logits)
    return p - y_true


@jax.jit
def hessian(y_true: jnp.ndarray, y_pred_logits: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Hessian of the log loss for binary classification.
    Shape: (m,) -> (m,).
    """
    p = sigmoid(y_pred_logits)
    return p * (1 - p)


@jax.jit
def compute_gh(y_true: jnp.ndarray, y_pred_logits: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the gradient and Hessian of the log loss for binary classification.
    Shape: (m,) -> (m, 2).
    """
    g = gradient(y_true, y_pred_logits)
    h = hessian(y_true, y_pred_logits)
    # concat g and h along the column
    return jnp.c_[g, h]


def build_bins_equi_width(x: jnp.ndarray, max_bin: int) -> jnp.ndarray:
    """
    Computes equi-width bin boundaries for a single feature vector.

    Args:
        x (jnp.ndarray): A single feature vector. Shape (m,).
        max_bin (int): The maximum number of bins to create.

    Returns:
        jnp.ndarray: The split points (boundaries). Shape (max_bin-1,).
    """
    n_samples = x.shape[0]
    n_splits = max_bin - 1
    inf_splits = jnp.full(shape=(n_splits,), fill_value=jnp.inf, dtype=x.dtype)

    def create_valid_bins():
        min_val, max_val = jnp.min(x), jnp.max(x)
        is_constant = (max_val - min_val) < 1e-9

        def generate_splits():
            # Create max_bin + 1 points, then take the inner max_bin - 1 as splits.
            boundaries = jnp.linspace(min_val, max_val, num=max_bin + 1)
            return boundaries[1:-1]

        # If feature is constant, return inf splits to avoid errors.
        return jax.lax.cond(is_constant, lambda: inf_splits, generate_splits)

    # If not enough samples, no need to bin.
    return jax.lax.cond(n_samples >= 2, create_valid_bins, lambda: inf_splits)


@jax.jit
def compute_bin_indices(x: jnp.ndarray, bins: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the bin index for each value in a feature vector.

    Args:
        x (jnp.ndarray): A single feature vector. Shape (m,).
        bins (jnp.ndarray): The bin boundaries for this feature. Shape (n_bins-1,).

    Returns:
        jnp.ndarray: The bin index for each sample. Shape (m,).
    """
    # jnp.digitize finds the index of the bin to which each value belongs.
    # right=True means bins are [left, right], so value <= boundary goes to left bin.
    return jnp.digitize(x, bins, right=True)


def local_build_histogram(
    gh: jnp.ndarray,
    bt_local: jnp.ndarray,
    bin_indices: jnp.ndarray,
    t: int,
    k: int,
) -> jnp.ndarray:
    """
    Builds histograms of gradients (G) and Hessians (H) for all features in a fully
    vectorized manner using jax.vmap, eliminating the Python for-loop.

    Args:
        gh (jnp.ndarray): Gradients and Hessians for each sample. Shape (m, 2).
        bt_local (jnp.ndarray): Local node index [0, t-1] for each sample. Shape (m,).
        bin_indices (jnp.ndarray): Bin index for each sample and feature. Shape (m, n).
        t (int): Number of tree nodes at the current level.
        k (int): Number of bins.

    Returns:
        jnp.ndarray: Histogram of gradients and Hessians. Shape (n, t, k, 2).
    """

    # 1. Define a function that computes the histogram for a SINGLE feature.
    # This function will be vectorized over all features using vmap.
    def histogram_for_one_feature(
        bin_indices_one_feature: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            bin_indices_one_feature: Shape (m,) - bin indices for one feature.
        Returns:
            A flattened histogram for one feature. Shape (t * k, 2).
        """
        # `bt_local` and `k` are available from the outer scope.
        combined_indices = bt_local * k + bin_indices_one_feature

        # Handle negative indices safely: mask out samples with bt_local < 0
        # These are samples that are already in leaf nodes from previous levels
        valid_mask = bt_local >= 0

        # Only process samples that are in the current level's nodes
        valid_gh = gh * valid_mask[:, None]  # Broadcast mask to match gh shape

        # `segment_sum` operates on the (m, 2) `gh` array. It sums rows,
        # preserving the 2 columns for g and h separately.
        return segment_sum(valid_gh, combined_indices, num_segments=t * k)

    # 2. Vectorize the function over the feature dimension of `bin_indices`.
    # in_axes=1 means we map over the columns (features) of `bin_indices`.
    # out_axes=0 means we stack the results along a new leading axis.
    # Input `bin_indices` shape: (m, n)
    # Output `flat_histograms` shape: (n, t * k, 2)
    flat_histograms = jax.vmap(histogram_for_one_feature, in_axes=1, out_axes=0)(
        bin_indices
    )

    # 3. Reshape the result to the desired (n, t, k, 2) format.
    # Final shape: (n, t, k, 2)
    GH_hist = flat_histograms.reshape((bin_indices.shape[1], t, k, 2))

    return GH_hist


def _compute_best_split_per_node(
    GH_node: jnp.ndarray,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the best split for a single node using a combined GH histogram.
    This version performs vectorized calculations on the stacked G and H data.

    Args:
        GH_node (jnp.ndarray): Combined Gradient/Hessian histogram. Shape (n, k, 2).
        reg_lambda (float): L2 regularization term.
        gamma (float): Minimum loss reduction required to make a split.
        min_child_weight (float): Minimum sum of Hessians required in a child.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - best_gain (jnp.ndarray): The maximal gain. Shape ().
            - best_feat (jnp.ndarray): Feature index for the best split. Shape ().
            - best_threshold_idx (jnp.ndarray): Bin index for the best split. Shape ().
    """
    # 1. Perform sum and cumsum ONCE on the combined GH tensor.
    # The operations are applied along axis=1 (bins), affecting G and H simultaneously.
    gh_total_node = jnp.sum(GH_node, axis=1)  # Shape: (n, 2)
    gh_left = jnp.cumsum(GH_node, axis=1)[:, :-1, :]  # Shape: (n, k-1, 2)

    # 2. Unpack the results just before the non-linear gain calculation.
    g_total_node = gh_total_node[..., 0]  # Shape: (n,)
    h_total_node = gh_total_node[..., 1]  # Shape: (n,)

    G_left = gh_left[..., 0]  # Shape: (n, k-1)
    H_left = gh_left[..., 1]  # Shape: (n, k-1)

    # --- The rest of the logic remains the same, but uses the unpacked values ---
    score_parent = jnp.square(g_total_node) / (h_total_node + reg_lambda + 1e-9)

    G_right = g_total_node[:, None] - G_left
    H_right = h_total_node[:, None] - H_left

    valid_split = (H_left >= min_child_weight) & (H_right >= min_child_weight)

    gain = (
        jnp.square(G_left) / (H_left + reg_lambda + 1e-9)
        + jnp.square(G_right) / (H_right + reg_lambda + 1e-9)
        - score_parent[:, None]
    ) / 2.0
    gain = jnp.where(valid_split, gain - gamma, -jnp.inf)

    flat_idx = jnp.argmax(gain)
    best_feat, best_threshold_idx = jnp.unravel_index(flat_idx, gain.shape)
    best_gain = jnp.max(gain)

    return best_gain, best_feat, best_threshold_idx


def local_compute_best_split(
    GH_hist: jnp.ndarray,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the best split for all nodes at the current tree level by vmapping
    over the combined GH histogram.

    Args:
        GH_hist (jnp.ndarray): Combined G/H histograms for all nodes. Shape (n, t, k, 2).
        reg_lambda (float): L2 regularization term.
        gamma (float): Minimum loss reduction to make a split.
        min_child_weight (float): Minimum sum of Hessians in a child.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - max_gains (jnp.ndarray): Best gain for each node. Shape (t,).
            - best_features (jnp.ndarray): Best feature index for each node. Shape (t,).
            - best_threshold_idxs (jnp.ndarray): Best threshold bin index for each node. Shape (t,).
    """
    # Transpose from (n, t, k, 2) to (t, n, k, 2) to vmap over the 't' dimension.
    GH_hist_trans = jnp.transpose(GH_hist, (1, 0, 2, 3))

    # Vectorize the new, optimized per-node computation.
    # Note that in_axes is now just (0, None, None, None) for a single tensor input.
    max_gains, best_features, best_threshold_idxs = jax.vmap(
        _compute_best_split_per_node, in_axes=(0, None, None, None)
    )(GH_hist_trans, reg_lambda, gamma, min_child_weight)

    return max_gains, best_features, best_threshold_idxs


# ==============================================================================
# Part 3: Modified Tree Building Function with Pruning Logic
# ==============================================================================


def _update_pred(y_pred_current, update, learning_rate: float):
    return y_pred_current + learning_rate * update


def _get_subgroup_map(bt_level: jnp.ndarray, group_size: int) -> jnp.ndarray:
    """
    Creates a one-hot-like mapping from a group index array.

    Args:
        bt_level (jnp.ndarray): An array of shape (m,) where each element is an
                                integer index in the range [0, group_size - 1].
        group_size (int): The number of groups (and the number of rows in the output).

    Returns:
        jnp.ndarray: A binary matrix of shape (group_size, m). The element at
                     (k, i) is 1 if bt_level[i] == k, and 0 otherwise.
    """
    # 1. Create a column vector of group indices [0, 1, ..., group_size-1].
    # Shape: (group_size, 1)
    group_indices = jnp.arange(group_size)[:, None]

    # 2. Compare the group indices vector with the input bt_level array.
    #    Due to broadcasting rules:
    #    - `group_indices` (group_size, 1) is broadcast to (group_size, m).
    #    - `bt_level` (m,) is broadcast to (group_size, m).
    #    The comparison happens element-wise.
    #    The result is a boolean matrix where result[k, i] is True if k == bt_level[i].
    # Shape: (group_size, m)
    mapping_bool = group_indices == bt_level

    # 3. Convert the boolean matrix to an integer matrix (0s and 1s).
    return mapping_bool.astype(jnp.int8)


@jax.jit
def _update_owned_party_ids(
    owned_party_ids: jnp.ndarray, cur_indices: jnp.ndarray, best_party_id: jnp.ndarray
) -> jnp.ndarray:
    return owned_party_ids.at[cur_indices].set(best_party_id)


@jax.jit
def _update_is_leaf(
    is_leaf: jnp.ndarray, max_gains: jnp.ndarray, cur_indices: jnp.ndarray
) -> jnp.ndarray:
    # A node becomes a leaf if the gain is <= 0 or if it's NaN/inf
    is_leaf_level = (max_gains <= 0.0) | (~jnp.isfinite(max_gains))
    return is_leaf.at[cur_indices].set(is_leaf_level)


def _update_best_features(
    feats: jnp.ndarray,  # (t, )
    best_feature: jnp.ndarray,  # (level_t, )
    cur_indices: jnp.ndarray,  # (level_t, )
    owned_party_ids: jnp.ndarray,  # (t, )
    is_leaf: jnp.ndarray,  # (t, )
    cur_rank: int,
) -> jnp.ndarray:
    tmp_feats = feats.at[cur_indices].set(best_feature)
    feats = jnp.where(is_leaf, -1, tmp_feats)
    return jnp.where(owned_party_ids == cur_rank, feats, -1)  # (t, )


def _update_best_thresholds(
    thresholds: jnp.ndarray,  # (t, )
    bins: jnp.ndarray,  # (n, k)
    best_feature: jnp.ndarray,  # (level_t, )
    best_threshold_idx: jnp.ndarray,  # (level_t, )
    cur_indices: jnp.ndarray,  # (level_t, )
    owned_party_ids: jnp.ndarray,  # (t, )
    is_leaf: jnp.ndarray,  # (t, )
    cur_rank: int,
) -> jnp.ndarray:
    best_threshold = bins[best_feature, best_threshold_idx]  # (level_t, )
    tmp_thresholds = thresholds.at[cur_indices].set(best_threshold)
    thresholds = jnp.where(is_leaf, jnp.inf, tmp_thresholds)
    return jnp.where(owned_party_ids == cur_rank, thresholds, jnp.inf)  # (t, )


@jax.jit
def _update_bt(
    bt: jnp.ndarray,  # (m, )
    cur_level_bt: jnp.ndarray,  # (m, )
    is_leaf: jnp.ndarray,  # (t, )
    bin_indices: jnp.ndarray,  # (m, n)
    best_feature: jnp.ndarray,  # (level_t, )
    best_threshold_idx: jnp.ndarray,  # (level_t, )
) -> jnp.ndarray:
    n_samples = bt.shape[0]

    # For each sample, find the best feature and threshold of the node it belongs to.
    best_feature_for_each_sample = best_feature[cur_level_bt]  # (m, )
    best_threshold_idx_for_each_sample = best_threshold_idx[cur_level_bt]  # (m, )

    # Get the actual feature value for each sample's assigned splitting feature.
    sample_bin_indices = bin_indices[
        jnp.arange(n_samples), best_feature_for_each_sample
    ]  # (m, )

    # Determine if each sample goes to the left or right child.
    left_mask = sample_bin_indices <= best_threshold_idx_for_each_sample
    # Calculate the global index of the next node for each sample.
    bt_next = jnp.where(left_mask, 2 * bt + 1, 2 * bt + 2)

    # Determine which samples are now in a leaf node (after this level's splits).
    is_sample_in_leaf_node = is_leaf[bt]

    # Update `bt` only for active samples; others stay in their leaf node.
    return jnp.where(is_sample_in_leaf_node, bt, bt_next)


def _find_best_split_for_one_node_from_flat_cumulative(
    cumulative_gh_flat: jnp.ndarray,
    n_features: int,
    max_bin: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the best split for a single node from its flat cumulative GH histogram.

    Args:
        cumulative_gh_flat (jnp.ndarray): A flat array of cumulative G/H sums.
                                           Shape: (n_features * max_bin, 2).
        n_features (int): The number of features.
        max_bin (int): The number of bins.
        reg_lambda (float): L2 regularization term.
        gamma (float): Minimum loss reduction required to make a split.
        min_child_weight (float): Minimum sum of Hessians required in a child.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - best_gain (jnp.ndarray): The maximal gain. Shape ().
            - best_feat (jnp.ndarray): Feature index for the best split. Shape ().
            - best_threshold_idx (jnp.ndarray): Bin index for the best split. Shape ().
    """
    # 1. Reshape the flat input into a more usable format.
    # Shape: (n_features * max_bin, 2) -> (n_features, max_bin, 2)
    cumulative_gh = cumulative_gh_flat.reshape(n_features, max_bin, 2)

    # 2. Unpack G and H cumulative sums.
    G_cumulative = cumulative_gh[..., 0]  # Shape: (n_features, max_bin)
    H_cumulative = cumulative_gh[..., 1]  # Shape: (n_features, max_bin)

    # 3. Get total G and H for each feature (value in the last bin of cumulative sum).
    G_total = G_cumulative[:, -1]  # Shape: (n_features,)
    H_total = H_cumulative[:, -1]  # Shape: (n_features,)

    # 4. Define candidate splits. A split can happen after each bin except the last one.
    # The cumulative sums directly give us the G/H for the left child.
    G_left = G_cumulative[:, :-1]  # Shape: (n_features, max_bin - 1)
    H_left = H_cumulative[:, :-1]  # Shape: (n_features, max_bin - 1)

    # 5. Calculate G/H for the corresponding right child by subtraction.
    # Use [:, None] to correctly broadcast totals for element-wise subtraction.
    G_right = G_total[:, None] - G_left
    H_right = H_total[:, None] - H_left

    # 6. Calculate gain for all possible splits.
    # Suppress division by zero warnings as we handle it with a small epsilon.
    score_parent = jnp.square(G_total) / (H_total + reg_lambda + 1e-9)
    score_left = jnp.square(G_left) / (H_left + reg_lambda + 1e-9)
    score_right = jnp.square(G_right) / (H_right + reg_lambda + 1e-9)

    gain = (score_left + score_right - score_parent[:, None]) / 2.0

    # 7. Apply constraints: min_child_weight and gamma.
    valid_split = (H_left >= min_child_weight) & (H_right >= min_child_weight)
    # Set gain of invalid splits to -inf so they are never chosen.
    gain = jnp.where(valid_split, gain - gamma, -jnp.inf)

    # 8. Find the best split across all features and bins.
    flat_idx = jnp.argmax(gain)
    best_feat, best_threshold_idx = jnp.unravel_index(flat_idx, gain.shape)
    best_gain = jnp.max(gain)

    return best_gain, best_feat, best_threshold_idx


def pp_compute_all_best_splits(
    list_of_histograms: list[jnp.ndarray],
    n_features: int,
    max_bin: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Efficiently computes the best split for all nodes in a tree level.

    Args:
        list_of_histograms (List[jnp.ndarray]): A list of flat cumulative histograms.
                                                List length is `t`. Each element has
                                                shape (n_features * max_bin, 2).
        n_features (int): The number of features.
        max_bin (int): The number of bins.
        reg_lambda (float): L2 regularization term.
        gamma (float): Minimum loss reduction required to make a split.
        min_child_weight (float): Minimum sum of Hessians required in a child.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - max_gains (jnp.ndarray): Best gain for each node. Shape (t,).
            - best_features (jnp.ndarray): Best feature index for each node. Shape (t,).
            - best_threshold_idxs (jnp.ndarray): Best threshold bin index for each node. Shape (t,).
    """
    # 1. Stack the list of arrays into a single JAX array.
    # This is the key step to enable vectorization.
    # Shape: (t, n_features * max_bin, 2)
    stacked_histograms = jnp.stack(list_of_histograms, axis=0)

    # 2. Vmap the single-node function over the stacked histograms.
    # `in_axes=(0, None, ...)` means we iterate over the first axis of the first argument
    # (the `t` dimension of `stacked_histograms`), while other args are broadcast.
    vmapped_splitter = jax.vmap(
        _find_best_split_for_one_node_from_flat_cumulative,
        in_axes=(0, None, None, None, None, None),
    )

    # 3. Execute the vectorized computation.
    max_gains, best_features, best_threshold_idxs = vmapped_splitter(
        stacked_histograms, n_features, max_bin, reg_lambda, gamma, min_child_weight
    )

    return max_gains, best_features, best_threshold_idxs


@jax.jit
def find_global_best_split_local_features(
    cur_level_best_gains: list[jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Finds the global best split for each node from results across multiple feature groups,
    returning the LOCAL feature index within the winning group.

    Args:
        cur_level_best_gains (List[jnp.ndarray]): A list of gain vectors.
            List length `p`. Each element has shape (t,).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - global_best_gains (jnp.ndarray): Global best gain for each node. Shape (t,).
            - best_group_indices (jnp.ndarray): The index of the feature group that provided the best split. Shape (t,).
    """
    # 1. Stack the lists into single JAX arrays.
    # This creates a new leading dimension representing the feature groups (parties).
    # Shape of each: (p, t)
    gains_stacked = jnp.stack(cur_level_best_gains, axis=0)

    # 2. Find the index of the best group for each node.
    # jnp.argmax along axis=0 (the group dimension) finds the index of the maximum gain.
    # Shape: (t,)
    best_group_indices = jnp.argmax(gains_stacked, axis=0)

    # 3. Use the `best_group_indices` to gather the global best values.
    # `jnp.take_along_axis` is perfect for this "advanced indexing".
    # We need to reshape `best_group_indices` to (1, t) to match the dimensions for gathering.
    best_group_indices_reshaped = best_group_indices[None, :]

    global_best_gains = jnp.take_along_axis(
        gains_stacked, best_group_indices_reshaped, axis=0
    ).squeeze(axis=0)

    return (
        global_best_gains,
        best_group_indices,
    )


def update_values(
    values: jnp.ndarray,
    gh_plaintext: jnp.ndarray,
    bt: jnp.ndarray,
    is_leaf: jnp.ndarray,
    reg_lambda: float,
    n_nodes: int,
) -> jnp.ndarray:
    # 1. Perform a single segment_sum on the combined matrix.
    # The result will have shape (n_nodes, 2).
    sum_gh_segments = segment_sum(gh_plaintext, bt, num_segments=n_nodes)

    # 2. Unpack the results.
    sum_g_segments = sum_gh_segments[:, 0]  # Shape: (n_nodes,)
    sum_h_segments = sum_gh_segments[:, 1]  # Shape: (n_nodes,)

    # 3. The rest of the logic remains the same.
    # Avoid division by zero for nodes that might not have any samples.
    safe_h_sum = jnp.where(sum_h_segments == 0, 1, sum_h_segments)
    leaf_values = -sum_g_segments / (safe_h_sum + reg_lambda)

    # Only update values for nodes that are actual leaves and received samples.
    mask_to_update = (is_leaf) & (sum_h_segments != 0)
    values = jnp.where(mask_to_update, leaf_values, values)

    return values


def build_tree(
    gh_plaintext: mp.MPObject,
    gh_encrypted: mp.MPObject,
    all_bins: list[jnp.ndarray],
    all_bin_indices: list[mp.MPObject],
    pkey: mp.MPObject,  # broadcasted public key
    skey: mp.MPObject,  # private secret key owned by active party
    active_party_id: int,
    passive_party_ids: list[int],
    max_depth: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> Tree:
    """
    Builds a single decision tree level-by-level (breadth-first).

    Args:
        gh_plaintext (mp.MPObject): The plaintext gradients and hessians. Shape (m, 2).
        gh_encrypted (mp.MPObject): The encrypted gradients and hessians. Shape (m, 2).
        all_bins (list[jnp.ndarray]): Bin boundaries. Shape (n, k-1).
        all_bin_indices (list[jnp.ndarray]): Binned input data. Shape (m, n).
        active_party_id (int): The active party id.
        passive_party_ids (list[int]): The passive party ids.
        max_depth (int): Maximum depth of the tree.
        reg_lambda (float): L2 regularization term.
        gamma (float): Minimum loss reduction to make a split.
        min_child_weight (float): Minimum sum of Hessians in a child.

    Returns:
        Tree: The completed tree object.
    """
    all_party_ids = [active_party_id, *passive_party_ids]
    all_party_mask = mp.Mask((1 << (len(all_party_ids))) - 1)

    m = gh_plaintext.shape[0]
    n_nodes = 2 ** (max_depth + 1) - 1

    # --- Initialize flat tree arrays ---
    all_feats = [
        mp.run_jax_at(party_id, lambda x: x, jnp.full(n_nodes, -1, dtype=jnp.int64))
        for party_id in all_party_ids
    ]
    all_thresholds = [
        mp.run_jax_at(
            party_id, lambda x: x, jnp.full(n_nodes, jnp.inf, dtype=jnp.float32)
        )
        for party_id in all_party_ids
    ]

    is_leaf = mp.run_jax(lambda x: x, jnp.full(n_nodes, 0, dtype=jnp.int64))
    # TODO: can this be known only by ap?
    owned_party_ids = mp.run_jax(lambda x: x, jnp.full(n_nodes, -1, dtype=jnp.int64))

    # only owned by active party
    values = mp.run_jax_at(
        active_party_id, lambda x: x, jnp.full(n_nodes, 0.0, dtype=jnp.float32)
    )

    # --- Initialize sample-to-node mapping ---
    # `bt` stores the GLOBAL node index for each sample.
    bt = mp.run_jax(lambda x: x, jnp.zeros(m, dtype=jnp.int64))

    # A tree of depth `d` has `d` levels of splits (level 0 to level d-1).
    for level in range(max_depth):
        n_nodes_level = 2**level
        # Note: here we assume all features have the same number of bins.
        # Number of bins `k` is number of splits + 1.
        k = all_bins[0].shape[1] + 1

        # Get global indices for nodes at the current level.
        cur_level_indices = mp.constant(jnp.arange(n_nodes_level) + (2**level - 1))

        # `bt` holds global node indices. We need local indices [0, t-1] for this level.
        # The first node at the current level has a global index of (2**level - 1).
        first_node_idx_of_level = 2**level - 1
        bt_levels = mp.run_jax(
            lambda bt, offset=first_node_idx_of_level: bt - offset, bt
        )

        # 1. Build the histogram & find the best split ,for the current level.
        # 1.1 ap can do the histogram and best split purely locally.
        # ap find the best split for each node at this level.
        ap_GH_hist = mp.run_jax_at(
            active_party_id,
            partial(local_build_histogram, t=n_nodes_level, k=k),
            gh_plaintext,
            bt_levels,
            all_bin_indices[0],
        )
        ap_max_gains, ap_best_features, ap_best_threshold_idxs = mp.run_jax_at(
            active_party_id,
            local_compute_best_split,
            ap_GH_hist,
            reg_lambda,
            gamma,
            min_child_weight,
        )

        cur_level_best_gains = [ap_max_gains]
        cur_level_best_features = [ap_best_features]
        cur_level_best_threshold_idxs = [ap_best_threshold_idxs]

        # 1.2 pp should compute the histogram locally, encrypt it, and send it to ap.
        for idx, pp_rank in enumerate(passive_party_ids):
            # 1.2.1 pp compute the accumulated histogram locally.
            cur_pp_rank = pp_rank

            # +1 because the first party is the active party
            cur_pp_bin_indices = all_bin_indices[idx + 1]

            # construct subgroup map from bt_level
            # bt_level: (m,), with values in (0,1,2,...,n_nodes_level-1)
            cur_pp_subgroup_map = mp.run_jax_at(
                cur_pp_rank,
                partial(_get_subgroup_map, group_size=n_nodes_level),
                bt_levels,
            )  # (n_nodes_level, m)

            # 1.2.2 pp encrypt the accumulated histogram.
            cur_pp_enc_hist_cumsum: list[mp.MPObject] = (
                batch_feature_wise_bucket_sum_mplang(
                    gh_encrypted,
                    cur_pp_subgroup_map,
                    cur_pp_bin_indices,
                    k,  # bucket_size
                    n_nodes_level,  # group_size
                    cur_pp_rank,  # rank
                )
            )

            assert len(cur_pp_enc_hist_cumsum) == n_nodes_level, (
                f"Expect {n_nodes_level} outputs, got {len(cur_pp_enc_hist_cumsum)}"
            )

            # 1.2.3 pp send the encrypted histogram to ap.
            cur_pp_enc_hist_cumsum: list[mp.MPObject] = p2p_list(
                cur_pp_rank, active_party_id, cur_pp_enc_hist_cumsum
            )

            # 1.2.4 ap decrypt the histogram.
            cur_pp_dec_hist_cumsum: list[mp.MPObject] = [
                mp.run_at(active_party_id, phe.decrypt, cur_pp_enc_hist_cumsum[i], skey)
                for i in range(len(cur_pp_enc_hist_cumsum))
            ]

            # 1.2.5 ap find the best split for each node at this level.
            (
                cur_pp_best_gains,
                cur_pp_best_features,
                cur_pp_best_threshold_idxs,
            ) = mp.run_jax_at(
                active_party_id,
                partial(
                    pp_compute_all_best_splits,
                    n_features=cur_pp_bin_indices.shape[1],
                    max_bin=k,
                    reg_lambda=reg_lambda,
                    gamma=gamma,
                    min_child_weight=min_child_weight,
                ),
                cur_pp_dec_hist_cumsum,
            )
            cur_level_best_gains.append(cur_pp_best_gains)
            cur_level_best_features.append(cur_pp_best_features)
            cur_level_best_threshold_idxs.append(cur_pp_best_threshold_idxs)

        # 1.3 ap find the best split for each node at this level by considering all the gains.
        assert (
            len(cur_level_best_gains)
            == len(cur_level_best_features)
            == len(cur_level_best_threshold_idxs)
        )
        assert len(cur_level_best_gains) == 1 + len(passive_party_ids)

        # all shape (t, )
        (
            global_best_gains,
            best_group_indices,
        ) = mp.run_jax_at(
            active_party_id,
            find_global_best_split_local_features,
            cur_level_best_gains,
        )

        # TODO: not all pp should know the group_indice , features, etc
        # update the is_leaf
        is_leaf = mp.run_jax_at(
            active_party_id,
            _update_is_leaf,
            is_leaf,
            global_best_gains,
            cur_level_indices,
        )
        is_leaf = mp.bcast_m(all_party_mask, active_party_id, is_leaf)

        # TODO: can this not be known by all parties?
        owned_party_ids = mp.run_jax_at(
            active_party_id,
            _update_owned_party_ids,
            owned_party_ids,
            cur_level_indices,
            best_group_indices,
        )
        owned_party_ids = mp.bcast_m(all_party_mask, active_party_id, owned_party_ids)

        all_cur_tmp_bt = []
        # update feats, thresholds
        for i in range(len(all_party_ids)):
            if i > 0:
                cur_level_best_features[i] = mp.p2p(
                    active_party_id,
                    passive_party_ids[i - 1],
                    cur_level_best_features[i],
                )
                cur_level_best_threshold_idxs[i] = mp.p2p(
                    active_party_id,
                    passive_party_ids[i - 1],
                    cur_level_best_threshold_idxs[i],
                )

            # temp bt for each party
            tmp = mp.run_jax_at(
                i,
                _update_bt,
                bt,
                bt_levels,
                is_leaf,
                all_bin_indices[i],
                cur_level_best_features[i],
                cur_level_best_threshold_idxs[i],
            )
            if i > 0:
                tmp = mp.p2p(passive_party_ids[i - 1], active_party_id, tmp)
            all_cur_tmp_bt.append(tmp)

            all_feats[i] = mp.run_jax_at(
                i,
                _update_best_features,
                all_feats[i],
                cur_level_best_features[i],
                cur_level_indices,
                owned_party_ids,
                is_leaf,
                i,
            )
            all_thresholds[i] = mp.run_jax_at(
                i,
                _update_best_thresholds,
                all_thresholds[i],
                all_bins[i],
                cur_level_best_features[i],
                cur_level_best_threshold_idxs[i],
                cur_level_indices,
                owned_party_ids,
                is_leaf,
                i,
            )

        # We need to update all_bt based on best_group_indices
        # For each sample, we check which node it's currently in, then check which party
        # provides the best split for that node, and use the corresponding tmp_bt

        def update_bt_with_best_splits(
            current_bt, all_tmp_bt, best_group_indices, cur_level_indices
        ):
            """
            Update bt based on best_group_indices using fully vectorized operations.

            Args:
                current_bt: Current bt array, shape (m,)
                all_tmp_bt: List of tmp_bt from all parties, each shape (m,)
                best_group_indices: Which party won for each node, shape (t,)
                cur_level_indices: Global indices for nodes at current level, shape (t,)

            Returns:
                Updated bt array, shape (m,)
            """
            # Stack all tmp_bt arrays into a single array: shape (n_parties, m)
            stacked_tmp_bt = jnp.stack(all_tmp_bt, axis=0)

            # Initialize updated_bt with current_bt
            updated_bt = current_bt

            # Create a mapping from global node index to local node index
            # We'll use a vectorized approach
            t = len(cur_level_indices)

            # For each local node index, create a mask for samples in that node
            # and update them with the corresponding winning party's tmp_bt
            def update_for_one_node(carry, i):
                updated_bt = carry
                global_node_idx = cur_level_indices[i]
                winning_party = best_group_indices[i]

                # Find samples that are currently in this node
                samples_in_node = current_bt == global_node_idx

                # Get the tmp_bt from the winning party
                winning_tmp_bt = stacked_tmp_bt[winning_party]

                # Update bt for samples in this node
                updated_bt = jnp.where(samples_in_node, winning_tmp_bt, updated_bt)

                return updated_bt, None

            # Use lax.scan to iterate over nodes in a JAX-compatible way
            updated_bt, _ = jax.lax.scan(update_for_one_node, updated_bt, jnp.arange(t))

            return updated_bt

        # Apply the update logic at AP (since AP has all tmp_bt and best_group_indices)
        bt = mp.run_jax_at(
            active_party_id,
            update_bt_with_best_splits,
            bt,
            all_cur_tmp_bt,
            best_group_indices,
            cur_level_indices,
        )

        # Share the updated bt to all passive parties
        bt = mp.bcast_m(all_party_mask, active_party_id, bt)

    # --- Force leaf nodes at maximum depth ---
    # After reaching max_depth, we need to mark all nodes at the final level as leaves.
    # Nodes at the final level that could be split are at levels max_depth-1
    # Their children would be at level max_depth, so we mark those children as leaves
    final_level_start = 2**max_depth - 1
    final_level_end = 2 ** (max_depth + 1) - 1
    final_level_indices = mp.constant(jnp.arange(final_level_start, final_level_end))

    def force_leaf_nodes(is_leaf_arr: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
        return is_leaf_arr.at[indices].set(1)

    def force_leaf_owner_ids(
        owner_id_arr: jnp.ndarray, indices: jnp.ndarray, owner_id: int
    ) -> jnp.ndarray:
        return owner_id_arr.at[indices].set(owner_id)

    # Force all nodes at max depth to be leaves
    is_leaf = mp.run_jax(force_leaf_nodes, is_leaf, final_level_indices)

    # Force owner_id for all forced leaf nodes to be active_party_id
    # (since leaf values are managed by active party)
    owned_party_ids = mp.run_jax(
        force_leaf_owner_ids, owned_party_ids, final_level_indices, active_party_id
    )

    # --- Final leaf value calculation ---
    # After the tree structure is built, calculate the value for every node based on final sample assignments.
    # `bt` now contains the final node index for every sample.
    values = mp.run_jax_at(
        active_party_id,
        partial(update_values, n_nodes=n_nodes, reg_lambda=reg_lambda),
        values,
        gh_plaintext,
        bt,
        is_leaf,
    )

    return Tree(
        feature=all_feats,
        threshold=all_thresholds,
        value=values,  # owned by ap only
        is_leaf=is_leaf,
        owned_party_id=owned_party_ids,
    )


def _local_try_to_predict(
    data: mp.MPObject,
    feature: mp.MPObject,
    threshold: mp.MPObject,
    is_leaf: mp.MPObject,
    owned_party_id: mp.MPObject,
    party_id: int,
) -> mp.MPObject:
    n_samples = data.shape[0]
    n_nodes = feature.shape[0]

    @jax.jit
    def _local_traverse_kernel(
        data: jnp.ndarray,
        feature: jnp.ndarray,
        thresholds: jnp.ndarray,
        is_leaf: jnp.ndarray,
        owned_party_id: jnp.ndarray,
    ):
        # initial_locations: (n_samples, n_nodes), 1 means the sample is "likely" to be stopped at that node.
        initial_locations = (
            jnp.zeros((n_samples, n_nodes), dtype=jnp.int64).at[:, 0].set(1)
        )

        def propagate(i, locations):
            # `i`: the absolute index of the potential split node we are processing.

            # We define ownership explicitly. A party acts on a node `i` if and only if:
            # 1. It is a split node (`is_leaf[i] == 0`).
            # 2. The party's ID matches the node's owner ID.
            is_my_split_node = (is_leaf[i] == 0) & (owned_party_id[i] == party_id)

            # This function is only executed if `is_my_split_node` is True.
            def process_my_split_node(locs):
                samples_at_this_node = locs[:, i]
                feat_idx = feature[i]
                threshold = thresholds[i]

                go_left = data[:, feat_idx] <= threshold
                move_to_left_child = samples_at_this_node * go_left
                move_to_right_child = samples_at_this_node * (1 - go_left)

                # Propagate and then clear the current node.
                locs = locs.at[:, 2 * i + 1].add(move_to_left_child)
                locs = locs.at[:, 2 * i + 2].add(move_to_right_child)
                return locs.at[:, i].set(0)

            # This function is executed for all other cases:
            # - It's another party's split node.
            # - It's a leaf node.
            def pass_through_or_propagate_unknown(locs):
                is_a_split_node = is_leaf[i] == 0

                def propagate_unknown(loc):
                    # It's a split node, but not mine. Propagate uncertainty to both children.
                    samples_at_this_node = loc[:, i]
                    loc = loc.at[:, 2 * i + 1].add(samples_at_this_node)
                    loc = loc.at[:, 2 * i + 2].add(samples_at_this_node)
                    return loc.at[:, i].set(0)

                def do_nothing(loc):
                    # It's a leaf node. Samples that land here should rest. Do nothing.
                    return loc

                # Is it a split node? If so, propagate uncertainty. If not (it's a leaf), do nothing.
                return jax.lax.cond(
                    is_a_split_node, propagate_unknown, do_nothing, locs
                )

            # The master switch: Did I own this split node or not?
            return jax.lax.cond(
                is_my_split_node,
                process_my_split_node,
                pass_through_or_propagate_unknown,
                locations,
            )

        # Iterate through all nodes that could potentially be parents.
        final_locations = jax.lax.fori_loop(
            0, n_nodes // 2, propagate, initial_locations
        )
        return final_locations

    local_pred = mp.run_jax_at(
        party_id,
        _local_traverse_kernel,
        data,
        feature,
        threshold,
        is_leaf,
        owned_party_id,
    )
    return local_pred


@jax.jit
def agg_prediction(
    all_masks: list[jnp.ndarray], is_leaf: jnp.ndarray, values: jnp.ndarray
):
    stacked_masks = jnp.stack(all_masks)
    consensus_mask = jnp.all(stacked_masks > 0, axis=0)
    final_leaf_mask = consensus_mask * is_leaf
    # argmax will always return the first non-zero index, which is the left and upper most leaf node.
    final_leaf_node_indices = jnp.argmax(final_leaf_mask, axis=1)
    predictions = values[final_leaf_node_indices]
    return predictions


def predict_tree(
    tree: Tree,
    all_datas: list[mp.MPObject],
    active_party_id: int,
    passive_party_ids: list[int],
) -> mp.MPObject:
    n_parties = len(passive_party_ids) + 1

    # ap/pps do the local "predict" according to their own data and the tree.
    ap_mask = _local_try_to_predict(
        all_datas[0],
        tree.feature[0],
        tree.threshold[0],
        tree.is_leaf,
        tree.owned_party_id,
        active_party_id,
    )

    all_masks = [ap_mask]
    for i in range(n_parties - 1):
        # PPs are parties 1, 2, ...
        pp_mask = _local_try_to_predict(
            all_datas[i + 1],
            tree.feature[i + 1],
            tree.threshold[i + 1],
            tree.is_leaf,
            tree.owned_party_id,
            passive_party_ids[i],
        )

        all_masks.append(mp.p2p(passive_party_ids[i], active_party_id, pp_mask))

    # Aggregation and Final Prediction in AP
    final_pred = mp.run_jax_at(
        active_party_id, agg_prediction, all_masks, tree.is_leaf, tree.value
    )
    return final_pred


def predict_ensemble(
    model: TreeEnsemble,
    all_datas: list[mp.MPObject],
    active_party_id: int,
    passive_party_ids: list[int],
    learning_rate: float,
) -> mp.MPObject:
    y_pred_logits = mp.run_jax_at(
        active_party_id,
        lambda init_y, m=all_datas[0].shape[0]: init_y * jnp.ones(m),
        model.initial_prediction,
    )

    for tree in model.trees:
        pred = predict_tree(tree, all_datas, active_party_id, passive_party_ids)
        y_pred_logits = mp.run_jax_at(
            active_party_id,
            partial(_update_pred, learning_rate=learning_rate),
            y_pred_logits,
            pred,
        )

    y_pred = mp.run_jax_at(active_party_id, sigmoid, y_pred_logits)

    return y_pred


@jax.jit
def agg_prediction_leaves(all_masks: list[jnp.ndarray]):
    # Original stacking: (n_parties, n_samples, n_nodes)
    stacked_masks = jnp.stack(all_masks)

    # Transpose to (n_samples, n_parties, n_nodes)
    transposed_masks = stacked_masks.transpose(1, 0, 2)

    # Reshape to (n_samples * n_parties, n_nodes) with desired ordering:
    # [AP_sample0, PP_sample0, AP_sample1, PP_sample1, ...]
    n_samples, n_parties, n_nodes = transposed_masks.shape
    reordered_masks = transposed_masks.reshape(n_samples * n_parties, n_nodes)

    return reordered_masks


def predict_tree_leaf(
    tree: Tree,
    all_datas: list[mp.MPObject],
    active_party_id: int,
    passive_party_ids: list[int],
) -> mp.MPObject:
    n_parties = len(passive_party_ids) + 1

    # ap/pps do the local "predict" according to their own data and the tree.
    ap_mask = _local_try_to_predict(
        all_datas[0],
        tree.feature[0],
        tree.threshold[0],
        tree.is_leaf,
        tree.owned_party_id,
        active_party_id,
    )

    all_masks = [ap_mask]
    for i in range(n_parties - 1):
        # PPs are parties 1, 2, ...
        pp_mask = _local_try_to_predict(
            all_datas[i + 1],
            tree.feature[i + 1],
            tree.threshold[i + 1],
            tree.is_leaf,
            tree.owned_party_id,
            passive_party_ids[i],
        )

        all_masks.append(mp.p2p(passive_party_ids[i], active_party_id, pp_mask))

    # Aggregation and Final Prediction in AP
    final_pred = mp.run_jax_at(active_party_id, agg_prediction_leaves, all_masks)
    return final_pred


def predict_leaves_ensemble(
    model: TreeEnsemble,
    all_datas: list[mp.MPObject],
    active_party_id: int,
    passive_party_ids: list[int],
) -> mp.MPObject:
    # debug only, so we only print one tree.
    assert len(model.trees) == 1

    tree = model.trees[0]
    leaves = predict_tree_leaf(tree, all_datas, active_party_id, passive_party_ids)
    return leaves


def fit_tree_ensemble(
    all_datas: list[mp.MPObject],
    y_data: mp.MPObject,
    all_bins: list[jnp.ndarray],
    all_bin_indices: list[mp.MPObject],
    initial_y_pred: mp.MPObject,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    active_party_id: int,
    passive_party_ids: list[int],
) -> TreeEnsemble:
    all_party_pmasks = mp.Mask((1 << (len(passive_party_ids) + 1)) - 1)
    m = y_data.shape[0]

    y_pred_current = mp.run_jax_at(
        active_party_id, lambda init_y, m=m: init_y * jnp.ones(m), initial_y_pred
    )

    trees: list[Tree] = []
    # TODO: add more controls of keygen, e.g., key size
    pkey, skey = mp.run_at(active_party_id, phe.keygen)
    world_mask = mp.Mask.all(1 + len(passive_party_ids))
    pkey = mp.bcast_m(world_mask, active_party_id, pkey)

    # TODO: to support early stopping, maybe we need something like `jax.lax.scan` to store all the trees?
    for _ in range(n_estimators):
        gh = mp.run_jax_at(active_party_id, compute_gh, y_data, y_pred_current)
        gh_encrypted = mp.run_at(active_party_id, phe.encrypt, gh, pkey)

        # known by all parties
        gh_encrypted = mp.bcast_m(all_party_pmasks, active_party_id, gh_encrypted)

        tree = build_tree(
            gh,
            gh_encrypted,
            all_bins,
            all_bin_indices,
            pkey,
            skey,
            active_party_id,
            passive_party_ids,
            max_depth,
            reg_lambda,
            gamma,
            min_child_weight,
        )

        update = predict_tree(
            tree,
            all_datas,
            active_party_id,
            passive_party_ids,
        )

        y_pred_current = mp.run_jax_at(
            active_party_id, _update_pred, y_pred_current, update, learning_rate
        )
        trees.append(tree)

    return TreeEnsemble(
        max_depth=max_depth, trees=trees, initial_prediction=initial_y_pred
    )


# ==============================================================================
# Part 4: High-level Classifier Class and Main Execution
# ==============================================================================


class SecureBoost:
    def __init__(
        self,
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        max_bin=8,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1.0,
        active_party_id=0,
        passive_party_ids=None,
    ):
        """
        This implements the SecureBoost algorithm building upon the mplang and JAX framework.

        Args:
            n_estimators: number of trees to fit
            learning_rate: shrinkage parameter to prevent overfitting
            max_depth: max depth of each tree
            max_bin: max number of bins for each feature
            reg_lambda: L2 regularization term on weights
            gamma: minimum loss reduction required to make a split
            min_child_weight: minimum sum of instance weights needed in a child
            active_party_id: id of the active party, which owns the labels
            passive_party_ids: ids of the passive parties
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        if max_bin < 2:
            raise ValueError(f"max_bin must be >= 2, but got {max_bin}")
        self.max_bin = max_bin

        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight

        self.active_party_id = active_party_id
        self.active_party_mask = mp.Mask(1 << self.active_party_id)

        if passive_party_ids is None:
            passive_party_ids = [1]

        assert isinstance(passive_party_ids, list)
        self.passive_party_ids = passive_party_ids
        self.passive_party_masks = [mp.Mask(1 << pid) for pid in passive_party_ids]

        # TODO: support more general party ids
        assert self.active_party_id == 0, "Only active party id 0 is supported now"
        assert self.passive_party_ids == list(range(1, len(passive_party_ids) + 1)), (
            f"Only passive party ids {list(range(1, len(passive_party_ids) + 1))} are supported now"
        )

        self.trees: TreeEnsemble | None = None

    def fit(self, all_datas: list[mp.MPObject], y_data: mp.MPObject):
        """
        Fit the SecureBoost model.

        Args:
            all_datas: a list of MPObjects, each of which is a data matrix of a party
                - the first element is the data matrix of the active party
                - the rest are the data matrices of the passive parties
            y_data: the label vector of the active party
        """
        self._check_all_datas(all_datas)
        assert y_data.pmask == self.active_party_mask, (
            f"y_data.pmask: {y_data.pmask}, self.active_party_mask: {self.active_party_mask}"
        )

        # 1. do the binning.
        # TODO: support more sophisticated binning schemes.
        # Note: We make several simplifying assumptions here:
        #   1. xgboost usually do the binning before the training of each tree, but we do it here only once for simplicity.
        #   2. we assume ALL FEATURES have the SAME number of bins.
        #   3. we assume ALL FEATURES are continuous.
        #   4. we use the easy equi-width binning scheme rather than the score-based binning scheme.

        # Vmapped versions for applying functions across all features.
        build_bins_vmapped = jax.vmap(
            partial(build_bins_equi_width, max_bin=self.max_bin), in_axes=1
        )
        compute_indices_vmapped = jax.vmap(
            compute_bin_indices, in_axes=(1, 0), out_axes=1
        )

        # We forcibly divide the binning process into two parts: AP and PPs,
        # in order to facilitate the subsequent addition of more complex and customized binning mechanisms.
        ap_data = all_datas[0]
        pp_datas = all_datas[1:]
        all_bins = [mp.run_jax_at(self.active_party_id, build_bins_vmapped, ap_data)]

        all_bin_indices = [
            mp.run_jax_at(
                self.active_party_id, compute_indices_vmapped, ap_data, all_bins[0]
            )
        ]
        for idx, pp_rank in enumerate(self.passive_party_ids):
            pp_bin = mp.run_jax_at(pp_rank, build_bins_vmapped, pp_datas[idx])
            all_bins.append(pp_bin)
            all_bin_indices.append(
                mp.run_jax_at(pp_rank, compute_indices_vmapped, pp_datas[idx], pp_bin)
            )

        # 2. init base pred
        initial_y_pred = mp.run_jax_at(self.active_party_id, compute_init_pred, y_data)

        self.trees = fit_tree_ensemble(
            all_datas,
            y_data,
            all_bins,
            all_bin_indices,
            initial_y_pred,
            self.n_estimators,
            self.learning_rate,
            self.max_depth,
            self.reg_lambda,
            self.gamma,
            self.min_child_weight,
            self.active_party_id,
            self.passive_party_ids,
        )

        return self

    def predict(self, all_datas: list[mp.MPObject]) -> mp.MPObject:
        """
        Predict the values/probabilities of the data.

        Args:
            all_datas: a list of MPObjects, each of which is a data matrix of a party
                - the first element is the data matrix of the active party
                - the rest are the data matrices of the passive parties
        """
        self._check_all_datas(all_datas)

        if self.trees is None:
            raise RuntimeError(
                "The model has not been fitted yet. Please call .fit() first."
            )

        return predict_ensemble(
            self.trees,
            all_datas,
            self.active_party_id,
            self.passive_party_ids,
            self.learning_rate,
        )

    def _check_all_datas(self, all_datas: list[mp.MPObject]):
        ap_data = all_datas[0]
        pp_datas = all_datas[1:]
        assert len(pp_datas) == len(self.passive_party_ids)
        assert ap_data.pmask == self.active_party_mask
        for i, pp_data in enumerate(pp_datas):
            assert pp_data.pmask == self.passive_party_masks[i]
        # check whether ap_data and pp_datas have the same number of rows
        for pp_data in pp_datas:
            assert pp_data.shape[0] == ap_data.shape[0], (
                "The number of rows of ap_data and pp_datas must be the same"
            )

    # debug only
    def predict_leaves(self, all_datas: list[mp.MPObject]) -> mp.MPObject:
        self._check_all_datas(all_datas)

        if self.trees is None:
            raise RuntimeError(
                "The model has not been fitted yet. Please call .fit() first."
            )
        assert len(self.trees.trees) == 1

        return predict_leaves_ensemble(
            self.trees,
            all_datas,
            self.active_party_id,
            self.passive_party_ids,
        )


# debug only
def pretty_print_ensemble(ensemble: TreeEnsemble, party_ids: list[int]):
    """
    Prints a TreeEnsemble object in a human-readable, raw format,
    reflecting the perspective of each party.

    Args:
        ensemble (TreeEnsemble): The trained TreeEnsemble object to print.
        party_ids (List[int]): A list of all party IDs, e.g., [0, 1].
    """
    # Create a mapping from rank to a more descriptive name
    party_names = {
        pid: f"Party {pid} (AP)" if pid == 0 else f"Party {pid} (PP)"
        for pid in party_ids
    }

    print("\n" + "=" * 25 + " Tree Ensemble Details " + "=" * 25)

    # --- 1. Print Ensemble-level information ---
    print("\n[Ensemble Info]")
    print(f"  - Max Depth: {ensemble.max_depth}")
    print(f"  - Initial Prediction (Logits): {ensemble.initial_prediction}")

    # --- 2. Iterate through and print each tree ---
    print(f"\nNumber of Trees: {len(ensemble.trees)}")

    for i, tree in enumerate(ensemble.trees):
        print("\n" + "-" * 20 + f" Tree {i} " + "-" * 20)

        # --- Iterate through each party's complete view of the tree ---
        for party_idx, party_id in enumerate(party_ids):
            print(f"\n  --- {party_names[party_id]}'s Complete View ---")

            # Print information about the MPObjects rather than trying to index them
            print(f"    - Feature mp.MPObject:     {tree.feature[party_idx]}")
            print(f"    - Threshold mp.MPObject:   {tree.threshold[party_idx]}")
            print(f"    - Is Leaf mp.MPObject:     {tree.is_leaf[party_idx]}")
            print(f"    - Owner ID mp.MPObject:    {tree.owned_party_id[party_idx]}")

        # --- Print AP's exclusive 'value' array ---
        # This is the only array that is not a list and belongs solely to the AP.
        print(f"\n  --- {party_names[0]}'s Exclusive Data ---")
        print(f"    - Leaf Values mp.MPObject: {tree.value}")

    print("\n" + "=" * 70)
