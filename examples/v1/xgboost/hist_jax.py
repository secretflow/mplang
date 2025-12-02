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

# ==============================================================================
# Naming Conventions for Shapes & Variables
# ==============================================================================
# m: number of samples (or examples)
# n: number of features
# k: number of bins for discretization
# t: number of nodes at the current tree level
# g: gradient of the loss function w.r.t. predictions
# h: hessian (second derivative) of the loss function
# bt: "branching of tree", an array where bt[i] stores the node index for sample i

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
    """

    feature: jnp.ndarray
    threshold: jnp.ndarray
    value: jnp.ndarray
    is_leaf: jnp.ndarray


class TreeEnsemble(NamedTuple):
    """
    Represents an entire XGBoost model.

    Attributes:
        max_depth (int): Maximum depth of the trees in the ensemble.
        trees (List[Tree]): A list of Tree NamedTuples.
        initial_prediction (float): The base prediction (logit or mean) for all samples.
        bins (jnp.ndarray): Shape (n, max_bin-1). The bin boundaries for each feature.
    """

    max_depth: int
    trees: list[Tree]
    initial_prediction: float
    bins: jnp.ndarray


# ==============================================================================
# Part 2: Core Mathematical & Binning Functions
# ==============================================================================


@jax.jit
def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + jnp.exp(-x))


# --- Gradient and Hessian Functions for Different Objectives ---


@jax.jit
def _gradient_clf(y_true: jnp.ndarray, y_pred_logits: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the gradient of the log loss for binary classification.
    Shape: (m,) -> (m,).
    """
    p = sigmoid(y_pred_logits)
    return p - y_true


@jax.jit
def _hessian_clf(y_true: jnp.ndarray, y_pred_logits: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Hessian of the log loss for binary classification.
    Shape: (m,) -> (m,).
    """
    p = sigmoid(y_pred_logits)
    return p * (1 - p)


@jax.jit
def _gradient_reg(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the gradient of the squared error loss.
    Shape: (m,) -> (m,).
    """
    return (y_pred - y_true).astype(jnp.float32)


@jax.jit
def _hessian_reg(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Hessian of the squared error loss, which is always 1.
    Shape: (m,) -> (m,).
    """
    return jnp.ones_like(y_true, dtype=jnp.float32)


@partial(jax.jit, static_argnames=("max_bins",))
def build_bins_equi_width(x: jnp.ndarray, max_bins: int) -> jnp.ndarray:
    """
    Computes equi-width bin boundaries for a single feature vector.

    Args:
        x (jnp.ndarray): A single feature vector. Shape (m,).
        max_bins (int): The maximum number of bins to create.

    Returns:
        jnp.ndarray: The split points (boundaries). Shape (max_bins-1,).
    """
    if max_bins < 2:
        raise ValueError(f"max_bins must be >= 2, but got {max_bins}")
    n_samples = x.shape[0]
    n_splits = max_bins - 1
    inf_splits = jnp.full(shape=(n_splits,), fill_value=jnp.inf, dtype=x.dtype)

    def create_valid_bins():
        min_val, max_val = jnp.min(x), jnp.max(x)
        is_constant = (max_val - min_val) < 1e-9

        def generate_splits():
            # Create max_bins + 1 points, then take the inner max_bins - 1 as splits.
            boundaries = jnp.linspace(min_val, max_val, num=max_bins + 1)
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


# Vmapped versions for applying functions across all features.
build_bins_vmapped = jax.vmap(build_bins_equi_width, in_axes=(1, None))
compute_indices_vmapped = jax.vmap(compute_bin_indices, in_axes=(1, 0), out_axes=1)


@partial(jax.jit, static_argnames=("t", "k"))
def build_histogram(
    g: jnp.ndarray,
    h: jnp.ndarray,
    bt_local: jnp.ndarray,
    bin_indices: jnp.ndarray,
    t: int,
    k: int,
) -> jnp.ndarray:
    """
    Builds histograms of gradients (G) and Hessians (H) for all features in a fully
    vectorized manner using jax.vmap, eliminating the Python for-loop.

    Args:
        g (jnp.ndarray): Gradients for each sample. Shape (m,).
        h (jnp.ndarray): Hessians for each sample. Shape (m,).
        bt_local (jnp.ndarray): Local node index [0, t-1] for each sample. Shape (m,).
        bin_indices (jnp.ndarray): Bin index for each sample and feature. Shape (m, n).
        t (int): Number of tree nodes at the current level.
        k (int): Number of bins.

    Returns:
        jnp.ndarray: Histogram of gradients and Hessians. Shape (n, t, k, 2).
    """
    # 1. Stack g and h into a single array.
    # Shape: (m,) + (m,) -> (m, 2)
    gh = jnp.stack([g, h], axis=1)

    # 2. Define a function that computes the histogram for a SINGLE feature.
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
        # `segment_sum` operates on the (m, 2) `gh` array. It sums rows,
        # preserving the 2 columns for g and h separately.
        return segment_sum(gh, combined_indices, num_segments=t * k)

    # 3. Vectorize the function over the feature dimension of `bin_indices`.
    # in_axes=1 means we map over the columns (features) of `bin_indices`.
    # out_axes=0 means we stack the results along a new leading axis.
    # Using vmap instead of a Python for-loop significantly improves compile performance.
    # Input `bin_indices` shape: (m, n)
    # Output `flat_histograms` shape: (n, t * k, 2)
    flat_histograms = jax.vmap(histogram_for_one_feature, in_axes=1, out_axes=0)(
        bin_indices
    )

    # 4. Reshape the result to the desired (n, t, k, 2) format.
    # Final shape: (n, t, k, 2)
    GH_hist = flat_histograms.reshape((bin_indices.shape[1], t, k, 2))

    return GH_hist


@partial(jax.jit, static_argnames=("reg_lambda", "gamma", "min_child_weight"))
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


@partial(jax.jit, static_argnames=("reg_lambda", "gamma", "min_child_weight"))
def compute_best_split(
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


@partial(
    jax.jit, static_argnames=("max_depth", "reg_lambda", "gamma", "min_child_weight")
)
def build_tree(
    bins: jnp.ndarray,
    bin_indices: jnp.ndarray,
    g: jnp.ndarray,
    h: jnp.ndarray,
    max_depth: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> Tree:
    """
    Builds a single decision tree level-by-level (breadth-first).

    Args:
        bins (jnp.ndarray): Bin boundaries. Shape (n, max_bin-1).
        bin_indices (jnp.ndarray): Binned input data. Shape (m, n).
        g (jnp.ndarray): Gradients. Shape (m,).
        h (jnp.ndarray): Hessians. Shape (m,).
        max_depth (int): Maximum depth of the tree.
        reg_lambda (float): L2 regularization term.
        gamma (float): Minimum loss reduction to make a split.
        min_child_weight (float): Minimum sum of Hessians in a child.

    Returns:
        Tree: The completed tree object.
    """
    n_samples, _ = bin_indices.shape
    n_nodes = 2 ** (max_depth + 1) - 1

    # --- Initialize flat tree arrays ---
    feats = jnp.full(n_nodes, -1, dtype=jnp.int32)
    thresholds = jnp.full(n_nodes, -1.0, dtype=jnp.float32)
    values = jnp.full(n_nodes, 0.0, dtype=jnp.float32)
    is_leaf = jnp.full(n_nodes, True, dtype=bool)

    # --- Initialize root node (index 0) ---
    values = values.at[0].set(-jnp.sum(g) / (jnp.sum(h) + reg_lambda + 1e-9))
    is_leaf = is_leaf.at[0].set(False)

    # --- Initialize sample-to-node mapping ---
    # `bt` stores the GLOBAL node index for each sample.
    bt = jnp.zeros(n_samples, dtype=jnp.int32)
    # `active_samples_mask` tracks which samples are in nodes that can still be split.
    active_samples_mask = jnp.full(n_samples, True, dtype=bool)

    # A tree of depth `d` has `d` levels of splits (level 0 to level d-1).
    for level in range(max_depth):
        n_nodes_level = 2**level
        # Number of bins `k` is number of splits + 1.
        k = bins.shape[1] + 1

        # Mask gradients and hessians to only include active samples.
        g_level = g * active_samples_mask
        h_level = h * active_samples_mask

        # `bt` holds global node indices. We need local indices [0, t-1] for this level.
        # The first node at the current level has a global index of (2**level - 1).
        first_node_idx_of_level = 2**level - 1
        bt_local = bt - first_node_idx_of_level
        GH_hist = build_histogram(
            g_level, h_level, bt_local, bin_indices, n_nodes_level, k
        )

        # Find the best split for each node at this level.
        max_gains, best_features, best_threshold_idxs = compute_best_split(
            GH_hist, reg_lambda, gamma, min_child_weight
        )

        # Get global indices for nodes at the current level.
        cur_indices = jnp.arange(n_nodes_level) + (2**level - 1)

        # Convert best bin indices to actual threshold values.
        best_features = best_features.astype(jnp.int32)
        best_thresholds = bins[best_features, best_threshold_idxs]

        # Update the tree structure arrays with split info for the current level.
        feats = feats.at[cur_indices].set(best_features)
        thresholds = thresholds.at[cur_indices].set(best_thresholds)

        # Pre-emptive pruning: A node becomes a leaf if its best gain is <= 0 or NaN/inf.
        is_leaf_level = (max_gains <= 0.0) | (~jnp.isfinite(max_gains))
        is_leaf = is_leaf.at[cur_indices].set(is_leaf_level)

        # --- Update sample-to-node mapping (`bt`) for the next level ---
        best_feature_for_each_sample = best_features[bt_local]
        best_threshold_idx_for_each_sample = best_threshold_idxs[bt_local]
        sample_bin_indices = bin_indices[
            jnp.arange(n_samples), best_feature_for_each_sample
        ]
        left_mask = sample_bin_indices <= best_threshold_idx_for_each_sample

        # Calculate the global index of the next node for each sample.
        bt_next = jnp.where(left_mask, 2 * bt + 1, 2 * bt + 2)

        # Determine which samples are now in a leaf node (after this level's splits).
        is_sample_in_leaf_node = is_leaf[bt]

        # Deactivate samples that have landed in a leaf node.
        active_samples_mask = active_samples_mask & ~is_sample_in_leaf_node

        # Update `bt` only for active samples; others stay in their leaf node.
        bt = jnp.where(active_samples_mask, bt_next, bt).astype(jnp.int32)

    # --- Force leaf nodes at maximum depth ---
    # After reaching max_depth, we need to mark all nodes at the final level as leaves.
    final_level_start = 2**max_depth - 1
    final_level_end = 2 ** (max_depth + 1) - 1
    final_level_indices = jnp.arange(final_level_start, final_level_end)

    # Force all nodes at max depth to be leaves
    is_leaf = is_leaf.at[final_level_indices].set(True)

    # --- Final leaf value calculation ---
    # After the tree structure is built, calculate the value for every node based on final sample assignments.
    # `bt` now contains the final node index for every sample.
    # 1. Stack g and h into a single (m, 2) matrix.
    gh = jnp.stack([g, h], axis=1)

    # 2. Perform a single segment_sum on the combined matrix.
    # The result will have shape (n_nodes, 2).
    sum_gh_segments = segment_sum(gh, bt, num_segments=n_nodes)

    # 3. Unpack the results.
    sum_g_segments = sum_gh_segments[:, 0]  # Shape: (n_nodes,)
    sum_h_segments = sum_gh_segments[:, 1]  # Shape: (n_nodes,)

    # 4. The rest of the logic remains the same.
    # Avoid division by zero for nodes that might not have any samples.
    safe_h_sum = jnp.where(sum_h_segments == 0, 1, sum_h_segments)
    leaf_values = -sum_g_segments / (safe_h_sum + reg_lambda)

    # Only update values for nodes that are actual leaves and received samples.
    mask_to_update = (is_leaf) & (sum_h_segments != 0)
    values = jnp.where(mask_to_update, leaf_values, values)

    return Tree(feature=feats, threshold=thresholds, value=values, is_leaf=is_leaf)


# --- Prediction and Ensemble Functions ---


def predict_tree(x: jnp.ndarray, model: Tree) -> jnp.ndarray:
    """
    Predicts using a single decision tree for a batch of samples.
    """

    def predict_single(sample_x: jnp.ndarray, model: Tree) -> jnp.ndarray:
        def go_left(node_idx: jnp.ndarray) -> jnp.ndarray:
            return 2 * node_idx + 1

        def go_right(node_idx: jnp.ndarray) -> jnp.ndarray:
            return 2 * node_idx + 2

        def body(state: tuple[jnp.ndarray, Tree]) -> tuple[jnp.ndarray, Tree]:
            cur_node, model = state
            feature_idx, threshold = model.feature[cur_node], model.threshold[cur_node]
            next_node = jax.lax.cond(
                sample_x[feature_idx] <= threshold, go_left, go_right, cur_node
            )
            return (next_node, model)

        def condition(state: tuple[jnp.ndarray, Tree]) -> jnp.ndarray:
            return ~model.is_leaf[state[0]]

        final_node, _ = jax.lax.while_loop(condition, body, (jnp.int32(0), model))
        return model.value[final_node]

    return jax.vmap(predict_single, in_axes=(0, None))(x, model)


@partial(jax.jit, static_argnames=("learning_rate", "objective"))
def predict_ensemble(
    x: jnp.ndarray, model: TreeEnsemble, learning_rate: float, objective: str
) -> jnp.ndarray:
    """
    Predicts using the entire ensemble, handling different objectives.
    """
    y_pred_logits = model.initial_prediction * jnp.ones(x.shape[0])
    for tree in model.trees:
        y_pred_logits += predict_tree(x, tree) * learning_rate

    return jax.lax.cond(
        objective == "binary:logistic",
        lambda: sigmoid(y_pred_logits),
        lambda: y_pred_logits,
    )


@partial(
    jax.jit,
    static_argnames=(
        "n_estimators",
        "max_bin",
        "learning_rate",
        "max_depth",
        "reg_lambda",
        "gamma",
        "min_child_weight",
        "objective",
    ),
)
def fit_tree_ensemble(
    x: jnp.ndarray,
    y: jnp.ndarray,
    n_estimators: int,
    max_bin: int,
    learning_rate: float,
    max_depth: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    objective: str,
) -> TreeEnsemble:
    """
    Fits the entire XGBoost tree ensemble. This function is JIT-compiled.
    """

    # TODO: for the sake of simplicity, we only do the binning once for the entire dataset.
    # In practice, we should do the binning for each tree.
    bins = build_bins_vmapped(x, max_bin)
    bin_indices = compute_indices_vmapped(x, bins)

    def init_clf_pred():
        p_base = jnp.clip(jnp.mean(y), 1e-15, 1 - 1e-15)
        return jnp.log(p_base / (1 - p_base))

    def init_reg_pred():
        return jnp.mean(y)

    initial_y_pred = jax.lax.cond(
        objective == "binary:logistic", init_clf_pred, init_reg_pred
    )

    y_pred = initial_y_pred * jnp.ones(x.shape[0])
    trees: list[Tree] = []

    # The Python for-loop is "unrolled" by JAX during JIT compilation.
    for _ in range(n_estimators):
        g = jax.lax.cond(
            objective == "binary:logistic",
            lambda: _gradient_clf(y, y_pred),
            lambda: _gradient_reg(y, y_pred),
        )
        h = jax.lax.cond(
            objective == "binary:logistic",
            lambda: _hessian_clf(y, y_pred),
            lambda: _hessian_reg(y, y_pred),
        )

        tree = build_tree(
            bins, bin_indices, g, h, max_depth, reg_lambda, gamma, min_child_weight
        )

        y_pred += predict_tree(x, tree) * learning_rate
        trees.append(tree)

    return TreeEnsemble(
        max_depth=max_depth, trees=trees, initial_prediction=initial_y_pred, bins=bins
    )


# ==============================================================================
# Part 4: High-level Generic Class and Main Execution
# ==============================================================================


class XGBoostJAX:
    """
    A user-friendly Scikit-Learn-style wrapper for a generic JAX XGBoost implementation.
    Supports 'binary:logistic' and 'reg:squarederror' objectives.
    """

    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        max_depth: int,
        max_bin: int,
        reg_lambda: float,
        gamma: float,
        min_child_weight: float,
        objective: str = "binary:logistic",  # NEW: Objective parameter
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bin = max_bin
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        if objective not in ["binary:logistic", "reg:squarederror"]:
            raise ValueError(
                "Objective must be 'binary:logistic' or 'reg:squarederror'"
            )
        self.objective = objective
        self.model: TreeEnsemble | None = None

    def fit(self, x: jnp.ndarray, y: jnp.ndarray):
        """Fits the model to the training data based on the objective."""
        self.model = fit_tree_ensemble(
            x,
            y,
            self.n_estimators,
            self.max_bin,
            self.learning_rate,
            self.max_depth,
            self.reg_lambda,
            self.gamma,
            self.min_child_weight,
            self.objective,
        )

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Makes predictions.
        - For 'binary:logistic', returns probabilities.
        - For 'reg:squarederror', returns predicted values.
        """
        if self.model is None:
            raise RuntimeError(
                "The model has not been fitted yet. Please call .fit() first."
            )
        return predict_ensemble(x, self.model, self.learning_rate, self.objective)

    def predict_class(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Predicts class labels (0 or 1). Only for 'binary:logistic' objective.
        """
        if self.objective != "binary:logistic":
            raise RuntimeError(
                "predict_class is only available for 'binary:logistic' objective."
            )
        probabilities = self.predict(x)
        return (probabilities > 0.5).astype(int)


# ==============================================================================
# Part 4: Debug Functions for Tree Visualization and Leaf Prediction
# ==============================================================================


def pretty_print_ensemble(ensemble: TreeEnsemble):
    """
    Prints a TreeEnsemble object in a human-readable format.

    Args:
        ensemble (TreeEnsemble): The trained TreeEnsemble object to print.
    """
    print("\n" + "=" * 25 + " Tree Ensemble Details " + "=" * 25)

    # --- 1. Print Ensemble-level information ---
    print("\n[Ensemble Info]")
    print(f"  - Max Depth: {ensemble.max_depth}")
    print(f"  - Initial Prediction: {ensemble.initial_prediction:.6f}")
    print(f"  - Number of Features: {ensemble.bins.shape[0]}")
    print(f"  - Number of Bins per Feature: {ensemble.bins.shape[1] + 1}")

    # --- 2. Iterate through and print each tree ---
    print(f"\nNumber of Trees: {len(ensemble.trees)}")

    for i, tree in enumerate(ensemble.trees):
        print("\n" + "-" * 20 + f" Tree {i} " + "-" * 20)

        n_nodes = len(tree.feature)
        max_depth = ensemble.max_depth

        print(f"  Total Nodes: {n_nodes}")
        print(f"  Max Depth: {max_depth}")

        # Print node-by-node details
        print("\n  Node Details:")
        print(
            f"  {'Node':>4} {'Depth':>5} {'Type':>6} {'Feature':>7} {'Threshold':>12} {'Value':>12}"
        )
        print(f"  {'-' * 4} {'-' * 5} {'-' * 6} {'-' * 7} {'-' * 12} {'-' * 12}")

        for node_idx in range(n_nodes):
            # Calculate depth from node index (assuming complete binary tree indexing)
            depth = int(jnp.floor(jnp.log2(node_idx + 1))) if node_idx > 0 else 0

            if tree.is_leaf[node_idx]:
                node_type = "Leaf"
                feature_str = "-"
                threshold_str = "-"
                value_str = f"{tree.value[node_idx]:.6f}"
            else:
                node_type = "Split"
                feature_str = f"{tree.feature[node_idx]}"
                threshold_str = f"{tree.threshold[node_idx]:.6f}"
                value_str = f"{tree.value[node_idx]:.6f}"

            print(
                f"  {node_idx:>4} {depth:>5} {node_type:>6} {feature_str:>7} {threshold_str:>12} {value_str:>12}"
            )

    print("\n" + "=" * 70)


@jax.jit
def predict_tree_leaves(x: jnp.ndarray, tree: Tree) -> jnp.ndarray:
    """
    Predicts which leaf node each sample reaches in a single tree.

    Args:
        x (jnp.ndarray): Input samples. Shape (m, n).
        tree (Tree): The decision tree.

    Returns:
        jnp.ndarray: Array where result[i] is the leaf node index that sample i reaches.
                     Shape (m,).
    """

    def predict_leaf_single(sample_x: jnp.ndarray) -> jnp.ndarray:
        def go_left(node_idx: jnp.ndarray) -> jnp.ndarray:
            return 2 * node_idx + 1

        def go_right(node_idx: jnp.ndarray) -> jnp.ndarray:
            return 2 * node_idx + 2

        def body(state: tuple[jnp.ndarray, Tree]) -> tuple[jnp.ndarray, Tree]:
            cur_node, model = state
            feature_idx, threshold = model.feature[cur_node], model.threshold[cur_node]
            next_node = jax.lax.cond(
                sample_x[feature_idx] <= threshold, go_left, go_right, cur_node
            )
            return (next_node, model)

        def condition(state: tuple[jnp.ndarray, Tree]) -> jnp.ndarray:
            return ~tree.is_leaf[state[0]]

        final_node, _ = jax.lax.while_loop(condition, body, (jnp.int32(0), tree))
        return final_node

    return jax.vmap(predict_leaf_single)(x)


def predict_ensemble_leaves(x: jnp.ndarray, ensemble: TreeEnsemble) -> jnp.ndarray:
    """
    Predicts which leaf nodes samples reach for all trees in the ensemble.

    Args:
        x (jnp.ndarray): Input samples. Shape (m, n).
        ensemble (TreeEnsemble): The trained ensemble.

    Returns:
        jnp.ndarray: Array where result[tree_idx, sample_idx] is the leaf node index
                     that sample_idx reaches in tree_idx. Shape (n_trees, m).
    """
    leaf_predictions = []
    for tree in ensemble.trees:
        leaves = predict_tree_leaves(x, tree)
        leaf_predictions.append(leaves)

    return jnp.stack(leaf_predictions, axis=0)


def print_leaf_predictions(
    x: jnp.ndarray,
    ensemble: TreeEnsemble,
    y_true: jnp.ndarray | None = None,
    max_samples: int = 10,
    learning_rate: float = 0.1,
    objective: str = "binary:logistic",
):
    """
    Prints detailed leaf prediction information for debugging.

    Args:
        x (jnp.ndarray): Input samples. Shape (m, n).
        ensemble (TreeEnsemble): The trained ensemble.
        y_true (Optional[jnp.ndarray]): True labels for comparison. Shape (m,).
        max_samples (int): Maximum number of samples to print details for.
        learning_rate (float): Learning rate used in training.
        objective (str): Objective function used in training.
    """
    # Get predictions and leaf paths
    y_pred = predict_ensemble(x, ensemble, learning_rate, objective)
    leaf_predictions = predict_ensemble_leaves(x, ensemble)

    n_samples = min(x.shape[0], max_samples)
    n_trees = len(ensemble.trees)

    print(f"\n📊 Leaf Node Predictions (Shape: {leaf_predictions.shape}):")
    print("=" * 80)

    for sample_idx in range(n_samples):
        print(f"Sample {sample_idx:2d}:")

        if y_true is not None:
            gt = int(y_true[sample_idx])
            pred_prob = y_pred[sample_idx]
            pred_class = (
                int(pred_prob > 0.5) if objective == "binary:logistic" else pred_prob
            )
            if objective == "binary:logistic":
                print(
                    f"  Ground Truth: {gt}, Predicted: {pred_class}, Probability: {pred_prob:.3f}"
                )
            else:
                print(f"  Ground Truth: {gt:.3f}, Predicted: {pred_prob:.3f}")
        else:
            pred_prob = y_pred[sample_idx]
            if objective == "binary:logistic":
                pred_class = int(pred_prob > 0.5)
                print(f"  Predicted: {pred_class}, Probability: {pred_prob:.3f}")
            else:
                print(f"  Predicted: {pred_prob:.3f}")

        print("  Tree → Leaf Nodes:")
        for tree_idx in range(n_trees):
            leaf_node = leaf_predictions[tree_idx, sample_idx]
            leaf_value = ensemble.trees[tree_idx].value[leaf_node]
            print(f"    Tree {tree_idx}: Node {leaf_node} (Value: {leaf_value:.6f})")

        print("─" * 50)


def count_samples_per_node(x: jnp.ndarray, tree: Tree) -> jnp.ndarray:
    """
    Counts the number of samples that reach each node in a single tree.

    Args:
        x (jnp.ndarray): Input samples. Shape (m, n).
        tree (Tree): The decision tree.

    Returns:
        jnp.ndarray: Array where result[i] is the number of samples reaching node i.
                     Shape (n_nodes,).
    """
    leaf_nodes = predict_tree_leaves(x, tree)
    n_nodes = len(tree.feature)

    # Use bincount to count occurrences of each node index
    counts = jnp.bincount(leaf_nodes, length=n_nodes)
    return counts


def count_samples_per_node_ensemble(
    x: jnp.ndarray, ensemble: TreeEnsemble
) -> jnp.ndarray:
    """
    Counts the number of samples that reach each node in all trees of the ensemble.

    Args:
        x (jnp.ndarray): Input samples. Shape (m, n).
        ensemble (TreeEnsemble): The trained ensemble.

    Returns:
        jnp.ndarray: Array where result[tree_idx, node_idx] is the number of samples
                     reaching node_idx in tree_idx. Shape (n_trees, n_nodes).
    """
    counts = []
    for tree in ensemble.trees:
        tree_counts = count_samples_per_node(x, tree)
        counts.append(tree_counts)

    return jnp.stack(counts, axis=0)


def print_node_sample_counts(
    x: jnp.ndarray,
    ensemble: TreeEnsemble,
    tree_idx: int | None = None,
    show_only_leaves: bool = False,
):
    """
    Prints a summary of sample counts for each node in the ensemble or a specific tree.

    Args:
        x (jnp.ndarray): Input samples. Shape (m, n).
        ensemble (TreeEnsemble): The trained ensemble.
        tree_idx (Optional[int]): If specified, only show counts for this tree index.
                                  If None, show counts for all trees.
        show_only_leaves (bool): If True, only show leaf nodes. If False, show all nodes.
    """
    if tree_idx is not None:
        # Show counts for a specific tree
        if tree_idx < 0 or tree_idx >= len(ensemble.trees):
            print(
                f"❌ Invalid tree index: {tree_idx}. Available: 0-{len(ensemble.trees) - 1}"
            )
            return

        tree = ensemble.trees[tree_idx]
        counts = count_samples_per_node(x, tree)

        print(f"\n📊 Sample Counts for Tree {tree_idx}:")
        print("=" * 60)
        print(f"  {'Node':>4} {'Type':>6} {'Count':>8} {'Percentage':>12}")
        print(f"  {'-' * 4} {'-' * 6} {'-' * 8} {'-' * 12}")

        total_samples = x.shape[0]
        for node_idx in range(len(tree.feature)):
            if show_only_leaves and not tree.is_leaf[node_idx]:
                continue

            node_type = "Leaf" if tree.is_leaf[node_idx] else "Split"
            count = counts[node_idx]
            percentage = (count / total_samples) * 100

            print(f"  {node_idx:>4} {node_type:>6} {count:>8} {percentage:>11.2f}%")
    else:
        # Show counts for all trees
        counts_all = count_samples_per_node_ensemble(x, ensemble)

        print("\n📊 Sample Counts for All Trees:")
        print("=" * 80)

        for i, tree in enumerate(ensemble.trees):
            print(f"\nTree {i}:")
            print(f"  {'Node':>4} {'Type':>6} {'Count':>8} {'Percentage':>12}")
            print(f"  {'-' * 4} {'-' * 6} {'-' * 8} {'-' * 12}")

            total_samples = x.shape[0]
            counts = counts_all[i]

            for node_idx in range(len(tree.feature)):
                if show_only_leaves and not tree.is_leaf[node_idx]:
                    continue

                node_type = "Leaf" if tree.is_leaf[node_idx] else "Split"
                count = counts[node_idx]
                percentage = (count / total_samples) * 100

                if count > 0:  # Only show nodes with samples
                    print(
                        f"  {node_idx:>4} {node_type:>6} {count:>8} {percentage:>11.2f}%"
                    )
