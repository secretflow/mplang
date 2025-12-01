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

import os
import sys
import time
import unittest
from contextlib import contextmanager

import jax.numpy as jnp
import xgboost as xgb
from hist_jax import (
    XGBoostJAX,
    _compute_best_split_per_node,
    build_bins_vmapped,
    build_histogram,
    compute_best_split,
    compute_indices_vmapped,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


def _calculate_gain_naively(G, H, reg_lambda, gamma, min_child_weight):
    """A simple, unoptimized reference implementation to calculate best split."""
    n_features, n_bins = G.shape
    best_gain = -jnp.inf
    best_feat = -1
    best_thresh_idx = -1

    for i in range(n_features):
        G_total = jnp.sum(G[i, :])
        H_total = jnp.sum(H[i, :])

        if H_total < min_child_weight:
            continue

        score_parent = G_total**2 / (H_total + reg_lambda)

        G_left, H_left = 0.0, 0.0
        for j in range(n_bins - 1):  # Iterate through all possible split points
            G_left += G[i, j]
            H_left += H[i, j]

            if H_left < min_child_weight:
                continue

            G_right = G_total - G_left
            H_right = H_total - H_left

            if H_right < min_child_weight:
                continue

            score_left = G_left**2 / (H_left + reg_lambda)
            score_right = G_right**2 / (H_right + reg_lambda)

            current_gain = 0.5 * (score_left + score_right - score_parent) - gamma

            if current_gain > best_gain:
                best_gain = current_gain
                best_feat = i
                best_thresh_idx = j

    return best_gain, best_feat, best_thresh_idx


@contextmanager
def suppress_stdout():
    """
    A context manager to temporarily suppress stdout.

    Usage:
        with suppress_stdout():
            print("This will not be seen")
        print("This will be seen")
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    original_stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)

    try:
        os.dup2(devnull_fd, original_stdout_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.close(devnull_fd)
        os.close(saved_stdout_fd)


class TestOptimizedXGBoostFunctions(unittest.TestCase):
    """
    Unit tests for the optimized JAX-based XGBoost functions.
    """

    def test_build_bins(self):
        """Tests bin creation for both varying and constant features."""
        x = jnp.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
        max_bins = 5
        # The function build_bins_equi_width is vmapped
        bins = build_bins_vmapped(x, max_bins)

        expected_splits_0 = jnp.linspace(1.0, 5.0, num=max_bins + 1)[1:-1]
        expected_splits_1 = jnp.full(
            max_bins - 1, jnp.inf
        )  # Constant features return inf splits

        self.assertEqual(bins.shape, (2, max_bins - 1))
        self.assertTrue(jnp.allclose(bins[0], expected_splits_0))
        self.assertTrue(jnp.allclose(bins[1], expected_splits_1))

    def test_compute_bin_indices(self):
        """Tests the mapping of values to bin indices, including edge cases."""
        x = jnp.array([[0.5, 8.0], [2.5, 5.0], [5.5, 3.0]])
        # Splits for feature 0: 2.0, 3.0, 4.0
        # Splits for feature 1: 4.0, 5.0, 6.0
        bins = jnp.array([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])

        bin_indices = compute_indices_vmapped(x, bins)

        # Feat 0: 0.5(<2)->0; 2.5(>2,<=3)->1; 5.5(>4)->3 (indices 0,1,2,3 for 4 bins)
        # Feat 1: 8.0(>6)->3; 5.0(>4,<=5)->1; 3.0(<4)->0
        expected_indices = jnp.array([[0, 3], [1, 1], [3, 0]])
        self.assertTrue(jnp.array_equal(bin_indices, expected_indices))

    def test_build_histogram_optimized(self):
        """Tests the fully vectorized histogram building."""
        g = jnp.array([1.0, 2.0, 3.0, 4.0])
        h = jnp.array([0.1, 0.2, 0.3, 0.4])
        bt_local = jnp.array([0, 0, 1, 1])
        bin_indices = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        t, k = 2, 2

        GH_hist = build_histogram(g, h, bt_local, bin_indices, t, k)
        self.assertEqual(GH_hist.shape, (2, 2, 2, 2))

        expected_G_hist = jnp.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 1.0], [4.0, 3.0]],
        ])
        expected_H_hist = jnp.array([
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.1], [0.4, 0.3]],
        ])

        self.assertTrue(jnp.allclose(GH_hist[..., 0], expected_G_hist))
        self.assertTrue(jnp.allclose(GH_hist[..., 1], expected_H_hist))

    def test_compute_best_split_per_node_optimized(self):
        """Tests the gain calculation for a single node from a combined GH histogram."""
        GH_node = jnp.array([
            [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
            [[4.0, 2.0], [5.0, 2.5], [6.0, 3.0]],
        ])
        reg_lambda, gamma, min_child_weight = 1.0, 0.0, 1.0

        best_gain, best_feat, best_thresh_idx = _compute_best_split_per_node(
            GH_node, reg_lambda, gamma, min_child_weight
        )

        # Get result from the naive "golden standard" function
        G, H = GH_node[..., 0], GH_node[..., 1]
        naive_gain, naive_feat, naive_thresh_idx = _calculate_gain_naively(
            G, H, reg_lambda, gamma, min_child_weight
        )

        self.assertTrue(jnp.allclose(best_gain, naive_gain))
        self.assertEqual(best_feat, naive_feat)
        self.assertEqual(best_thresh_idx, naive_thresh_idx)

    def test_compute_best_split_optimized(self):
        """Tests the vectorized best split computation over multiple nodes."""
        GH_hist = jnp.array([
            [[[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]],
            [[[10.0, 5.0], [20.0, 10.0], [30.0, 15.0]]],
        ]).transpose((1, 0, 2, 3))

        reg_lambda, gamma, min_child_weight = 1.0, 0.0, 0.0

        max_gains, best_features, best_threshold_idxs = compute_best_split(
            GH_hist, reg_lambda, gamma, min_child_weight
        )

        self.assertEqual(max_gains.shape, (2,))
        self.assertEqual(best_features.shape, (2,))
        self.assertEqual(best_threshold_idxs.shape, (2,))

        _, feat0, idx0 = _compute_best_split_per_node(
            GH_hist[:, 0, :, :], reg_lambda, gamma, min_child_weight
        )
        self.assertEqual(best_features[0], feat0)
        self.assertEqual(best_threshold_idxs[0], idx0)

        _, feat1, idx1 = _compute_best_split_per_node(
            GH_hist[:, 1, :, :], reg_lambda, gamma, min_child_weight
        )
        self.assertEqual(best_features[1], feat1)
        self.assertEqual(best_threshold_idxs[1], idx1)

    def test_classification(self):
        """Encapsulated test for the classification task."""
        print("\n" + "=" * 50 + "\n--- Running Classification Test ---\n" + "=" * 50)
        # Load and preprocess data
        X_np, y_np = make_classification(
            n_samples=50_000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42,
        )
        X = jnp.array(X_np, dtype=jnp.float32)
        y = jnp.array(y_np, dtype=jnp.float32)

        # Model hyperparameters
        params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 3,
            "max_bin": 64,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "min_child_weight": 1.0,
        }

        print("--- 1. Testing JAX Classifier ---")
        model_jax = XGBoostJAX(**params, objective="binary:logistic")

        start_run = time.time()
        model_jax.fit(X, y)
        with suppress_stdout():
            print(model_jax.model)
        print(
            f"JAX Classifier compiled + first fit in: {time.time() - start_run:.4f} seconds"
        )

        start_run = time.time()
        model_jax.fit(X, y)
        with suppress_stdout():
            print(model_jax.model)
        print(f"JAX Classifier execution time: {time.time() - start_run:.4f} seconds")

        start_run = time.time()
        with suppress_stdout():
            print(model_jax.predict(X))
        print(
            f"JAX Classifier prediction compiled time: {time.time() - start_run:.4f} seconds"
        )

        start_run = time.time()
        y_pred_proba = model_jax.predict(X)
        with suppress_stdout():
            print(y_pred_proba)
        print(f"JAX Classifier prediction time: {time.time() - start_run:.4f} seconds")

        accuracy = accuracy_score(y, (y_pred_proba > 0.5).astype(int))
        auc = roc_auc_score(y, y_pred_proba)
        print(f"JAX Model -> Accuracy: {accuracy:.4f}, AUC: {auc:.4f}\n")

        print("--- 2. Testing Official XGBoost for comparison ---")
        xgb_params = params.copy()
        model_xgb = xgb.XGBClassifier(
            **xgb_params,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
        )

        start_run = time.time()
        model_xgb.fit(X_np, y_np)
        print(f"Official XGBoost fit time: {time.time() - start_run:.4f} seconds")
        y_pred_proba_xgb = model_xgb.predict_proba(X_np)[:, 1]
        accuracy_xgb = accuracy_score(y_np, (y_pred_proba_xgb > 0.5).astype(int))
        auc_xgb = roc_auc_score(y_np, y_pred_proba_xgb)
        print(f"Official XGBoost -> Accuracy: {accuracy_xgb:.4f}, AUC: {auc_xgb:.4f}\n")

    def test_regression(self):
        """Encapsulated test for the regression task with comparison."""
        print("\n" + "=" * 50 + "\n--- Running Regression Test ---\n" + "=" * 50)

        # Fix: make_regression may return 3 values, we only need X and y
        result = make_regression(
            n_samples=50000,
            n_features=20,
            n_informative=10,
            noise=25.0,
            random_state=42,
        )
        X_np, y_np = result[0], result[1]
        X, y = jnp.array(X_np, dtype=jnp.float32), jnp.array(y_np, dtype=jnp.float32)

        params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 3,
            "max_bin": 64,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "min_child_weight": 1.0,
        }

        print("--- 1. Testing JAX Regressor ---")
        model_jax = XGBoostJAX(**params, objective="reg:squarederror")

        start_run = time.time()
        model_jax.fit(X, y)
        print(
            f"JAX Regressor compiled + first fit in: {time.time() - start_run:.4f} seconds"
        )
        start_run = time.time()
        model_jax.fit(X, y)
        print(f"JAX Regressor execution time: {time.time() - start_run:.4f} seconds")

        y_pred_jax = model_jax.predict(X)
        mse_jax = mean_squared_error(y, y_pred_jax)
        print(f"JAX Regressor MSE: {mse_jax:.4f}\n")

        print("--- 2. Testing Official XGBoost Regressor for comparison ---")
        xgb_params = params.copy()
        model_xgb = xgb.XGBRegressor(
            **xgb_params,
            tree_method="hist",
            objective="reg:squarederror",
            eval_metric="rmse",
        )

        start_run = time.time()
        model_xgb.fit(X_np, y_np)
        print(f"Official XGBoost fit time: {time.time() - start_run:.4f} seconds")
        y_pred_xgb = model_xgb.predict(X_np)
        mse_xgb = mean_squared_error(y_np, y_pred_xgb)
        print(f"Official XGBoost MSE: {mse_xgb:.4f}\n")

    def test_debug_functions(self):
        """Tests the debug functions for tree visualization and leaf prediction."""
        print("\n" + "=" * 50 + "\n--- Testing Debug Functions ---\n" + "=" * 50)

        # Import the debug functions
        from hist_jax import (
            count_samples_per_node,
            count_samples_per_node_ensemble,
            predict_ensemble_leaves,
            predict_tree_leaves,
            pretty_print_ensemble,
            print_leaf_predictions,
            print_node_sample_counts,
        )

        # Create a small dataset for testing
        X_np, y_np = make_classification(
            n_samples=21,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_classes=2,
            random_state=42,
        )
        X = jnp.array(X_np, dtype=jnp.float32)
        y = jnp.array(y_np, dtype=jnp.float32)

        # Train a simple model
        params = {
            "n_estimators": 2,
            "learning_rate": 0.1,
            "max_depth": 2,
            "max_bin": 4,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "min_child_weight": 1.0,
        }

        model = XGBoostJAX(**params, objective="binary:logistic")
        model.fit(X, y)

        # Ensure model is fitted
        assert model.model is not None, "Model should be fitted"

        print("--- 1. Testing pretty_print_ensemble ---")
        pretty_print_ensemble(model.model)

        print("\n--- 2. Testing predict_tree_leaves ---")
        tree = model.model.trees[0]
        leaf_nodes = predict_tree_leaves(X[:5], tree)  # Test with first 5 samples
        print(f"Leaf nodes for first 5 samples in tree 0: {leaf_nodes}")

        # Verify the shape
        self.assertEqual(leaf_nodes.shape, (5,))
        # Verify all leaf nodes are valid indices
        max_node_idx = len(tree.feature) - 1
        self.assertTrue(jnp.all(leaf_nodes >= 0))
        self.assertTrue(jnp.all(leaf_nodes <= max_node_idx))

        print("\n--- 3. Testing predict_ensemble_leaves ---")
        ensemble_leaves = predict_ensemble_leaves(X[:5], model.model)
        print(f"Ensemble leaf shape: {ensemble_leaves.shape}")

        # Verify the shape: (n_trees, n_samples)
        self.assertEqual(ensemble_leaves.shape, (2, 5))

        print("\n--- 4. Testing print_leaf_predictions ---")
        print_leaf_predictions(
            X,
            model.model,
            y,
            max_samples=5,
            learning_rate=params["learning_rate"],
            objective="binary:logistic",
        )

        print("\n--- 5. Testing count_samples_per_node ---")
        tree_counts = count_samples_per_node(X, tree)
        print(f"Sample counts for tree 0: {tree_counts}")

        # Verify the shape: should match number of nodes in tree
        n_nodes = len(tree.feature)
        self.assertEqual(tree_counts.shape, (n_nodes,))

        # Verify total count equals number of samples (all samples should reach some leaf)
        total_count = jnp.sum(tree_counts)
        self.assertEqual(total_count, X.shape[0])

        # Verify only leaf nodes have non-zero counts
        for node_idx in range(n_nodes):
            if tree.is_leaf[node_idx]:
                # Leaf nodes can have zero or more samples
                self.assertTrue(tree_counts[node_idx] >= 0)
            else:
                # Split nodes should have zero samples (samples only stay at leaves)
                self.assertEqual(tree_counts[node_idx], 0)

        print("\n--- 6. Testing count_samples_per_node_ensemble ---")
        ensemble_counts = count_samples_per_node_ensemble(X, model.model)
        print(f"Ensemble counts shape: {ensemble_counts.shape}")

        # Verify the shape: (n_trees, n_nodes)
        n_trees = len(model.model.trees)
        self.assertEqual(ensemble_counts.shape, (n_trees, n_nodes))

        # Verify each tree's total count equals number of samples
        for tree_idx in range(n_trees):
            tree_total = jnp.sum(ensemble_counts[tree_idx])
            self.assertEqual(tree_total, X.shape[0])

        # Verify consistency with single tree function
        tree_0_counts_individual = count_samples_per_node(X, model.model.trees[0])
        tree_0_counts_ensemble = ensemble_counts[0]
        self.assertTrue(jnp.allclose(tree_0_counts_individual, tree_0_counts_ensemble))

        print("\n--- 7. Testing print_node_sample_counts ---")
        # Test for specific tree
        print("\n7a. Printing counts for Tree 0 (leaves only):")
        print_node_sample_counts(X, model.model, tree_idx=0, show_only_leaves=True)

        print("\n7b. Printing counts for Tree 0 (all nodes):")
        print_node_sample_counts(X, model.model, tree_idx=0, show_only_leaves=False)

        print("\n7c. Printing counts for all trees (leaves only):")
        print_node_sample_counts(X, model.model, tree_idx=None, show_only_leaves=True)

        # Test error handling for invalid tree index
        print("\n7d. Testing invalid tree index:")
        print_node_sample_counts(X, model.model, tree_idx=99, show_only_leaves=True)

        print("\n--- 8. Advanced validation ---")
        # Cross-validation: ensure consistency between different methods

        # Method 1: Using predict_tree_leaves + manual counting
        leaves_method1 = predict_tree_leaves(X, tree)
        unique_leaves, counts_method1 = jnp.unique(leaves_method1, return_counts=True)

        # Method 2: Using count_samples_per_node
        counts_method2 = count_samples_per_node(X, tree)

        # Verify that non-zero counts match
        for i, leaf_idx in enumerate(unique_leaves):
            expected_count = counts_method1[i]
            actual_count = counts_method2[leaf_idx]
            self.assertEqual(
                expected_count,
                actual_count,
                f"Mismatch for leaf {leaf_idx}: expected {expected_count}, got {actual_count}",
            )

        print("✅ All sample counting validations passed!")

        print("\n--- Debug Functions Test Completed Successfully ---")


if __name__ == "__main__":
    unittest.main()
