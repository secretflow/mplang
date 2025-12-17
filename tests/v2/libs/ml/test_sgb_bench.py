#!/usr/bin/env python3
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
"""
Benchmark SecureBoost v2 on Home Credit Default Risk dataset and OpenML datasets.

This script benchmarks the performance of SecureBoost (SGB) running on a simplified
MPC simulator against a plaintext XGBoost baseline.

Dataset Sources:
- Kaggle Home Credit: https://www.kaggle.com/c/home-credit-default-risk
- OpenML: https://www.openml.org/ (e.g., Covertype)

Usage:
  # Standalone run (command line)
  python sgb_bench.py --dataset openml:covertype --estimators 10 --depth 3

  # Via CLI submit (uses __mp_main__)
  cli submit examples/v2/sgb_bench.py
"""

import argparse
import time
import urllib.error

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import mplang.v2 as mp
from mplang.v2.libs.ml.sgb import SecureBoost


def to_np(d):
    """Convert data to numpy float32 array."""
    return d.values.astype(np.float32) if hasattr(d, "values") else d.astype(np.float32)


def load_backend_or_exit():
    """Load the BFV backend or exit if it fails."""
    from mplang.v2.backends import load_backend

    try:
        load_backend("mplang.v2.backends.bfv_impl")
        print("✓ BFV backend loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to load BFV backend: {e}")
        import sys

        sys.exit(1)


def load_data(path_or_id: str, sample_size: int | None = None):
    """
    Load and preprocess data from local CSV or OpenML.

    Args:
        path_or_id: File path or 'openml:<name/id>'
        sample_size: Optional number of rows to sample

    Returns:
        X (pd.DataFrame or np.ndarray), y (np.ndarray)
    """
    print(f"Loading data from {path_or_id}...")

    # 1. Fetch Data
    if path_or_id.startswith("openml:"):
        dataset_name = path_or_id.split(":", 1)[1]

        # Robust handling for Covertype
        if dataset_name.lower() in ["covertype", "covtype", "1597", "150"]:
            print(f"Fetching OpenML dataset: {dataset_name} (using fetch_covtype)")
            from sklearn.datasets import fetch_covtype

            bunch = fetch_covtype()
            X, y = bunch.data, bunch.target
        else:
            print(f"Fetching OpenML dataset: {dataset_name}")
            try:
                # Try by name, then ID
                bunch = fetch_openml(name=dataset_name, as_frame=True, parser="auto")
            except (ValueError, urllib.error.URLError):
                if dataset_name.isdigit():
                    bunch = fetch_openml(
                        data_id=int(dataset_name), as_frame=True, parser="auto"
                    )
                else:
                    raise ValueError(
                        f"Could not fetch OpenML dataset: {dataset_name}"
                    ) from None
            X, y = bunch.data, bunch.target

        # Helper: Convert categorical target
        if hasattr(y, "dtype") and (y.dtype == "object" or str(y.dtype) == "category"):
            y = LabelEncoder().fit_transform(y)
    else:
        # Local CSV
        df = pd.read_csv(path_or_id)
        if "TARGET" not in df.columns:
            raise ValueError("Column 'TARGET' not found in local CSV.")
        y = df["TARGET"].values
        X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

    # 2. Sample if requested
    if sample_size and sample_size < len(X):
        print(f"Sampling {sample_size} rows...")
        if isinstance(X, pd.DataFrame):
            X = X.sample(n=sample_size, random_state=42)
            y = y[X.index]
        else:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
            y = y[indices]

    # 3. Preprocess (Missing Values & Encoding)
    # Convert to DataFrame for easier preprocessing if it's not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    print("Preprocessing: handling missing values & encoding...")
    le = LabelEncoder()
    for col in X.columns:
        if pd.api.types.is_object_dtype(
            X[col].dtype
        ) or pd.api.types.is_categorical_dtype(X[col].dtype):
            X[col] = X[col].fillna(X[col].mode()[0])
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
        else:
            X[col] = X[col].fillna(X[col].median())

    # 4. Handle Target (Binarize if multiclass)
    unique_y = np.unique(y)
    if len(unique_y) > 2:
        print(
            f"Dataset has {len(unique_y)} classes. Binarizing (Class {unique_y[0]} vs Rest)..."
        )
        y = (y == unique_y[0]).astype(np.float32)
    elif set(unique_y) == {1, 2}:
        y = y - 1

    y = y.astype(np.float32)
    return X, y


def run_sgb_benchmark(
    sim,
    X_train,
    X_test,
    y_train,
    y_test,
    params,
    pp_feature_ratio=0.5,
):
    """
    Train and evaluate SecureBoost using MPLang simulation.

    Args:
        sim: MPLang Simulator instance
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        params: Dict with 'estimators', 'depth', 'max_bin' keys
        pp_feature_ratio: Ratio of features assigned to passive party

    Returns:
        (auc, total_time) tuple
    """
    print("\n" + "=" * 60)
    print("Starting Training (SecureBoost v2)")
    print("=" * 60)

    # Split features for Active Party (AP) and Passive Party (PP)
    n_features = X_train.shape[1]
    n_pp = int(n_features * pp_feature_ratio)
    n_ap = n_features - n_pp
    print(f"Feature split: AP={n_ap}, PP={n_pp}")

    # Ensure numpy arrays (float32)
    X_train_np, X_test_np = to_np(X_train), to_np(X_test)

    X_ap_train, X_pp_train = X_train_np[:, :n_ap], X_train_np[:, n_ap:]
    X_ap_test, X_pp_test = X_test_np[:, :n_ap], X_test_np[:, n_ap:]

    n_samples_train = len(y_train)
    n_samples_test = len(y_test)

    # Define the job function for tracing
    def job():
        # Place data on respective parties
        d_ap = mp.put("P0", X_ap_train)
        d_y = mp.put("P0", y_train)
        d_pp = mp.put("P1", X_pp_train)

        model = SecureBoost(
            n_estimators=params["estimators"],
            max_depth=params["depth"],
            max_bin=params.get("max_bin", 8),
            learning_rate=0.1,
            ap_rank=0,
            pp_ranks=[1],
        )

        model.fit(
            [d_ap, d_pp],
            d_y,
            n_samples=n_samples_train,
            n_features_per_party=[n_ap, n_pp],
        )

        # Predict on test set
        d_ap_test = mp.put("P0", X_ap_test)
        d_pp_test = mp.put("P1", X_pp_test)
        y_prob = model.predict([d_ap_test, d_pp_test], n_samples=n_samples_test)

        return y_prob

    # --- 1. Compile (Trace) ---
    start_time = time.perf_counter()
    traced = mp.compile(job, context=sim)
    trace_time = time.perf_counter() - start_time
    print(
        f"Graph tracing finished in {trace_time:.2f}s ({len(traced.graph.operations)} ops)"
    )

    # --- 2. Execute Graph ---
    print("\nExecuting graph (Simulating 2 parties)...")
    exec_start = time.perf_counter()
    y_prob_obj = mp.evaluate(traced, context=sim)
    exec_time = time.perf_counter() - exec_start
    print(f"Execution finished in {exec_time:.2f}s")

    # --- 3. Fetch and Evaluate ---
    y_pred = mp.fetch(y_prob_obj)
    if isinstance(y_pred, list):
        y_pred = y_pred[0]

    if np.isnan(y_pred).any():
        print("WARNING: Predictions contain NaNs. Filling with 0.")
        y_pred = np.nan_to_num(y_pred)

    auc = roc_auc_score(y_test, y_pred)
    total_time = trace_time + exec_time

    return auc, total_time


def train_eval_xgboost(X_train, X_test, y_train, y_test, params):
    """
    Train and evaluate a vanilla XGBoost baseline.
    """
    print("\n" + "=" * 60)
    print("Starting Training (XGBoost Baseline)")
    print("=" * 60)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost not installed. Skipping baseline.")
        return None, 0.0

    # Ensure numpy
    X_train_np, X_test_np = to_np(X_train), to_np(X_test)

    start = time.perf_counter()
    model = XGBClassifier(
        n_estimators=params["estimators"],
        max_depth=params["depth"],
        learning_rate=0.1,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train_np, y_train)
    elapsed = time.perf_counter() - start

    y_pred = model.predict_proba(X_test_np)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    print(f"XGBoost training finished in {elapsed:.2f}s")
    return auc, elapsed


def run_benchmark_demo(sim):
    """Demo benchmark with synthetic data for CLI submit.

    This is the entry point for `cli submit examples/v2/sgb_bench.py`.
    """
    print("=" * 60)
    print("SecureBoost Benchmark Demo (Synthetic Data)")
    print("=" * 60)

    load_backend_or_exit()

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 20

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] + X[:, 5] > 0).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {"estimators": 3, "depth": 3, "max_bin": 8}

    print(f"Samples: train={len(y_train)}, test={len(y_test)}")
    print(f"Features: {n_features}")
    print(f"Params: {params}")

    # Run SecureBoost
    sgb_auc, sgb_time = run_sgb_benchmark(sim, X_train, X_test, y_train, y_test, params)

    # Run XGBoost Baseline
    xgb_auc, xgb_time = train_eval_xgboost(X_train, X_test, y_train, y_test, params)

    # Report
    print("\n" + "=" * 60)
    print("DEMO RESULTS")
    print("=" * 60)
    print(f"SecureBoost AUC: {sgb_auc:.4f}  (Time: {sgb_time:.2f}s)")

    if xgb_auc is not None:
        print(f"XGBoost AUC:     {xgb_auc:.4f}  (Time: {xgb_time:.2f}s)")
        print(f"Diff (SGB-XGB):  {sgb_auc - xgb_auc:.4f}")
    print("=" * 60)


# Entry point for CLI submit
__mp_main__ = run_benchmark_demo


def main():
    """Main entry point for standalone command-line execution."""
    parser = argparse.ArgumentParser(description="SecureBoost Benchmark")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to CSV or 'openml:<name>'"
    )
    parser.add_argument("--estimators", type=int, default=10, help="Number of trees")
    parser.add_argument("--depth", type=int, default=3, help="Max depth")
    parser.add_argument("--sample", type=int, default=None, help="Sample size")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    load_backend_or_exit()

    # Create simulator
    sim = mp.make_simulator(2)

    if args.dataset is None:
        # Run demo with synthetic data
        run_benchmark_demo(sim)
    else:
        # Run with specified dataset
        X, y = load_data(args.dataset, args.sample)
        print(f"Total samples: {len(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        params = {"estimators": args.estimators, "depth": args.depth}

        # Run SecureBoost
        sgb_auc, sgb_time = run_sgb_benchmark(
            sim, X_train, X_test, y_train, y_test, params
        )

        # Run XGBoost Baseline
        xgb_auc, xgb_time = train_eval_xgboost(X_train, X_test, y_train, y_test, params)

        # Report
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"SecureBoost AUC: {sgb_auc:.4f}  (Time: {sgb_time:.2f}s)")

        if xgb_auc is not None:
            print(f"XGBoost AUC:     {xgb_auc:.4f}  (Time: {xgb_time:.2f}s)")
            print(f"Diff (SGB-XGB):  {sgb_auc - xgb_auc:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
