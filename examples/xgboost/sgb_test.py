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

import time

import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

import mplang as mp
from examples.xgboost.sgb import (
    SecureBoost,
    batch_feature_wise_bucket_sum_mplang,
    pretty_print_ensemble,
)
from mplang.ops import phe


def load_dataset(
    n_samples=400,
    n_total_features=10,
    n_features_ap=4,
    pp_parties=1,
    random_state=42,
):
    """
    Generates and preprocesses a synthetic dataset for vertical federated learning.
    This version does NOT perform a train/test split, using the entire dataset for
    both training and validation to focus on implementation correctness.
    """
    from sklearn.datasets import make_classification

    ap_id = 0
    pp_ids = [i + 1 for i in range(pp_parties)]

    n_features_pp = n_total_features - n_features_ap
    n_features_pp_per_party = n_features_pp // pp_parties

    X_np, y_np = make_classification(
        n_samples=n_samples,
        n_features=n_total_features,
        n_informative=n_total_features - 1,
        n_redundant=0,
        random_state=random_state,
        n_clusters_per_class=1,
    )

    X_ap_np = X_np[:, :n_features_ap]
    X_pp_np = X_np[:, n_features_ap:]

    X_pp_np_per_party = []
    for i in range(pp_parties - 1):
        X_pp_np_per_party.append(
            X_pp_np[:, i * n_features_pp_per_party : (i + 1) * n_features_pp_per_party]
        )
    X_pp_np_per_party.append(X_pp_np[:, (pp_parties - 1) * n_features_pp_per_party :])

    X_ap_jax = jnp.array(X_ap_np, dtype=jnp.float32)
    X_pp_jax = [
        jnp.array(X_pp_np_per_party[i], dtype=jnp.float32) for i in range(pp_parties)
    ]
    y_jax = jnp.array(y_np, dtype=jnp.float32)

    X_parts = {
        ap_id: X_ap_jax,
        **{pp_id: X_pp_jax[i] for i, pp_id in enumerate(pp_ids)},
    }
    all_party_ids_list = [ap_id, *pp_ids]

    print("\n--- Step 1: Loaded Dataset ---")
    print("=" * 80)
    print(
        f"Total Samples: {n_samples}, total Features: {n_total_features}, PP Parties: {pp_parties}"
    )
    print(f"AP data shape: {X_ap_jax.shape}, y shape: {y_jax.shape}")
    for i, pp_id in enumerate(pp_ids):
        print(f"PP{pp_id} data shape: {X_pp_jax[i].shape}")
    print("=" * 80)

    return all_party_ids_list, X_parts, y_jax, X_np, y_np


def run_plaintext_xgboost_benchmark(X_np: np.ndarray, y_np: np.ndarray, params: dict):
    """
    Trains a standard XGBoost classifier on the full plaintext data and prints metrics.
    This serves as a benchmark for the SecureBoost implementation.
    The parameters are mapped directly from the SecureBoost parameters.
    """
    print("\n--- Step 2: Running Plaintext XGBoost Benchmark ---")
    print("=" * 80)

    xgb_params = params.copy()
    xgb_model = XGBClassifier(**xgb_params, eval_metric="logloss")
    xgb_model.fit(X_np, y_np)

    y_pred_xgb = xgb_model.predict(X_np)
    y_pred_proba_xgb = xgb_model.predict_proba(X_np)[:, 1]

    accuracy = accuracy_score(y_np, y_pred_xgb)
    auc = roc_auc_score(y_np, y_pred_proba_xgb)

    print(f"Standard XGBoost Train Accuracy: {accuracy:.4f}")
    print(f"Standard XGBoost Train AUC: {auc:.4f}")
    print("=" * 80)

    return accuracy, auc


@mp.function
def run_sgb(
    model: SecureBoost,
    X_parts: dict,
    y_jax: jnp.ndarray,
    all_party_ids_list: list[int],
    pred_leaves: bool = False,
):
    # 1. load data
    all_datas = [
        mp.run_jax_at(
            all_party_ids_list[0], lambda x: x, X_parts[all_party_ids_list[0]]
        ),  # AP
        *[
            mp.run_jax_at(pp_id, lambda x: x, X_parts[pp_id])
            for pp_id in all_party_ids_list[1:]
        ],  # PPs
    ]
    y_data = mp.run_jax_at(all_party_ids_list[0], lambda x: x, y_jax)

    # 2. train process
    model = model.fit(all_datas, y_data)
    pred = model.predict(all_datas)

    if pred_leaves:
        leaves = model.predict_leaves(all_datas)
    else:
        leaves = None

    return model.trees, pred, leaves


@pytest.fixture(scope="module")
def test_setup():
    """Setup fixture for all tests"""
    print(" ========= start test of jit_sgb package ========= \n")

    sim2 = mp.Simulator.simple(2)
    sim3 = mp.Simulator.simple(3)

    # fixed dataset params
    n_samples = 10
    n_total_features = 3
    n_features_ap = 1
    random_state = 42

    # fixed xgboost params
    XGB_PARAMS = {
        "n_estimators": 1,
        "learning_rate": 0.1,
        "max_depth": 2,
        "max_bin": 4,
        "reg_lambda": 0.1,
        "gamma": 0.1,
        "min_child_weight": 1.0,
    }

    # fixed debug params
    DEBUG_SAMPLES = 2

    yield {
        "sim2": sim2,
        "sim3": sim3,
        "n_samples": n_samples,
        "n_total_features": n_total_features,
        "n_features_ap": n_features_ap,
        "random_state": random_state,
        "XGB_PARAMS": XGB_PARAMS,
        "DEBUG_SAMPLES": DEBUG_SAMPLES,
    }

    print(" ========= end test of jit_sgb package ========= \n")


def _sgb_run_main(test_setup, world_size: int, need_debug_leaves: bool):
    print(
        f"========= start test of jit_sgb package with world_size {world_size}, need_debug_leaves {need_debug_leaves} ========= \n"
    )
    assert world_size in [2, 3], "world_size must be 2 or 3"

    # Step 1: setup phase
    sim = test_setup["sim2"] if world_size == 2 else test_setup["sim3"]
    params = test_setup["XGB_PARAMS"].copy()
    # if need debug leaves, set n_estimators to 1 to avoid too many trees
    if need_debug_leaves:
        params["n_estimators"] = 1

    # Step 2: load dataset
    (
        all_party_ids_list,
        X_parts,
        y_jax,
        X_plaintext,
        y_plaintext,
    ) = load_dataset(
        n_samples=test_setup["n_samples"],
        n_total_features=test_setup["n_total_features"],
        n_features_ap=test_setup["n_features_ap"],
        pp_parties=world_size - 1,
        random_state=test_setup["random_state"],
    )
    assert all_party_ids_list == list(range(world_size))

    # Step 3: run plaintext xgboost benchmark
    start_time = time.time()
    xgb_acc, xgb_auc = run_plaintext_xgboost_benchmark(X_plaintext, y_plaintext, params)
    end_time = time.time()
    print(f"Plaintext XGBoost Benchmark Time: {end_time - start_time:.2f} seconds")

    # Step 4: run secureboost
    print("=" * 100)
    ap_id = 0
    pp_ids = list(range(1, world_size))
    secure_boost = SecureBoost(
        active_party_id=ap_id,
        passive_party_ids=pp_ids,
        **params,
    )
    start_time = time.time()
    out = mp.evaluate(
        sim,
        run_sgb,
        secure_boost,
        X_parts,
        y_jax,
        all_party_ids_list,
        need_debug_leaves,
    )
    ret = mp.fetch(sim, out)
    assert len(ret) == 3  # trees, pred, leaves (if need)
    print("SecureBoost training and prediction completed.")
    # Calculate and print accuracy metrics
    prob = ret[1][0]
    predictions = (prob > 0.5).astype(int)
    sgb_acc = accuracy_score(y_plaintext, predictions)
    sgb_auc = roc_auc_score(y_plaintext, prob)
    print(f"- SecureBoost {world_size}PC Train Accuracy: {sgb_acc:.4f}")
    print(f"- SecureBoost {world_size}PC Train AUC: {sgb_auc:.4f}")
    end_time = time.time()
    print(f"SecureBoost 2PC Benchmark Time: {end_time - start_time:.2f} seconds")

    # xgb and sgb should have similar accuracy and auc
    assert abs(sgb_acc - xgb_acc) < 2e-2
    assert abs(sgb_auc - xgb_auc) < 2e-2

    # step 5: debug phase (if need)
    if need_debug_leaves:
        pretty_print_ensemble(ret[0], all_party_ids_list)
        leaves_data = ret[2][0]  # Shape: (n_samples * n_parties, n_nodes)
        n_parties = world_size

        print(f"\n📊 Leaf Node Predictions (Shape: {leaves_data.shape}):")
        print("=" * 80)

        wrong_nodes = 0

        for sample_idx in range(test_setup["DEBUG_SAMPLES"]):
            ap_row_idx = sample_idx * n_parties + 0  # AP row
            pp1_row_idx = sample_idx * n_parties + 1  # PP1 row
            if n_parties == 3:
                pp2_row_idx = sample_idx * n_parties + 2  # PP2 row

            ap_leaves = leaves_data[ap_row_idx]
            pp1_leaves = leaves_data[pp1_row_idx]
            if n_parties == 3:
                pp2_leaves = leaves_data[pp2_row_idx]

            # Find non-zero leaf nodes for each party
            ap_active_nodes = [i for i, val in enumerate(ap_leaves) if val > 0]
            pp1_active_nodes = [i for i, val in enumerate(pp1_leaves) if val > 0]
            if n_parties == 3:
                pp2_active_nodes = [i for i, val in enumerate(pp2_leaves) if val > 0]

            # Get ground truth and prediction for context
            gt = int(y_plaintext[sample_idx])
            pred = predictions[sample_idx]
            prob_val = prob[sample_idx]

            print(f"Sample {sample_idx:2d} │ GT:{gt} Pred:{pred} Prob:{prob_val:.3f}")
            print(f"         │ AP  → Nodes: {ap_active_nodes}")
            print(f"         │ PP1 → Nodes: {pp1_active_nodes}")
            if n_parties == 3:
                print(f"         │ PP2 → Nodes: {pp2_active_nodes}")

            # Show consensus (nodes where both parties agree)
            consensus_nodes = [i for i in ap_active_nodes if i in pp1_active_nodes]
            if n_parties == 3:
                consensus_nodes = [i for i in consensus_nodes if i in pp2_active_nodes]
            if consensus_nodes:
                print(f"         │ 🤝  → Consensus: {consensus_nodes}")
            else:
                print("         │ ❌  → No consensus")

            if len(consensus_nodes) != 1:
                wrong_nodes += 1

            print("─" * 50)

        assert wrong_nodes == 0, f"wrong nodes: {wrong_nodes}"

    print(
        f"========= end test of jit_sgb package with world_size {world_size}, need_debug_leaves {need_debug_leaves} ========= \n"
    )


def test_sgb_2pc(test_setup):
    _sgb_run_main(test_setup, world_size=2, need_debug_leaves=False)


def test_sgb_2pc_debug_leaves(test_setup):
    _sgb_run_main(test_setup, world_size=2, need_debug_leaves=True)


def test_sgb_3pc(test_setup):
    _sgb_run_main(test_setup, world_size=3, need_debug_leaves=False)


def test_sgb_3pc_debug_leaves(test_setup):
    _sgb_run_main(test_setup, world_size=3, need_debug_leaves=True)


@mp.function
def run_bucket_sum_2_groups():
    """Test batch feature-wise bucket sum with 2 groups"""
    # sample_size = 6
    # feature_size = 2
    # gh_size = 2
    bucket_num = 3
    group_size = 2

    # shape: (sample_size, gh_size)
    m1_np = np.array([
        [1, 10],  # sample 0
        [2, 20],  # sample 1
        [3, 30],  # sample 2
        [4, 40],  # sample 3
        [5, 50],  # sample 4
        [6, 60],  # sample 5
    ])
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

    pkey, skey = mp.run_at(0, phe.keygen)
    world_mask = mp.Mask.all(2)
    pkey_bcasted = mp.bcast_m(world_mask, 0, pkey)

    m1 = mp.run_jax_at(0, lambda x: x, m1_np)

    encrypted_arr = mp.run_at(0, phe.encrypt, m1, pkey_bcasted)
    encrypted_arr = mp.p2p(0, 1, encrypted_arr)

    subgroup_map = mp.run_jax_at(1, lambda x: x, subgroup_map)
    order_map = mp.run_jax_at(1, lambda x: x, order_map)

    bucket_sum_list = batch_feature_wise_bucket_sum_mplang(
        encrypted_arr, subgroup_map, order_map, bucket_num, group_size, rank=1
    )

    # Decrypt each group result separately
    decrypted_results = []
    for group_idx in range(group_size):
        decrypted_group = mp.run_at(
            0, phe.decrypt, mp.p2p(1, 0, bucket_sum_list[group_idx]), skey
        )
        decrypted_results.append(decrypted_group)

    return decrypted_results


@mp.function
def run_bucket_sum_3_groups():
    """Test batch feature-wise bucket sum with 3 groups"""
    # sample_size = 9
    # feature_size = 2
    # gh_size = 2
    bucket_num = 3
    group_size = 3

    # shape: (sample_size, gh_size)
    m1_np = np.array([
        [1, 10],  # sample 0 - group 0
        [2, 20],  # sample 1 - group 1
        [3, 30],  # sample 2 - group 0
        [4, 40],  # sample 3 - group 1
        [5, 50],  # sample 4 - group 2
        [6, 60],  # sample 5 - group 0
        [7, 70],  # sample 6 - group 1
        [8, 80],  # sample 7 - group 2
        [9, 90],  # sample 8 - group 2
    ])
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

    pkey, skey = mp.run_at(0, phe.keygen)
    world_mask = mp.Mask.all(2)
    pkey_bcasted = mp.bcast_m(world_mask, 0, pkey)

    m1 = mp.run_jax_at(0, lambda x: x, m1_np)

    encrypted_arr = mp.run_at(0, phe.encrypt, m1, pkey_bcasted)
    encrypted_arr = mp.p2p(0, 1, encrypted_arr)

    subgroup_map = mp.run_jax_at(1, lambda x: x, subgroup_map)
    order_map = mp.run_jax_at(1, lambda x: x, order_map)

    bucket_sum_list = batch_feature_wise_bucket_sum_mplang(
        encrypted_arr, subgroup_map, order_map, bucket_num, group_size, rank=1
    )

    # Decrypt each group result separately
    decrypted_results = []
    for group_idx in range(group_size):
        decrypted_group = mp.run_at(
            0, phe.decrypt, mp.p2p(1, 0, bucket_sum_list[group_idx]), skey
        )
        decrypted_results.append(decrypted_group)

    return decrypted_results


def test_batch_feature_wise_bucket_sum_2_groups(test_setup):
    """Test batch feature-wise bucket sum implementation with 2 groups"""
    print("=== Testing batch_feature_wise_bucket_sum with 2 groups ===")

    sim = test_setup["sim2"]
    result_2_groups = mp.evaluate(sim, run_bucket_sum_2_groups)
    fetched_2_groups = mp.fetch(sim, result_2_groups)

    # fetched_2_groups is [[group0_from_rank0, None], [group1_from_rank0, None]]
    print(f"2-group PHE sum completed. Number of groups: {len(fetched_2_groups)}")
    for i, group_item in enumerate(fetched_2_groups):
        group_result = group_item[0] if group_item[0] is not None else group_item[1]
        if group_result is not None:
            print(f"Group {i} shape: {group_result.shape}")
        else:
            raise ValueError(f"Group {i} result is None")

    # Extract group results
    out_2_0 = fetched_2_groups[0][0]  # First group's result
    out_2_1 = fetched_2_groups[1][0]  # Second group's result

    print(f"group 0 sum: {out_2_0}")
    print(f"group 1 sum: {out_2_1}")

    # Verify 2-group test correctness
    expected_2_0 = np.array([
        [4, 40],  # bucket 0 for feature 0: samples 0,2 (buckets <=0)
        [6, 60],  # bucket 1 for feature 0: samples 0,1,2 (buckets <=1)
        [6, 60],  # bucket 2 for feature 0: samples 0,1,2 (buckets <=2)
        [3, 30],  # bucket 0 for feature 1: sample 2 (bucket <=0)
        [4, 40],  # bucket 1 for feature 1: samples 0,2 (buckets <=1)
        [6, 60],  # bucket 2 for feature 1: samples 0,1,2 (buckets <=2)
    ])

    expected_2_1 = np.array([
        [6, 60],  # bucket 0 for feature 0: sample 5 (bucket <=0)
        [11, 110],  # bucket 1 for feature 0: samples 4,5 (buckets <=1)
        [15, 150],  # bucket 2 for feature 0: samples 3,4,5 (buckets <=2)
        [5, 50],  # bucket 0 for feature 1: sample 4 (bucket <=0)
        [9, 90],  # bucket 1 for feature 1: samples 3,4 (buckets <=1)
        [15, 150],  # bucket 2 for feature 1: samples 3,4,5 (buckets <=2)
    ])

    np.testing.assert_array_equal(out_2_0, expected_2_0)
    np.testing.assert_array_equal(out_2_1, expected_2_1)
    print("✓ 2-group test passed!")


def test_batch_feature_wise_bucket_sum_3_groups(test_setup):
    """Test batch feature-wise bucket sum implementation with 3 groups"""
    print("=== Testing batch_feature_wise_bucket_sum with 3 groups ===")

    sim = test_setup["sim2"]
    result_3_groups = mp.evaluate(sim, run_bucket_sum_3_groups)
    fetched_3_groups = mp.fetch(sim, result_3_groups)

    # fetched_3_groups is [[group0_from_rank0, None], [group1_from_rank0, None], [group2_from_rank0, None]]
    print(f"3-group PHE sum completed. Number of groups: {len(fetched_3_groups)}")
    for i, group_item in enumerate(fetched_3_groups):
        group_result = group_item[0] if group_item[0] is not None else group_item[1]
        if group_result is not None:
            print(f"Group {i} shape: {group_result.shape}")
        else:
            raise ValueError(f"Group {i} result is None")

    # Extract group results
    out_3_0 = fetched_3_groups[0][0]  # First group's result
    out_3_1 = fetched_3_groups[1][0]  # Second group's result
    out_3_2 = fetched_3_groups[2][0]  # Third group's result

    print(f"group 0 sum: {out_3_0}")
    print(f"group 1 sum: {out_3_1}")
    print(f"group 2 sum: {out_3_2}")

    # Verify 3-group test correctness
    # Group 0: samples 0,2,5 with values [1,10], [3,30], [6,60]
    # order_map: [0,1], [0,0], [0,2]
    expected_3_0 = np.array([
        [10, 100],  # bucket 0 for feature 0: samples 0,2,5 (buckets <=0)
        [10, 100],  # bucket 1 for feature 0: samples 0,2,5 (buckets <=1)
        [10, 100],  # bucket 2 for feature 0: samples 0,2,5 (buckets <=2)
        [3, 30],  # bucket 0 for feature 1: sample 2 (bucket <=0)
        [4, 40],  # bucket 1 for feature 1: samples 0,2 (buckets <=1)
        [10, 100],  # bucket 2 for feature 1: samples 0,2,5 (buckets <=2)
    ])

    # Group 1: samples 1,3,6 with values [2,20], [4,40], [7,70]
    # order_map: [1,2], [2,1], [1,0]
    expected_3_1 = np.array([
        [0, 0],  # bucket 0 for feature 0: no samples (no buckets <=0)
        [9, 90],  # bucket 1 for feature 0: samples 1,6 (buckets <=1)
        [13, 130],  # bucket 2 for feature 0: samples 1,3,6 (buckets <=2)
        [7, 70],  # bucket 0 for feature 1: sample 6 (bucket <=0)
        [11, 110],  # bucket 1 for feature 1: samples 3,6 (buckets <=1)
        [13, 130],  # bucket 2 for feature 1: samples 1,3,6 (buckets <=2)
    ])

    # Group 2: samples 4,7,8 with values [5,50], [8,80], [9,90]
    # order_map: [1,0], [2,1], [0,2]
    expected_3_2 = np.array([
        [9, 90],  # bucket 0 for feature 0: sample 8 (bucket <=0)
        [14, 140],  # bucket 1 for feature 0: samples 4,8 (buckets <=1)
        [22, 220],  # bucket 2 for feature 0: samples 4,7,8 (buckets <=2)
        [5, 50],  # bucket 0 for feature 1: sample 4 (bucket <=0)
        [13, 130],  # bucket 1 for feature 1: samples 4,7 (buckets <=1)
        [22, 220],  # bucket 2 for feature 1: samples 4,7,8 (buckets <=2)
    ])

    np.testing.assert_array_equal(out_3_0, expected_3_0)
    np.testing.assert_array_equal(out_3_1, expected_3_1)
    np.testing.assert_array_equal(out_3_2, expected_3_2)
    print("✓ 3-group test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
