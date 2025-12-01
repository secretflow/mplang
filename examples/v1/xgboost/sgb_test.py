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
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from examples.xgboost.sgb import (
    SecureBoost,
    batch_feature_wise_bucket_sum_fhe_vector,
    pretty_print_ensemble,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

import mplang.v1 as mp
from mplang.v1.ops import fhe


def extract_ap_values(x):
    """
    Recursively extract the first non-None value for party 0 (Active Party).

    Two-party case: leaf MPObject fetch yields [val_at_p0, None].
    Multi-party case: may have nested lists with None values.
    """
    if isinstance(x, (list, tuple)):
        # Two-party case: leaf MPObject fetch yields [val_at_p0, None]
        if len(x) == 2 and x[1] is None:
            return extract_ap_values(x[0])
        return [extract_ap_values(e) for e in x]
    return x


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
            pp2_row_idx = None
            if n_parties == 3:
                pp2_row_idx = sample_idx * n_parties + 2  # PP2 row

            ap_leaves = leaves_data[ap_row_idx]
            pp1_leaves = leaves_data[pp1_row_idx]
            pp2_leaves = None
            if n_parties == 3 and pp2_row_idx is not None:
                pp2_leaves = leaves_data[pp2_row_idx]

            # Find non-zero leaf nodes for each party
            ap_active_nodes = [i for i, val in enumerate(ap_leaves) if val > 0]
            pp1_active_nodes = [i for i, val in enumerate(pp1_leaves) if val > 0]
            pp2_active_nodes = []
            if n_parties == 3 and pp2_leaves is not None:
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
def _run_pp_fhe_cumulative_once(X_parts, y_jax, all_party_ids_list, params):
    """Return decrypted PP cumulative GH for level-0 (t=1) using FHE path, for debugging."""
    ap_id = all_party_ids_list[0]
    pp_id = all_party_ids_list[1]

    # Build bins and bin indices for AP/PP in their parties

    from examples.xgboost.sgb import (
        DEFAULT_FXP_BITS,
        batch_feature_wise_bucket_sum_fhe_vector,
        build_bins_equi_width,
        compute_bin_indices,
        compute_gh,
        compute_init_pred,
        quantize_gh,
    )

    build_bins_vmapped = jax.vmap(
        partial(build_bins_equi_width, max_bin=params["max_bin"]), in_axes=1
    )
    compute_indices_vmapped = jax.vmap(compute_bin_indices, in_axes=(1, 0), out_axes=1)

    X_ap = mp.run_jax_at(ap_id, lambda x: x, X_parts[ap_id])
    X_pp = mp.run_jax_at(pp_id, lambda x: x, X_parts[pp_id])
    y_data = mp.run_jax_at(ap_id, lambda x: x, y_jax)

    _ = mp.run_jax_at(
        ap_id, build_bins_vmapped, X_ap
    )  # not used downstream, keep binning consistent
    bins_pp = mp.run_jax_at(pp_id, build_bins_vmapped, X_pp)
    bin_idx_pp = mp.run_jax_at(pp_id, compute_indices_vmapped, X_pp, bins_pp)

    # GH at AP
    init_pred = mp.run_jax_at(ap_id, compute_init_pred, y_data)
    logits0 = mp.run_jax_at(
        ap_id, lambda p, m=X_ap.shape[0]: p * jnp.ones(m), init_pred
    )
    GH = mp.run_jax_at(ap_id, compute_gh, y_data, logits0)

    # Quantize and encrypt at AP
    fxp_scale = 1 << DEFAULT_FXP_BITS
    Q = mp.run_jax_at(ap_id, quantize_gh, GH, fxp_scale)
    qg = mp.run_jax_at(ap_id, lambda a: a[:, 0].astype(jnp.int64), Q)
    qh = mp.run_jax_at(ap_id, lambda a: a[:, 1].astype(jnp.int64), Q)
    priv_ctx, pub_ctx, _ = mp.run_at(ap_id, fhe.keygen, scheme="BFV")
    g_ct = mp.run_at(ap_id, fhe.encrypt, qg, pub_ctx)
    h_ct = mp.run_at(ap_id, fhe.encrypt, qh, pub_ctx)

    # Level-0 subgroup map (all samples in group 0)
    subgroup_map = mp.run_jax_at(
        pp_id, lambda m: jnp.ones((1, m), dtype=jnp.int8), X_pp.shape[0]
    )

    # Compute cumulative via FHE dot; return decrypted lists
    g_lists, h_lists = batch_feature_wise_bucket_sum_fhe_vector(
        g_ct,
        h_ct,
        subgroup_map,
        bin_idx_pp,
        params["max_bin"],
        1,
        rank=pp_id,
        ap_rank=ap_id,
    )
    enc_g_list = g_lists[0]
    enc_h_list = h_lists[0]

    dec_g = [mp.run_at(ap_id, fhe.decrypt, ct, priv_ctx) for ct in enc_g_list]
    dec_h = [mp.run_at(ap_id, fhe.decrypt, ct, priv_ctx) for ct in enc_h_list]

    return dec_g, dec_h, fxp_scale, bins_pp.shape[1]


def test_pp_fhe_cumulative_matches_plain(test_setup):
    # Setup a 2PC environment and dataset
    sim = test_setup["sim2"]
    params = test_setup["XGB_PARAMS"].copy()
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
        pp_parties=1,
        random_state=test_setup["random_state"],
    )

    out = mp.evaluate(
        sim, _run_pp_fhe_cumulative_once, X_parts, y_jax, all_party_ids_list, params
    )
    dec_g, dec_h, fxp_scale, n_features_pp = mp.fetch(sim, out)
    print("Debug dec_g sample:", dec_g[:3])
    # Assemble vectors in Python
    # Each decrypt fetch returns [value, None]; take the first element
    g_vec_q = np.array(extract_ap_values(dec_g)).astype(np.int64).reshape(-1)
    h_vec_q = np.array(extract_ap_values(dec_h)).astype(np.int64).reshape(-1)
    g_vec = g_vec_q.astype(np.float32) / fxp_scale
    h_vec = h_vec_q.astype(np.float32) / fxp_scale
    gh_flat_fhe = np.stack([g_vec, h_vec], axis=1)

    # Plaintext cumulative construction
    k = params["max_bin"]
    # Recompute plaintext GH and PP bin indices using numpy to avoid MP fetch shape issues
    y_np = y_plaintext
    X_pp_np = X_plaintext[:, test_setup["n_features_ap"] :]
    p_base = np.clip(np.mean(y_np), 1e-15, 1 - 1e-15)
    init_pred = np.log(p_base / (1 - p_base))
    p = 1 / (1 + np.exp(-init_pred))
    g = p - y_np
    h_scalar = p * (1 - p)
    h = np.full_like(g, h_scalar)
    GH_np = np.stack([g, h], axis=1).astype(np.float32)
    # Bins and indices for PP features
    n_features_pp = X_pp_np.shape[1]
    bin_idx_np = np.zeros((X_pp_np.shape[0], n_features_pp), dtype=np.int64)
    for f in range(n_features_pp):
        x = X_pp_np[:, f]
        if x.shape[0] >= 2:
            min_val, max_val = np.min(x), np.max(x)
            if max_val - min_val < 1e-9:
                bins = np.full((k - 1,), np.inf, dtype=np.float32)
            else:
                boundaries = np.linspace(min_val, max_val, num=k + 1)
                bins = boundaries[1:-1]
        else:
            bins = np.full((k - 1,), np.inf, dtype=np.float32)
        bin_idx_np[:, f] = np.digitize(x, bins, right=True)
    flat_cumul = []
    for f in range(n_features_pp):
        bins_f = bin_idx_np[:, f]
        per_bin = np.zeros((k, 2), dtype=np.float32)
        for i in range(GH_np.shape[0]):
            b = int(bins_f[i])
            per_bin[b, 0] += GH_np[i, 0]
            per_bin[b, 1] += GH_np[i, 1]
        cumul = np.cumsum(per_bin, axis=0)
        flat_cumul.append(cumul)
    gh_flat_plain = np.stack(flat_cumul, axis=0).reshape((n_features_pp * k, 2))

    diff = np.linalg.norm(gh_flat_fhe - gh_flat_plain)
    print("PP cumulative L2 diff:", diff)
    print("FHE g first 12:", g_vec[:12])
    print("Plain g first 12:", gh_flat_plain[:12, 0])
    assert diff < 1e-4


@mp.function
def run_bucket_sum_2_groups():
    """Test batch feature-wise bucket sum with 2 groups using FHE(BFV) vector dot"""
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

    # Keygen (BFV)
    priv_ctx, pub_ctx, _ = mp.run_at(0, fhe.keygen, scheme="BFV")

    # Prepare g/h integer vectors at AP
    m1 = mp.run_jax_at(0, lambda x: x, m1_np)
    g = mp.run_jax_at(0, lambda a: a[:, 0].astype(jnp.int64), m1)
    h = mp.run_jax_at(0, lambda a: a[:, 1].astype(jnp.int64), m1)

    # Encrypt and send to PP(1)
    g_ct = mp.run_at(0, fhe.encrypt, g, pub_ctx)
    h_ct = mp.run_at(0, fhe.encrypt, h, pub_ctx)

    # Move masks and order map to PP(1)
    subgroup_map = mp.run_jax_at(1, lambda x: x, subgroup_map)
    order_map = mp.run_jax_at(1, lambda x: x, order_map)

    # Compute encrypted sums via vector dot
    g_lists, h_lists = batch_feature_wise_bucket_sum_fhe_vector(
        g_ct,
        h_ct,
        subgroup_map,
        order_map,
        bucket_num,
        group_size,
        rank=1,
        ap_rank=0,
    )

    # Decrypt each group's results at AP and return lists of scalars per group
    decrypted_results = []
    for group_idx in range(group_size):
        enc_g_list = g_lists[group_idx]
        enc_h_list = h_lists[group_idx]

        # Decrypt scalars
        dec_g_scalars = [mp.run_at(0, fhe.decrypt, ct, priv_ctx) for ct in enc_g_list]
        dec_h_scalars = [mp.run_at(0, fhe.decrypt, ct, priv_ctx) for ct in enc_h_list]
        # Append decrypted scalar lists; we'll assemble vectors on fetch side in Python
        decrypted_results.append((dec_g_scalars, dec_h_scalars))

    return decrypted_results


@mp.function
def run_bucket_sum_3_groups():
    """Test batch feature-wise bucket sum with 3 groups using FHE(BFV) vector dot"""
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

    # Keygen (BFV)
    priv_ctx, pub_ctx, _ = mp.run_at(0, fhe.keygen, scheme="BFV")

    # Prepare g/h integer vectors at AP
    m1 = mp.run_jax_at(0, lambda x: x, m1_np)
    g = mp.run_jax_at(0, lambda a: a[:, 0].astype(jnp.int64), m1)
    h = mp.run_jax_at(0, lambda a: a[:, 1].astype(jnp.int64), m1)

    # Encrypt and send to PP(1)
    g_ct = mp.run_at(0, fhe.encrypt, g, pub_ctx)
    h_ct = mp.run_at(0, fhe.encrypt, h, pub_ctx)

    # Move masks and order map to PP(1)
    subgroup_map = mp.run_jax_at(1, lambda x: x, subgroup_map)
    order_map = mp.run_jax_at(1, lambda x: x, order_map)

    # Compute encrypted sums via vector dot
    g_lists, h_lists = batch_feature_wise_bucket_sum_fhe_vector(
        g_ct,
        h_ct,
        subgroup_map,
        order_map,
        bucket_num,
        group_size,
        rank=1,
        ap_rank=0,
    )

    # Decrypt each group's results at AP and return lists of scalars per group
    decrypted_results = []
    for group_idx in range(group_size):
        enc_g_list = g_lists[group_idx]
        enc_h_list = h_lists[group_idx]

        # Decrypt scalars
        dec_g_scalars = [mp.run_at(0, fhe.decrypt, ct, priv_ctx) for ct in enc_g_list]
        dec_h_scalars = [mp.run_at(0, fhe.decrypt, ct, priv_ctx) for ct in enc_h_list]
        # Append decrypted scalar lists; assemble vectors on fetch side
        decrypted_results.append((dec_g_scalars, dec_h_scalars))

    return decrypted_results


def test_batch_feature_wise_bucket_sum_2_groups(test_setup):
    """Test batch feature-wise bucket sum implementation with 2 groups"""
    print("=== Testing batch_feature_wise_bucket_sum with 2 groups ===")

    sim = test_setup["sim2"]
    print("[dbg] evaluating run_bucket_sum_2_groups...")
    result_2_groups = mp.evaluate(sim, run_bucket_sum_2_groups)
    print("[dbg] fetch run_bucket_sum_2_groups result...")
    fetched_2_groups = mp.fetch(sim, result_2_groups)
    print("[dbg] fetched run_bucket_sum_2_groups result")

    # fetched_2_groups is [[(g_list,h_list) from rank0, None], ...]
    print(f"2-group FHE sum completed. Number of groups: {len(fetched_2_groups)}")

    # Build numpy arrays for each group from decrypted scalar lists
    gh_groups = []
    for i, group_item in enumerate(fetched_2_groups):
        print(f"[dbg] group_item[{i}] type={type(group_item)}, len={len(group_item)}")
        assert isinstance(group_item, (list, tuple)) and len(group_item) == 2
        g_item, h_item = group_item
        print(f"[dbg] g_item[{i}] type={type(g_item)}; h_item type={type(h_item)}")
        g_list = extract_ap_values(g_item)
        h_list = extract_ap_values(h_item)
        g_vec = np.array(g_list, dtype=np.int64)
        h_vec = np.array(h_list, dtype=np.int64)
        gh_flat = np.stack([g_vec, h_vec], axis=1)
        print(f"Group {i} gh_flat shape: {gh_flat.shape}")
        gh_groups.append(gh_flat)

    out_2_0 = gh_groups[0]
    out_2_1 = gh_groups[1]

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
    print("[dbg] evaluating run_bucket_sum_3_groups...")
    result_3_groups = mp.evaluate(sim, run_bucket_sum_3_groups)
    print("[dbg] fetch run_bucket_sum_3_groups result...")
    fetched_3_groups = mp.fetch(sim, result_3_groups)
    print("[dbg] fetched run_bucket_sum_3_groups result")

    # fetched_3_groups is [[(g_list,h_list) from rank0, None], ...]
    print(f"3-group FHE sum completed. Number of groups: {len(fetched_3_groups)}")

    gh_groups = []
    for i, group_item in enumerate(fetched_3_groups):
        assert isinstance(group_item, (list, tuple)) and len(group_item) == 2
        g_item, h_item = group_item
        g_list = extract_ap_values(g_item)
        h_list = extract_ap_values(h_item)
        g_vec = np.array(g_list, dtype=np.int64)
        h_vec = np.array(h_list, dtype=np.int64)
        gh_flat = np.stack([g_vec, h_vec], axis=1)
        print(f"Group {i} gh_flat shape: {gh_flat.shape}")
        gh_groups.append(gh_flat)

    out_3_0 = gh_groups[0]
    out_3_1 = gh_groups[1]
    out_3_2 = gh_groups[2]

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
