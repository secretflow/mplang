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
import unittest
from typing import List

import jax.numpy as jnp
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

import mplang
import mplang.simp as simp
from examples.xgboost.sgb import (
    SecureBoost,
    pretty_print_ensemble,
)


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


@mplang.function
def run_sgb(
    model: SecureBoost,
    X_parts: dict,
    y_jax: jnp.ndarray,
    all_party_ids_list: List[int],
    pred_leaves: bool = False,
):
    # 1. load data
    all_datas = [
        simp.runAt(all_party_ids_list[0], lambda x: x)(
            X_parts[all_party_ids_list[0]]
        ),  # AP
        *[
            simp.runAt(pp_id, lambda x: x)(X_parts[pp_id])
            for pp_id in all_party_ids_list[1:]
        ],  # PPs
    ]
    y_data = simp.runAt(all_party_ids_list[0], lambda x: x)(y_jax)

    # 2. train process
    model = model.fit(all_datas, y_data)
    pred = model.predict(all_datas)

    if pred_leaves:
        leaves = model.predict_leaves(all_datas)
    else:
        leaves = None

    return model.trees, pred, leaves


class TestJitSgb(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(" ========= start test of jit_sgb package ========= \n")

        cls.sim2 = mplang.Simulator(2)
        cls.sim3 = mplang.Simulator(3)

        # fixed dataset params
        cls.n_samples = 1_00
        cls.n_total_features = 10
        cls.n_features_ap = 4
        cls.random_state = 42

        # fixed xgboost params
        cls.XGB_PARAMS = {
            "n_estimators": 1,
            "learning_rate": 0.1,
            "max_depth": 3,
            "max_bin": 8,
            "reg_lambda": 0.1,
            "gamma": 0.1,
            "min_child_weight": 1.0,
        }

        # fixed debug params
        cls.DEBUG_SAMPLES = 10

    @classmethod
    def tearDownClass(cls):
        print(" ========= end test of jit_sgb package ========= \n")

    def _sgb_run_main(self, world_size: int, need_debug_leaves: bool):
        print(
            f"========= start test of jit_sgb package with world_size {world_size}, need_debug_leaves {need_debug_leaves} ========= \n"
        )
        assert world_size in [2, 3], "world_size must be 2 or 3"

        # Step 1: setup phase
        sim = self.sim2 if world_size == 2 else self.sim3
        params = self.XGB_PARAMS.copy()
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
            n_samples=self.n_samples,
            n_total_features=self.n_total_features,
            n_features_ap=self.n_features_ap,
            pp_parties=world_size - 1,
            random_state=self.random_state,
        )
        self.assertEqual(all_party_ids_list, list(range(world_size)))

        # Step 3: run plaintext xgboost benchmark
        start_time = time.time()
        xgb_acc, xgb_auc = run_plaintext_xgboost_benchmark(
            X_plaintext, y_plaintext, params
        )
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
        out = mplang.evaluate(
            sim,
            run_sgb,
            secure_boost,
            X_parts,
            y_jax,
            all_party_ids_list,
            need_debug_leaves,
        )
        ret = mplang.fetch(sim, out)
        assert len(ret) == 3  # trees, pred, leaves (if need)

        # Calculate and print accuracy metrics
        pred_results = ret[1][0]
        prob = pred_results[0]  # Extract probabilities from AP
        predictions = (prob > 0.5).astype(int)
        sgb_acc = accuracy_score(y_plaintext, predictions)
        sgb_auc = roc_auc_score(y_plaintext, prob)
        print(f"- SecureBoost {world_size}PC Train Accuracy: {sgb_acc:.4f}")
        print(f"- SecureBoost {world_size}PC Train AUC: {sgb_auc:.4f}")
        end_time = time.time()
        print(f"SecureBoost 2PC Benchmark Time: {end_time - start_time:.2f} seconds")

        # xgb and sgb should have similar accuracy and auc
        self.assertAlmostEqual(sgb_acc, xgb_acc, delta=2e-2)
        self.assertAlmostEqual(sgb_auc, xgb_auc, delta=2e-2)

        # step 5: debug phase (if need)
        if need_debug_leaves:
            pretty_print_ensemble(ret[0][0], all_party_ids_list)
            pred_leaves = ret[2][0]
            leaves_data = pred_leaves[0]  # Shape: (n_samples * n_parties, n_nodes)
            n_parties = world_size

            print(f"\n📊 Leaf Node Predictions (Shape: {leaves_data.shape}):")
            print("=" * 80)

            wrong_nodes = 0

            for sample_idx in range(self.DEBUG_SAMPLES):
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
                    pp2_active_nodes = [
                        i for i, val in enumerate(pp2_leaves) if val > 0
                    ]

                # Get ground truth and prediction for context
                gt = int(y_plaintext[sample_idx])
                pred = predictions[sample_idx]
                prob_val = prob[sample_idx]

                print(
                    f"Sample {sample_idx:2d} │ GT:{gt} Pred:{pred} Prob:{prob_val:.3f}"
                )
                print(f"         │ AP  → Nodes: {ap_active_nodes}")
                print(f"         │ PP1 → Nodes: {pp1_active_nodes}")
                if n_parties == 3:
                    print(f"         │ PP2 → Nodes: {pp2_active_nodes}")

                # Show consensus (nodes where both parties agree)
                consensus_nodes = [i for i in ap_active_nodes if i in pp1_active_nodes]
                if n_parties == 3:
                    consensus_nodes = [
                        i for i in consensus_nodes if i in pp2_active_nodes
                    ]
                if consensus_nodes:
                    print(f"         │ 🤝  → Consensus: {consensus_nodes}")
                else:
                    print(f"         │ ❌  → No consensus")

                if len(consensus_nodes) != 1:
                    wrong_nodes += 1

                print("─" * 50)

            self.assertEqual(wrong_nodes, 0, f"wrong nodes: {wrong_nodes}")

        print(
            f"========= end test of jit_sgb package with world_size {world_size}, need_debug_leaves {need_debug_leaves} ========= \n"
        )

    def test_sgb_2pc(self):
        self._sgb_run_main(world_size=2, need_debug_leaves=False)

    # def test_sgb_2pc_debug_leaves(self):
    #     self._sgb_run_main(world_size=2, need_debug_leaves=True)

    # def test_sgb_3pc(self):
    #     self._sgb_run_main(world_size=3, need_debug_leaves=False)

    # def test_sgb_3pc_debug_leaves(self):
    #     self._sgb_run_main(world_size=3, need_debug_leaves=True)


if __name__ == "__main__":
    unittest.main()
