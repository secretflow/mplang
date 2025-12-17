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

"""Unit tests for SecureBoost (SGB) ML library."""

import numpy as np
import pytest

import mplang.v2 as mp
from mplang.v2.libs.ml.sgb import SecureBoost


@pytest.fixture
def sim():
    """Create a 2-party simulator for SGB tests."""
    return mp.make_simulator(2)


class TestSecureBoost:
    """Test suite for SecureBoost algorithm."""

    def test_two_party_basic(self, sim):
        """Test basic 2-party SecureBoost with BFV FHE."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features_ap = 3  # Active party features
        n_features_pp = 2  # Passive party features

        # Linearly separable data
        X_all = np.random.randn(n_samples, n_features_ap + n_features_pp).astype(
            np.float32
        )
        y = (X_all[:, 0] + X_all[:, 1] + X_all[:, 3] > 0).astype(np.float32)

        # Split features by party
        X_ap = X_all[:, :n_features_ap]
        X_pp = X_all[:, n_features_ap:]

        # Define training job
        def job(
            X_ap=X_ap,
            X_pp=X_pp,
            y=y,
            n_samples=n_samples,
            n_features_ap=n_features_ap,
            n_features_pp=n_features_pp,
        ):
            data_ap = mp.put("P0", X_ap)
            data_pp = mp.put("P1", X_pp)
            y_data = mp.put("P0", y)

            model = SecureBoost(
                n_estimators=2,
                max_depth=2,
                learning_rate=0.1,
                max_bin=8,
                ap_rank=0,
                pp_ranks=[1],
            )

            model.fit(
                [data_ap, data_pp],
                y_data,
                n_samples=n_samples,
                n_features_per_party=[n_features_ap, n_features_pp],
            )

            y_prob = model.predict([data_ap, data_pp], n_samples=n_samples)
            return y_prob

        # Execute
        y_prob_obj = mp.evaluate(job, context=sim)

        # Fetch and validate
        y_pred_probs = mp.fetch(y_prob_obj, context=sim)
        if isinstance(y_pred_probs, list):
            y_pred_probs = y_pred_probs[0]

        # Basic sanity checks
        assert y_pred_probs is not None
        assert len(y_pred_probs) == n_samples
        assert np.all((y_pred_probs >= 0) & (y_pred_probs <= 1)), "Probs in [0,1]"

        # Check accuracy (should be reasonably good for linearly separable data)
        y_pred_class = (y_pred_probs > 0.5).astype(np.float32)
        accuracy = np.mean(y_pred_class == y)
        assert accuracy > 0.6, f"Accuracy {accuracy:.2%} should be > 60%"

    def test_single_party_no_fhe(self, sim):
        """Test single-party mode (no FHE, just local computation)."""
        np.random.seed(42)
        n_samples = 50
        n_features = 4

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

        def job(X=X, y=y, n_samples=n_samples, n_features=n_features):
            data = mp.put("P0", X)
            y_data = mp.put("P0", y)

            model = SecureBoost(
                n_estimators=2,
                max_depth=2,
                learning_rate=0.1,
                max_bin=8,
                ap_rank=0,
                pp_ranks=[],  # No passive parties = no FHE
            )

            model.fit(
                [data],
                y_data,
                n_samples=n_samples,
                n_features_per_party=[n_features],
            )

            y_prob = model.predict([data], n_samples=n_samples)
            return y_prob

        # Execute
        y_prob_obj = mp.evaluate(job, context=sim)
        y_pred_probs = mp.fetch(y_prob_obj, context=sim)
        if isinstance(y_pred_probs, list):
            y_pred_probs = y_pred_probs[0]

        # Validate
        assert y_pred_probs is not None
        assert len(y_pred_probs) == n_samples

        # Single-party should work well on simple data
        y_pred_class = (y_pred_probs > 0.5).astype(np.float32)
        accuracy = np.mean(y_pred_class == y)
        assert accuracy > 0.55, f"Single-party accuracy {accuracy:.2%} should be > 55%"

    def test_model_parameters(self):
        """Test SecureBoost parameter initialization."""
        model = SecureBoost(
            n_estimators=5,
            max_depth=4,
            learning_rate=0.05,
            max_bin=16,
            ap_rank=0,
            pp_ranks=[1, 2],
        )

        assert model.n_estimators == 5
        assert model.max_depth == 4
        assert model.learning_rate == 0.05
        assert model.max_bin == 16
        assert model.ap_rank == 0
        assert model.pp_ranks == [1, 2]
