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

import numpy as np


class DecisionTree:
    def __init__(self, max_depth=3, lambda_=1.0, gamma=0.0, min_child_weight=1.0):
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.tree = None

    def _compute_best_split(self, X, g, h, lambda_, gamma, min_child_weight):
        _n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = -1
        best_threshold = 0.0

        for feature in range(n_features):
            X_feature = X[:, feature]
            sorted_indices = np.argsort(X_feature)
            X_sorted = X_feature[sorted_indices]
            g_sorted = g[sorted_indices]
            h_sorted = h[sorted_indices]

            G_cum = np.cumsum(g_sorted)
            H_cum = np.cumsum(h_sorted)
            total_G = G_cum[-1]
            total_H = H_cum[-1]

            G_left = G_cum[:-1]
            H_left = H_cum[:-1]
            G_right = total_G - G_left
            H_right = total_H - H_left

            gain = (
                G_left**2 / (H_left + lambda_)
                + G_right**2 / (H_right + lambda_)
                - total_G**2 / (total_H + lambda_)
            ) / 2 - gamma

            valid = (H_left >= min_child_weight) & (H_right >= min_child_weight)
            gain = np.where(valid, gain, -np.inf)

            max_gain_idx = np.argmax(gain)
            max_gain = gain[max_gain_idx]

            if max_gain > best_gain:
                best_gain = max_gain
                best_feature = feature
                threshold = (
                    (X_sorted[max_gain_idx] + X_sorted[max_gain_idx + 1]) / 2
                    if max_gain_idx + 1 < len(X_sorted)
                    else X_sorted[max_gain_idx]
                )  # handle edge case for last element
                best_threshold = threshold

        return best_gain, best_feature, best_threshold

    def _build_tree(self, X, g, h, depth=0):
        node = {
            "value": -np.sum(g) / (np.sum(h) + self.lambda_),
            "is_leaf": False,
        }

        if depth >= self.max_depth:
            node["is_leaf"] = True
            return node

        if X.shape[0] < 2:
            node["is_leaf"] = True
            return node

        best_gain, best_feature, best_threshold = self._compute_best_split(
            X, g, h, self.lambda_, self.gamma, self.min_child_weight
        )

        if best_gain <= 0:
            node["is_leaf"] = True
            return node

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node["feature"] = best_feature
        node["threshold"] = best_threshold
        node["left"] = self._build_tree(
            X[left_mask], g[left_mask], h[left_mask], depth + 1
        )
        node["right"] = self._build_tree(
            X[right_mask], g[right_mask], h[right_mask], depth + 1
        )
        return node

    def fit(self, X, g, h):
        self.tree = self._build_tree(X, g, h)
        return self

    def predict_single(self, x, tree):
        if tree["is_leaf"]:
            return tree["value"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self.predict_single(x, tree["left"])
        else:
            return self.predict_single(x, tree["right"])

    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])


class XGBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    @staticmethod
    def _gradient(y_true, y_pred):
        return y_pred - y_true  # MSE gradient

    @staticmethod
    def _hessian(y_true, y_pred):
        return np.ones_like(y_true)  # MSE hessian

    def fit(self, X, y):
        y_pred = np.full(y.shape, np.mean(y))  # Initial prediction
        for _ in range(self.n_estimators):
            g = self._gradient(y, y_pred)
            h = self._hessian(y, y_pred)

            tree = DecisionTree(max_depth=self.max_depth).fit(X, g, h)
            self.trees.append(tree)

            update = tree.predict(X) * self.learning_rate
            y_pred = y_pred + update
        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += tree.predict(X) * self.learning_rate
        return y_pred


if __name__ == "__main__":
    np.random.seed(0)  # for reproducibility
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = np.sin(X) + np.random.normal(size=(100, 1)) * 0.3
    y = y.flatten()

    model = XGBoost(n_estimators=20, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    y_pred = model.predict(X)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label="Data with Noise")
    plt.plot(X, np.sin(X), label="True function")
    plt.plot(X, y_pred, label="XGBoost Prediction")
    plt.legend()
    plt.title("XGBoost Regression with NumPy")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()
