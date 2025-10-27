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

"""Prepare a small vertical-split dataset for tutorials.

This script writes two CSV files into tutorials/data:
- alice.csv: columns f1, f2
- bob.csv:   columns f3, f4

By default we use sklearn's iris dataset (150x4). If sklearn isn't available,
we fall back to a simple synthetic 150x4 float matrix.

Usage:
  uv run python tutorials/data/prepare_vertical_iris.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_iris_like() -> tuple[pd.DataFrame, int]:
    """Return a DataFrame with 4 float columns named f1..f4 and row count.

    Tries sklearn.datasets.load_iris; falls back to synthetic data if needed.
    """
    try:
        from sklearn import datasets  # type: ignore

        iris = datasets.load_iris()
        raw = getattr(iris, "data", None)
        if raw is None:
            if isinstance(iris, (tuple, list)):
                raw = iris[0]
            else:
                try:
                    raw = iris["data"]  # type: ignore[index]
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(
                        "Unexpected sklearn iris structure; cannot extract data"
                    ) from exc
        X = np.asarray(raw, dtype=np.float64)
        n_rows = int(X.shape[0])
        df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
        return df, n_rows
    except Exception:
        # Fallback: deterministic synthetic data (150x4)
        n_rows = 150
        X = np.linspace(0.0, 14.9, n_rows * 4, dtype=np.float64).reshape(n_rows, 4)
        df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
        return df, n_rows


def prepare(output_dir: str | os.PathLike[str]) -> tuple[str, str, int]:
    os.makedirs(output_dir, exist_ok=True)
    df, n_rows = load_iris_like()
    alice_path = str(Path(output_dir) / "alice.csv")
    bob_path = str(Path(output_dir) / "bob.csv")
    df[["f1", "f2"]].to_csv(alice_path, index=False)
    df[["f3", "f4"]].to_csv(bob_path, index=False)
    return alice_path, bob_path, n_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare vertical iris dataset")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent),
        help="Output directory (default: tutorials/data)",
    )
    args = parser.parse_args()
    alice, bob, n_rows = prepare(args.out)
    print(f"Wrote {n_rows} rows to:\n  {alice}\n  {bob}")


if __name__ == "__main__":
    main()
