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

"""Device: Hybrid JAX + SQL I/O pipeline

Learning objectives:
1. Use basic.read/write for table/tensor I/O on devices
2. Convert PPU tables to tensors and move to SPU
3. Run a JAX numeric computation on SPU and persist results

Pipeline overview:
- On P0 and P1: write small tables to file:// (CSV) using basic.write
- Read them back on P0/P1 with basic.read (schema provided)
- Convert tables to dense tensors (basic.table_to_tensor)
- Move tensors to SPU and run JAX computation
- Bring result back to P0 and write to file:// as .npy via basic.write
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import pandas as pd

import mplang.v1 as mp
from mplang.v1.core.dtypes import FLOAT64, INT64
from mplang.v1.ops import basic as basic_ops

cluster_spec = mp.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61920"},
        {"name": "node_1", "endpoint": "127.0.0.1:61921"},
        {"name": "node_2", "endpoint": "127.0.0.1:61922"},
    ],
    "devices": {
        "SP0": {
            "kind": "SPU",
            "members": ["node_0", "node_1", "node_2"],
            "config": {"protocol": "SEMI2K", "field": "FM128"},
        },
        "P0": {"kind": "PPU", "members": ["node_0"]},
        "P1": {"kind": "PPU", "members": ["node_1"]},
        "TEE0": {"kind": "TEE", "members": ["node_2"]},
    },
})


def prepare_inputs():
    """Create small tables on P0/P1 and persist via basic.write (CSV files).

    This stage is separated to ensure writes complete before subsequent reads.
    """
    _ = os.makedirs("tmp", exist_ok=True)
    base_dir = os.path.abspath("tmp")
    df_p0 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_p1 = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
    # Host-side seed of CSV inputs for P0/P1
    df_p0.to_csv(f"{base_dir}/p0_table.csv", index=False)
    df_p1.to_csv(f"{base_dir}/p1_table.csv", index=False)


@mp.function
def simple_pipeline():
    """Simple pipeline: PPU tables -> read -> tensor -> SPU JAX compute -> write result.

    Steps:
    1) Read tables on P0/P1 with basic.read + explicit schema
    2) Convert tables to dense tensors using basic.table_to_tensor
    3) Move tensors to SPU and compute a scalar sum using JAX
    4) Return result on P0 and persist as .npy via basic.write
    """
    # 1) Read back as tables on each PPU with explicit schema
    schema = mp.TableType.from_dict({"a": INT64, "b": INT64})
    base_dir = os.path.abspath("tmp")
    tbl_p0 = mp.device("P0")(basic_ops.read)(path=f"{base_dir}/p0_table.csv", ty=schema)
    tbl_p1 = mp.device("P1")(basic_ops.read)(path=f"{base_dir}/p1_table.csv", ty=schema)

    # 2) Convert to dense tensors (shape: (N, 2))
    t_p0 = mp.device("P0")(basic_ops.table_to_tensor)(tbl_p0, number_rows=3)
    t_p1 = mp.device("P1")(basic_ops.table_to_tensor)(tbl_p1, number_rows=3)

    # 3) Move tensors to SPU and compute scalar sum via JAX
    t_spu_0 = mp.put("SP0", t_p0)
    t_spu_1 = mp.put("SP0", t_p1)
    sum_spu = mp.device("SP0")(lambda a, b: jnp.sum(a) + jnp.sum(b))(t_spu_0, t_spu_1)

    # 4) Bring result to P0 and write to file as .npy
    res_p0 = mp.put("P0", sum_spu)
    base_dir = os.path.abspath("tmp")
    _ = mp.device("P0")(basic_ops.write)(res_p0, path=f"{base_dir}/hybrid_sum.npy")

    return res_p0


@mp.function
def ml_pipeline(
    alice_csv: str, bob_csv: str, n_rows: int
) -> tuple[mp.MPObject, mp.MPObject, mp.MPObject]:
    """Read vertically split features and run a tiny logistic regression on SPU.

    - Reads alice.csv (f1,f2) on P0 and bob.csv (f3,f4) on P1
    - Converts to tensors and moves to SPU
    - Creates a synthetic binary label from feature f4 threshold
    - Trains a simple logistic regression with gradient descent
    - Returns learned weights, bias and training accuracy back to P0
    """
    # Read CSVs on respective PPUs as tables with explicit FLOAT64 schema
    schema_alice = mp.TableType.from_dict({"f1": FLOAT64, "f2": FLOAT64})
    schema_bob = mp.TableType.from_dict({"f3": FLOAT64, "f4": FLOAT64})
    tbl_a = mp.device("P0")(basic_ops.read)(path=alice_csv, ty=schema_alice)
    tbl_b = mp.device("P1")(basic_ops.read)(path=bob_csv, ty=schema_bob)

    # Convert to dense tensors (N,2) on each party
    ta = mp.device("P0")(basic_ops.table_to_tensor)(tbl_a, number_rows=n_rows)
    tb = mp.device("P1")(basic_ops.table_to_tensor)(tbl_b, number_rows=n_rows)

    # Move tensors to SPU and run a tiny logistic regression using JAX
    ta_s = mp.put("SP0", ta)
    tb_s = mp.put("SP0", tb)

    def train_logistic_regression(a: jnp.ndarray, b: jnp.ndarray):
        X = jnp.concatenate([a, b], axis=1).astype(jnp.float64)  # (N, 4)
        # Synthetic binary label on raw features: 1 if f4 > mean(f4), else 0
        y = (X[:, 3] > jnp.mean(X[:, 3])).astype(jnp.float64)  # (N,)

        # Standardize features to improve conditioning
        x_mean = jnp.mean(X, axis=0)
        x_std = jnp.std(X, axis=0) + 1e-6
        Xn = (X - x_mean) / x_std

        # Initialize parameters
        w = jnp.zeros((Xn.shape[1],), dtype=jnp.float64)
        b0 = jnp.array(0.0, dtype=jnp.float64)

        def loss_fn(w_, b_):
            z = Xn @ w_ + b_
            p = 1.0 / (1.0 + jnp.exp(-z))
            eps = 1e-7
            ce = -jnp.mean(y * jnp.log(p + eps) + (1.0 - y) * jnp.log(1.0 - p + eps))
            l2 = 1e-3 * jnp.sum(w_ * w_)
            return ce + l2

        # Gradient descent with compiled loop for faster compile/run
        lr = 0.1
        steps = 20
        grad_loss = jax.grad(loss_fn, argnums=(0, 1))

        def body_fun(_, carry):
            ww, bb = carry
            dw, db = grad_loss(ww, bb)
            return (ww - lr * dw, bb - lr * db)

        w, b0 = jax.lax.fori_loop(0, steps, body_fun, (w, b0))

        # Compute training accuracy
        z = Xn @ w + b0
        p = 1.0 / (1.0 + jnp.exp(-z))
        acc = jnp.mean((p > 0.5) == (y > 0.5))
        return w, b0, acc

    w_s, b_s, acc_s = mp.device("SP0")(train_logistic_regression)(ta_s, tb_s)

    # Return to P0 for fetching/printing
    return mp.put("P0", w_s), mp.put("P0", b_s), mp.put("P0", acc_s)


def main():
    print("=" * 70)
    print("Device: Hybrid JAX + SQL I/O pipeline")
    print("=" * 70)

    sim = mp.Simulator(cluster_spec)

    print("\n--- Stage 1: Prepare inputs (write CSV via basic.write) ---")
    prepare_inputs()

    print("\n--- Stage 2: Simple pipeline: PPU read + SPU JAX + write result ---")
    r = mp.evaluate(sim, simple_pipeline)
    out = mp.fetch(sim, r)
    print("SPU JAX scalar sum:")
    print(out)
    print("Result persisted to file: tmp/hybrid_sum.npy (NumPy .npy)")

    # ---------------------------------------------------------------
    # New: Vertical split dataset from sklearn -> CSV -> SPU logistic regression
    # ---------------------------------------------------------------
    print("\n--- Stage 3: Vertical split (alice/bob) from tutorials/data ---")
    # Expect prepared CSVs under tutorials/data. If missing, instruct user to run
    # tutorials/data/prepare_vertical_iris.py
    alice_csv = "tutorials/data/alice.csv"
    bob_csv = "tutorials/data/bob.csv"
    assert os.path.exists(alice_csv) and os.path.exists(bob_csv)

    # Determine row count by reading CSV (minus header)
    # Note:
    #   MPLang requires static (compile-time known) shapes. When converting a table
    #   to a tensor we must know the number of rows (n_rows) upfront.
    #   CSV files don't carry row-count metadata, so we do a one-pass scan here.
    #   This is fine for a tutorial but not ideal for production.
    #   Prefer columnar formats with rich metadata (e.g., Parquet, Arrow/Feather)
    #   where n_rows is available without scanning, which better fits static-shape needs.
    import csv

    with open(alice_csv) as f:
        n_rows = sum(1 for _ in csv.reader(f)) - 1
    r_w, r_b, r_acc = mp.evaluate(sim, ml_pipeline, alice_csv, bob_csv, n_rows)
    w_np = mp.fetch(sim, r_w)
    b_np = mp.fetch(sim, r_b)
    acc_np = mp.fetch(sim, r_acc)
    print("\nLogistic regression on SPU (from tutorials/data/*.csv):")
    print("weights (w):", w_np)
    print("bias (b):", b_np)
    print("train accuracy:", acc_np)

    # Finished Stage 3

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. basic.read/write: device-side resource I/O (file://, mem://)")
    print("2. table_to_tensor: bridge tables to tensors for JAX/ML pipelines")
    print("3. SPU: secure multi-party compute for JAX ops; use mp.put for movement")
    print(
        "4. Vertical split workflow: per-party CSV -> read -> tensor -> SPU ML (logreg)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
