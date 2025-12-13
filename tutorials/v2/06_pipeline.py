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

"""Hybrid JAX + Table I/O Pipeline (MPLang2 version)

Learning objectives:
1. Use table.read/write for table I/O on devices
2. Convert PPU tables to tensors and move to SPU
3. Run a JAX numeric computation on SPU
4. Persist results back to file

Pipeline overview:
- Prepare CSV files on host
- Read tables on P0/P1 with table.read (schema required)
- Convert tables to tensors (table.table2tensor)
- Move tensors to SPU and run JAX computation
- Return result to P0 and write to file via table.write

Migration notes (mplang -> mplang2):
- basic.read/write → table.read/write
- basic.table_to_tensor → table.table2tensor
- mp.TableType.from_dict() → TableType({"col": dtype})
- Schema uses scalar types directly (i64, f64) not TensorType wrappers
"""

from __future__ import annotations

import csv
import os

import jax
import jax.numpy as jnp
import pandas as pd

import mplang.v2 as mp
from mplang.v2.dialects import table
from mplang.v2.edsl.typing import TableType, f64, i64

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
    },
})


def prepare_inputs():
    """Create small CSV tables for P0/P1."""
    os.makedirs("tmp", exist_ok=True)
    base_dir = os.path.abspath("tmp")

    df_p0 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_p1 = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

    df_p0.to_csv(f"{base_dir}/p0_table.csv", index=False)
    df_p1.to_csv(f"{base_dir}/p1_table.csv", index=False)

    print(f"Created: {base_dir}/p0_table.csv")
    print(f"Created: {base_dir}/p1_table.csv")


@mp.function
def simple_pipeline():
    """Simple pipeline: read tables -> convert to tensor -> SPU compute -> write result.

    Steps:
    1) Read tables on P0/P1 with table.read + explicit schema
    2) Convert tables to tensors using table.table2tensor
    3) Move tensors to SPU and compute a scalar sum using JAX
    4) Return result to P0
    """
    # Schema definition - use scalar types directly
    schema = TableType({"a": i64, "b": i64})
    base_dir = os.path.abspath("tmp")

    # 1) Read tables on each PPU
    @mp.device("P0")
    def read_p0():
        return table.read(f"{base_dir}/p0_table.csv", schema=schema, format="csv")

    @mp.device("P1")
    def read_p1():
        return table.read(f"{base_dir}/p1_table.csv", schema=schema, format="csv")

    tbl_p0 = read_p0()
    tbl_p1 = read_p1()

    # 2) Convert to dense tensors (shape: (3, 2))
    @mp.device("P0")
    def to_tensor_p0(tbl):
        return table.table2tensor(tbl, number_rows=3)

    @mp.device("P1")
    def to_tensor_p1(tbl):
        return table.table2tensor(tbl, number_rows=3)

    t_p0 = to_tensor_p0(tbl_p0)
    t_p1 = to_tensor_p1(tbl_p1)

    # 3) Move tensors to SPU and compute scalar sum via JAX
    t_spu_0 = mp.put("SP0", t_p0)
    t_spu_1 = mp.put("SP0", t_p1)

    @mp.device("SP0")
    def spu_sum(a, b):
        return jnp.sum(a) + jnp.sum(b)

    sum_spu = spu_sum(t_spu_0, t_spu_1)

    # 4) Return result to P0
    return mp.put("P0", sum_spu)


@mp.function
def ml_pipeline(alice_csv: str, bob_csv: str, n_rows: int):
    """Read vertically split features and run logistic regression on SPU.

    - Reads alice.csv (f1, f2) on P0 and bob.csv (f3, f4) on P1
    - Converts to tensors and moves to SPU
    - Creates synthetic binary label from feature f4 threshold
    - Trains logistic regression with gradient descent
    - Returns learned weights, bias and training accuracy
    """
    # Schema definitions
    schema_alice = TableType({"f1": f64, "f2": f64})
    schema_bob = TableType({"f3": f64, "f4": f64})

    # Read tables on respective PPUs
    @mp.device("P0")
    def read_alice():
        return table.read(alice_csv, schema=schema_alice, format="csv")

    @mp.device("P1")
    def read_bob():
        return table.read(bob_csv, schema=schema_bob, format="csv")

    tbl_a = read_alice()
    tbl_b = read_bob()

    # Convert to dense tensors (N, 2) on each party
    @mp.device("P0")
    def to_tensor_a(tbl):
        return table.table2tensor(tbl, number_rows=n_rows)

    @mp.device("P1")
    def to_tensor_b(tbl):
        return table.table2tensor(tbl, number_rows=n_rows)

    ta = to_tensor_a(tbl_a)
    tb = to_tensor_b(tbl_b)

    # Move tensors to SPU
    ta_s = mp.put("SP0", ta)
    tb_s = mp.put("SP0", tb)

    @mp.device("SP0")
    def train_logistic_regression(a, b):
        X = jnp.concatenate([a, b], axis=1).astype(jnp.float64)  # (N, 4)
        # Synthetic binary label: 1 if f4 > mean(f4), else 0
        y = (X[:, 3] > jnp.mean(X[:, 3])).astype(jnp.float64)  # (N,)

        # Standardize features
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

        # Gradient descent
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

    w_s, b_s, acc_s = train_logistic_regression(ta_s, tb_s)

    # Return to P0
    return mp.put("P0", w_s), mp.put("P0", b_s), mp.put("P0", acc_s)


def main():
    print("=" * 70)
    print("Hybrid JAX + Table I/O Pipeline (MPLang2)")
    print("=" * 70)

    sim = mp.make_simulator(3, cluster_spec=cluster_spec)
    mp.set_root_context(sim)

    # Stage 1: Prepare inputs
    print("\n--- Stage 1: Prepare inputs (write CSV files) ---")
    prepare_inputs()

    # Stage 2: Simple pipeline
    print("\n--- Stage 2: Simple pipeline: read -> tensor -> SPU JAX -> result ---")
    r = mp.evaluate(simple_pipeline)
    out = mp.fetch(r)
    if isinstance(out, list):
        out = out[0]
    print(f"SPU JAX scalar sum: {out}")
    # Expected: sum([1,2,3,4,5,6]) + sum([7,8,9,10,11,12]) = 21 + 57 = 78

    # Stage 3: ML pipeline with vertical split
    print("\n--- Stage 3: Vertical split logistic regression on SPU ---")
    alice_csv = "tutorials/data/alice.csv"
    bob_csv = "tutorials/data/bob.csv"

    if os.path.exists(alice_csv) and os.path.exists(bob_csv):
        # Count rows (excluding header)
        with open(alice_csv) as f:
            n_rows = sum(1 for _ in csv.reader(f)) - 1

        r_w, r_b, r_acc = mp.evaluate(ml_pipeline, alice_csv, bob_csv, n_rows)
        w_np = mp.fetch(r_w)
        b_np = mp.fetch(r_b)
        acc_np = mp.fetch(r_acc)

        print("\nLogistic regression on SPU (from tutorials/data/*.csv):")
        print("weights (w):", w_np)
        print("bias (b):", b_np)
        print("train accuracy:", acc_np)
    else:
        print(f"Skipping: {alice_csv} or {bob_csv} not found.")
        print("Run: python tutorials/data/prepare_vertical_iris.py to generate data.")

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. table.read/write: device-side file I/O (CSV, Parquet)")
    print("2. table.table2tensor: convert tables to tensors for JAX/ML")
    print("3. SPU: secure multi-party compute; use mp.put() for data movement")
    print("4. Vertical split: per-party data -> read -> tensor -> SPU ML")
    print("5. Schema: TableType with scalar types (i64, f64) required for read")
    print("=" * 70)


if __name__ == "__main__":
    main()
