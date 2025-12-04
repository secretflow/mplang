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

"""Device: SQL on PPU and TEE

Learning objectives:
1. Execute SQL queries on PPU for plaintext table processing
2. Execute SQL queries on TEE for secure data processing
3. Understand table placement and privacy domains

Key concepts:
- PPU: fast SQL, no privacy protection
- TEE: SQL in trusted execution environment
- TableType: define table schema for SQL operations (schema can be inferred)
- run_sql: execute DuckDB SQL queries with automatic schema inference
"""

import pandas as pd

import mplang.v1 as mp
from mplang.v1.core.dtypes import INT64
from mplang.v1.ops import sql_cc

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


@mp.function
def sql_on_ppu():
    """Pattern 1: Simple SQL query on plaintext data."""
    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10, 20, 30, 40, 50],
    })

    # For demonstration, place the table on P0 from host
    # In real scenarios, data may load from P0 directly
    input_table = mp.put("P0", data)

    # Run SQL on P0: double the values
    query = "SELECT id, value * 2 as doubled FROM input_table"
    # Schema is inferred automatically by run_sql
    result = mp.device("P0")(sql_cc.run_sql)(query, input_table=input_table)

    return result


@mp.function
def sql_on_tee():
    """Pattern 2: Execute SQL on TEE from a device-layer function.

    This function demonstrates placing each party's plaintext tables onto TEE and
    then invoking SQL inside the TEE device context via ``mp.device("TEE0")(...)``.
    It returns the UNION result table directly from the TEE.
    """
    # Two parties provide private data (constructed on host, then placed)
    raw_p0 = pd.DataFrame({
        "user_id": [1, 2],
        "amount": [100, 200],
    })

    raw_p1 = pd.DataFrame({
        "user_id": [3, 4],
        "amount": [150, 250],
    })

    # Place original data to P0 and P1 first.
    # In real scenarios, each party would load/generate data on its own PPU.
    data_p0 = mp.put("P0", raw_p0)
    data_p1 = mp.put("P1", raw_p1)

    # Move the placed tables to TEE0 via device-to-device transfer.
    # The current implementation supports PPU -> TEE table transfer (via _d2d with
    # transparent encryption handshake and transport), so we migrate PPU tables
    # to TEE directly here.
    tbl_tee_0 = mp.put("TEE0", data_p0)
    tbl_tee_1 = mp.put("TEE0", data_p1)

    # Run SQL UNION on TEE (inside device context) and return the result table
    query = "SELECT * FROM tbl1 UNION ALL SELECT * FROM tbl2 ORDER BY user_id"
    result = mp.device("TEE0")(sql_cc.run_sql)(query, tbl1=tbl_tee_0, tbl2=tbl_tee_1)

    return result


# Removed: Pattern 4 (single-party CASE query) to keep tutorial concise


def main():
    print("=" * 70)
    print("Device: SQL on PPU and TEE")
    print("=" * 70)

    # Helper to check inferred schema matches expectation
    def assert_schema(
        mpobj: mp.MPObject, expected: dict[str, mp.DType], label: str
    ) -> None:
        exp = mp.TableType.from_dict(expected)
        got = mpobj.schema
        if got != exp:
            raise AssertionError(f"{label} schema mismatch: expected {exp}, got {got}")

    sim = mp.Simulator(cluster_spec)

    # Pattern 1: PPU
    print("\n--- Pattern 1: SQL on PPU ---")
    r1 = mp.evaluate(sim, sql_on_ppu)
    # Check inferred schema
    assert_schema(r1, {"id": INT64, "doubled": INT64}, "Pattern 1")
    result1 = mp.fetch(sim, r1)
    print("Result (doubled values):")
    print(result1)

    # Pattern 2: TEE
    print("\n--- Pattern 2: SQL on TEE ---")
    # Bind mock TEE attest/quote for local simulation
    tee_bindings = {
        "tee.quote_gen": "mock_tee.quote_gen",
        "tee.attest": "mock_tee.attest",
    }
    for n in cluster_spec.nodes.values():
        n.runtime_info.op_bindings.update(tee_bindings)
    sim_tee = mp.Simulator(cluster_spec)
    # sql_on_tee already executes the UNION query inside TEE and returns the result table
    r2 = mp.evaluate(sim_tee, sql_on_tee)
    assert_schema(r2, {"user_id": INT64, "amount": INT64}, "Pattern 2")
    result2 = mp.fetch(sim_tee, r2)
    print("TEE UNION result (combined rows from P0 and P1):")
    print(result2)

    # Note: See tutorials/device/05_pipeline.py for a hybrid
    # PPU->SPU pipeline demo using basic.read/write and JAX on SPU.

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. PPU: fast plaintext SQL, no privacy")
    print("2. TEE: SQL in trusted isolated environment")
    print(
        "3. sql_cc.run_sql: execute DuckDB queries on tables (schema inferred via sqlglot)"
    )
    print("4. TableType: define schema for SQL inputs/outputs")
    print("5. mp.put: move tables between devices safely")
    print("6. SQL ops: WHERE, UNION, SUM, AVG, ORDER BY")
    print("=" * 70)


if __name__ == "__main__":
    main()
