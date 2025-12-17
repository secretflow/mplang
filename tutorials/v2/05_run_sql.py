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

"""SQL on PPU and TEE (MPLang2 version)

Learning objectives:
1. Execute SQL queries on PPU for plaintext table processing
2. Execute SQL queries on TEE for secure data processing
3. Understand table placement and schema definition

Key concepts:
- PPU: fast SQL, no privacy protection
- TEE: SQL in trusted execution environment
- TableType: define table schema for SQL operations
- table.run_sql: execute DuckDB SQL queries

API patterns:
- table.constant({"col": [values]}): create table from dict
- table.run_sql(query, out_type=schema, tbl=table): run SQL
- TableType({"col": dtype}) or Table[{"col": dtype}]: define schema

Migration notes (mplang -> mplang2):
- from mplang2.dialects import table (instead of mplang.ops.sql_cc)
- Explicit out_type parameter required (no auto-inference in v2)
- Use table.constant() instead of mp.put() for creating tables in trace
"""

import mplang.v2 as mp
from mplang.v2.dialects import table
from mplang.v2.edsl.typing import TableType, i64

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


# Define reusable schemas using TableType() or Table[{}] syntax
# Schema values should be scalar types (i64, f64, STRING) not TensorType
input_schema = TableType({
    "id": i64,
    "value": i64,
})

doubled_schema = TableType({
    "id": i64,
    "doubled": i64,
})

union_schema = TableType({
    "user_id": i64,
    "amount": i64,
})


@mp.function
def sql_on_ppu():
    """Pattern 1: Simple SQL query on plaintext data."""

    # Create table and run SQL on P0
    @mp.device("P0")
    def run_query():
        # Create table constant inside device context
        input_table = table.constant({
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })

        # Run SQL: double the values
        query = "SELECT id, value * 2 as doubled FROM input_table"
        result = table.run_sql(
            query,
            out_type=doubled_schema,
            input_table=input_table,
        )
        return result

    return run_query()


@mp.function
def sql_on_tee():
    """Pattern 2: Execute SQL on TEE with data from multiple parties.

    Two parties provide private data, which is transferred to TEE
    for secure SQL processing.
    """

    @mp.device("TEE0")
    def run_tee_query():
        # Create tables inside TEE context
        tbl1 = table.constant({
            "user_id": [1, 2],
            "amount": [100, 200],
        })

        tbl2 = table.constant({
            "user_id": [3, 4],
            "amount": [150, 250],
        })

        # Run SQL UNION on TEE
        query = "SELECT * FROM tbl1 UNION ALL SELECT * FROM tbl2 ORDER BY user_id"
        result = table.run_sql(
            query,
            out_type=union_schema,
            tbl1=tbl1,
            tbl2=tbl2,
        )
        return result

    return run_tee_query()


def main():
    print("=" * 70)
    print("SQL on PPU and TEE (MPLang2)")
    print("=" * 70)

    # Pattern 1: PPU
    print("\n--- Pattern 1: SQL on PPU ---")
    sim_ppu = mp.make_simulator(3, cluster_spec=cluster_spec)
    with sim_ppu:
        r1 = mp.evaluate(sql_on_ppu)
        # TODO: mp.fetch(follow_device=True) doesn't work here because Object
        # attributes (including __device__) are lost across mp.evaluate boundary.
        # TracedFunction doesn't preserve Object attributes when reconstructing outputs.
        # Workaround: manually index by rank (P0 = rank 0)
        result1 = mp.fetch(r1)[0]
        print("Result (doubled values):")
        print(result1.to_pandas())

        # Verify schema
        print(f"Schema: {r1.type}")

    # Pattern 2: TEE
    print("\n--- Pattern 2: SQL on TEE ---")
    # Bind mock TEE attest/quote for local simulation
    tee_bindings = {
        "tee.quote_gen": "mock_tee.quote_gen",
        "tee.attest": "mock_tee.attest",
    }
    for n in cluster_spec.nodes.values():
        n.runtime_info.op_bindings.update(tee_bindings)

    sim_tee = mp.make_simulator(3, cluster_spec=cluster_spec)
    with sim_tee:
        r2 = mp.evaluate(sql_on_tee)
        # Workaround: manually index by rank (TEE0 = rank 2)
        result2 = mp.fetch(r2)[2]
        print("TEE UNION result (combined rows from P0 and P1):")
        if result2 is not None:
            print(result2.to_pandas())
        else:
            print("(TEE result is None - mock TEE may not fully support table ops)")
            print("Note: TEE table operations require proper attestation setup")

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. PPU: fast plaintext SQL, no privacy")
    print("2. TEE: SQL in trusted isolated environment")
    print("3. table.run_sql: execute DuckDB queries on tables")
    print("4. TableType(): define schema for SQL inputs/outputs")
    print("5. table.constant(): create table values in traced functions")
    print("=" * 70)


if __name__ == "__main__":
    main()
