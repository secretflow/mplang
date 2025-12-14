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

"""Device: IR Dump and Graph Inspection

Learning objectives:
1. Compile functions to Graph IR for inspection
2. Understand the traced IR representation
3. Debug and understand multi-party program structure

Key tools:
- mp.compile: generate Graph IR representation without executing
- traced.compiler_ir(): get human-readable IR text
- mp.format_graph: format graph for debugging

Migrated from mplang v1 to mplang2.
"""

import mplang.v2 as mp

cluster_spec = mp.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61930"},
        {"name": "node_1", "endpoint": "127.0.0.1:61931"},
    ],
    "devices": {
        "SP0": {
            "kind": "SPU",
            "members": ["node_0", "node_1"],
            "config": {"protocol": "SEMI2K", "field": "FM128"},
        },
        "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
        "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
    },
})


def millionaire():
    """Simple millionaire problem for IR inspection."""
    x = mp.put("P0", 100)
    y = mp.put("P1", 200)
    z = mp.device("SP0")(lambda a, b: a < b)(x, y)
    r = mp.put("P0", z)
    return x, y, z, r


@mp.function
def jitted_millionaire():
    """JIT version of millionaire for comparison."""
    x = mp.put("P0", 100)
    y = mp.put("P1", 200)
    z = mp.device("SP0")(lambda a, b: a < b)(x, y)
    r = mp.put("P0", z)
    return x, y, z, r


def main():
    print("=" * 70)
    print("Device: IR Dump and Graph Inspection")
    print("=" * 70)

    sim = mp.make_simulator(2, cluster_spec=cluster_spec)
    mp.set_root_context(sim)

    # Pattern 1: Compile without executing to get IR
    print("\n--- Pattern 1: Compile to IR ---")
    traced = mp.compile(millionaire)
    print("Traced function name:", traced.name)
    print("Number of graph inputs:", len(traced.graph.inputs))
    print("Number of graph outputs:", len(traced.graph.outputs))
    print("Number of operations:", len(traced.graph.operations))

    # Pattern 2: Print human-readable IR
    print("\n--- Pattern 2: Human-readable IR ---")
    ir_text = traced.compiler_ir()
    print(ir_text)

    # Pattern 3: Verbose IR with type annotations
    print("\n--- Pattern 3: Verbose IR with types ---")
    ir_verbose = traced.compiler_ir(verbose=True)
    # Just print first 20 lines to keep output manageable
    lines = ir_verbose.split("\n")[:20]
    for line in lines:
        print(line)
    if len(ir_verbose.split("\n")) > 20:
        print("  ...")

    # Pattern 4: Execute and compare
    print("\n--- Pattern 4: Execute and verify ---")
    x, y, z, r = mp.evaluate(millionaire)
    print(
        f"Results: x={mp.fetch(x)}, y={mp.fetch(y)}, z={mp.fetch(z)}, r={mp.fetch(r)}"
    )

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. mp.compile: trace function to IR without executing")
    print("2. compiler_ir(): inspect operations, inputs, outputs")
    print("3. IR helps debug data flow and device placement")
    print("=" * 70)


if __name__ == "__main__":
    main()
