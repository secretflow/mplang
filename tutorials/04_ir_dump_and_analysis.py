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

Migrated from mplang v1 to mplang.
"""

import mplang as mp

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
    """@mp.function-wrapped version for comparison.

    Note: @mp.function is a multi-party wrapper (pcall_static over all parties),
    not a traditional single-process JIT compiler. It changes the IR structure
    by introducing a pcall region.
    """
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

    # Pattern 1: Compile without executing (backend) to get IR
    print("\n--- Pattern 1: Compile to IR (plain function) ---")
    traced_plain = mp.compile(millionaire)
    print("Traced function name:", traced_plain.name)
    print("Number of graph inputs:", len(traced_plain.graph.inputs))
    print("Number of graph outputs:", len(traced_plain.graph.outputs))
    print("Number of operations:", len(traced_plain.graph.operations))

    # Pattern 1b: Compare with @mp.function wrapper
    print("\n--- Pattern 1b: Compile to IR (@mp.function wrapper) ---")
    traced_pcall = mp.compile(jitted_millionaire)
    print("Traced function name:", traced_pcall.name)
    print("Number of graph inputs:", len(traced_pcall.graph.inputs))
    print("Number of graph outputs:", len(traced_pcall.graph.outputs))
    print("Number of operations:", len(traced_pcall.graph.operations))

    # Pattern 2: Human-readable IR (compact)
    print("\n--- Pattern 2: Human-readable IR (compact) ---")
    print("[plain]")
    print(traced_plain.compiler_ir())
    print("\n[@mp.function]")
    print(traced_pcall.compiler_ir())

    # Pattern 3: Human-readable IR (verbose types)
    print("\n--- Pattern 3: Human-readable IR (verbose types) ---")
    print("[plain]")
    ir_verbose_plain = traced_plain.compiler_ir(verbose=True)
    all_lines = ir_verbose_plain.split("\n")
    lines = all_lines[:20]
    for line in lines:
        print(line)
    if len(all_lines) > 20:
        print("  ...")

    print("\n[@mp.function]")
    ir_verbose_pcall = traced_pcall.compiler_ir(verbose=True)
    lines = ir_verbose_pcall.split("\n")[:20]
    for line in lines:
        print(line)
    if len(ir_verbose_pcall.split("\n")) > 20:
        print("  ...")

    # Pattern 3b: Graph pretty-print (compact vs verbose, with attrs)
    print("\n--- Pattern 3b: mp.format_graph (compact vs verbose, attrs) ---")
    print("[plain / compact]")
    print(mp.format_graph(traced_plain.graph, show_types=False, show_attrs=False))
    print("\n[plain / verbose]")
    print(mp.format_graph(traced_plain.graph, show_types=True, show_attrs=True))
    print("\n[@mp.function / compact]")
    print(mp.format_graph(traced_pcall.graph, show_types=False, show_attrs=False))
    print("\n[@mp.function / verbose]")
    print(mp.format_graph(traced_pcall.graph, show_types=True, show_attrs=True))

    # Pattern 4: Execute and verify
    print("\n--- Pattern 4: Execute and verify ---")
    x, y, z, r = mp.evaluate(millionaire)
    print(
        f"[direct] x={mp.fetch(x)}, y={mp.fetch(y)}, z={mp.fetch(z)}, r={mp.fetch(r)}"
    )

    # Pattern 4b: Execute compiled (traced) graphs
    print("\n--- Pattern 4b: Execute traced graphs (compare with direct) ---")
    x2, y2, z2, r2 = mp.evaluate(traced_plain)
    print(
        f"[traced/plain] x={mp.fetch(x2)}, y={mp.fetch(y2)}, z={mp.fetch(z2)}, r={mp.fetch(r2)}"
    )

    x3, y3, z3, r3 = mp.evaluate(traced_pcall)
    print(
        f"[traced/@mp.function] x={mp.fetch(x3)}, y={mp.fetch(y3)}, z={mp.fetch(z3)}, r={mp.fetch(r3)}"
    )

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. mp.compile: trace function to IR without executing backend")
    print(
        "2. @mp.function: wraps your function into simp.pcall_static(ALL, fn, ...) and adds a region"
    )
    print(
        "3. compiler_ir()/format_graph: inspect operations, inputs/outputs, regions, attrs"
    )
    print("4. mp.evaluate can execute TracedFunction graphs directly")
    print("5. IR helps debug data flow and device placement")
    print("=" * 70)


if __name__ == "__main__":
    main()
