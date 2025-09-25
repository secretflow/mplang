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

"""Tutorial 10: IR & Graph Analysis Report Generation.

This tutorial demonstrates how to generate a unified analysis report for a
multi-party device function using:
  * textual compiler IR
  * structural graph statistics (always included)
  * optional Mermaid sequence / flowchart diagrams

Artifacts written:
  * simple.mpir - raw MPIR graph proto (text form)
  * simple.md   - markdown report (IR + stats + diagrams)

The previous Mermaid-focused tutorial (10_mermaid.py) has been replaced by this
more general analysis-centric version.
"""

from __future__ import annotations

import random

import mplang
import mplang.device as mpd

cluster_spec = mplang.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61930"},
        {"name": "node_1", "endpoint": "127.0.0.1:61931"},
        {"name": "node_2", "endpoint": "127.0.0.1:61932"},
    ],
    "devices": {
        "SP0": {
            "kind": "SPU",
            "members": ["node_0", "node_1", "node_2"],
            "config": {"protocol": "SEMI2K", "field": "FM128"},
        },
        "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
        "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
        "TEE0": {"kind": "TEE", "members": ["node_2"], "config": {}},
    },
})


@mpd.function
def millionaire_device():
    x = mpd.device("P0")(random.randint)(0, 100)
    y = mpd.device("P1")(random.randint)(0, 100)
    z = mpd.device("TEE0")(lambda a, b: a < b)(x, y)
    r = mpd.put("P0", z)
    return x, y, z, r


if __name__ == "__main__":
    sim = mplang.Simulator(cluster_spec)
    traced_dev = mplang.compile(sim, millionaire_device)
    mplang.analysis.dump(
        traced_dev,
        cluster_spec=cluster_spec,
        sequence=True,
        flow=True,
        include_ir=True,
        report_path="simple.md",
        mpir_path="simple.mpir",
        title="Millionaire Device Analysis",
        flow_opts={
            "direction": "TB",
            "cluster_by_party": True,
        },
        seq_opts={
            "collapse_local": False,
        },
    )
    print("Generated simple.mpir and simple.md")
