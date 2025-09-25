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

import re

from mplang.analysis import diagram as analysis
from mplang.core.cluster import ClusterSpec
from mplang.core.mpir import Writer
from mplang.core.tracer import TraceContext, TracedFunction, trace
from mplang.simp import prandint, reveal, sealFrom, srun


def _toy():  # simple function for viz
    x = prandint(0, 10)
    y = prandint(0, 10)
    x_ = sealFrom(x, 0)
    y_ = sealFrom(y, 1)
    z_ = srun(lambda a, b: a < b)(x_, y_)
    z = reveal(z_)
    return z


def test_sequence_diagram_basic():
    cluster = ClusterSpec.simple(2)
    tctx = TraceContext(cluster)
    traced: TracedFunction = trace(tctx, _toy)
    graph = Writer().dumps(traced.make_expr())
    diagram = analysis.to_sequence_diagram(graph, world_size=2)
    assert "sequenceDiagram" in diagram
    # Accept any backend-specific naming for sealing and revealing primitives.
    seal_tokens = ("seal", "makeshare", "makeshares")
    reveal_tokens = ("reveal", "reconstruct")
    assert any(t in diagram for t in seal_tokens), diagram
    assert any(t in diagram for t in reveal_tokens), diagram


def _dummy_fn(a, b):
    # a simple function to trace; multiplication ensures an op appears
    return a + b * 2


def test_dump_markdown_sections(tmp_path):
    # Trace the function using analysis internal mechanisms via to_flowchart (side effect: requires a traced fn)
    # We indirectly obtain a traced function by using the diagram utilities which accept python callables.
    # However, dump expects a traced function object; importing here to avoid broad public surface changes.
    # Build a simple MPContext and trace manually to avoid importing heavy frontends
    cluster = ClusterSpec.simple(1)
    tctx = TraceContext(cluster)
    traced: TracedFunction = trace(tctx, _dummy_fn, 1, 2)

    md_path = tmp_path / "report.md"
    mpir_path = tmp_path / "graph.mpir"
    result = analysis.dump(
        traced,
        cluster_spec=cluster,
        sequence=True,
        flow=True,
        include_ir=True,
        report_path=str(md_path),
        mpir_path=str(mpir_path),
        title="Test Report",
        seq_opts={"collapse_local": True},
        flow_opts={"direction": "LR"},
    )

    content = result.markdown

    # Load from file to ensure write path executed
    written = md_path.read_text(encoding="utf-8")
    assert written == content

    # Core sections
    assert "## Compiler IR (text)" in content
    assert "## Graph Structure Analysis" in content

    # Mermaid fenced blocks (at least one sequence and one flowchart)
    mermaid_blocks = re.findall(r"```mermaid[\s\S]*?```", content)
    assert len(mermaid_blocks) >= 1

    # Ensure mpir file was written and is non-empty
    assert mpir_path.exists()
    assert mpir_path.read_text(encoding="utf-8").strip() != ""

    # Structured result fields sanity
    assert result.ir is not None
    assert result.sequence is not None
    assert result.flow is not None
    assert "sequenceDiagram" in result.sequence
