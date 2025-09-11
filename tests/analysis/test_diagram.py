import re

import mplang
import mplang.simp as simp
from mplang import analysis
from mplang.core.cluster import ClusterSpec
from mplang.core.tracer import TraceContext, TracedFunction, trace


def _toy():  # simple function for viz
    x = simp.prandint(0, 10)
    y = simp.prandint(0, 10)
    x_ = simp.sealFrom(x, 0)
    y_ = simp.sealFrom(y, 1)
    z_ = simp.srun(lambda a, b: a < b)(x_, y_)
    z = simp.reveal(z_)
    return z


def test_sequence_diagram_basic():
    copts = mplang.CompileOptions.simple(2)
    traced = mplang.compile(copts, _toy)
    diagram = analysis.to_sequence_diagram(traced, world_size=2)
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
    content = analysis.dump(
        traced,
        world_size=1,
        sequence=True,
        flow=True,
        include_ir=True,
        report_path=str(md_path),
        mpir_path=str(mpir_path),
        title="Test Report",
    )

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
