"""Diagram rendering (Mermaid) and markdown dump helpers.

Moved from mplang.utils.mermaid to dedicated analysis namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mplang.core import TracedFunction
from mplang.core.mask import Mask
from mplang.core.mpir import Writer, get_graph_statistics
from mplang.protos.v1alpha1 import mpir_pb2

# ----------------------------- Core helpers (copied) -----------------------------


@dataclass
class Event:
    kind: str
    lines: list[str]


def _pmask_to_ranks(pmask: int) -> list[int]:
    return list(Mask(pmask)) if pmask >= 0 else []


def _collect_world_size(
    graph: mpir_pb2.GraphProto, explicit_world_size: int | None
) -> int:
    if explicit_world_size is not None:
        return explicit_world_size
    max_rank = -1
    for node in graph.nodes:
        for out in node.outs_info:
            if out.pmask >= 0:
                for r in _pmask_to_ranks(out.pmask):
                    max_rank = max(max_rank, r)
    return max_rank + 1 if max_rank >= 0 else 0


def _node_output_pmasks(node: mpir_pb2.NodeProto) -> list[int]:
    return [out.pmask for out in node.outs_info]


def _value_producers(graph: mpir_pb2.GraphProto) -> dict[str, mpir_pb2.NodeProto]:
    prod: dict[str, mpir_pb2.NodeProto] = {}
    for n in graph.nodes:
        base = n.name
        arity = max(1, len(n.outs_info))
        for i in range(arity):
            key = f"{base}" if arity == 1 else f"{base}:{i}"
            prod[key] = n
    return prod


# ----------------------------- Public APIs -----------------------------


def to_sequence_diagram(
    traced: TracedFunction,
    *,
    world_size: int | None = None,
    collapse_local: bool = False,
    show_compute: bool = False,
    max_local_batch: int = 8,
    show_meta: bool = False,
) -> str:
    """Render a traced function as a Mermaid sequenceDiagram."""
    expr = traced.make_expr()
    graph = Writer().dumps(expr)

    wsize = _collect_world_size(graph, world_size)
    value_producer = _value_producers(graph)

    participants: list[str] = [f"participant P{i}" for i in range(wsize)]
    events: list[Event] = []

    def emit(kind: str, *lines: str):
        events.append(Event(kind=kind, lines=list(lines)))

    local_buffer: list[str] = []

    def flush_local_buffer():
        nonlocal local_buffer
        if not local_buffer:
            return
        if collapse_local and not show_compute:

            def _strip(s: str) -> str:
                parts = s.split(": ", 1)
                return parts[1] if len(parts) == 2 else s

            sample = [_strip(s) for s in local_buffer[:max_local_batch]]
            more = (
                ""
                if len(local_buffer) <= max_local_batch
                else f" (+{len(local_buffer) - max_local_batch} more)"
            )
            emit("note", f"note over P0,P{wsize - 1}: {', '.join(sample)}{more}")
        else:
            for line in local_buffer:
                emit("local", line)
        local_buffer = []

    def owners_of(node: mpir_pb2.NodeProto) -> set[int]:
        ranks: set[int] = set()
        for pm in _node_output_pmasks(node):
            if pm >= 0:
                ranks.update(_pmask_to_ranks(pm))
        return ranks

    for node in graph.nodes:
        own = owners_of(node)
        input_ranks: set[int] = set()
        for val in node.inputs:
            prod = value_producer.get(val.split(":")[0])
            if prod:
                input_ranks.update(owners_of(prod))
        pfunc = node.attrs.get("pfunc") if node.op_type == "eval" else None
        if node.op_type == "eval" and pfunc:
            fn_name = pfunc.func.name or pfunc.func.type or "eval"
            label = f"{fn_name} {node.name}"
        elif node.op_type == "access":
            label = ""
        else:
            label = f"{node.op_type} {node.name}"
        value_suffix = ""
        if len(node.outs_info) > 1:
            value_suffix = f" -> {len(node.outs_info)} outs"
        if node.op_type == "access":
            continue
        if not show_meta and node.op_type in {"tuple", "func_def"}:
            continue
        if node.op_type == "shfl_s":
            flush_local_buffer()
            pmask_attr = node.attrs.get("pmask")
            src_ranks_attr = node.attrs.get("src_ranks")
            if pmask_attr and src_ranks_attr:
                dst_ranks = _pmask_to_ranks(pmask_attr.i)
                src_ranks = list(src_ranks_attr.ints)
                for dst, src in zip(dst_ranks, src_ranks, strict=False):
                    emit("comm", f"P{src}->>P{dst}: {node.name}")
            else:
                emit("comm", f"note over P0,P{wsize - 1}: send %${node.name}")
            continue
        if node.op_type in {"cond", "while"}:
            flush_local_buffer()
            emit("note", f"note over P0,P{wsize - 1}: {node.op_type} {node.name}")
            continue
        cross = False
        if input_ranks and own and (own - input_ranks or input_ranks - own):
            cross = True
        if cross and own and input_ranks:
            flush_local_buffer()
            for s in sorted(input_ranks):
                for t in sorted(own):
                    if s != t:
                        emit("comm", f"P{s}->>P{t}: {label}{value_suffix}")
            continue
        local_desc = f"{label}{value_suffix}".strip()
        if not local_desc:
            continue
        if own:
            repr_rank = min(own)
            local_line = f"P{repr_rank}-->>P{repr_rank}: {local_desc}"
        else:
            local_line = f"note over P0,P{wsize - 1}: {local_desc} (dyn)"
        local_buffer.append(local_line)

    flush_local_buffer()

    out_lines: list[str] = ["sequenceDiagram"]
    out_lines.extend(participants)
    for ev in events:
        out_lines.extend(ev.lines)
    return "\n".join(out_lines)


def to_flowchart(
    traced: TracedFunction,
    *,
    group_local: bool = True,
    world_size: int | None = None,
    show_meta: bool = False,
    direction: str = "LR",
    cross_edge_color: str = "#ff6a00",
    cluster_by_party: bool = False,
    shared_cluster_name: str = "Shared",
) -> str:
    """Render a traced function as a Mermaid flowchart (DAG view)."""
    expr = traced.make_expr()
    graph = Writer().dumps(expr)
    value_to_node: dict[str, mpir_pb2.NodeProto] = {}
    for n in graph.nodes:
        base = n.name
        arity = max(1, len(n.outs_info))
        for i in range(arity):
            key = f"{base}" if arity == 1 else f"{base}:{i}"
            value_to_node[key] = n

    def owners_of(node: mpir_pb2.NodeProto) -> set[int]:
        rs: set[int] = set()
        for out in node.outs_info:
            if out.pmask >= 0:
                rs.update(_pmask_to_ranks(out.pmask))
        return rs

    node_labels: list[str] = []
    per_party_nodes: dict[int, list[str]] = {}
    shared_nodes: list[str] = []
    node_id_map: dict[str, str] = {}
    node_map: dict[str, mpir_pb2.NodeProto] = {n.name: n for n in graph.nodes}
    id_to_owners: dict[str, set[int]] = {}
    for n in graph.nodes:
        if n.op_type == "access":
            continue
        if not show_meta and n.op_type in {"tuple", "func_def"}:
            continue
        node_id = f"n{n.name[1:]}"
        node_id_map[n.name] = node_id
        pfunc = n.attrs.get("pfunc") if n.op_type == "eval" else None
        if pfunc:
            op_label = pfunc.func.name or pfunc.func.type or n.op_type
        else:
            op_label = n.op_type
        arity = len(n.outs_info)
        arity_suffix = f"/{arity}" if arity > 1 else ""
        owners = owners_of(n)
        owners_str = (
            "" if not owners else " @" + ",".join(f"P{r}" for r in sorted(owners))
        )
        label_line = f'{node_id}["{op_label}{arity_suffix}{owners_str}"]'
        if cluster_by_party:
            if len(owners) == 1:
                (owner_rank,) = tuple(owners)
                per_party_nodes.setdefault(owner_rank, []).append(label_line)
            else:
                shared_nodes.append(label_line)
        else:
            node_labels.append(label_line)
        id_to_owners[node_id] = owners

    def resolve_sources(val: str, seen: set[str] | None = None) -> set[str]:
        if seen is None:
            seen = set()
        base = val.split(":")[0]
        node = value_to_node.get(val) or value_to_node.get(base)
        if not node:
            return set()
        if node.name in seen:
            return set()
        seen.add(node.name)
        if node.op_type == "access":
            srcs: set[str] = set()
            for upstream in node.inputs:
                srcs |= resolve_sources(upstream, seen)
            return srcs
        return {node.name}

    edge_set: set[tuple[str, str]] = set()
    for n in graph.nodes:
        if n.op_type == "access":
            continue
        if not show_meta and n.op_type in {"tuple", "func_def"}:
            continue
        dst_id = node_id_map.get(n.name)
        if not dst_id:
            continue
        for inp in n.inputs:
            for src_name in resolve_sources(inp):
                if (not show_meta) and node_map[src_name].op_type in {
                    "tuple",
                    "func_def",
                }:
                    continue
                src_id = node_id_map.get(src_name)
                if not src_id or src_id == dst_id:
                    continue
                edge_set.add((src_id, dst_id))

    ordered_edges = sorted(edge_set)
    edges = [f"{s} --> {t}" for s, t in ordered_edges]
    cross_indices: list[int] = []
    for idx, (s, t) in enumerate(ordered_edges):
        so = id_to_owners.get(s, set())
        to = id_to_owners.get(t, set())
        if so and to and so != to:
            cross_indices.append(idx)

    _ = group_local

    dir_norm = direction.upper()
    if dir_norm == "TD":
        dir_norm = "TB"
    if dir_norm not in {"LR", "TB", "RL", "BT"}:
        dir_norm = "LR"
    result_lines = [f"graph {dir_norm};"]
    result_lines.append("")
    if cluster_by_party:
        wsize = _collect_world_size(graph, world_size)
        for r in range(wsize):
            nodes = per_party_nodes.get(r)
            if not nodes:
                continue
            result_lines.append(f"    subgraph P{r}")
            for ln in nodes:
                result_lines.append(f"        {ln}")
            result_lines.append("    end")
            result_lines.append("")
        if shared_nodes:
            result_lines.append(f"    subgraph {shared_cluster_name}")
            for ln in shared_nodes:
                result_lines.append(f"        {ln}")
            result_lines.append("    end")
            result_lines.append("")
    else:
        for lbl in node_labels:
            if lbl:
                result_lines.append("    " + lbl)
    if node_labels:
        result_lines.append("")
    for edge in edges:
        result_lines.append("    " + edge)
    for ci in cross_indices:
        result_lines.append(
            f"    linkStyle {ci} stroke:{cross_edge_color},stroke-width:2px;"
        )
    return "\n".join(result_lines)


# ----------------------------- Markdown dump -----------------------------


def dump(
    traced: TracedFunction,
    *,
    world_size: int | None = None,
    sequence: bool = True,
    flow: bool = True,
    include_ir: bool = True,
    report_path: str | Path | None = None,
    mpir_path: str | Path | None = None,
    title: str | None = None,
    **diagram_kwargs,
) -> str:
    """Produce a composite analysis artifact (currently markdown).

    Sections included:
      * Title (optional)
      * Compiler IR (plain text fenced block) if include_ir
      * Mermaid sequenceDiagram (optional)
      * Mermaid flowchart (optional)

    Args:
        traced: TracedFunction to visualize.
        world_size: Override party count for diagrams.
        sequence / flow: Toggle diagram kinds.
        include_ir: Whether to embed textual IR.
        report_path: If provided, write the markdown report here.
        mpir_path: If provided, also write raw mpir text (graph proto text form).
        title: Optional heading title.
        **diagram_kwargs: Forwarded to diagram functions.
    """
    if report_path is None and mpir_path is None:
        raise ValueError("must provide at least one of report_path or mpir_path")

    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")

    graph_proto = None
    if include_ir:
        parts.append("## Compiler IR (text)\n")
        parts.append("```")
        parts.append(traced.compiler_ir())
        parts.append("```")

    # Always include graph statistics section
    if graph_proto is None:
        expr = traced.make_expr()
        graph_proto = Writer().dumps(expr)
    stats = get_graph_statistics(graph_proto)
    parts.append("## Graph Structure Analysis\n")
    parts.append("```")
    parts.append(stats)
    parts.append("```")

    if sequence:
        # Filter kwargs for sequence diagram (avoid passing flow-only args like direction/cluster_by_party)
        seq_allowed = {"collapse_local", "show_compute", "max_local_batch", "show_meta"}
        seq_kwargs = {k: v for k, v in diagram_kwargs.items() if k in seq_allowed}
        seq = to_sequence_diagram(traced, world_size=world_size, **seq_kwargs)
        parts.append("## Mermaid Sequence Diagram")
        parts.append("```mermaid")
        parts.append(seq)
        parts.append("```")
    if flow:
        flw = to_flowchart(traced, world_size=world_size, **diagram_kwargs)
        parts.append("## Mermaid Flowchart (DAG)")
        parts.append("```mermaid")
        parts.append(flw)
        parts.append("```")

    content = "\n\n".join(parts) + "\n"

    if mpir_path:
        if graph_proto is None:
            expr = traced.make_expr()
            graph_proto = Writer().dumps(expr)
        Path(mpir_path).write_text(str(graph_proto), encoding="utf-8")

    if report_path:
        Path(report_path).write_text(content, encoding="utf-8")
    return content


__all__ = ["dump", "to_flowchart", "to_sequence_diagram"]
