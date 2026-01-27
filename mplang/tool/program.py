from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Literal

import mplang
from mplang.edsl import serde
from mplang.edsl.graph import Graph
from mplang.edsl.program import (
    CompiledProgram,
    FlatIOSignature,
    compute_graph_digest,
    utc_now_iso,
)
from mplang.edsl.tracer import TracedFunction, trace

DEFAULT_MAX_ARTIFACT_JSON_BYTES = 512 * 1024 * 1024  # 512 MiB


def _iter_graphs(root: Graph) -> list[Graph]:
    # Use an explicit stack to avoid Python recursion limits.
    # Also guard against potential region graph cycles.
    out: list[Graph] = []
    stack: list[Graph] = [root]
    visited: set[int] = set()
    while stack:
        graph = stack.pop()
        graph_id = id(graph)
        if graph_id in visited:
            continue
        visited.add(graph_id)
        out.append(graph)

        for op in graph.operations:
            if op.regions:
                stack.extend(op.regions)

    return out


def _collect_opcodes(graph: Graph) -> set[str]:
    opcodes: set[str] = set()
    for g in _iter_graphs(graph):
        for op in g.operations:
            opcodes.add(op.opcode)
    return opcodes


def _collect_parties(graph: Graph) -> set[int]:
    parties: set[int] = set()
    for g in _iter_graphs(graph):
        for op in g.operations:
            raw = op.attrs.get("parties")
            if raw is None:
                continue

            if not isinstance(raw, (list, tuple, set)):
                raise TypeError(
                    "Invalid 'parties' attribute: expected list/tuple/set of ints, "
                    f"got {type(raw).__name__}"
                )
            for p in raw:
                p_int = int(p)
                if p_int < 0:
                    raise ValueError("Invalid 'parties' attribute: negative party id")
                parties.add(p_int)
    return parties


def _compute_required_world_size(graph: Graph) -> int:
    parties = _collect_parties(graph)
    if not parties:
        return 0
    return max(parties) + 1


def _validate_traced_for_artifact(traced: TracedFunction) -> None:
    # Restriction: no closure captures
    if traced.captured:
        raise ValueError(
            "CompiledProgram does not support closure captures; "
            "please refactor to pass all values explicitly."
        )

    # Restriction: no constant outputs (out_imms)
    if traced.out_imms:
        raise ValueError(
            "CompiledProgram does not support constant outputs (out_imms); "
            "return only traced values (graph outputs)."
        )

    # Restriction: signature is flat positional list I/O.
    # We do not preserve (args, kwargs) pytree metadata.
    # We therefore require all runtime-provided inputs correspond exactly to graph.inputs.
    if len(traced.graph.inputs) != len(traced.in_var_pos):
        raise ValueError(
            "CompiledProgram requires flat positional inputs that map 1:1 to graph.inputs; "
            f"got graph.inputs={len(traced.graph.inputs)} but in_var_pos={len(traced.in_var_pos)}."
        )


def _validate_program(program: CompiledProgram) -> None:
    if program.signature.kind != FlatIOSignature.kind:
        raise ValueError(f"Unsupported signature kind: {program.signature.kind}")

    if program.signature.input_arity != len(program.graph.inputs):
        raise ValueError(
            "Signature input_arity does not match graph.inputs: "
            f"input_arity={program.signature.input_arity}, inputs={len(program.graph.inputs)}"
        )
    if program.signature.output_arity != len(program.graph.outputs):
        raise ValueError(
            "Signature output_arity does not match graph.outputs: "
            f"output_arity={program.signature.output_arity}, outputs={len(program.graph.outputs)}"
        )

    expected_opcodes = sorted(_collect_opcodes(program.graph))
    if sorted(program.required_opcodes) != expected_opcodes:
        raise ValueError(
            "required_opcodes mismatch with graph content; "
            "artifact may be corrupted or constructed inconsistently."
        )

    actual_digest = compute_graph_digest(program.graph)
    if program.graph_digest and program.graph_digest != actual_digest:
        raise ValueError(
            "Graph digest mismatch: "
            f"expected={program.graph_digest}, actual={actual_digest}"
        )

    expected_world_size = _compute_required_world_size(program.graph)
    if (
        program.required_world_size is not None
        and program.required_world_size != expected_world_size
    ):
        raise ValueError(
            "required_world_size mismatch with graph content; "
            f"expected={expected_world_size}, got={program.required_world_size}."
        )

    # Ensure JSON serialization works (fail fast for non-serde attrs).
    serde.to_json(program)


def compile_program(
    fn_or_traced: Any,
    *args: Any,
    context: Any | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> CompiledProgram:
    """Compile (trace) into a source-free executable `CompiledProgram`.

    Restrictions (enforced):
    - no closure captures
    - no constant outputs (`out_imms` must be empty)
    - signature is flat list (positional) I/O

    Note: `in_imms` (compile-time constants) are allowed: they are baked into the graph.
    """

    traced: TracedFunction
    if isinstance(fn_or_traced, TracedFunction):
        traced = fn_or_traced
    else:
        if context is not None:
            with context:
                traced = trace(fn_or_traced, *args, **kwargs)
        else:
            traced = trace(fn_or_traced, *args, **kwargs)

    _validate_traced_for_artifact(traced)

    signature = FlatIOSignature(
        input_arity=len(traced.graph.inputs),
        output_arity=len(traced.graph.outputs),
    )

    required_opcodes = sorted(_collect_opcodes(traced.graph))
    graph_digest = compute_graph_digest(traced.graph)
    required_world_size = _compute_required_world_size(traced.graph)

    program = CompiledProgram(
        graph=traced.graph,
        signature=signature,
        required_opcodes=required_opcodes,
        graph_digest=graph_digest,
        required_world_size=required_world_size,
        created_at=utc_now_iso(),
        mplang_version=getattr(mplang, "__version__", None),
        name=name or traced.name,
    )
    _validate_program(program)
    return program


def pack(program: CompiledProgram, *, compress: bool = True) -> bytes:
    """Pack a `CompiledProgram` into portable bytes.

    Container format (recommended): a `tar.gz` archive containing a single
    human-readable JSON file `artifact.json`.

    This allows users to inspect artifacts via:
        `tar -xzf program.tar.gz && cat artifact.json`

    If `compress=False`, returns an uncompressed tar archive (still extractable
    via `tar -xf`).
    """

    artifact_json = json.dumps(
        serde.to_json(program),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

    buf = io.BytesIO()
    mode: Literal["w:gz", "w"] = "w:gz" if compress else "w"
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        info = tarfile.TarInfo(name="artifact.json")
        info.size = len(artifact_json)
        tf.addfile(info, io.BytesIO(artifact_json))

    return buf.getvalue()


def pack_to_path(
    program: CompiledProgram, path: str | Path, *, compress: bool = True
) -> Path:
    """Pack and write artifact to disk.

    Args:
        program: Program to pack.
        path: Output path (typically ends with `.tar.gz`).
        compress: Whether to gzip the tar archive.

    Returns:
        The resolved output path.
    """

    out_path = Path(path).expanduser().resolve()
    out_path.write_bytes(pack(program, compress=compress))
    return out_path


def unpack(
    data: bytes, *, max_artifact_json_bytes: int = DEFAULT_MAX_ARTIFACT_JSON_BYTES
) -> CompiledProgram:
    """Unpack bytes into a `CompiledProgram`.

    Supported container format: tar(.gz) containing `artifact.json`.
    """

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            member = tf.getmember("artifact.json")

            if not member.isfile():
                raise ValueError("artifact.json is not a regular file")

            if member.size < 0:
                raise ValueError("Invalid artifact.json size in tar header")

            if member.size > max_artifact_json_bytes:
                raise ValueError(
                    "artifact.json is too large to unpack safely: "
                    f"size={member.size} bytes, limit={max_artifact_json_bytes} bytes"
                )

            f = tf.extractfile(member)
            if f is None:
                raise ValueError("artifact.json not found in tar archive")
            payload = json.loads(f.read().decode("utf-8"))
    except (tarfile.ReadError, KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Invalid artifact container: expected tar(.gz) with artifact.json"
        ) from exc

    program = serde.from_json(payload)
    if not isinstance(program, CompiledProgram):
        raise TypeError(
            f"Expected artifact.json to deserialize to CompiledProgram, got {type(program).__name__}"
        )

    _validate_program(program)
    return program


def unpack_path(path: str | Path) -> CompiledProgram:
    """Read an artifact from disk and unpack it."""

    in_path = Path(path).expanduser().resolve()
    return unpack(in_path.read_bytes())


def inspect_artifact(data: bytes) -> dict[str, Any]:
    """Return a JSON-friendly inspection report without executing."""

    program = unpack(data)
    return {
        "schema_version": program.schema_version,
        "name": program.name,
        "mplang_version": program.mplang_version,
        "created_at": program.created_at,
        "graph_digest": program.graph_digest,
        "required_world_size": program.required_world_size,
        "signature": program.signature.to_json(),
        "required_opcodes": program.required_opcodes,
        "graph": {
            "inputs": len(program.graph.inputs),
            "ops": len(program.graph.operations),
            "outputs": len(program.graph.outputs),
            "region_count": sum(len(op.regions) for op in program.graph.operations),
        },
    }
