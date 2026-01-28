# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

import mplang as mp
from mplang.edsl.graph import Graph
from mplang.edsl.program import CompiledProgram, FlatIOSignature, compute_graph_digest
from mplang.edsl.tracer import trace
from mplang.edsl.typing import TensorType, f32
from mplang.runtime.interpreter import InterpObject, Interpreter


def test_artifact_roundtrip_and_execute_identity() -> None:
    interp = Interpreter()

    x_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_obj = InterpObject(x_val, TensorType(f32, (3,)), interp)

    def identity(x: object) -> object:
        return x

    traced = trace(identity, x_obj)
    program = mp.tool.compile_program(traced)

    blob = mp.tool.pack(program)
    program2 = mp.tool.unpack(blob)

    assert program2.signature.input_arity == 1
    assert program2.signature.output_arity == 1
    assert program2.graph_digest == program.graph_digest

    out = mp.evaluate(program, x_val, context=interp)
    fetched = mp.fetch(out, context=interp)
    assert isinstance(fetched, list)
    np.testing.assert_allclose(fetched[0], x_val)

    out2 = mp.evaluate(program2, x_val, context=interp)
    fetched2 = mp.fetch(out2, context=interp)
    assert isinstance(fetched2, list)
    np.testing.assert_allclose(fetched2[0], x_val)


def test_pack_to_path_and_tar_extractable(tmp_path: Path) -> None:
    interp = Interpreter()
    x_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_obj = InterpObject(x_val, TensorType(f32, (3,)), interp)

    def identity(x: object) -> object:
        return x

    program = mp.tool.compile_program(trace(identity, x_obj))

    out_path = tmp_path / "program.tar.gz"
    mp.tool.pack_to_path(program, out_path)
    loaded = mp.tool.unpack_path(out_path)
    assert loaded.graph_digest == program.graph_digest


def test_rejects_closure_captures() -> None:
    interp = Interpreter()

    x_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_obj = InterpObject(x_val, TensorType(f32, (3,)), interp)

    captured = x_obj

    def fn(a: object) -> tuple[object, object]:
        return a, captured

    traced = trace(fn, x_obj)
    with pytest.raises(ValueError, match="does not support closure captures"):
        mp.tool.compile_program(traced)


def test_rejects_constant_outputs() -> None:
    interp = Interpreter()

    x_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_obj = InterpObject(x_val, TensorType(f32, (3,)), interp)

    def fn(x: object) -> tuple[object, float]:
        return x, 2.0

    traced = trace(fn, x_obj)
    with pytest.raises(ValueError, match="does not support constant outputs"):
        mp.tool.compile_program(traced)


def test_required_world_size_roundtrip_and_validation() -> None:
    graph = Graph()
    x = graph.add_input("x", TensorType(f32, ()))
    (y,) = graph.add_op(
        "dummy",
        [x],
        output_types=[TensorType(f32, ())],
        attrs={"parties": [0, 2]},
    )
    graph.add_output(y)

    program = CompiledProgram(
        graph=graph,
        signature=FlatIOSignature(input_arity=1, output_arity=1),
        required_opcodes=sorted({"dummy"}),
        graph_digest=compute_graph_digest(graph),
        required_world_size=3,
    )

    blob = mp.tool.pack(program)
    loaded = mp.tool.unpack(blob)
    assert loaded.required_world_size == 3

    bad = CompiledProgram(
        graph=graph,
        signature=FlatIOSignature(input_arity=1, output_arity=1),
        required_opcodes=sorted({"dummy"}),
        graph_digest=compute_graph_digest(graph),
        required_world_size=2,
    )
    with pytest.raises(ValueError, match="required_world_size mismatch"):
        mp.tool.unpack(mp.tool.pack(bad))


def test_collect_helpers_handle_deeply_nested_regions_without_recursionerror() -> None:
    # Regression test: these helpers used to recurse through regions.
    # A crafted (or accidental) deep nesting could trigger RecursionError (DoS).
    from mplang.tool import program as program_tool

    depth = 2000
    graphs = [Graph() for _ in range(depth)]

    for i in range(depth - 1):
        # Add a single op whose region points to the next graph.
        (out,) = graphs[i].add_op(
            "dummy",
            [],
            output_types=[TensorType(f32, ())],
            attrs={"parties": [i % 3]} if i == depth - 2 else {},
            regions=[graphs[i + 1]],
        )
        graphs[i].add_output(out)

    # Leaf graph: just one op.
    (leaf_out,) = graphs[-1].add_op(
        "leaf",
        [],
        output_types=[TensorType(f32, ())],
        attrs={},
        regions=[],
    )
    graphs[-1].add_output(leaf_out)

    opcodes = program_tool._collect_opcodes(graphs[0])
    parties = program_tool._collect_parties(graphs[0])

    assert "dummy" in opcodes
    assert "leaf" in opcodes
    assert parties.issubset({0, 1, 2})


def test_unpack_rejects_oversized_artifact_json_header_size() -> None:
    # Security regression: prevent tar(.gz) "zip bomb" style DoS.
    # We simulate this by setting a tiny max_artifact_json_bytes.
    from mplang.tool import program as program_tool

    artifact_json = b"{}"  # small payload

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="artifact.json")
        info.size = len(artifact_json)
        tf.addfile(info, io.BytesIO(artifact_json))

    with pytest.raises(ValueError, match="artifact\\.json is too large"):
        program_tool.unpack(buf.getvalue(), max_artifact_json_bytes=1)
