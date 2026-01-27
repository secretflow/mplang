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

    out = mp.evaluate(program2, x_val, context=interp)
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
