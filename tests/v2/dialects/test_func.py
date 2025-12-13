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
from textwrap import dedent

import numpy as np

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects.func import call, func
from mplang.v2.dialects.tensor import run_jax
from mplang.v2.runtime.interpreter import InterpObject


def _scale_add(x, y):
    return run_jax(lambda a, b: a * 2 + b, x, y)


def _complex_body(a, b):
    doubled = run_jax(lambda v: v * 2, a)
    summed = run_jax(lambda lhs, rhs: lhs + rhs, doubled, b)
    residual = run_jax(lambda lhs, rhs: lhs - rhs, summed, a)
    return {"nested": (residual, {"orig": b}), "sum": summed}


def test_func_call_emits_region():
    x = InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])
    y = InterpObject(np.array(3.0), elt.Tensor[elt.f32, ()])

    def wrapper(a, b):
        fn = func(_scale_add, a, b)
        return call(fn, a, b)

    traced = el.trace(wrapper, x, y)
    graph = traced.graph
    func_ops = [op for op in graph.operations if op.opcode == "func.func"]
    call_ops = [op for op in graph.operations if op.opcode == "func.call"]
    assert func_ops and call_ops
    # func.call now uses function handle as first input, not callee attr
    assert len(call_ops[0].inputs) >= 1  # function handle + args


def test_func_define_returns_traceobject():
    x = InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])
    y = InterpObject(np.array(3.0), elt.Tensor[elt.f32, ()])

    def wrapper(a, b):
        return func(_scale_add, a, b)

    traced = el.trace(wrapper, x, y)
    graph = traced.graph
    func_ops = [op for op in graph.operations if op.opcode == "func.func"]
    assert func_ops, "expected a func.func definition"


def test_func_call_handles_complex_pytree_output():
    x = InterpObject(np.array(2.0), elt.Tensor[elt.f32, ()])
    y = InterpObject(np.array(5.0), elt.Tensor[elt.f32, ()])

    def wrapper(a, b):
        nested_fn = func(_complex_body, a, b)
        return call(nested_fn, a, b)

    traced = el.trace(wrapper, x, y)
    graph = traced.graph
    func_ops = [op for op in graph.operations if op.opcode == "func.func"]
    call_ops = [op for op in graph.operations if op.opcode == "func.call"]
    assert len(func_ops) == 1
    assert len(call_ops) == 1

    complex_region = func_ops[0].regions[0]
    tensor_ops = [
        op for op in complex_region.operations if op.opcode == "tensor.run_jax"
    ]
    assert len(tensor_ops) == 3

    call_op = call_ops[0]
    assert len(call_op.outputs) == 3

    out_tree = traced.out_tree
    assert out_tree is not None
    assert out_tree.num_leaves == 3
    placeholders = [f"leaf{i}" for i in range(out_tree.num_leaves)]
    assert out_tree.unflatten(placeholders) == {
        "nested": ("leaf0", {"orig": "leaf1"}),
        "sum": "leaf2",
    }

    formatted = el.format_graph(graph)
    normalized = re.sub(
        r"text_ref='[^']+'", "text_ref='<ID>'", formatted, flags=re.MULTILINE
    )
    normalized = re.sub(
        r"fn=<function .*? at 0x[\da-f]+>", "fn='<FN>'", normalized, flags=re.MULTILINE
    )
    expected_ir = dedent(
        """\
            (%arg0: Tensor[f32, ()], %arg1: Tensor[f32, ()]) {
              %0 = func.func() {in_imms=[], in_tree=PyTreeDef(((*, *), {})), in_var_pos=[0, 1], out_imms=[], out_tree=PyTreeDef({'nested': (*, {'orig': *}), 'sum': *}), out_var_pos=[0, 1, 2], output_types=[Tensor[f32, ()], Tensor[f32, ()], Tensor[f32, ()]], sym_name='_complex_body'} : Custom[function] {
                (%arg0: Tensor[f32, ()], %arg1: Tensor[f32, ()]) {
                  %0 = tensor.run_jax(%arg0) {arg_keep_map=None, ir_type='stablehlo', stablehlo_code='<ID>', text_ref='<ID>'} : Tensor[f32, ()]
                  %1 = tensor.run_jax(%0, %arg1) {arg_keep_map=None, ir_type='stablehlo', stablehlo_code='<ID>', text_ref='<ID>'} : Tensor[f32, ()]
                  %2 = tensor.run_jax(%1, %arg0) {arg_keep_map=None, ir_type='stablehlo', stablehlo_code='<ID>', text_ref='<ID>'} : Tensor[f32, ()]
                  return %2, %arg1, %1
                }
              }
              [%1, %2, %3] = func.call(%0, %arg0, %arg1) : (Tensor[f32, ()], Tensor[f32, ()], Tensor[f32, ()])
              return %1, %2, %3
            }"""
    )  # Normalize stablehlo_code to <ID> for comparison
    normalized = re.sub(
        r"stablehlo_code='[^']+'",
        "stablehlo_code='<ID>'",
        normalized,
        flags=re.MULTILINE,
    )

    assert normalized == expected_ir
