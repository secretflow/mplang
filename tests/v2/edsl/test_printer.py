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

"""Tests for the EDSL graph printer."""

from textwrap import dedent

from mplang.v2.edsl.graph import Graph
from mplang.v2.edsl.printer import GraphPrinter, format_graph
from mplang.v2.edsl.typing import Tensor, f32, i32


def _build_simple_graph() -> Graph:
    g = Graph()
    x = g.add_input("x", Tensor[f32, (1,)])
    const = g.add_op("constant", [], output_types=[f32], attrs={"value": 1.0})[0]
    [y] = g.add_op("add", [x, const], output_types=[Tensor[f32, (1,)]])
    g.add_output(y)
    return g


def test_graph_printer_basic():
    graph = _build_simple_graph()
    output = GraphPrinter().format(graph)
    assert "(x: Tensor[f32, (1)])" in output
    assert "constant" in output
    assert "add" in output
    assert "return" in output
    expected = dedent(
        """\
        (x: Tensor[f32, (1)]) {
          %0 = constant() {value=1.0} : f32
          %1 = add(x, %0) : Tensor[f32, (1)]
          return %1
        }"""
    )
    assert output == expected


def test_graph_printer_with_regions():
    g = Graph()
    pred = g.add_input("pred", i32)
    data = g.add_input("x", Tensor[f32, ()])

    then_graph = Graph()
    then_x = then_graph.add_input("x", Tensor[f32, ()])
    [then_out] = then_graph.add_op("negate", [then_x], output_types=[Tensor[f32, ()]])
    then_graph.add_output(then_out)

    else_graph = Graph()
    else_x = else_graph.add_input("x", Tensor[f32, ()])
    zero = else_graph.add_op("constant", [], output_types=[f32], attrs={"value": 0.0})[
        0
    ]
    [else_out] = else_graph.add_op(
        "add", [else_x, zero], output_types=[Tensor[f32, ()]]
    )
    else_graph.add_output(else_out)

    [result] = g.add_op(
        "cond",
        [pred, data],
        output_types=[Tensor[f32, ()]],
        regions=[then_graph, else_graph],
    )
    g.add_output(result)

    output = format_graph(g)
    assert "cond" in output
    assert "return" in output

    expected = dedent(
        """\
        (pred: i32, x: Tensor[f32, ()]) {
          %0 = cond(pred, x) : Tensor[f32, ()] {
            (x: Tensor[f32, ()]) {
              %0 = negate(x) : Tensor[f32, ()]
              return %0
            }
            (x: Tensor[f32, ()]) {
              %0 = constant() {value=0.0} : f32
              %1 = add(x, %0) : Tensor[f32, ()]
              return %1
            }
          }
          return %0
        }"""
    )
    assert output == expected
