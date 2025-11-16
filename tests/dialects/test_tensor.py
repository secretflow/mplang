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

import numpy as np

import mplang.edsl as el
import mplang.edsl.typing as elt
from mplang.dialects.tensor import get_run_jax_compilation, run_jax


def _add_fn(x):
    return x + 2


def test_tensor_run_jax_op_emitted():
    value = el.InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])

    def wrapper(x):
        return run_jax(_add_fn, x)

    traced = el.trace(wrapper, value)
    graph = traced.graph

    assert len(graph.operations) == 1
    op = graph.operations[0]
    assert op.opcode == "tensor.run_jax"
    assert op.attrs["ir_type"] == "stablehlo"
    assert "text_ref" in op.attrs
    assert len(op.outputs) == 1

    compilation = get_run_jax_compilation(op.attrs["text_ref"])
    assert compilation.stablehlo.strip().startswith("module")
    assert len(compilation.output_types) == 1
