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

from mplang.dialects.tensor import run_jax
from mplang.edsl.interpreter import InterpObject
from mplang.edsl.tracer import trace
from mplang.edsl.typing import Tensor, f32


def _add_fn(x):
    return x + 2


def test_tensor_run_jax_op_emitted():
    value = InterpObject(np.array(1.0), Tensor[f32, ()])

    def wrapper(x):
        return run_jax(_add_fn, x, out_types=Tensor[f32, ()])

    traced = trace(wrapper, value)
    graph = traced.graph

    assert len(graph.operations) == 1
    op = graph.operations[0]
    assert op.opcode == "tensor.run_jax"
    assert "add_fn" in op.attrs["fn"]
    assert op.attrs["backend"] == "plaintext"
    assert len(op.outputs) == 1
