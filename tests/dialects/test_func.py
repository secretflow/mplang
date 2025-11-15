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

from mplang.dialects.func import call, func
from mplang.dialects.tensor import run_jax
from mplang.edsl.interpreter import InterpObject
from mplang.edsl.tracer import trace
from mplang.edsl.typing import Tensor, f32


def _scale_add(x, y):
    return run_jax(lambda a, b: a * 2 + b, x, y, out_types=Tensor[f32, ()])


def test_func_call_emits_region():
    x = InterpObject(np.array(1.0), Tensor[f32, ()])
    y = InterpObject(np.array(3.0), Tensor[f32, ()])

    def wrapper(a, b):
        fn = func(_scale_add, a, b)
        return call(fn, a, b)

    traced = trace(wrapper, x, y)
    graph = traced.graph
    func_ops = [op for op in graph.operations if op.opcode == "func.func"]
    call_ops = [op for op in graph.operations if op.opcode == "func.call"]
    assert func_ops and call_ops
    assert call_ops[0].attrs["callee"] == func_ops[0].attrs["sym_name"]


def test_func_define_returns_traceobject():
    x = InterpObject(np.array(1.0), Tensor[f32, ()])
    y = InterpObject(np.array(3.0), Tensor[f32, ()])

    def wrapper(a, b):
        return func(_scale_add, a, b)

    traced = trace(wrapper, x, y)
    graph = traced.graph
    func_ops = [op for op in graph.operations if op.opcode == "func.func"]
    assert func_ops, "expected a func.func definition"
