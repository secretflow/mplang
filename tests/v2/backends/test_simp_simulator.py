# Copyright 2025 Ant Group Co., Ltd.
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

import jax.numpy as jnp

# Register runtimes
import mplang.v2.backends.tensor_impl  # noqa: F401
from mplang.v2.backends.simp_host import HostVar
from mplang.v2.backends.simp_simulator import SimpSimulator
from mplang.v2.dialects import simp
from mplang.v2.dialects.simp import pcall_static, uniform_cond
from mplang.v2.dialects.tensor import run_jax
from mplang.v2.edsl.interpreter import InterpObject


def add(x, y):
    return run_jax(jnp.add, x, y)


def mul(x, y):
    return run_jax(jnp.multiply, x, y)


def test_pcall_static():
    # Define a function to run on parties
    def my_func(x, y):
        return add(x, y)

    # Create interpreter
    interp = SimpSimulator(world_size=3)

    # Create inputs (HostVars)
    # World size 3

    # Call pcall_static
    with interp:
        x0 = simp.constant((0,), 1)
        x1 = simp.constant((1,), 2)
        x2 = simp.constant((2,), 3)
        x_obj = simp.converge(x0, x1, x2)

        y0 = simp.constant((0,), 10)
        y1 = simp.constant((1,), 20)
        y2 = simp.constant((2,), 30)
        y_obj = simp.converge(y0, y1, y2)

        res = pcall_static((0, 1, 2), my_func, x_obj, y_obj)

    assert isinstance(res, InterpObject)
    assert isinstance(res.runtime_obj, HostVar)
    # Note: run_jax returns numpy arrays (or jax arrays), so we compare values
    # HostVar holds list of values.
    # 1+10=11, 2+20=22, 3+30=33
    assert res.runtime_obj.values == [11, 22, 33]


def test_uniform_cond():
    interp = SimpSimulator(world_size=2)

    def then_fn(x):
        return pcall_static((0, 1), lambda a: add(a, a), x)

    def else_fn(x):
        return pcall_static((0, 1), lambda a: mul(a, a), x)

    with interp:
        x0 = simp.constant((0,), 1)
        x1 = simp.constant((1,), 2)
        x_obj = simp.converge(x0, x1)

        # True condition
        pred_true = simp.constant((0, 1), True)
        res = uniform_cond(pred_true, then_fn, else_fn, x_obj)

    assert res.runtime_obj.values == [2, 4]

    # Test False case
    with interp:
        pred_obj_false = simp.constant((0, 1), False)
        res_false = uniform_cond(pred_obj_false, then_fn, else_fn, x_obj)
    assert res_false.runtime_obj.values == [1, 4]


def test_while_loop_eager():
    """Test simp.while_loop eager execution."""

    def cond(val):
        # val is TraceObject during tracing
        def local_cond(x):
            return run_jax(lambda a: a < 10, x)

        return pcall_static((0, 1), local_cond, val)

    def body(val):
        def local_body(x):
            return run_jax(lambda a: a + 1, x)

        return pcall_static((0, 1), local_body, val)

    # Setup runtime
    # Reset global context to ensure world_size=2
    interp = SimpSimulator(world_size=2)

    # Eager call
    with interp:
        start_obj = simp.constant((0, 1), 0)
        res = simp.while_loop(cond, body, start_obj)

    assert isinstance(res, InterpObject)
    assert res.runtime_obj.values == [10, 10]
