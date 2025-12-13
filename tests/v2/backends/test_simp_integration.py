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
import numpy as np

import mplang.v2 as mp

# Register runtimes
# Register runtimes
import mplang.v2.backends.tensor_impl  # noqa: F401
from mplang.v2.backends.simp_driver import DriverVar
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import simp
from mplang.v2.dialects.simp import pcall_static, uniform_cond
from mplang.v2.dialects.tensor import run_jax
from mplang.v2.runtime.interpreter import InterpObject


def _unwrap_values(values: list) -> list:
    """Unwrap TensorValue objects in a list."""
    result = []
    for v in values:
        if isinstance(v, (TensorValue, InterpObject)):
            # Convert to scalar if possible
            arr = v.data
            if isinstance(arr, (jnp.ndarray, np.ndarray, np.generic)):
                result.append(arr.item())
            else:
                result.append(arr)
        elif isinstance(v, (jnp.ndarray, np.ndarray, np.generic)):
            result.append(v.item())
        else:
            result.append(v)
    return result


def add(x, y):
    return run_jax(jnp.add, x, y)


def mul(x, y):
    return run_jax(jnp.multiply, x, y)


def test_pcall_static():
    # Define a function to run on parties
    def my_func(x, y):
        return add(x, y)

    # Create interpreter
    sim = simp.make_simulator(world_size=3)

    # Create inputs (DriverVars)
    # World size 3

    # Call pcall_static
    with sim:
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
        assert isinstance(res.runtime_obj, DriverVar)
        # Note: run_jax returns numpy arrays (or jax arrays), so we compare values
        # DriverVar holds list of values.
        # 1+10=11, 2+20=22, 3+30=33
        values = mp.fetch(res)
        assert _unwrap_values(values) == [11, 22, 33]


def test_uniform_cond():
    sim = simp.make_simulator(world_size=2)

    def then_fn(x):
        return pcall_static((0, 1), lambda a: add(a, a), x)

    def else_fn(x):
        return pcall_static((0, 1), lambda a: mul(a, a), x)

    with sim:
        x0 = simp.constant((0,), 1)
        x1 = simp.constant((1,), 2)
        x_obj = simp.converge(x0, x1)

        # True condition
        pred_true = simp.constant((0, 1), True)
        res = uniform_cond(pred_true, then_fn, else_fn, x_obj)

        values = mp.fetch(res)
        assert _unwrap_values(values) == [2, 4]

    # Test False case
    with sim:
        x0 = simp.constant((0,), 1)
        x1 = simp.constant((1,), 2)
        x_obj = simp.converge(x0, x1)
        pred_obj_false = simp.constant((0, 1), False)
        res_false = uniform_cond(pred_obj_false, then_fn, else_fn, x_obj)

        values = mp.fetch(res_false)
        assert _unwrap_values(values) == [1, 4]


def test_while_loop_eager():
    """Test simp.while_loop eager execution."""

    def cond(val):
        # val is Tensor during worker execution
        return run_jax(lambda a: a < 10, val)

    def body(val):
        # val is Tensor during worker execution
        return run_jax(lambda a: a + 1, val)

    # Setup runtime
    # Reset global context to ensure world_size=2
    sim = simp.make_simulator(world_size=2)

    # Eager call
    with sim:
        start_obj = simp.constant((0, 1), 0)
        # Note: while_loop and uniform_cond are registered in WORKER_HANDLERS
        # The test tries to run them on the simulator (Host).
        # But they are Worker-side ops (control flow inside a party).
        # We need to wrap them in pcall_static to run them on workers?
        # OR we need to register Host-side implementations for them?
        # Originally, SimpHost had handlers for these?
        # Checking simp_impl.py, HOST_HANDLERS only has pcall/shuffle/converge.
        # So we CANNOT run while_loop on the Host.
        # We must run it ON the worker via pcall.

        def run_loop(start_val):
            return simp.while_loop(cond, body, start_val)

        res = simp.pcall_static((0, 1), run_loop, start_obj)

        assert isinstance(res, InterpObject)
        # mp.fetch can fetch the result directly from the wrapper if needed,
        # but here res is InterpObject. fetch needs (sim, obj).
        values = mp.fetch(res)
        assert _unwrap_values(values) == [10, 10]
