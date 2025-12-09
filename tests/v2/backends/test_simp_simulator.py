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

import time

import jax.numpy as jnp
import pytest

# Register runtimes
import mplang.v2.backends.tensor_impl  # noqa: F401
from mplang.v2.backends.simp_host import HostVar
from mplang.v2.backends.simp_simulator import SimpSimulator
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import simp
from mplang.v2.dialects.simp import pcall_static, uniform_cond
from mplang.v2.dialects.tensor import run_jax
from mplang.v2.edsl.graph import Graph
from mplang.v2.runtime.interpreter import InterpObject


def _unwrap_values(values: list) -> list:
    """Unwrap TensorValue objects in a list."""
    result = []
    for v in values:
        if isinstance(v, TensorValue):
            # Convert 0-d array to scalar
            arr = v.data
            result.append(arr.item() if arr.ndim == 0 else arr)
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
    values = interp.fetch(res.runtime_obj)
    assert _unwrap_values(values) == [11, 22, 33]


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

    values = interp.fetch(res.runtime_obj)
    assert _unwrap_values(values) == [2, 4]

    # Test False case
    with interp:
        x0 = simp.constant((0,), 1)
        x1 = simp.constant((1,), 2)
        x_obj = simp.converge(x0, x1)
        pred_obj_false = simp.constant((0, 1), False)
        res_false = uniform_cond(pred_obj_false, then_fn, else_fn, x_obj)

    values = interp.fetch(res_false.runtime_obj)
    assert _unwrap_values(values) == [1, 4]


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
    values = interp.fetch(res.runtime_obj)
    assert _unwrap_values(values) == [10, 10]


class FaultySimpSimulator(SimpSimulator):
    def _run_party(self, rank, graph, inputs, job_id=None):
        if rank == 0:
            # Fail immediately
            raise RuntimeError("Rank 0 crashed!")
        elif rank == 1:
            # Sleep to simulate long running task
            time.sleep(2)
            return "Rank 1 result"
        return "Rank X result"


def test_simulator_fail_fast():
    """Test that simulator fails fast when one worker crashes."""
    sim = FaultySimpSimulator(world_size=2)
    graph = Graph()  # Dummy graph

    start_time = time.time()
    with pytest.raises(RuntimeError, match="Rank 0 crashed!"):
        sim.evaluate_graph(graph, [])
    end_time = time.time()

    duration = end_time - start_time
    # It should fail much faster than the 2s sleep
    assert duration < 1.0, f"Simulator took {duration}s to fail, expected < 1.0s"

    # Cleanup to avoid affecting other tests
    sim.shutdown()
