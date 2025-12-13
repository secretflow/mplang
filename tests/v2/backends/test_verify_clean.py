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

import pytest

import mplang.v2 as mp
import mplang.v2.edsl.typing as elt
from mplang.v2.backends.simp_driver import DriverVar
from mplang.v2.dialects import simp, tensor


def test_object_store_put_get():
    """Test ObjectStore put and get (Clean ver)."""
    sim = simp.make_simulator(world_size=2)
    workers = sim._simp_cluster.workers
    worker0 = workers[0]
    store0 = worker0.store

    key = "mem://test_key"
    data = "test_data"

    # Store with explicit URI
    store0.put(data, uri=key)
    assert store0.get(key) == data
    hasattr(sim, "_simp_cluster") and sim._simp_cluster.shutdown()


def test_object_store_host_var_storage(tmp_path):
    """Test storing DriverVar."""
    from mplang.v2.runtime.object_store import ObjectStore

    store = ObjectStore(fs_root=tmp_path)
    hv = DriverVar([1, 2, 3])
    # Let it generate URI
    uri = store.put(hv)
    val = store.get(uri)
    assert val.values == [1, 2, 3]


@pytest.mark.skip(
    reason="Logic expects host to compute on remote specific URIs automatically"
)
def test_simulator_object_store_flow():
    """Test Simulator URI flow."""
    sim = simp.make_simulator(world_size=2)
    workers = sim._simp_cluster.workers

    uri_x0 = workers[0].store.put(10)
    uri_x1 = workers[1].store.put(20)
    x_var = DriverVar([uri_x0, uri_x1])

    # Wrap in InterpObject for tracing
    from mplang.v2.edsl.typing import MPType, TensorType
    from mplang.v2.runtime.interpreter import InterpObject

    # Define type for x_var: MP[Tensor[i32], {0, 1}]
    # elt.i32 is instance of IntegerType
    x_type = MPType(TensorType(elt.i32, ()), (0, 1))
    # InterpObject(value, type, context) - sim is now an Interpreter directly
    x_interp = InterpObject(x_var, x_type, sim)

    # Graph: y = x + 1
    def fn(x):
        return tensor.run_jax(lambda a: a + 1, x)

    graph = mp.trace(fn, x_interp).graph
    y_var = sim.evaluate_graph(graph, [x_var])

    assert isinstance(y_var, DriverVar)
    assert len(y_var.values) == 2
    assert isinstance(y_var.values[0], str) and "://" in y_var.values[0]

    results = mp.fetch(y_var)
    # Cast to int
    results = [int(r) for r in results]
    assert results == [11, 21]
    hasattr(sim, "_simp_cluster") and sim._simp_cluster.shutdown()


def test_uniform_cond_clean():
    """Test uniform_cond (Clean ver)."""
    sim = simp.make_simulator(world_size=2)

    with sim:
        # Check if pcall_static handling returns DriverVar
        # Manually invoke pcall_static first
        simp.pcall_static((0, 1), lambda: tensor.constant(True))

        x0 = simp.constant((0,), 1)
        x1 = simp.constant((1,), 2)
        x_obj = simp.converge(x0, x1)

        def then_fn(x):
            return simp.pcall_static(
                (0, 1), lambda a: tensor.run_jax(lambda v: v + v, a), x
            )

        def else_fn(x):
            return simp.pcall_static(
                (0, 1), lambda a: tensor.run_jax(lambda v: v * v, a), x
            )

        pred_true = simp.constant((0, 1), True)
        # uniform_cond
        res = simp.uniform_cond(pred_true, then_fn, else_fn, x_obj)

        values = mp.fetch(res)
        values = [
            int(v) if not hasattr(v, "shape") or v.shape == () else v for v in values
        ]
        assert values == [2, 4]
    hasattr(sim, "_simp_cluster") and sim._simp_cluster.shutdown()
