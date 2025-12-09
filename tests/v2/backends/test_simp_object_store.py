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


from mplang.v2.backends.simp_host import HostVar
from mplang.v2.backends.simp_simulator import SimpSimulator
from mplang.v2.edsl.graph import Graph


def test_simulator_object_store_flow():
    """Test that Simulator uses ObjectStore correctly (URI passing)."""
    sim = SimpSimulator(world_size=2)

    # 1. Manually put data into workers' stores to simulate previous computation
    # Worker 0: x=10
    # Worker 1: x=20
    uri_x0 = sim.workers[0].store.put(10)
    uri_x1 = sim.workers[1].store.put(20)

    # Create a HostVar pointing to these URIs
    x_var = HostVar([uri_x0, uri_x1])

    # 2. Create a simple graph: y = x + 5
    # We need to construct a Graph manually or use tracing.
    # Let's use a simple manual construction for control.
    graph = Graph()

    # Inputs
    g_x = graph.add_input("x", type=None)  # Type doesn't matter for SIMP execution

    # Operation: add 5
    # We need a primitive. Let's use a dummy one or reuse tensor.run_jax
    # But run_jax expects JAX arrays.
    # Let's define a simple python function and wrap it.

    def add_five(x):
        return x + 5

    # We can't easily inject a python function into the graph without a Primitive.
    # So let's use the existing infrastructure but mock the execution?
    # No, let's use the real tensor dialect if possible, or just verify the flow.

    # Actually, SimpWorker.evaluate_graph calls super().evaluate_graph.
    # If we use a graph with no ops, just input->output, we can test identity.
    graph.outputs = [g_x]

    # 3. Execute identity graph
    # The simulator should:
    # - Pass URIs [uri_x0, uri_x1] to workers
    # - Workers resolve URIs -> get 10, 20
    # - Workers execute (identity) -> get 10, 20
    # - Workers store results -> get new URIs [uri_y0, uri_y1]
    # - Simulator returns HostVar([uri_y0, uri_y1])

    y_var = sim.evaluate_graph(graph, [x_var])

    assert isinstance(y_var, HostVar)
    assert len(y_var.values) == 2

    # Verify results are URIs
    assert isinstance(y_var.values[0], str) and "://" in y_var.values[0]
    assert isinstance(y_var.values[1], str) and "://" in y_var.values[1]

    # Verify URIs are different from inputs (new objects)
    assert y_var.values[0] != uri_x0
    assert y_var.values[1] != uri_x1

    # 4. Fetch results
    # This should trigger _submit_fetch -> worker.store.get
    results = sim.fetch(y_var)

    assert results == [10, 20]

    sim.shutdown()


def test_simulator_fetch_mixed_values():
    """Test fetch with mixed URIs and direct values."""
    sim = SimpSimulator(world_size=2)

    uri_0 = sim.workers[0].store.put(100)
    val_1 = 200  # Direct value

    var = HostVar([uri_0, val_1])

    results = sim.fetch(var)
    assert results == [100, 200]

    sim.shutdown()
