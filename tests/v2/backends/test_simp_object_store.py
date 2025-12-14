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

from mplang.v2.backends.simp_driver import DriverVar
from mplang.v2.dialects import simp


def test_object_store_put_get():
    """Test ObjectStore put and get."""
    sim = simp.make_simulator(world_size=2)

    # Access workers via exposed client_ctx
    workers = sim._simp_cluster.workers
    worker0 = workers[0]
    store0 = worker0.store

    key = "test_key"
    data = "test_data"

    # Fixed put usage
    uri_0 = f"mem://{key}"
    store0.put(data, uri_0)

    val = store0.get(uri_0)
    assert val == data

    # Store with generated URI
    uri_gen = store0.put("some_data")
    assert "mem://" in uri_gen
    assert store0.get(uri_gen) == "some_data"
    val = store0.get(uri_0)
    assert val == data
    hasattr(sim, "_simp_cluster") and sim._simp_cluster.shutdown()


def test_object_store_host_var_storage(tmp_path):
    """Test storing DriverVar."""
    from mplang.v2.runtime.object_store import ObjectStore

    store = ObjectStore(fs_root=tmp_path)
    hv = DriverVar([1, 2, 3])
    uri = store.put(hv)

    hv_out = store.get(uri)
    assert isinstance(hv_out, DriverVar)
    assert hv_out.values == hv.values


def test_simulator_object_store_flow():
    """Test ObjectStore URI flow in Simulator context."""
    sim = simp.make_simulator(world_size=2)
    workers = sim._simp_cluster.workers
    simp_state = sim.get_dialect_state("simp")

    # 1. Put data into workers' stores
    data_0, data_1 = 10, 20
    uri_x0 = workers[0].store.put(data_0)
    uri_x1 = workers[1].store.put(data_1)

    # 2. Verify URIs are valid format
    assert isinstance(uri_x0, str) and "://" in uri_x0
    assert isinstance(uri_x1, str) and "://" in uri_x1

    # 3. Create DriverVar
    x_var = DriverVar([uri_x0, uri_x1])
    assert len(x_var.values) == 2

    # 4. Verify data can be retrieved via store.get
    assert workers[0].store.get(uri_x0) == data_0
    assert workers[1].store.get(uri_x1) == data_1

    # 5. Verify fetch via simp state
    fetched_0 = simp_state.fetch(0, uri_x0).result()
    fetched_1 = simp_state.fetch(1, uri_x1).result()
    assert fetched_0 == data_0
    assert fetched_1 == data_1

    hasattr(sim, "_simp_cluster") and sim._simp_cluster.shutdown()


def test_simulator_fetch_mixed_values():
    """Test fetch with mixed URIs and direct values."""
    sim = simp.make_simulator(world_size=2)
    workers = sim._simp_cluster.workers
    simp_state = sim.get_dialect_state("simp")

    uri_0 = workers[0].store.put(100)
    val_1 = 200  # Direct value (not stored)

    # Manually fetch URI value
    res0 = simp_state.fetch(0, uri_0).result()
    assert res0 == 100
    assert val_1 == 200

    hasattr(sim, "_simp_cluster") and sim._simp_cluster.shutdown()
