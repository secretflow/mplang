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

"""Utils Tests using clean simp.constant API."""

import numpy as np

import mplang.v2 as mp
from mplang.v2.dialects import simp, tensor
from mplang.v2.edsl import trace
from mplang.v2.libs.mpc import _utils as utils


def run_protocol(sim, protocol_fn):
    """Helper to trace and run a protocol."""
    traced = trace(protocol_fn)
    # Simulator doesn't have evaluate_graph directly on it in v2 API wrapper,
    # but backend (Interpreter) does. However we should use public API.
    return mp.evaluate(traced)


def test_bits_conversion():
    """Test bytes_to_bits and bits_to_bytes type inference."""
    sim = simp.make_simulator(world_size=2)
    mp.set_root_context(sim)

    # 1 byte: 0b10101010 = 170
    data = np.array([170], dtype=np.uint8)

    def protocol():
        # Use pcall_static to work with local tensor ops
        def local_bits_conversion():
            t_data = tensor.constant(data)
            t_bits = utils.bytes_to_bits(t_data)
            t_back = utils.bits_to_bytes(t_bits)
            return t_bits, t_back

        return simp.pcall_static((0,), local_bits_conversion)

    # Just trace to verify type inference
    traced = trace(protocol)

    # Check output types (wrapped in MPType)
    assert traced.graph.outputs[0].type.value_type.shape == (8,)
    assert traced.graph.outputs[1].type.value_type.shape == (1,)

    hasattr(sim, "_simp_cluster") and sim._simp_cluster.shutdown()


def test_cuckoo_hash():
    """Test CuckooHash class type inference."""
    sim = simp.make_simulator(world_size=2)
    mp.set_root_context(sim)

    items = np.array([1, 2, 3], dtype=np.int64)

    def protocol():
        def local_hash():
            t_items = tensor.constant(items)
            cuckoo = utils.CuckooHash(num_bins=10, num_hash_functions=3)
            return cuckoo.hash(t_items, seed=123)

        return simp.pcall_static((0,), local_hash)

    # Just trace to verify type inference
    traced = trace(protocol)

    # Shape should be (3,) wrapped in MPType
    assert traced.graph.outputs[0].type.value_type.shape == (3,)

    hasattr(sim, "_simp_cluster") and sim._simp_cluster.shutdown()
