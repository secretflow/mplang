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

"""PSI Tests using psi_unbalanced for proper intersection verification."""

import numpy as np

from mplang.v2.backends.simp_simulator import SimpSimulator
from mplang.v2.dialects import simp
from mplang.v2.edsl import trace
from mplang.v2.libs.mpc.psi import psi_unbalanced


def run_protocol(sim: SimpSimulator, protocol_fn):
    """Helper to trace and run a protocol."""
    traced = trace(protocol_fn)
    return sim.evaluate_graph(traced.graph, [])


def test_psi_full_overlap():
    """Test PSI with identical sets (full overlap)."""
    sim = SimpSimulator(world_size=2)

    # Both parties have the same items
    server_items = np.arange(100, dtype=np.uint64)
    client_items = np.arange(100, dtype=np.uint64)

    SERVER, CLIENT = 0, 1

    def protocol():
        s_items = simp.constant((SERVER,), server_items)
        c_items = simp.constant((CLIENT,), client_items)
        return psi_unbalanced(SERVER, CLIENT, 100, 100, s_items, c_items)

    result = run_protocol(sim, protocol)

    # Result is intersection mask on client
    mask = result[CLIENT].unwrap()

    # All items should be in intersection
    assert mask.shape == (100,)
    assert np.sum(mask) == 100, (
        f"Expected 100 matches for full overlap, got {np.sum(mask)}"
    )
    print(f"Full overlap: {np.sum(mask)}/100 matches")

    sim.shutdown()


def test_psi_no_overlap():
    """Test PSI with disjoint sets (no overlap)."""
    sim = SimpSimulator(world_size=2)

    # Disjoint sets
    server_items = np.arange(0, 100, dtype=np.uint64)
    client_items = np.arange(200, 300, dtype=np.uint64)

    SERVER, CLIENT = 0, 1

    def protocol():
        s_items = simp.constant((SERVER,), server_items)
        c_items = simp.constant((CLIENT,), client_items)
        return psi_unbalanced(SERVER, CLIENT, 100, 100, s_items, c_items)

    result = run_protocol(sim, protocol)

    # Result is intersection mask on client
    mask = result[CLIENT].unwrap()

    # No items should match
    assert mask.shape == (100,)
    assert np.sum(mask) == 0, f"Expected 0 matches for no overlap, got {np.sum(mask)}"
    print(f"No overlap: {np.sum(mask)}/100 matches")

    sim.shutdown()


def test_psi_partial_overlap():
    """Test PSI with 50% overlap."""
    sim = SimpSimulator(world_size=2)

    # Server has 0-99, client has 50-149 (overlap: 50-99)
    server_items = np.arange(0, 100, dtype=np.uint64)
    client_items = np.arange(50, 150, dtype=np.uint64)

    SERVER, CLIENT = 0, 1

    def protocol():
        s_items = simp.constant((SERVER,), server_items)
        c_items = simp.constant((CLIENT,), client_items)
        return psi_unbalanced(SERVER, CLIENT, 100, 100, s_items, c_items)

    result = run_protocol(sim, protocol)

    # Result is intersection mask on client
    mask = result[CLIENT].unwrap()

    # First 50 items of client (50-99) should be in intersection
    expected_matches = 50
    assert mask.shape == (100,)
    actual_matches = np.sum(mask)
    assert actual_matches == expected_matches, (
        f"Expected {expected_matches} matches, got {actual_matches}"
    )

    # Verify specific positions: first 50 items should match
    assert np.all(mask[:50] == 1), "First 50 client items should be in intersection"
    assert np.all(mask[50:] == 0), "Last 50 client items should NOT be in intersection"

    print(f"Partial overlap: {actual_matches}/100 matches")

    sim.shutdown()
