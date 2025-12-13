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

"""Shared fixtures for device API tests.

This module provides common cluster configurations and context management
fixtures used across device-related test files.
"""

import pytest

from mplang.v2.dialects import simp
from mplang.v2.edsl.context import pop_context, push_context
from mplang.v2.libs.device import ClusterSpec

# =============================================================================
# Cluster Configuration Fixtures
# =============================================================================


@pytest.fixture
def cluster_3pc_aby3():
    """3-party computation cluster with ABY3 protocol.

    Layout:
    - 4 nodes: node_0, node_1, node_2, node_3
    - SP0 (SPU): 3-party ABY3 on node_0, node_1, node_2
    - P0, P1 (PPU): Single-party on node_0 and node_3
    - TEE0: Trusted execution on node_1
    """
    config = {
        "nodes": [
            {"name": "node_0", "endpoint": "http://127.0.0.1:61920"},
            {"name": "node_1", "endpoint": "http://127.0.0.1:61921"},
            {"name": "node_2", "endpoint": "http://127.0.0.1:61922"},
            {"name": "node_3", "endpoint": "http://127.0.0.1:61923"},
        ],
        "devices": {
            "SP0": {
                "kind": "SPU",
                "members": ["node_0", "node_1", "node_2"],
                "config": {"protocol": "ABY3", "field": "FM64"},
            },
            "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
            "P1": {"kind": "PPU", "members": ["node_3"], "config": {}},
            "P2": {"kind": "PPU", "members": ["node_2"], "config": {}},
            "TEE0": {"kind": "TEE", "members": ["node_1"], "config": {}},
        },
    }
    return ClusterSpec.from_dict(config)


@pytest.fixture
def cluster_2pc_semi2k():
    """2-party computation cluster with SEMI2K protocol.

    Layout:
    - 2 nodes: alice, bob
    - SP0 (SPU): 2-party SEMI2K
    - P_alice, P_bob (PPU): Single-party each
    """
    config = {
        "nodes": [
            {"name": "alice", "endpoint": "http://127.0.0.1:9000"},
            {"name": "bob", "endpoint": "http://127.0.0.1:9001"},
        ],
        "devices": {
            "SP0": {
                "kind": "SPU",
                "members": ["alice", "bob"],
                "config": {"protocol": "SEMI2K", "field": "FM128"},
            },
            "P_alice": {"kind": "PPU", "members": ["alice"], "config": {}},
            "P_bob": {"kind": "PPU", "members": ["bob"], "config": {}},
        },
    }
    return ClusterSpec.from_dict(config)


@pytest.fixture
def cluster_4pc_multi_spu():
    """4-party cluster with multiple SPU devices.

    Layout:
    - 4 nodes: p0, p1, p2, p3
    - SP0 (SPU): 2-party CHEETAH on p0, p1
    - SP1 (SPU): 2-party SEMI2K on p2, p3
    - P0, P1, P2, P3 (PPU): One per node
    """
    config = {
        "nodes": [
            {"name": "p0", "endpoint": "http://127.0.0.1:8000"},
            {"name": "p1", "endpoint": "http://127.0.0.1:8001"},
            {"name": "p2", "endpoint": "http://127.0.0.1:8002"},
            {"name": "p3", "endpoint": "http://127.0.0.1:8003"},
        ],
        "devices": {
            "SP0": {
                "kind": "SPU",
                "members": ["p0", "p1"],
                "config": {"protocol": "CHEETAH", "field": "FM64"},
            },
            "SP1": {
                "kind": "SPU",
                "members": ["p2", "p3"],
                "config": {"protocol": "SEMI2K", "field": "FM128"},
            },
            "P0": {"kind": "PPU", "members": ["p0"], "config": {}},
            "P1": {"kind": "PPU", "members": ["p1"], "config": {}},
            "P2": {"kind": "PPU", "members": ["p2"], "config": {}},
            "P3": {"kind": "PPU", "members": ["p3"], "config": {}},
        },
    }
    return ClusterSpec.from_dict(config)


@pytest.fixture
def cluster_ppu_only():
    """PPU-only cluster (no secure devices).

    Layout:
    - 2 nodes: node_0, node_1
    - P0, P1 (PPU): One per node
    """
    config = {
        "nodes": [
            {"name": "node_0", "endpoint": "http://127.0.0.1:5000"},
            {"name": "node_1", "endpoint": "http://127.0.0.1:5001"},
        ],
        "devices": {
            "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
            "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
        },
    }
    return ClusterSpec.from_dict(config)


# =============================================================================
# Context Management Fixtures
# =============================================================================


@pytest.fixture
def ctx_3pc(cluster_3pc_aby3):
    """Set up 3-party cluster with simulator context."""
    interp = simp.make_simulator(
        len(cluster_3pc_aby3.nodes), cluster_spec=cluster_3pc_aby3
    )
    push_context(interp)
    yield cluster_3pc_aby3
    pop_context()
    cluster = getattr(interp, "_simp_cluster", None)
    if cluster:
        cluster.shutdown()


@pytest.fixture
def ctx_2pc(cluster_2pc_semi2k):
    """Set up 2-party cluster with simulator context."""
    interp = simp.make_simulator(
        len(cluster_2pc_semi2k.nodes), cluster_spec=cluster_2pc_semi2k
    )
    push_context(interp)
    yield cluster_2pc_semi2k
    pop_context()
    cluster = getattr(interp, "_simp_cluster", None)
    if cluster:
        cluster.shutdown()


@pytest.fixture
def ctx_4pc(cluster_4pc_multi_spu):
    """Set up 4-party multi-SPU cluster with simulator context."""
    interp = simp.make_simulator(
        len(cluster_4pc_multi_spu.nodes), cluster_spec=cluster_4pc_multi_spu
    )
    push_context(interp)
    yield cluster_4pc_multi_spu
    pop_context()
    cluster = getattr(interp, "_simp_cluster", None)
    if cluster:
        cluster.shutdown()


@pytest.fixture
def ctx_ppu_only(cluster_ppu_only):
    """Set up PPU-only cluster with simulator context."""
    interp = simp.make_simulator(
        len(cluster_ppu_only.nodes), cluster_spec=cluster_ppu_only
    )
    push_context(interp)
    yield cluster_ppu_only
    pop_context()
    cluster = getattr(interp, "_simp_cluster", None)
    if cluster:
        cluster.shutdown()


@pytest.fixture
def cluster_multi_tee():
    """Cluster with multiple TEE devices for testing TEE-to-TEE transfers.

    Layout:
    - 3 nodes: node_0, node_1, node_2
    - TEE0, TEE1: Two separate TEE devices
    - P0 (PPU): Single-party for data input
    """
    config = {
        "nodes": [
            {"name": "node_0", "endpoint": "http://127.0.0.1:7000"},
            {"name": "node_1", "endpoint": "http://127.0.0.1:7001"},
            {"name": "node_2", "endpoint": "http://127.0.0.1:7002"},
        ],
        "devices": {
            "TEE0": {"kind": "TEE", "members": ["node_0"], "config": {}},
            "TEE1": {"kind": "TEE", "members": ["node_1"], "config": {}},
            "P0": {"kind": "PPU", "members": ["node_2"], "config": {}},
        },
    }
    return ClusterSpec.from_dict(config)


@pytest.fixture
def ctx_multi_tee(cluster_multi_tee):
    """Set up cluster with multiple TEEs and simulator context."""
    interp = simp.make_simulator(
        len(cluster_multi_tee.nodes), cluster_spec=cluster_multi_tee
    )
    push_context(interp)
    yield cluster_multi_tee
    pop_context()
    cluster = getattr(interp, "_simp_cluster", None)
    if cluster:
        cluster.shutdown()
