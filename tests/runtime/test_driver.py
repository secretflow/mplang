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

"""Tests for the HttpDriver."""

import asyncio

import pytest

from mplang.core.cluster import ClusterSpec, Device, Node, RuntimeInfo
from mplang.runtime.driver import Driver
from tests.utils.server_fixtures import http_servers  # noqa: F401


def create_test_cluster_spec(node_addrs: dict[str, str]) -> ClusterSpec:
    """Create a ClusterSpec for testing with the given node addresses."""
    nodes = {}
    for node_id, addr in node_addrs.items():
        rank = int(node_id)
        nodes[f"node{rank}"] = Node(
            name=f"node{rank}",
            rank=rank,
            endpoint=addr,  # Keep the full HTTP URL as endpoint
            runtime_info=RuntimeInfo(
                version="test",
                platform="test",
                backends=["__all__"],
            ),
        )

    # Create local devices for each node
    local_devices = {}
    for _node_name, node in nodes.items():
        local_devices[f"local_{node.rank}"] = Device(
            name=f"local_{node.rank}",
            kind="local",
            members=[node],
        )

    # Create SPU device with all nodes
    spu_device = Device(
        name="SPU_0",
        kind="SPU",
        members=list(nodes.values()),
        config={
            "protocol": "SEMI2K",
            "field": "FM128",
        },
    )

    devices = {**local_devices, "SPU_0": spu_device}

    return ClusterSpec(nodes=nodes, devices=devices)


@pytest.mark.parametrize("http_servers", [3], indirect=True)
def test_http_driver_initialization(http_servers):  # type: ignore  # noqa: F811
    """Test HttpDriver initialization and basic properties."""
    # Build mapping 0..n-1 -> address
    node_addrs = {str(i): addr for i, addr in enumerate(http_servers.addresses)}

    cluster_spec = create_test_cluster_spec(node_addrs)
    driver = Driver(cluster_spec)

    # Test basic properties
    assert driver.world_size() == 3
    assert len(driver.node_addrs) == 3

    # Test that _create_clients method works correctly
    clients = driver._create_clients()
    assert len(clients) == 3
    assert "node0" in clients
    assert "node1" in clients
    assert "node2" in clients

    # Clean up clients
    asyncio.run(driver._close_clients(clients))


@pytest.mark.parametrize("http_servers", [3], indirect=True)
def test_session_creation(http_servers):  # type: ignore  # noqa: F811
    """Test session creation across multiple HTTP servers."""
    # Take only two nodes for this test
    node_addrs = {str(i): addr for i, addr in enumerate(http_servers.addresses[:2])}

    cluster_spec = create_test_cluster_spec(node_addrs)
    driver = Driver(cluster_spec)

    # Create session using async method
    import asyncio

    session_id = asyncio.run(driver._get_or_create_session())
    assert isinstance(session_id, str)
    assert len(session_id) > 0

    # Should return same session ID on subsequent calls
    session_id2 = asyncio.run(driver._get_or_create_session())
    assert session_id == session_id2


@pytest.mark.parametrize("http_servers", [1], indirect=True)
def test_unique_name_generation(http_servers):  # type: ignore  # noqa: F811
    """Test unique name generation for executions."""
    # Use only one node
    node_addrs = {"0": http_servers.addresses[0]}
    cluster_spec = create_test_cluster_spec(node_addrs)
    driver = Driver(cluster_spec)

    # Generate multiple names
    name1 = driver.new_name()
    name2 = driver.new_name("exec")
    name3 = driver.new_name()

    assert name1 != name2
    assert name1 != name3
    assert name2 != name3
    assert "var_" in name1
    assert "exec_" in name2


@pytest.mark.parametrize("http_servers", [2], indirect=True)
def test_driver_context_properties(http_servers):  # type: ignore  # noqa: F811
    """Test HttpDriver as InterpContext."""
    node_addrs = {str(i): addr for i, addr in enumerate(http_servers.addresses)}

    cluster_spec = create_test_cluster_spec(node_addrs)
    driver = Driver(cluster_spec)

    # Test context properties
    assert driver.world_size() == 2
