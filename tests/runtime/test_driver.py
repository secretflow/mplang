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

"""
Tests for the HttpDriver.
"""

import multiprocessing
import time

import httpx
import pytest
import uvicorn

from mplang.core.cluster import ClusterSpec, LogicalDevice, PhysicalNode, RuntimeInfo
from mplang.runtime.driver import Driver
from mplang.runtime.server import app
from tests.conftest import get_free_ports

# Global state for servers
distributed_server_processes: dict[int, multiprocessing.Process] = {}


def create_test_cluster_spec(node_addrs: dict[str, str]) -> ClusterSpec:
    """Create a ClusterSpec for testing with the given node addresses."""
    nodes = {}
    for node_id, addr in node_addrs.items():
        rank = int(node_id)
        nodes[f"node{rank}"] = PhysicalNode(
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
        local_devices[f"local_{node.rank}"] = LogicalDevice(
            name=f"local_{node.rank}",
            kind="local",
            members=[node],
        )

    # Create SPU device with all nodes
    spu_device = LogicalDevice(
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


def run_distributed_server(port: int):
    """Function to run a uvicorn server on a specific port for distributed testing."""
    config = uvicorn.Config(
        app,
        host="localhost",
        port=port,
        log_level="critical",
        ws="none",  # Disable websockets to avoid deprecation warnings
    )
    server = uvicorn.Server(config)
    server.run()


@pytest.fixture(scope="module", autouse=True)
def start_servers():
    """Start 3 HTTP servers on dynamic ports and return node address mapping."""
    ports = get_free_ports(3)
    node_ids = ["0", "1", "2"]
    node_addrs = {
        nid: f"http://localhost:{p}" for nid, p in zip(node_ids, ports, strict=True)
    }

    for port in ports:
        process = multiprocessing.Process(target=run_distributed_server, args=(port,))
        process.daemon = True
        distributed_server_processes[port] = process
        process.start()

    # Wait for servers to be ready via health check
    for port in ports:
        ready = False
        for _ in range(60):  # up to ~6s
            try:
                r = httpx.get(f"http://localhost:{port}/health", timeout=0.25)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.1)
        if not ready:
            raise RuntimeError(f"Server on port {port} failed to start in time")

    yield node_addrs

    for port in ports:
        process = distributed_server_processes.get(port)
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()


def test_http_driver_initialization(start_servers):  # type: ignore
    """Test HttpDriver initialization and basic properties."""
    node_addrs = start_servers

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
    import asyncio

    asyncio.run(driver._close_clients(clients))


def test_session_creation(start_servers):  # type: ignore
    """Test session creation across multiple HTTP servers."""
    # Take only two nodes for this test
    full_nodes = start_servers
    node_addrs = {k: full_nodes[k] for k in list(full_nodes.keys())[:2]}

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


def test_unique_name_generation(start_servers):  # type: ignore
    """Test unique name generation for executions."""
    # Use only one node
    full_nodes = start_servers
    first_key = next(iter(full_nodes.keys()))
    node_addrs = {first_key: full_nodes[first_key]}
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


def test_driver_context_properties(start_servers):  # type: ignore
    """Test HttpDriver as InterpContext."""
    full_nodes = start_servers
    node_keys = list(full_nodes.keys())[:2]
    node_addrs = {k: full_nodes[k] for k in node_keys}

    cluster_spec = create_test_cluster_spec(node_addrs)
    driver = Driver(cluster_spec)

    # Test context properties
    assert driver.world_size() == 2
