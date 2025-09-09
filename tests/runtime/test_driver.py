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
    """Fixture to start servers for HttpDriver testing."""
    # Start distributed servers for driver tests
    ports = [9001, 9002, 9003]
    for port in ports:
        process = multiprocessing.Process(target=run_distributed_server, args=(port,))
        process.daemon = True
        distributed_server_processes[port] = process
        process.start()

    # Wait for servers to be ready via health check
    for port in ports:
        ready = False
        for _ in range(50):  # up to ~5s
            try:
                r = httpx.get(f"http://localhost:{port}/health", timeout=0.2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.1)
        if not ready:
            raise RuntimeError(f"Server on port {port} failed to start in time")

    yield

    # Teardown: stop all server processes
    for port in ports:
        if port in distributed_server_processes:
            process = distributed_server_processes[port]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)  # Wait up to 5 seconds
                if process.is_alive():
                    process.kill()  # Force kill if still alive


def test_http_driver_initialization():
    """Test HttpDriver initialization and basic properties."""
    node_addrs = {
        "0": "http://localhost:9001",
        "1": "http://localhost:9002",
        "2": "http://localhost:9003",
    }

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


def test_session_creation():
    """Test session creation across multiple HTTP servers."""
    node_addrs = {
        "0": "http://localhost:9001",
        "1": "http://localhost:9002",
    }

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


def test_unique_name_generation():
    """Test unique name generation for executions."""
    node_addrs = {"0": "http://localhost:9001"}
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


def test_driver_context_properties():
    """Test HttpDriver as InterpContext."""
    node_addrs = {
        "0": "http://localhost:9001",
        "1": "http://localhost:9002",
    }

    cluster_spec = create_test_cluster_spec(node_addrs)
    driver = Driver(cluster_spec)

    # Test context properties
    assert driver.world_size() == 2
