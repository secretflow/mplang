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

import threading
import time
from typing import Any

import pytest
import uvicorn

from mplang.runtime.http_backend.driver import HttpDriver
from mplang.runtime.http_backend.server import app

# Global state for servers
distributed_servers: dict[int, Any] = {}
distributed_server_threads: dict[int, threading.Thread] = {}


def run_distributed_server(port: int):
    """Function to run a uvicorn server on a specific port for distributed testing."""
    config = uvicorn.Config(app, host="localhost", port=port, log_level="critical")
    server = uvicorn.Server(config)
    distributed_servers[port] = server
    server.run()


@pytest.fixture(scope="module", autouse=True)
def start_servers():
    """Fixture to start servers for HttpDriver testing."""
    # Start distributed servers for driver tests
    ports = [9001, 9002, 9003]
    for port in ports:
        thread = threading.Thread(
            target=run_distributed_server, args=(port,), daemon=True
        )
        distributed_server_threads[port] = thread
        thread.start()

    # Give servers time to start up
    time.sleep(3)

    yield

    # Teardown: stop all servers
    for port in ports:
        if port in distributed_servers:
            distributed_servers[port].should_exit = True
            if port in distributed_server_threads:
                distributed_server_threads[port].join(timeout=2)


def test_http_driver_initialization():
    """Test HttpDriver initialization and basic properties."""
    node_addrs = {
        0: "http://localhost:9001",
        1: "http://localhost:9002",
        2: "http://localhost:9003",
    }

    driver = HttpDriver(node_addrs)

    # Test basic properties
    assert driver.world_size == 3
    assert len(driver.party_addrs) == 3

    # clients dictionary uses integer keys
    assert len(driver.clients) == 3
    assert 0 in driver.clients
    assert 1 in driver.clients
    assert 2 in driver.clients


def test_session_creation():
    """Test session creation across multiple HTTP servers."""
    node_addrs = {
        0: "http://localhost:9001",
        1: "http://localhost:9002",
    }

    driver = HttpDriver(node_addrs)

    # Create session
    session_id = driver.get_or_create_session()
    assert isinstance(session_id, str)
    assert len(session_id) > 0

    # Should return same session ID on subsequent calls
    session_id2 = driver.get_or_create_session()
    assert session_id == session_id2


def test_unique_name_generation():
    """Test unique name generation for executions."""
    node_addrs = {0: "http://localhost:9001"}
    driver = HttpDriver(node_addrs)

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
        0: "http://localhost:9001",
        1: "http://localhost:9002",
    }

    driver = HttpDriver(node_addrs, custom_attr="test_value")

    # Test context properties
    assert driver.world_size == 2
    assert driver.attr("custom_attr") == "test_value"
