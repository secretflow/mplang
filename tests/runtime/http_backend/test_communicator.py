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
Tests for the HttpCommunicator.
"""

import multiprocessing
import time
from typing import Any

import httpx
import pytest
import uvicorn

from mplang.runtime.http_backend.communicator import HttpCommunicator
from mplang.runtime.http_backend.server import app

# Global state for servers
distributed_servers: dict[int, Any] = {}
distributed_server_processes: dict[int, multiprocessing.Process] = {}


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
    distributed_servers[port] = server
    server.run()


@pytest.fixture(scope="module", autouse=True)
def start_servers():
    """Fixture to start servers for testing."""
    # Start distributed servers for distributed tests
    ports = [8001, 8002, 8003]
    for port in ports:
        process = multiprocessing.Process(
            target=run_distributed_server, args=(port,), daemon=True
        )
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

    # Teardown: stop all servers
    for port in ports:
        if port in distributed_server_processes:
            process = distributed_server_processes[port]
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

    # No file logging cleanup needed


def test_distributed_send_recv():
    """
    Test distributed communication where each party has its own server.
    This uses a simple HTTP validation approach since proper single-process-per-party
    testing requires more complex multiprocess coordination.
    """
    # For now, test the HTTP endpoints work correctly for communication
    # The actual distributed test requires the single-process-per-party architecture

    # Test that we can call the health endpoint on all servers

    endpoints = [
        "http://localhost:8001",
        "http://localhost:8002",
    ]

    for endpoint in endpoints:
        response = httpx.get(f"{endpoint}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # Test that session creation works on each server independently
    # Each server will create its own session with the rank appropriate for that server
    session_name = "test_session"

    for i, endpoint in enumerate(endpoints):
        rank = i  # Server 0 gets rank 0, server 1 gets rank 1
        response = httpx.post(
            f"{endpoint}/sessions",
            json={
                "name": session_name,
                "rank": rank,
                "endpoints": endpoints,
            },
        )
        assert response.status_code == 200
        assert (
            response.json()["name"] == session_name
        )  # This test verifies the basic HTTP communication works
    # For full bidirectional communication testing, see single_process_party.py


def test_distributed_multiple_messages():
    """Test multiple messages validation by ensuring HTTP endpoints respond correctly."""
    # This test validates that the HTTP server can handle multiple session creation requests
    # without interfering with each other

    # Test multiple independent sessions can be created
    base_endpoints = [
        "http://localhost:8001",
        "http://localhost:8002",
    ]

    # Create multiple sessions on the servers to test isolation
    for session_idx in range(3):
        session_name = f"multi_session_{session_idx}"

        for rank, endpoint in enumerate(base_endpoints):
            response = httpx.post(
                f"{endpoint}/sessions",
                json={"name": session_name, "rank": rank, "endpoints": base_endpoints},
            )
            assert response.status_code == 200
            assert response.json()["name"] == session_name


def test_communicator_properties():
    """Test the communicator properties and interface compliance."""
    session_name = "properties_test"
    endpoints = ["http://localhost:8001", "http://localhost:8002"]

    comm = HttpCommunicator(session_name, rank=0, endpoints=endpoints)

    # Test properties
    assert comm.rank == 0
    assert comm.world_size == 2

    # Test new_id method
    id1 = comm.new_id()
    id2 = comm.new_id()
    assert isinstance(id1, str)
    assert isinstance(id2, str)
    assert id1 != id2  # Should generate unique IDs


def run_party_e2e_process(rank: int, return_dict: dict):
    """
    Run a complete party process with server and communication logic.
    This is the proper single-process-per-party architecture.
    """
    import logging
    import sys
    import threading

    # Configure logging for this process (stdout only to avoid file conflicts)
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"%(asctime)s - E2E_PARTY{rank} - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(f"e2e_party{rank}")

    try:
        port = 25000 + rank

        # Create session and communicator
        session_name = "e2e_test_session"
        endpoints = {0: "http://localhost:25000", 1: "http://localhost:25001"}

        # Import after logging is configured
        from contextlib import asynccontextmanager

        from mplang.runtime.http_backend import resource

        # Create session in the resource manager
        logger.info(f"Creating session: {session_name}")
        session = resource.create_session(
            name=session_name, rank=rank, endpoints=list(endpoints.values())
        )

        # Save communicator for server to use
        global party_communicator
        party_communicator = session.communicator

        # Define server lifespan
        @asynccontextmanager
        async def lifespan(app):
            import asyncio

            logger.info(f"Server starting for party {rank}")
            await asyncio.sleep(0)
            yield
            await asyncio.sleep(0)
            logger.info(f"Server shutting down for party {rank}")

        # Import server app and set lifespan
        from mplang.runtime.http_backend.server import app

        app.router.lifespan_context = lifespan

        # Start server in background thread
        def start_server():
            logger.info(f"Starting HTTP server on port {port}")
            config = uvicorn.Config(
                app,
                host="localhost",
                port=port,
                log_level="error",
                ws="none",  # Disable websockets to avoid deprecation warnings
            )
            server = uvicorn.Server(config)
            server.run()

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        # Wait for server to be ready
        server_ready = False
        for _ in range(50):  # Try for ~5 seconds
            try:
                response = httpx.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    logger.info(f"Party {rank} server ready")
                    server_ready = True
                    break
            except Exception:
                pass
            time.sleep(0.1)
        if not server_ready:
            raise RuntimeError("Server failed to start within timeout")

        # Run party-specific communication logic
        if rank == 0:
            # Party 0: Send message to Party 1
            logger.info("Party 0: Sending message to party 1")
            test_data = {"message": "Hello from Party 0", "test_value": 42}

            session.communicator.send(to=1, key="test_message", data=test_data)
            logger.info("Party 0: Message sent, waiting for response")

            # Wait for response
            response = session.communicator.recv(frm=1, key="response_message")
            logger.info(f"Party 0: Received response: {response}")

            return_dict[rank] = {"status": "success", "received": response}

        else:  # rank == 1
            # Party 1: Wait for message from Party 0
            logger.info("Party 1: Waiting for message from party 0")

            received_data = session.communicator.recv(frm=0, key="test_message")
            logger.info(f"Party 1: Received message: {received_data}")

            # Send response back
            logger.info("Party 1: Sending response to party 0")
            response_data = {"status": "received", "original_message": received_data}

            session.communicator.send(to=0, key="response_message", data=response_data)
            logger.info("Party 1: Response sent")

            return_dict[rank] = {"status": "success", "sent_response": response_data}

        logger.info(f"Party {rank} test completed successfully")

    except Exception as e:
        logger.error(f"Party {rank} test failed: {e}", exc_info=True)
        return_dict[rank] = {"status": "error", "error": str(e)}


def test_end_to_end_communication():
    """
    Test complete end-to-end communication between two parties.
    This validates the full single-process-per-party architecture.
    """
    # No log file cleanup needed

    # Use multiprocessing to run two party processes
    with multiprocessing.Manager() as manager:
        return_dict = manager.dict()

        # Start both party processes
        processes = []
        for rank in [0, 1]:
            process = multiprocessing.Process(
                target=run_party_e2e_process, args=(rank, return_dict)
            )
            process.start()
            processes.append(process)

        # Wait for both processes to complete
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

        # Validate results
        for rank in [0, 1]:
            assert rank in return_dict, f"Party {rank} did not complete"
            result = return_dict[rank]
            assert result.get("status") == "success", f"Party {rank} failed: {result}"

        # Validate message content
        party0_result = return_dict[0]
        party1_result = return_dict[1]

        # Party 0 should have received the response
        assert "received" in party0_result
        received_response = party0_result["received"]
        assert received_response["status"] == "received"
        assert received_response["original_message"]["message"] == "Hello from Party 0"
        assert received_response["original_message"]["test_value"] == 42

        # Party 1 should have sent the response
        assert "sent_response" in party1_result
        sent_response = party1_result["sent_response"]
        assert sent_response["status"] == "received"
