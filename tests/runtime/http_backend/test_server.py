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
Tests for the HTTP server endpoints.
"""

import base64

import cloudpickle as pickle
from fastapi.testclient import TestClient

from mplang.runtime.http_backend.server import app

client = TestClient(app)


def test_create_and_get_session():
    """Test creating and retrieving a session."""
    # Create a session with a specific name, rank, and endpoints
    response = client.post(
        "/sessions",
        json={
            "name": "test_session_1",
            "rank": 0,
            "endpoints": ["http://localhost:8000", "http://localhost:8001"],
        },
    )
    assert response.status_code == 200
    assert response.json()["name"] == "test_session_1"

    # Get the session
    response = client.get("/sessions/test_session_1")
    assert response.status_code == 200
    assert response.json()["name"] == "test_session_1"

    # Try to get a non-existent session
    response = client.get("/sessions/non_existent_session")
    assert response.status_code == 404


def test_create_computation():
    """Test creating a computation within a session."""
    # First, create a session
    client.post(
        "/sessions",
        json={
            "name": "test_session_2",
            "rank": 0,
            "endpoints": ["http://localhost:8000", "http://localhost:8001"],
        },
    )

    # Create a computation
    # The mpprogram is a dummy base64 string that won't parse correctly
    # but we just want to test the basic HTTP mechanics for now
    mpprogram_b64 = "dGVzdA=="  # "test" in base64
    response = client.post(
        "/sessions/test_session_2/computations",
        json={
            "computation_id": "test_computation",
            "mpprogram": mpprogram_b64,
            "input_names": [],
            "output_names": ["result"],
        },
    )
    # This should fail with 400 because "test" is not valid protobuf
    assert response.status_code == 400

    # Try to create a computation in a non-existent session
    response = client.post(
        "/sessions/non_existent_session/computations",
        json={
            "computation_id": "test_computation_2",
            "mpprogram": mpprogram_b64,
            "input_names": [],
            "output_names": ["result"],
        },
    )
    # This will fail with 400 on protobuf parsing before session check
    assert response.status_code == 400


def test_create_and_get_symbol():
    """Test creating and retrieving a symbol."""
    # Create session first
    response = client.post(
        "/sessions",
        json={
            "name": "test_session_3",
            "rank": 0,
            "endpoints": ["http://localhost:8000", "http://localhost:8001"],
        },
    )
    assert response.status_code == 200

    # Create valid pickled data for a simple integer
    test_value = 42
    pickled_data = pickle.dumps(test_value)
    encoded_data = base64.b64encode(pickled_data).decode("utf-8")

    symbol_data = {
        "name": "my_symbol",
        "mptype": {"scalar_type": {"type": "SCALAR_TYPE_I32"}},
        "data": encoded_data,
    }
    response = client.post(
        "/sessions/test_session_3/symbols",
        json=symbol_data,
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["name"] == "my_symbol"

    # Retrieve the symbol (using session-level path, not computation)
    response = client.get("/sessions/test_session_3/symbols/my_symbol")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["name"] == "my_symbol"
