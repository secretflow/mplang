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

from mplang.runtime.server import app
from tests.utils.server_fixtures import get_free_ports

client = TestClient(app)


def _endpoints(n: int) -> list[str]:
    return [f"http://localhost:{port}" for port in get_free_ports(n)]


def test_create_and_get_session():
    """Test creating and retrieving a session."""
    # Create a session with a specific name, rank, and endpoints
    eps = _endpoints(2)
    response = client.put(
        "/sessions/test_session_1",
        json={
            "rank": 0,
            "endpoints": eps,
            "spu_mask": -1,
            "spu_protocol": "SEMI2K",
            "spu_field": "FM64",
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
    eps = _endpoints(2)
    client.put(
        "/sessions/test_session_2",
        json={
            "rank": 0,
            "endpoints": eps,
            "spu_mask": -1,
            "spu_protocol": "SEMI2K",
            "spu_field": "FM64",
        },
    )

    # Create a computation
    # The mpprogram is a dummy base64 string that won't parse correctly
    # but we just want to test the basic HTTP mechanics for now
    mpprogram_b64 = "dGVzdA=="  # "test" in base64
    response = client.put(
        "/sessions/test_session_2/computations/test_computation",
        json={
            "mpprogram": mpprogram_b64,
            "input_names": [],
            "output_names": ["result"],
        },
    )
    # This should fail with 400 because "test" is not valid protobuf
    assert response.status_code == 400

    # Try to create a computation in a non-existent session
    response = client.put(
        "/sessions/non_existent_session/computations/test_computation_2",
        json={
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
    eps = _endpoints(2)
    response = client.put(
        "/sessions/test_session_3",
        json={
            "rank": 0,
            "endpoints": eps,
            "spu_mask": -1,
            "spu_protocol": "SEMI2K",
            "spu_field": "FM64",
        },
    )
    assert response.status_code == 200

    # Create valid pickled data for a simple integer
    test_value = 42
    pickled_data = pickle.dumps(test_value)
    encoded_data = base64.b64encode(pickled_data).decode("utf-8")

    symbol_data = {
        "mptype": {"scalar_type": {"type": "SCALAR_TYPE_I32"}},
        "data": encoded_data,
    }
    response = client.put(
        "/sessions/test_session_3/symbols/my_symbol",
        json=symbol_data,
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["name"] == "my_symbol"


def test_global_symbol_crud():
    # Create global symbol
    import numpy as np

    arr = np.arange(6, dtype=np.int32)
    data_b64 = base64.b64encode(pickle.dumps(arr)).decode("utf-8")
    resp = client.put(
        "/api/v1/symbols/gx",
        json={"mptype": {"tensor": {"dtype": "I32", "shape": [6]}}, "data": data_b64},
    )
    assert resp.status_code == 200

    # Get global symbol
    resp = client.get("/api/v1/symbols/gx")
    assert resp.status_code == 200
    payload = resp.json()
    fetched = pickle.loads(base64.b64decode(payload["data"]))
    assert np.array_equal(fetched, arr)

    # List contains symbol
    resp = client.get("/api/v1/symbols")
    assert resp.status_code == 200
    assert "gx" in resp.json().get("symbols", [])

    # Delete
    resp = client.delete("/api/v1/symbols/gx")
    assert resp.status_code == 200

    # Get after delete -> 404
    resp = client.get("/api/v1/symbols/gx")
    assert resp.status_code == 404
