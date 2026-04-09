# Copyright 2026 Ant Group Co., Ltd.
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

"""Tests for async exec endpoints (/exec/async and /exec/{exec_id}/status)."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import mplang.edsl as el
from mplang.backends.simp_worker.http import create_worker_app
from mplang.dialects import simp
from mplang.edsl import serde


@pytest.fixture
def single_worker_client():
    """Create a TestClient with a single-worker app (rank=0, world_size=1)."""
    endpoints = ["http://127.0.0.1:19999"]
    app = create_worker_app(rank=0, world_size=1, endpoints=endpoints)
    with TestClient(app) as client:
        yield client


def _make_constant_graph():
    """Build a simple graph: constant → output."""

    def workflow():
        return simp.constant((0,), np.array([1.0, 2.0]))

    traced = el.trace(workflow)
    return traced.graph


class TestExecAsync:
    """Tests for POST /exec/async."""

    def test_submit_returns_exec_id(self, single_worker_client):
        client = single_worker_client
        graph = _make_constant_graph()
        payload = {
            "graph": serde.dumps_b64(graph),
            "inputs": serde.dumps_b64([]),
            "job_id": "test-job-1",
        }

        resp = client.post("/exec/async", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["exec_id"] == "test-job-1"

    def test_submit_bad_graph_returns_400(self, single_worker_client):
        client = single_worker_client
        payload = {
            "graph": "not-valid-base64-graph",
            "inputs": serde.dumps_b64([]),
            "job_id": "test-bad",
        }

        resp = client.post("/exec/async", json=payload)
        assert resp.status_code == 400

    def test_submit_missing_job_id_returns_400(self, single_worker_client):
        client = single_worker_client
        graph = _make_constant_graph()
        payload = {
            "graph": serde.dumps_b64(graph),
            "inputs": serde.dumps_b64([]),
        }

        resp = client.post("/exec/async", json=payload)
        assert resp.status_code == 400
        assert "job_id is required" in resp.json()["detail"]


class TestExecStatus:
    """Tests for GET /exec/{exec_id}/status."""

    def test_poll_until_success(self, single_worker_client):
        client = single_worker_client
        graph = _make_constant_graph()
        payload = {
            "graph": serde.dumps_b64(graph),
            "inputs": serde.dumps_b64([]),
            "job_id": "test-poll-1",
        }

        # Submit
        resp = client.post("/exec/async", json=payload)
        assert resp.json()["exec_id"] == "test-poll-1"

        # Poll until terminal state (with timeout)
        deadline = time.monotonic() + 30
        status = None
        while time.monotonic() < deadline:
            resp = client.get("/exec/test-poll-1/status")
            assert resp.status_code == 200
            data = resp.json()
            status = data["status"]
            if status in ("SUCCESS", "FAILED"):
                break
            time.sleep(0.1)

        assert status == "SUCCESS"
        assert data["result"] is not None

    def test_poll_nonexistent_returns_404(self, single_worker_client):
        client = single_worker_client

        resp = client.get("/exec/nonexistent-id/status")
        assert resp.status_code == 404
