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

"""Unit tests for base.py (Request handles and batch operations).

These tests verify the Request handle infrastructure that is shared
by both HttpCommunicator and ThreadCommunicator.
"""

import concurrent.futures

import pytest

from mplang.backends.simp_worker.base import (
    Request,
    RequestStatus,
    SendRequest,
    testall,
    testany,
    wait_all,
    wait_any,
)

# ---------------------------------------------------------------------------
# Request Tests
# ---------------------------------------------------------------------------


class TestRequest:
    """Tests for base Request class."""

    def test_completed_request(self):
        """Test request with completed future."""
        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        future.set_result(42)

        req = Request(future, "test op")
        assert req.done is True
        assert req.status == RequestStatus.COMPLETED
        assert req.wait() == 42

    def test_pending_request(self):
        """Test request with pending future."""
        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        req = Request(future, "test op")

        assert req.done is False
        assert req.status == RequestStatus.PENDING

        # Complete it
        future.set_result(123)
        assert req.done is True
        assert req.wait() == 123

    def test_error_request(self):
        """Test request with failed future."""
        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        future.set_exception(ValueError("test error"))

        req = Request(future, "test op")
        assert req.done is True
        assert req.status == RequestStatus.ERROR

        with pytest.raises(ValueError, match="test error"):
            req.wait()

    def test_wait_timeout(self):
        """Test wait with timeout on pending request."""
        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        req = Request(future, "test op")

        with pytest.raises(TimeoutError):
            req.wait(timeout=0.01)

    def test_test_method(self):
        """Test the non-blocking test() method."""
        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        req = Request(future, "test op")

        done, result = req.test()
        assert done is False
        assert result is None

        future.set_result(99)
        done, result = req.test()
        assert done is True
        assert result == 99

    def test_cancel(self):
        """Test cancellation of pending request."""
        # Note: Regular Future can be cancelled before it starts
        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        req = Request(future, "test op")

        # Futures not submitted to executor can be cancelled
        assert req.cancel() is True
        assert req.status == RequestStatus.CANCELLED


class TestSendRequest:
    """Tests for SendRequest class."""

    def test_attributes(self):
        """Test SendRequest stores to/key attributes."""
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        future.set_result(None)

        req = SendRequest(future, to=2, key="msg_key")
        assert req.to == 2
        assert req.key == "msg_key"
        assert "send to rank 2" in req._operation


# ---------------------------------------------------------------------------
# Batch Operation Tests
# ---------------------------------------------------------------------------


class TestWaitAll:
    """Tests for wait_all function."""

    def test_empty_list(self):
        """Test wait_all with empty list returns empty."""
        results = wait_all([])
        assert results == []

    def test_all_completed(self):
        """Test wait_all with all completed requests."""
        futures = [concurrent.futures.Future() for _ in range(3)]
        for i, f in enumerate(futures):
            f.set_result(i * 10)

        requests = [Request(f, f"op{i}") for i, f in enumerate(futures)]
        results = wait_all(requests)

        assert results == [0, 10, 20]

    def test_timeout(self):
        """Test wait_all with timeout."""
        f1: concurrent.futures.Future[int] = concurrent.futures.Future()
        f1.set_result(1)
        f2: concurrent.futures.Future[int] = (
            concurrent.futures.Future()
        )  # Never completes

        requests = [Request(f1, "op1"), Request(f2, "op2")]

        with pytest.raises(TimeoutError, match="wait_all timed out"):
            wait_all(requests, timeout=0.01)


class TestWaitAny:
    """Tests for wait_any function."""

    def test_empty_list_raises(self):
        """Test wait_any with empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            wait_any([])

    def test_one_completed(self):
        """Test wait_any returns first completed."""
        f1: concurrent.futures.Future[str] = concurrent.futures.Future()
        f2: concurrent.futures.Future[str] = concurrent.futures.Future()
        f2.set_result("second")

        requests = [Request(f1, "op1"), Request(f2, "op2")]
        idx, result = wait_any(requests)

        assert idx == 1
        assert result == "second"


class TestTestall:
    """Tests for testall function."""

    def test_all_done(self):
        """Test testall when all are done."""
        futures = [concurrent.futures.Future() for _ in range(2)]
        futures[0].set_result("a")
        futures[1].set_result("b")

        requests = [Request(f, f"op{i}") for i, f in enumerate(futures)]
        done, results = testall(requests)

        assert done is True
        assert results == ["a", "b"]

    def test_not_all_done(self):
        """Test testall when not all are done."""
        f1: concurrent.futures.Future[str] = concurrent.futures.Future()
        f1.set_result("done")
        f2: concurrent.futures.Future[str] = concurrent.futures.Future()

        requests = [Request(f1, "op1"), Request(f2, "op2")]
        done, results = testall(requests)

        assert done is False
        assert results is None


class TestTestany:
    """Tests for testany function."""

    def test_one_done(self):
        """Test testany when one is done."""
        f1: concurrent.futures.Future[str] = concurrent.futures.Future()
        f2: concurrent.futures.Future[str] = concurrent.futures.Future()
        f1.set_result("first")

        requests = [Request(f1, "op1"), Request(f2, "op2")]
        idx, result = testany(requests)

        assert idx == 0
        assert result == "first"

    def test_none_done(self):
        """Test testany when none are done."""
        f1: concurrent.futures.Future[str] = concurrent.futures.Future()
        f2: concurrent.futures.Future[str] = concurrent.futures.Future()

        requests = [Request(f1, "op1"), Request(f2, "op2")]
        idx, result = testany(requests)

        assert idx is None
        assert result is None
