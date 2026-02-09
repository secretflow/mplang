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

"""Unit tests for HttpCommunicator, CommConfig, CommStats, and timeout exceptions.

These tests use mocks to avoid actual HTTP communication, focusing on
the communicator's internal logic.
"""

import concurrent.futures
import threading
import time
from unittest.mock import Mock, patch

import pytest

from mplang.backends.simp_worker.http import (
    CommConfig,
    CommStats,
    HttpCommunicator,
    RecvTimeoutError,
    SendTimeoutError,
)

# ---------------------------------------------------------------------------
# CommConfig Tests
# ---------------------------------------------------------------------------


class TestCommConfig:
    """Tests for CommConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CommConfig()
        assert config.default_send_timeout == 60.0
        assert config.default_recv_timeout == 600.0  # 10 minutes
        assert config.http_timeout == 60.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CommConfig(
            default_send_timeout=30.0,
            default_recv_timeout=10.0,
            http_timeout=5.0,
        )
        assert config.default_send_timeout == 30.0
        assert config.default_recv_timeout == 10.0
        assert config.http_timeout == 5.0


# ---------------------------------------------------------------------------
# CommStats Tests
# ---------------------------------------------------------------------------


class TestCommStats:
    """Tests for CommStats dataclass."""

    def test_initial_values(self):
        """Test initial statistics are zero."""
        stats = CommStats()
        assert stats.messages_sent == 0
        assert stats.messages_received == 0
        assert stats.bytes_sent == 0
        assert stats.bytes_received == 0
        assert stats.send_errors == 0
        assert stats.recv_timeouts == 0
        assert stats.total_send_time_ms == 0.0
        assert stats.total_recv_wait_time_ms == 0.0

    def test_record_send(self):
        """Test recording send operations."""
        stats = CommStats()
        stats.record_send(1000, 50.0)
        stats.record_send(2000, 100.0)

        assert stats.messages_sent == 2
        assert stats.bytes_sent == 3000
        assert stats.total_send_time_ms == 150.0

    def test_record_recv(self):
        """Test recording receive operations."""
        stats = CommStats()
        stats.record_recv(500, 25.0)
        stats.record_recv(1500, 75.0)

        assert stats.messages_received == 2
        assert stats.bytes_received == 2000
        assert stats.total_recv_wait_time_ms == 100.0

    def test_record_send_error(self):
        """Test recording send errors."""
        stats = CommStats()
        stats.record_send_error()
        stats.record_send_error()

        assert stats.send_errors == 2

    def test_record_recv_timeout(self):
        """Test recording receive timeouts."""
        stats = CommStats()
        stats.record_recv_timeout()
        stats.record_recv_timeout()
        stats.record_recv_timeout()

        assert stats.recv_timeouts == 3

    def test_avg_send_latency_ms(self):
        """Test average send latency calculation."""
        stats = CommStats()
        assert stats.avg_send_latency_ms == 0.0  # No sends yet

        stats.record_send(100, 20.0)
        stats.record_send(100, 40.0)
        stats.record_send(100, 60.0)

        assert stats.avg_send_latency_ms == 40.0

    def test_avg_recv_wait_time_ms(self):
        """Test average receive wait time calculation."""
        stats = CommStats()
        assert stats.avg_recv_wait_time_ms == 0.0  # No receives yet

        stats.record_recv(100, 10.0)
        stats.record_recv(100, 30.0)

        assert stats.avg_recv_wait_time_ms == 20.0

    def test_snapshot(self):
        """Test snapshot returns correct dict."""
        stats = CommStats()
        stats.record_send(1000, 50.0)
        stats.record_recv(500, 25.0)
        stats.record_send_error()
        stats.record_recv_timeout()

        snapshot = stats.snapshot()

        assert snapshot["messages_sent"] == 1
        assert snapshot["messages_received"] == 1
        assert snapshot["bytes_sent"] == 1000
        assert snapshot["bytes_received"] == 500
        assert snapshot["send_errors"] == 1
        assert snapshot["recv_timeouts"] == 1
        assert snapshot["avg_send_latency_ms"] == 50.0
        assert snapshot["avg_recv_wait_time_ms"] == 25.0

    def test_reset(self):
        """Test reset clears all statistics."""
        stats = CommStats()
        stats.record_send(1000, 50.0)
        stats.record_recv(500, 25.0)
        stats.record_send_error()
        stats.record_recv_timeout()

        stats.reset()

        assert stats.messages_sent == 0
        assert stats.messages_received == 0
        assert stats.bytes_sent == 0
        assert stats.bytes_received == 0
        assert stats.send_errors == 0
        assert stats.recv_timeouts == 0
        assert stats.total_send_time_ms == 0.0
        assert stats.total_recv_wait_time_ms == 0.0

    def test_thread_safety(self):
        """Test that stats operations are thread-safe."""
        stats = CommStats()
        num_threads = 10
        ops_per_thread = 100

        def record_operations():
            for _ in range(ops_per_thread):
                stats.record_send(100, 1.0)
                stats.record_recv(50, 0.5)

        threads = [
            threading.Thread(target=record_operations) for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_messages = num_threads * ops_per_thread
        assert stats.messages_sent == expected_messages
        assert stats.messages_received == expected_messages


# ---------------------------------------------------------------------------
# Exception Tests
# ---------------------------------------------------------------------------


class TestRecvTimeoutError:
    """Tests for RecvTimeoutError exception."""

    def test_attributes(self):
        """Test exception attributes are set correctly."""
        exc = RecvTimeoutError(frm=1, key="test_key", timeout=5.0)

        assert exc.frm == 1
        assert exc.key == "test_key"
        assert exc.timeout == 5.0

    def test_message(self):
        """Test exception message format."""
        exc = RecvTimeoutError(frm=2, key="data_key", timeout=10.0)

        assert "10.0s" in str(exc)
        assert "rank 2" in str(exc)
        assert "data_key" in str(exc)

    def test_is_timeout_error(self):
        """Test that RecvTimeoutError is a TimeoutError subclass."""
        exc = RecvTimeoutError(frm=0, key="k", timeout=1.0)
        assert isinstance(exc, TimeoutError)


class TestSendTimeoutError:
    """Tests for SendTimeoutError exception."""

    def test_attributes(self):
        """Test exception attributes are set correctly."""
        exc = SendTimeoutError(to=2, key="test_key", timeout=30.0)

        assert exc.to == 2
        assert exc.key == "test_key"
        assert exc.timeout == 30.0

    def test_message(self):
        """Test exception message format."""
        exc = SendTimeoutError(to=3, key="msg_key", timeout=15.0)

        assert "15.0s" in str(exc)
        assert "rank 3" in str(exc)
        assert "msg_key" in str(exc)

    def test_is_timeout_error(self):
        """Test that SendTimeoutError is a TimeoutError subclass."""
        exc = SendTimeoutError(to=0, key="k", timeout=1.0)
        assert isinstance(exc, TimeoutError)


# ---------------------------------------------------------------------------
# HttpCommunicator Tests (with mocks)
# ---------------------------------------------------------------------------


class TestHttpCommunicatorInit:
    """Tests for HttpCommunicator initialization."""

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_default_config(self, mock_client):
        """Test communicator with default config."""
        comm = HttpCommunicator(
            rank=0,
            world_size=3,
            endpoints=[
                "http://localhost:8000",
                "http://localhost:8001",
                "http://localhost:8002",
            ],
        )

        assert comm.rank == 0
        assert comm.world_size == 3
        assert comm.config.default_send_timeout == 60.0
        assert comm.config.default_recv_timeout == 600.0  # 10 minutes
        assert isinstance(comm.stats, CommStats)

        comm.shutdown()

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_custom_config(self, mock_client):
        """Test communicator with custom config."""
        config = CommConfig(default_recv_timeout=10.0, http_timeout=5.0)
        comm = HttpCommunicator(
            rank=1,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
            config=config,
        )

        assert comm.config.default_recv_timeout == 10.0
        assert comm.config.http_timeout == 5.0

        comm.shutdown()


class TestHttpCommunicatorRecvTimeout:
    """Tests for recv() timeout behavior."""

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_recv_timeout_raises_error(self, mock_client):
        """Test that recv raises RecvTimeoutError when timeout expires."""
        comm = HttpCommunicator(
            rank=0,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
        )

        # recv with a short timeout should raise RecvTimeoutError
        with pytest.raises(RecvTimeoutError) as exc_info:
            comm.recv(frm=1, key="nonexistent", timeout=0.1)

        assert exc_info.value.frm == 1
        assert exc_info.value.key == "nonexistent"
        assert exc_info.value.timeout == 0.1

        # Stats should record the timeout
        assert comm.stats.recv_timeouts == 1

        comm.shutdown()

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_recv_uses_config_default_timeout(self, mock_client):
        """Test that recv uses config.default_recv_timeout when timeout not specified."""
        config = CommConfig(default_recv_timeout=0.1)
        comm = HttpCommunicator(
            rank=0,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
            config=config,
        )

        # Should use default_recv_timeout from config
        with pytest.raises(RecvTimeoutError) as exc_info:
            comm.recv(frm=1, key="test")

        assert exc_info.value.timeout == 0.1

        comm.shutdown()

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_recv_explicit_timeout_overrides_config(self, mock_client):
        """Test that explicit timeout parameter overrides config default."""
        config = CommConfig(default_recv_timeout=10.0)
        comm = HttpCommunicator(
            rank=0,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
            config=config,
        )

        # Explicit timeout should override config
        with pytest.raises(RecvTimeoutError) as exc_info:
            comm.recv(frm=1, key="test", timeout=0.05)

        assert exc_info.value.timeout == 0.05

        comm.shutdown()

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_recv_no_timeout_waits_for_data(self, mock_client):
        """Test that recv without timeout waits until data arrives."""
        comm = HttpCommunicator(
            rank=0,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
        )

        test_data = {"key": "value"}

        def delayed_deliver():
            time.sleep(0.1)
            comm.on_receive(from_rank=1, key="delayed", data=test_data)

        t = threading.Thread(target=delayed_deliver)
        t.start()

        # Should block until data arrives
        result = comm.recv(frm=1, key="delayed")
        t.join()

        assert result == test_data
        assert comm.stats.messages_received == 1

        comm.shutdown()


class TestHttpCommunicatorOnReceive:
    """Tests for on_receive() method."""

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_on_receive_stores_data(self, mock_client):
        """Test that on_receive stores data in mailbox."""
        comm = HttpCommunicator(
            rank=0,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
        )

        test_data = [1, 2, 3]
        comm.on_receive(from_rank=1, key="test_key", data=test_data)

        # Data should be retrievable via recv
        result = comm.recv(frm=1, key="test_key", timeout=1.0)
        assert result == test_data

        comm.shutdown()

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_on_receive_mailbox_overflow(self, mock_client):
        """Test that duplicate key raises RuntimeError."""
        comm = HttpCommunicator(
            rank=0,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
        )

        comm.on_receive(from_rank=1, key="dup_key", data="first")

        with pytest.raises(RuntimeError, match="Mailbox overflow"):
            comm.on_receive(from_rank=1, key="dup_key", data="second")

        comm.shutdown()


class TestHttpCommunicatorStats:
    """Tests for statistics tracking in HttpCommunicator."""

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_recv_records_stats(self, mock_client):
        """Test that successful recv records statistics."""
        comm = HttpCommunicator(
            rank=0,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
        )

        # Pre-populate mailbox
        comm.on_receive(from_rank=1, key="k1", data=b"hello")
        comm.on_receive(from_rank=1, key="k2", data="world")

        comm.recv(frm=1, key="k1", timeout=1.0)
        comm.recv(frm=1, key="k2", timeout=1.0)

        assert comm.stats.messages_received == 2
        # bytes data should record size
        assert comm.stats.bytes_received == 5  # len(b"hello")

        comm.shutdown()


class TestHttpCommunicatorSendSync:
    """Tests for send_sync() behavior."""

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_send_sync_success_records_stats(self, mock_client):
        """Test that send_sync succeeds and records stats."""
        mock_response = mock_client.return_value.put.return_value
        mock_response.raise_for_status.return_value = None

        comm = HttpCommunicator(
            rank=0,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
        )

        comm.send_sync(to=1, key="k", data=b"hi")

        assert comm.stats.messages_sent == 1
        assert comm.stats.send_errors == 0
        assert comm.stats.bytes_sent == 4  # len(base64.b64encode(b"hi"))

        _, kwargs = mock_client.return_value.put.call_args
        assert kwargs["json"]["is_raw_bytes"] is True

        comm.shutdown()

    @patch("mplang.backends.simp_worker.http.httpx.Client")
    def test_send_sync_timeout_raises_error(self, _mock_client):
        """Test that send_sync raises SendTimeoutError on timeout."""
        comm = HttpCommunicator(
            rank=0,
            world_size=2,
            endpoints=["http://localhost:8000", "http://localhost:8001"],
        )

        mock_future = Mock(spec=concurrent.futures.Future)
        mock_future.result.side_effect = concurrent.futures.TimeoutError()
        mock_future.cancel = Mock()

        comm._send_executor.submit = Mock(return_value=mock_future)  # type: ignore[method-assign]

        with pytest.raises(SendTimeoutError) as exc_info:
            comm.send_sync(to=1, key="k", data="payload", timeout=0.01)

        assert exc_info.value.to == 1
        assert exc_info.value.key == "k"
        assert exc_info.value.timeout == 0.01
        assert comm.stats.send_errors == 1

        comm.shutdown()
