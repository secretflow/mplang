# Copyright 2025 Ant Group Co., Ltd.
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

"""Base infrastructure for SIMP Worker communicators.

This module contains shared components used by both HttpCommunicator and
ThreadCommunicator:
- CommunicatorProtocol: Structural interface (Protocol) for communicators
- Request handles: RequestStatus, Request, SendRequest
- Batch operations: wait_all, wait_any, testall, testany
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Communicator Protocol
# ---------------------------------------------------------------------------


class CommunicatorProtocol(Protocol):
    """Structural interface for SIMP communicators.

    Both ThreadCommunicator and HttpCommunicator implement this interface.
    Uses Protocol (structural subtyping) rather than ABC to allow duck typing
    and avoid coupling implementations to a base class.

    Attributes:
        rank: This communicator's rank in the cluster.
        world_size: Total number of workers in the cluster.
    """

    rank: int
    world_size: int

    def send(
        self, to: int, key: str, data: Any, *, is_raw_bytes: bool = False
    ) -> SendRequest:
        """Send data to another rank (non-blocking).

        Args:
            to: Target rank.
            key: Message key for matching send/recv pairs.
            data: Payload to send.
            is_raw_bytes: If True, treat data as raw bytes (skip serde).

        Returns:
            SendRequest handle for tracking completion.
        """
        ...

    def send_sync(
        self,
        to: int,
        key: str,
        data: Any,
        *,
        is_raw_bytes: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Send data to another rank (blocking).

        Args:
            to: Target rank.
            key: Message key for matching send/recv pairs.
            data: Payload to send.
            is_raw_bytes: If True, treat data as raw bytes (skip serde).
            timeout: Maximum time to wait (seconds). None means wait forever.

        Raises:
            TimeoutError: If timeout expires before send completes.
        """
        ...

    def recv(self, frm: int, key: str, *, timeout: float | None = None) -> Any:
        """Receive data from another rank (blocking).

        Args:
            frm: Source rank.
            key: Message key for matching send/recv pairs.
            timeout: Maximum time to wait (seconds). None means wait forever.

        Returns:
            The received data.

        Raises:
            TimeoutError: If timeout expires before message arrives.
        """
        ...


# ---------------------------------------------------------------------------
# Request Handles (MPI-style non-blocking operations)
# ---------------------------------------------------------------------------


class RequestStatus:
    """Status of a non-blocking request."""

    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class Request:
    """Base class for non-blocking communication request handles.

    Similar to MPI_Request, allows tracking and waiting on async operations.

    Example:
        >>> req = comm.send(to=1, key="data", data=payload)
        >>> # ... do other work ...
        >>> req.wait()  # Block until complete

        >>> reqs = [comm.send(...), comm.send(...)]
        >>> wait_all(reqs)  # Wait for all
    """

    def __init__(self, future: concurrent.futures.Future[Any], operation: str):
        """Initialize request.

        Args:
            future: The underlying Future from executor.
            operation: Description of the operation (e.g., "send to rank 1").
        """
        self._future = future
        self._operation = operation
        self._result: Any = None
        self._error: Exception | None = None
        self._status = RequestStatus.PENDING

    @property
    def status(self) -> str:
        """Current status of the request."""
        if self._status != RequestStatus.PENDING:
            return self._status

        if self._future.done():
            self._finalize()
        return self._status

    @property
    def done(self) -> bool:
        """Check if the request is complete (non-blocking)."""
        return self.status != RequestStatus.PENDING

    def _finalize(self) -> None:
        """Finalize the request after future completes."""
        if self._status != RequestStatus.PENDING:
            return

        if self._future.cancelled():
            self._status = RequestStatus.CANCELLED
        else:
            try:
                self._result = self._future.result(timeout=0)
                self._status = RequestStatus.COMPLETED
            except Exception as e:
                self._error = e
                self._status = RequestStatus.ERROR

    def wait(self, timeout: float | None = None) -> Any:
        """Wait for the request to complete.

        Args:
            timeout: Maximum time to wait (seconds). None means wait forever.

        Returns:
            The result of the operation (for recv requests).

        Raises:
            TimeoutError: If timeout expires before completion.
            Exception: If the operation failed.
        """
        try:
            self._result = self._future.result(timeout=timeout)
            self._status = RequestStatus.COMPLETED
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError(
                f"Request timed out after {timeout}s: {self._operation}"
            ) from e
        except concurrent.futures.CancelledError:
            self._status = RequestStatus.CANCELLED
            raise
        except Exception as e:
            self._error = e
            self._status = RequestStatus.ERROR
            raise

        return self._result

    def test(self) -> tuple[bool, Any | None]:
        """Test if the request is complete (non-blocking).

        Returns:
            Tuple of (completed: bool, result: Any | None).
            If completed is True, result contains the operation result.
            If completed is False, result is None.

        Raises:
            Exception: If the operation completed with an error.
        """
        if not self._future.done():
            return False, None

        self._finalize()

        if self._status == RequestStatus.ERROR:
            raise self._error  # type: ignore[misc]

        return True, self._result

    def cancel(self) -> bool:
        """Attempt to cancel the request.

        Returns:
            True if successfully cancelled, False otherwise.
        """
        if self._future.cancel():
            self._status = RequestStatus.CANCELLED
            return True
        return False

    def __repr__(self) -> str:
        return f"Request({self._operation}, status={self.status})"


class SendRequest(Request):
    """Request handle for non-blocking send operations."""

    def __init__(
        self,
        future: concurrent.futures.Future[None],
        to: int,
        key: str,
    ):
        super().__init__(future, f"send to rank {to} key={key}")
        self.to = to
        self.key = key


# ---------------------------------------------------------------------------
# Batch Operations
# ---------------------------------------------------------------------------


def wait_all(requests: list[Request], timeout: float | None = None) -> list[Any]:
    """Wait for all requests to complete.

    Args:
        requests: List of Request objects.
        timeout: Maximum total time to wait (seconds). None means wait forever.

    Returns:
        List of results in the same order as requests.

    Raises:
        TimeoutError: If timeout expires before all complete.
        Exception: If any request failed.
    """
    if not requests:
        return []

    futures = [req._future for req in requests]
    _done, not_done = concurrent.futures.wait(
        futures,
        timeout=timeout,
        return_when=concurrent.futures.ALL_COMPLETED,
    )

    if not_done:
        raise TimeoutError(
            f"wait_all timed out: {len(not_done)}/{len(requests)} requests pending"
        )

    # Finalize all and collect results
    results = []
    for req in requests:
        req._finalize()
        if req._status == RequestStatus.ERROR:
            raise req._error  # type: ignore[misc]
        results.append(req._result)

    return results


def wait_any(requests: list[Request], timeout: float | None = None) -> tuple[int, Any]:
    """Wait for any one request to complete.

    Args:
        requests: List of Request objects.
        timeout: Maximum time to wait (seconds). None means wait forever.

    Returns:
        Tuple of (index, result) for the first completed request.

    Raises:
        TimeoutError: If timeout expires before any complete.
        ValueError: If requests list is empty.
        Exception: If the completed request failed.
    """
    if not requests:
        raise ValueError("requests list cannot be empty")

    futures = [req._future for req in requests]
    done, _ = concurrent.futures.wait(
        futures,
        timeout=timeout,
        return_when=concurrent.futures.FIRST_COMPLETED,
    )

    if not done:
        raise TimeoutError("wait_any timed out: no requests completed")

    # Find the first completed request
    for i, req in enumerate(requests):
        if req._future in done:
            req._finalize()
            if req._status == RequestStatus.ERROR:
                raise req._error  # type: ignore[misc]
            return i, req._result

    # Should not reach here
    raise RuntimeError("Unexpected state in wait_any")


def testall(requests: list[Request]) -> tuple[bool, list[Any] | None]:
    """Test if all requests are complete (non-blocking).

    Args:
        requests: List of Request objects.

    Returns:
        Tuple of (all_done: bool, results: list | None).
        If all_done is True, results contains all operation results.
        If all_done is False, results is None.

    Raises:
        Exception: If any completed request had an error.
    """
    if not requests:
        return True, []

    all_done = all(req._future.done() for req in requests)
    if not all_done:
        return False, None

    results = []
    for req in requests:
        req._finalize()
        if req._status == RequestStatus.ERROR:
            raise req._error  # type: ignore[misc]
        results.append(req._result)

    return True, results


def testany(requests: list[Request]) -> tuple[int | None, Any | None]:
    """Test if any request is complete (non-blocking).

    Args:
        requests: List of Request objects.

    Returns:
        Tuple of (index: int | None, result: Any | None).
        If a request is complete, returns its index and result.
        If none are complete, returns (None, None).

    Raises:
        Exception: If the completed request had an error.
    """
    if not requests:
        return None, None

    for i, req in enumerate(requests):
        if req._future.done():
            req._finalize()
            if req._status == RequestStatus.ERROR:
                raise req._error  # type: ignore[misc]
            return i, req._result

    return None, None
