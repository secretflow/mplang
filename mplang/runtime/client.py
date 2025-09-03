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
HTTP Executor Client Library.

This module provides a clean HTTP client interface for interacting with
HTTP-based executor services. It handles all HTTP communication details
and provides domain-specific methods for session, computation, and symbol management.
"""

from __future__ import annotations

import base64
from typing import Any

import cloudpickle as pickle
import httpx


class ExecutionStatus:
    """Status of a computation execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HttpExecutorClient:
    """HTTP client for interacting with HTTP-based executor services."""

    def __init__(self, endpoint: str, timeout: int = 60):
        """Initialize the HTTP executor client.

        Args:
            endpoint: The base URL of the HTTP executor service
            timeout: Default timeout for HTTP requests in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(base_url=self.endpoint, timeout=self.timeout)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # Internal helpers
    def _raise_http_error(self, action: str, e: Exception) -> RuntimeError:
        if isinstance(e, httpx.HTTPStatusError):
            # Extract detailed error message from response
            error_detail = "Unknown error"
            try:
                error_response = e.response.json()
                error_detail = error_response.get("detail", str(e))
            except Exception:
                error_detail = str(e)
            return RuntimeError(f"Failed to {action}: {error_detail}")
        elif isinstance(e, httpx.RequestError):
            return RuntimeError(f"Failed to {action}: {e}")
        else:
            return RuntimeError(f"Failed to {action}: {e}")

    # Session Management
    async def create_session(
        self,
        name: str,
        rank: int,
        endpoints: list[str],
        *,
        spu_mask: int = -1,
        spu_protocol: int = 2,  # SEMI2K
        spu_field: int = 2,  # FM64
    ) -> str:
        """Create a new session.

        Args:
            name: Session name/ID.
            rank: The rank of this party in the session.
            endpoints: List of endpoint URLs for all parties, indexed by rank.
            spu_mask: SPU mask for the session, -1 means all parties construct SPU.
            spu_protocol: SPU protocol for the session.
            spu_field: SPU field for the session.

        Returns:
            The session name/ID

        Raises:
            RuntimeError: If session creation fails
        """
        url = f"/sessions/{name}"
        payload: dict[str, Any] = {
            "rank": rank,
            "endpoints": endpoints,
            "spu_mask": spu_mask,
            "spu_protocol": spu_protocol,
            "spu_field": spu_field,
        }

        try:
            response = await self._client.put(url, json=payload)
            response.raise_for_status()
            return str(response.json()["name"])
        except httpx.HTTPStatusError as e:
            # Extract detailed error message from response
            error_detail = "Unknown error"
            try:
                error_response = e.response.json()
                error_detail = error_response.get("detail", str(e))
            except Exception:
                error_detail = str(e)
            raise RuntimeError(f"Failed to create session: {error_detail}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to create session: {e}") from e

    async def get_session(self, session_name: str) -> dict[str, Any]:
        """Get session information.

        Args:
            session_id: The session name/ID

        Returns:
            Session information dictionary

        Raises:
            RuntimeError: If session retrieval fails
        """
        url = f"/sessions/{session_name}"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return dict(response.json())
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to get session {session_name}: {e}") from e

    # Computation Management
    async def create_and_execute_computation(
        self,
        session_id: str,
        computation_id: str,
        program: bytes,
        input_names: list[str],
        output_names: list[str],
    ) -> str:
        """Create a new computation in a session.

        Args:
            session_id: The session name/ID
            computation_id: The computation name/ID
            program: Serialized computation program (protobuf bytes)
            input_names: List of input symbol names
            output_names: List of output symbol names

        Returns:
            The computation name/ID

        Raises:
            RuntimeError: If computation creation fails
        """
        url = f"/sessions/{session_id}/computations/{computation_id}"
        program_data = base64.b64encode(program).decode("utf-8")
        payload = {
            "mpprogram": program_data,
            "input_names": input_names,
            "output_names": output_names,
        }

        try:
            response = await self._client.put(url, json=payload)
            response.raise_for_status()
            return str(response.json()["name"])
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error("create computation", e) from e

    async def get_computation(
        self, session_id: str, computation_id: str
    ) -> dict[str, Any]:
        """Get computation information.

        Args:
            session_id: The session name/ID
            computation_id: The computation name/ID

        Returns:
            Computation information dictionary

        Raises:
            RuntimeError: If computation retrieval fails
        """
        url = f"/sessions/{session_id}/computations/{computation_id}"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return dict(response.json())
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(f"get computation {computation_id}", e) from e

    # Symbol Management
    async def create_symbol(
        self, session_name: str, symbol_name: str, data: Any, mptype: dict | None = None
    ) -> None:
        """Create a symbol with data.

        Args:
            session_name: The session name/ID
            symbol_name: The symbol name/ID
            data: The data to store
            mptype: Optional type information

        Raises:
            RuntimeError: If symbol creation fails
        """
        url = f"/sessions/{session_name}/symbols/{symbol_name}"

        # Serialize data
        data_bytes = pickle.dumps(data)
        data_b64 = base64.b64encode(data_bytes).decode("utf-8")

        payload = {"data": data_b64, "mptype": mptype or {}}

        try:
            response = await self._client.put(url, json=payload)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(f"create symbol {symbol_name}", e) from e

    async def get_symbol(self, session_name: str, symbol_name: str) -> Any:
        """Get symbol data.

        Args:
            session_name: The session name/ID
            symbol_name: The symbol name/ID

        Returns:
            The deserialized symbol data

        Raises:
            RuntimeError: If symbol retrieval fails
        """
        # For simple symbol names (no slashes), we can use them directly in the URL
        url = f"/sessions/{session_name}/symbols/{symbol_name}"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            symbol_data = response.json()

            # Deserialize data
            data_bytes = base64.b64decode(symbol_data["data"])
            return pickle.loads(data_bytes)

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(f"get symbol {symbol_name}", e) from e

    async def health_check(self) -> bool:
        """Perform a health check on the HTTP executor service.

        Returns:
            True if the service is healthy, False otherwise

        Raises:
            RuntimeError: If the health check fails
        """
        url = "/health"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            result = response.json().get("status") == "ok"
            return bool(result)  # Ensure we return a bool type
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error("perform health check", e) from e

    async def list_symbols(self, session_name: str) -> list[str]:
        """List all symbols in a session.

        Args:
            session_name: The session name/ID

        Returns:
            List of symbol names

        Raises:
            RuntimeError: If symbol listing fails
        """
        url = f"/sessions/{session_name}/symbols"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return list(response.json()["symbols"])
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error("list symbols", e) from e
