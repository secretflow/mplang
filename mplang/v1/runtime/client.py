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

import httpx

from mplang.v1.kernels.value import Value, decode_value, encode_value


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
        # Ensure endpoint has a protocol prefix
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"

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
        cluster_spec: dict,
    ) -> str:
        """Create a new session.

        Args:
            name: Session name/ID.
            rank: This party's rank.
            cluster_spec: Full cluster specification dict (ClusterSpec.to_dict()).

        Returns:
            The session name/ID

        Raises:
            RuntimeError: If session creation fails
        """
        url = f"/sessions/{name}"
        payload: dict[str, Any] = {"rank": rank, "cluster_spec": cluster_spec}

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
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(f"get session {session_name}", e) from e

    async def delete_session(self, session_name: str) -> None:
        """Delete a session and all its associated resources.

        Args:
            session_name: The session name/ID

        Raises:
            RuntimeError: If session deletion fails
        """
        url = f"/sessions/{session_name}"

        try:
            response = await self._client.delete(url)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(f"delete session {session_name}", e) from e

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

    async def delete_computation(self, session_id: str, computation_id: str) -> None:
        """Delete a computation from a session.

        Args:
            session_id: The session name/ID
            computation_id: The computation name/ID

        Raises:
            RuntimeError: If computation deletion fails
        """
        url = f"/sessions/{session_id}/computations/{computation_id}"

        try:
            response = await self._client.delete(url)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(
                f"delete computation {computation_id}", e
            ) from e

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

        # Serialize data using Value envelope
        if not isinstance(data, Value):
            raise TypeError(f"Data must be a Value instance, got {type(data)}")
        data_bytes = encode_value(data)
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

            # Deserialize data using Value envelope
            data_bytes = base64.b64decode(symbol_data["data"])
            return decode_value(data_bytes)

        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise self._raise_http_error(f"get symbol {symbol_name}", e) from e
        except httpx.RequestError as e:
            raise self._raise_http_error(f"get symbol {symbol_name}", e) from e

    async def delete_symbol(self, session_name: str, symbol_name: str) -> None:
        """Delete a symbol from a session.

        Args:
            session_name: The session name/ID
            symbol_name: The symbol name/ID

        Raises:
            RuntimeError: If symbol deletion fails
        """
        url = f"/sessions/{session_name}/symbols/{symbol_name}"

        try:
            response = await self._client.delete(url)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(f"delete symbol {symbol_name}", e) from e

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

    async def list_sessions(self) -> list[str]:
        """List all sessions on this node.

        Returns:
            List of session names

        Raises:
            RuntimeError: If session listing fails
        """
        url = "/sessions"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return list(response.json()["sessions"])
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error("list sessions", e) from e

    async def list_computations(self, session_name: str) -> list[str]:
        """List all computations in a session.

        Args:
            session_name: The session name/ID

        Returns:
            List of computation names

        Raises:
            RuntimeError: If computation listing fails
        """
        url = f"/sessions/{session_name}/computations"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return list(response.json()["computations"])
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(
                f"list computations for session {session_name}", e
            ) from e

    # ---------------- Global Symbols (process-level) ----------------
    async def create_global_symbol(
        self, symbol_name: str, data: Any, mptype: dict | None = None
    ) -> None:
        """Create or replace a process-global symbol.

        Args:
            symbol_name: Identifier
            data: Python object to store (pickle based)
            mptype: Optional metadata dict
        """
        url = f"/api/v1/symbols/{symbol_name}"
        try:
            # Serialize using Value envelope
            if not isinstance(data, Value):
                raise TypeError(f"Data must be a Value instance, got {type(data)}")
            data_bytes = encode_value(data)
            payload = {
                "data": base64.b64encode(data_bytes).decode("utf-8"),
                "mptype": mptype or {},
            }
            resp = await self._client.put(url, json=payload)
            resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(
                f"create global symbol {symbol_name}", e
            ) from e

    async def get_global_symbol(self, symbol_name: str) -> Any:
        url = f"/api/v1/symbols/{symbol_name}"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            payload = resp.json()
            data_bytes = base64.b64decode(payload["data"])
            return decode_value(data_bytes)
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(f"get global symbol {symbol_name}", e) from e

    async def delete_global_symbol(self, symbol_name: str) -> None:
        url = f"/api/v1/symbols/{symbol_name}"
        try:
            resp = await self._client.delete(url)
            resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error(
                f"delete global symbol {symbol_name}", e
            ) from e

    async def list_global_symbols(self) -> list[str]:
        url = "/api/v1/symbols"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            return list(resp.json().get("symbols", []))
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise self._raise_http_error("list global symbols", e) from e
