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
import requests


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

    # Session Management
    def create_session(
        self,
        name: str | None = None,
        rank: int = 0,
        endpoints: dict[int, str] | None = None,
    ) -> str:
        """Create a new session.

        Args:
            name: Optional session name. If None, server will generate one.
            rank: The rank of this party in the session.
            endpoints: Dictionary mapping rank to endpoint URL for all parties.

        Returns:
            The session name/ID

        Raises:
            RuntimeError: If session creation fails
        """
        url = f"{self.endpoint}/sessions"
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        payload["rank"] = rank
        if endpoints is not None:
            payload["endpoints"] = endpoints

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return str(response.json()["name"])
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to create session: {e}") from e

    def get_session(self, session_name: str) -> dict[str, Any]:
        """Get session information.

        Args:
            session_name: The session name/ID

        Returns:
            Session information dictionary

        Raises:
            RuntimeError: If session retrieval fails
        """
        url = f"{self.endpoint}/sessions/{session_name}"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return dict(response.json())
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get session {session_name}: {e}") from e

    # Computation Management
    def create_computation(
        self,
        session_name: str,
        program: bytes,
        input_names: list[str],
        output_names: list[str],
    ) -> str:
        """Create a new computation in a session.

        Args:
            session_name: The session name/ID
            program: Serialized computation program (protobuf bytes)
            input_names: List of input symbol names
            output_names: List of output symbol names

        Returns:
            The computation name/ID

        Raises:
            RuntimeError: If computation creation fails
        """
        url = f"{self.endpoint}/sessions/{session_name}/computations"
        program_data = base64.b64encode(program).decode("utf-8")
        payload = {
            "mpprogram": program_data,
            "input_names": input_names,
            "output_names": output_names,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return str(response.json()["name"])
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to create computation: {e}") from e

    def get_computation(
        self, session_name: str, computation_name: str
    ) -> dict[str, Any]:
        """Get computation information.

        Args:
            session_name: The session name/ID
            computation_name: The computation name/ID

        Returns:
            Computation information dictionary

        Raises:
            RuntimeError: If computation retrieval fails
        """
        url = f"{self.endpoint}/sessions/{session_name}/computations/{computation_name}"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return dict(response.json())
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to get computation {computation_name}: {e}"
            ) from e

    # Symbol Management
    def create_symbol(
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
        url = f"{self.endpoint}/sessions/{session_name}/symbols"

        # Serialize data
        data_bytes = pickle.dumps(data)
        data_b64 = base64.b64encode(data_bytes).decode("utf-8")

        payload = {"name": symbol_name, "data": data_b64, "mptype": mptype or {}}

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to create symbol {symbol_name}: {e}") from e

    def get_symbol(self, session_name: str, symbol_name: str) -> Any:
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
        url = f"{self.endpoint}/sessions/{session_name}/symbols/{symbol_name}"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            symbol_data = response.json()

            # Deserialize data
            data_bytes = base64.b64decode(symbol_data["data"])
            return pickle.loads(data_bytes)

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get symbol {symbol_name}: {e}") from e

    def list_symbols(self, session_name: str) -> list[str]:
        """List all symbols in a session.

        Args:
            session_name: The session name/ID

        Returns:
            List of symbol names

        Raises:
            RuntimeError: If symbol listing fails
        """
        url = f"{self.endpoint}/sessions/{session_name}/symbols"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return list(response.json()["symbols"])
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to list symbols: {e}") from e
