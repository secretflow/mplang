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
HTTP-based driver implementation for distributed execution.

This module provides an HTTP-based alternative to the gRPC Driver,
using REST APIs for distributed multi-party computation coordination.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from collections.abc import Sequence
from typing import Any

from mplang.core.cluster import ClusterSpec
from mplang.core.expr.ast import Expr
from mplang.core.interp import InterpContext, InterpVar
from mplang.core.mask import Mask
from mplang.core.mpir import Writer
from mplang.core.mpobject import MPObject
from mplang.core.mptype import MPType
from mplang.runtime.client import HttpExecutorClient


def new_uuid() -> str:
    """Generates a short UUID using URL-safe Base64 encoding."""
    u = uuid.uuid4()
    # Get the 16 bytes of the UUID
    uuid_bytes = u.bytes
    # Encode using URL-safe Base64
    encoded_bytes = base64.urlsafe_b64encode(uuid_bytes)
    # Decode to UTF-8 string, remove padding, and take first 8 characters
    encoded_string = encoded_bytes.decode("utf-8").rstrip("=")[:8]
    return encoded_string


class DriverVar(InterpVar):
    """A variable that references a value in distributed HTTP executor nodes.

    This represents a symbol stored on remote HTTP servers that can be
    retrieved via REST API calls.
    """

    def __init__(
        self,
        ctx: Driver,
        symbol_name: str,
        mptype: MPType,
    ) -> None:
        super().__init__(ctx, mptype)
        self.symbol_name = symbol_name

    @property
    def mptype(self) -> MPType:
        """The type of this variable."""
        return self._mptype

    def __repr__(self) -> str:
        return f"HttpDriverVar(symbol_name={self.symbol_name}, mptype={self.mptype})"


class Driver(InterpContext):
    """Driver for distributed execution using HTTP-based services.

    Args:
        cluster_spec: The cluster specification defining the distributed environment.
        trace_ranks: List of ranks to trace execution for debugging.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        cluster_spec: ClusterSpec,
        *,
        trace_ranks: list[int] | None = None,
        timeout: int = 120,
    ) -> None:
        """Initialize a driver with the given cluster specification.

        Args:
            cluster_spec: The cluster specification defining the distributed environment.
            trace_ranks: List of ranks to trace execution for debugging.
            timeout: HTTP request timeout in seconds.
        """
        super().__init__(cluster_spec)
        self._trace_ranks = trace_ranks or []
        self.timeout = timeout

        self._session_id: str | None = None
        self._counter = 0

        self.node_addrs = {
            node_id: node.endpoint for node_id, node in cluster_spec.nodes.items()
        }

        # Get SPU configuration from cluster_spec
        spu_devices = cluster_spec.get_devices_by_kind("SPU")
        if not spu_devices:
            raise ValueError("No SPU device found in the cluster specification")
        if len(spu_devices) > 1:
            raise ValueError("Multiple SPU devices found in the cluster specification")
        spu_device = spu_devices[0]

        # Store SPU configuration as strings for better readability
        self.spu_protocol_str = spu_device.config["protocol"]
        self.spu_field_str = spu_device.config["field"]

        # Compute spu_mask from spu_device members
        spu_mask = Mask.from_ranks([member.rank for member in spu_device.members])
        self.spu_mask_int = int(spu_mask)

    def _create_clients(self) -> dict[str, HttpExecutorClient]:
        """Create HTTP clients for all endpoints."""
        clients = {}
        for node_id, endpoint in self.node_addrs.items():
            clients[node_id] = HttpExecutorClient(endpoint, self.timeout)
        return clients

    async def _close_clients(self, clients: dict[str, HttpExecutorClient]) -> None:
        """Close all provided HTTP clients."""
        await asyncio.gather(*[client.close() for client in clients.values()])

    def new_name(self, prefix: str = "var") -> str:
        """Generate a unique execution name."""
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    async def _get_or_create_session(self) -> str:
        """Get existing session or create a new one across all HTTP servers."""
        if self._session_id is None:
            new_session_id = new_uuid()
            endpoints_list = list(self.node_addrs.values())

            # Create temporary clients for session creation
            clients = self._create_clients()
            try:
                # Create session on all HTTP servers concurrently
                tasks = []
                for node_id, client in clients.items():
                    # Convert node_id to rank for the session creation
                    rank = list(self.node_addrs.keys()).index(node_id)
                    task = client.create_session(
                        name=new_session_id,
                        rank=rank,
                        endpoints=endpoints_list,
                        spu_mask=self.spu_mask_int,
                        spu_protocol=self.spu_protocol_str,
                        spu_field=self.spu_field_str,
                    )
                    tasks.append(task)

                try:
                    results = await asyncio.gather(*tasks)
                    for session_id in results:
                        assert session_id == new_session_id
                    self._session_id = new_session_id
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Failed to create session on one or more parties: {e}"
                    ) from e
            finally:
                await self._close_clients(clients)

        assert self._session_id is not None
        return self._session_id

    async def _evaluate(
        self, expr: Expr, bindings: dict[str, MPObject]
    ) -> Sequence[MPObject]:
        """Async implementation to evaluate an expression."""
        session_id = await self._get_or_create_session()

        # Prepare input names from bindings
        var_names = []
        party_symbol_names = []
        for name, var in bindings.items():
            if var.ctx is not self:
                raise ValueError(f"Variable {name} not in this context, got {var.ctx}.")
            assert isinstance(var, DriverVar), (
                f"Expected HttpDriverVar, got {type(var)}"
            )
            var_names.append(name)
            party_symbol_names.append(var.symbol_name)

        var_name_mapping = dict(zip(var_names, party_symbol_names, strict=True))

        writer = Writer(var_name_mapping)
        program_proto = writer.dumps(expr)

        output_symbols = [self.new_name() for _ in range(expr.num_outputs)]

        # Create temporary clients for computation execution
        clients = self._create_clients()
        try:
            # Concurrently create and execute computation on all parties
            tasks = []
            computation_id = new_uuid()
            for _rank, client in clients.items():
                task = client.create_and_execute_computation(
                    session_id,
                    computation_id,
                    program_proto.SerializeToString(),
                    party_symbol_names,
                    output_symbols,
                )
                tasks.append(task)

            try:
                await asyncio.gather(*tasks)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to create and execute computation on one or more parties: {e}"
                ) from e
        finally:
            await self._close_clients(clients)

        # Create HttpDriverVar objects for each output
        driver_vars = []
        for symbol_name, mptype in zip(output_symbols, expr.mptypes, strict=True):
            driver_var = DriverVar(self, symbol_name, mptype)
            driver_vars.append(driver_var)

        return driver_vars

    def evaluate(self, expr: Expr, bindings: dict[str, MPObject]) -> Sequence[MPObject]:
        """Evaluate an expression using distributed HTTP execution."""
        return asyncio.run(self._evaluate(expr, bindings))

    async def _fetch(self, obj: MPObject) -> list[Any]:
        """Async implementation to fetch results."""
        if not isinstance(obj, DriverVar):
            raise ValueError(f"Expected HttpDriverVar, got {type(obj)}")

        session_id = await self._get_or_create_session()
        symbol_full_name = obj.symbol_name

        # Create temporary clients for fetching
        clients = self._create_clients()
        try:
            # Concurrently fetch symbol from all parties
            tasks = []
            for _rank, client in clients.items():
                task = client.get_symbol(session_id, symbol_full_name)
                tasks.append(task)

            try:
                # The results will be in the same order as the clients (ranks)
                results = await asyncio.gather(*tasks)
                return list(results)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to fetch symbol from one or more parties: {e}"
                ) from e
        finally:
            await self._close_clients(clients)

    def fetch(self, obj: MPObject) -> list[Any]:
        """Fetch results from the distributed HTTP execution."""
        return asyncio.run(self._fetch(obj))

    async def _ping(self, node_id: str) -> bool:
        """Async implementation to ping a node.

        Args:
            node_id: The ID of the node to ping

        Returns:
            True if the node is healthy, False otherwise
        """
        # Create a temporary client for the node
        if node_id not in self.node_addrs:
            raise ValueError(f"Node {node_id} not found in party addresses")

        endpoint = self.node_addrs[node_id]
        client = HttpExecutorClient(endpoint, self.timeout)

        try:
            # Perform health check
            return await client.health_check()
        except Exception:
            # Any exception means the node is not healthy
            return False
        finally:
            await client.close()

    def ping(self, node_id: str) -> bool:
        """Ping a node to check if it's healthy.

        Args:
            node_id: The ID of the node to ping

        Returns:
            True if the node is healthy, False otherwise
        """
        return asyncio.run(self._ping(node_id))
