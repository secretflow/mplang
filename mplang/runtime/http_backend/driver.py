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

This module provides an HTTP-based alternative to the gRPC ExecutorDriver,
using REST APIs for distributed multi-party computation coordination.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Sequence
from typing import Any

import spu.libspu as libspu

from mplang.core.interp import InterpContext, InterpVar
from mplang.core.mask import Mask
from mplang.core.mpir import Writer
from mplang.core.mpobject import MPObject
from mplang.core.mptype import MPType, Rank
from mplang.expr.ast import Expr
from mplang.runtime.http_backend.client import HttpExecutorClient


def new_uuid() -> str:
    """Generates a short UUID using URL-safe Base64 encoding."""
    u = uuid.uuid4()
    # Get the 16 bytes of the UUID
    uuid_bytes = u.bytes
    # Encode using URL-safe Base64
    encoded_bytes = base64.urlsafe_b64encode(uuid_bytes)
    # Decode to UTF-8 string and remove the '==' padding
    encoded_string = encoded_bytes.decode("utf-8").rstrip("=")
    return encoded_string


class HttpDriverVar(InterpVar):
    """A variable that references a value in distributed HTTP executor nodes.

    This represents a symbol stored on remote HTTP servers that can be
    retrieved via REST API calls.
    """

    def __init__(
        self,
        ctx: HttpDriver,
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


class HttpDriver(InterpContext):
    """Driver for distributed execution using HTTP-based services."""

    def __init__(
        self,
        node_addrs: dict[str, str] | dict[int, str],
        *,
        spu_protocol: libspu.ProtocolKind = libspu.ProtocolKind.SEMI2K,
        spu_field: libspu.FieldType = libspu.FieldType.FM64,
        spu_mask: Mask | None = None,
        trace_ranks: list[Rank] | None = None,
        timeout: int = 60,
        **attrs: Any,
    ) -> None:
        if trace_ranks is None:
            trace_ranks = []

        self.world_size = len(node_addrs)
        self.party_addrs = node_addrs
        self.timeout = timeout

        # Create HTTP clients for each endpoint
        self.clients = {}
        for key, endpoint in node_addrs.items():
            rank = int(key) if isinstance(key, str) else key
            self.clients[rank] = HttpExecutorClient(endpoint, timeout)

        self._session_id: str | None = None
        self._counter = 0

        spu_mask = spu_mask or Mask.all(self.world_size)
        executor_attrs = {
            "spu_protocol": int(spu_protocol),
            "spu_field": int(spu_field),
            "spu_mask": spu_mask,
            "trace_ranks": trace_ranks,
            **attrs,
        }

        super().__init__(self.world_size, executor_attrs)

    def new_name(self, prefix: str = "var") -> str:
        """Generate a unique execution name."""
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def get_or_create_session(self) -> str:
        """Get existing session or create a new one across all HTTP servers."""
        if self._session_id is None:
            new_session_id = new_uuid()

            # NOTE: Assumes node_addrs in __init__ was provided with keys in rank order.
            # Python dictionaries preserve insertion order since 3.7+.
            endpoints_list = list(self.party_addrs.values())

            # Create session on all HTTP servers using clients
            for rank, client in self.clients.items():
                try:
                    session_id = client.create_session(
                        name=new_session_id,
                        rank=rank,
                        endpoints=endpoints_list,
                    )
                    assert session_id == new_session_id
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Failed to create session on rank {rank}: {e}"
                    ) from e

            self._session_id = new_session_id

        assert self._session_id is not None
        return self._session_id

    def evaluate(self, expr: Expr, bindings: dict[str, MPObject]) -> Sequence[MPObject]:
        """Evaluate an expression using distributed HTTP execution."""
        session_id = self.get_or_create_session()

        # Prepare input names from bindings
        var_names = []
        party_symbol_names = []
        for name, var in bindings.items():
            if var.ctx is not self:
                raise ValueError(f"Variable {name} not in this context, got {var.ctx}.")
            assert isinstance(
                var, HttpDriverVar
            ), f"Expected HttpDriverVar, got {type(var)}"
            var_names.append(name)
            party_symbol_names.append(var.symbol_name)

        # Create variable name mapping from DAG variable names to remote symbol names
        var_name_mapping = dict(zip(var_names, party_symbol_names, strict=False))

        # Serialize the expression using mpir.proto
        writer = Writer(var_name_mapping)
        program_proto = writer.dumps(expr)

        # Generate output names for the execution
        # Use simple symbol names instead of full paths for easier URL handling
        output_symbols = [self.new_name() for _ in range(expr.num_outputs)]

        # Create computation on all HTTP servers using clients
        for rank, client in self.clients.items():
            try:
                input_names = (
                    party_symbol_names  # Use the already extracted symbol names
                )
                output_names = output_symbols  # Use simple names directly
                client.create_computation(
                    session_id,
                    program_proto.SerializeToString(),
                    input_names,
                    output_names,
                )
                # For now, we don't track individual computation IDs per rank
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to create computation on rank {rank}: {e}"
                ) from e

        # Create HttpDriverVar objects for each output
        driver_vars = []
        for symbol_name, mptype in zip(output_symbols, expr.mptypes, strict=False):
            driver_var = HttpDriverVar(
                self,
                symbol_name,
                mptype,
            )
            driver_vars.append(driver_var)

        return driver_vars

    def fetch(self, obj: MPObject) -> list[Any]:
        """Fetch results from the distributed HTTP execution."""
        if not isinstance(obj, HttpDriverVar):
            raise ValueError(f"Expected HttpDriverVar, got {type(obj)}")

        # Fetch symbol by resource name from all HTTP servers using clients
        results = []
        for rank, client in self.clients.items():
            try:
                # Use session-level symbol access
                session_id = self._session_id or "default"
                # Symbol name is already a simple string
                symbol_full_name = obj.symbol_name
                # For now, just use the full name as the symbol name - the server should handle this
                result = client.get_symbol(session_id, symbol_full_name)
                results.append(result)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to fetch symbol from rank {rank}: {e}"
                ) from e

        return results
