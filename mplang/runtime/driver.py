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
Executor client implementation.

This module contains the client-side implementation for communicating with
the executor service, including the driver for distributed execution.
"""

from __future__ import annotations

import base64
import uuid
from typing import Any, cast

import cloudpickle as pickle
import grpc
import spu.libspu as libspu

from mplang.core.base import Mask, MPObject, MPType, Rank
from mplang.core.interp import InterpContext, InterpVar
from mplang.core.mpir import Writer
from mplang.expr.ast import Expr
from mplang.protos import executor_pb2, executor_pb2_grpc
from mplang.runtime.executor.resource import SessionName, SymbolName


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


def make_stub(
    addr: str, max_message_length: int = 1024 * 1024 * 1024
) -> executor_pb2_grpc.ExecutorServiceStub:
    """Create a gRPC stub for the executor service."""
    channel = grpc.insecure_channel(
        addr,
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
        ],
    )
    return executor_pb2_grpc.ExecutorServiceStub(channel)


class DriverVar(InterpVar):
    """A variable that references a value in distributed executor nodes.

    DriverVar represents a value that has been computed and exists
    across multiple executor nodes, identified by a SymbolName.
    """

    def __init__(self, ctx: InterpContext, symbol_name: SymbolName, mptype: MPType):
        self._ctx = ctx
        self._symbol_name = symbol_name
        self._mptype = mptype

    @property
    def ctx(self) -> InterpContext:
        """The executor driver context this variable belongs to."""
        return self._ctx

    @property
    def symbol_name(self) -> SymbolName:
        """The symbol name of this variable across executor nodes."""
        return self._symbol_name

    @property
    def mptype(self) -> MPType:
        """The type of this variable."""
        return self._mptype

    def __repr__(self) -> str:
        return f"DriverVar(symbol_name={self.symbol_name}, mptype={self.mptype})"


class ExecutorDriver(InterpContext):
    """Driver for distributed execution using gRPC-based executor services."""

    def __init__(
        self,
        node_addrs: dict[str, str],
        *,
        spu_protocol: libspu.ProtocolKind = libspu.ProtocolKind.SEMI2K,
        spu_field: libspu.FieldType = libspu.FieldType.FM64,
        spu_mask: Mask | None = None,
        trace_ranks: list[Rank] | None = None,
        max_message_length: int = 1024 * 1024 * 1024,
        **attrs: Any,
    ) -> None:
        if trace_ranks is None:
            trace_ranks = []
        self.world_size = len(node_addrs)
        self.peer_addrs = node_addrs
        self.max_message_length = max_message_length
        self._stubs = [
            make_stub(addr, max_message_length) for addr in node_addrs.values()
        ]
        self._session_id: str | None = None
        self._counter = 0

        spu_mask = spu_mask or ((1 << self.world_size) - 1)
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
        if self._session_id is None:
            new_session_id = new_uuid()
            metadata: dict[str, str] = {}
            session = executor_pb2.Session(
                name=new_session_id, peer_addrs=self.peer_addrs, metadata=metadata
            )
            request = executor_pb2.CreateSessionRequest(parent="", session=session)
            futures = []
            for stub in self._stubs:
                futures.append(stub.CreateSession.future(request))

            # Wait for all futures to complete
            _ = [future.result() for future in futures]

            # Store the new session ID
            self._session_id = new_session_id
        assert self._session_id is not None, (
            "Session ID should not be None after get_or_create_session"
        )
        return self._session_id

    # override
    def evaluate(self, expr: Expr, bindings: dict[str, MPObject]) -> list[MPObject]:
        """Evaluate an expression using distributed execution."""
        session_id = self.get_or_create_session()
        execution_id = new_uuid()

        # Prepare input names from bindings
        # e.g. $0, $1, ...
        var_names = []
        # e.g. /session/{session_id}/execution/{execution_id}/input/{name}
        party_symbol_names = []
        for name, var in bindings.items():
            if var.ctx is not self:
                raise ValueError(f"Variable {name} not in this context, got {var.ctx}.")
            assert isinstance(var, DriverVar), f"Expected DriverVar, got {type(var)}"
            var_names.append(name)
            party_symbol_names.append(var.symbol_name.to_string())

        # Create variable name mapping from DAG variable names to remote symbol names
        var_name_mapping = dict(zip(var_names, party_symbol_names, strict=False))

        # Serialize the expression using mpir.proto
        writer = Writer(var_name_mapping)
        program = writer.dumps(expr).SerializeToString()

        # Generate output names for the execution
        output_symbols = [
            SymbolName.execution_symbol(
                session_id=session_id,
                execution_id=execution_id,
                symbol_id=self.new_name(),
            )
            for _ in range(expr.num_outputs)
        ]
        output_names = [s.to_string() for s in output_symbols]

        # Create the execution object
        execution = executor_pb2.Execution(
            name=execution_id,
            program=program,
            input_names=party_symbol_names,
            output_names=output_names,
        )

        # Set attributes for the execution
        execution.attrs["session_id"].string_value = session_id
        execution.attrs["spu_mask"].number_value = Mask(self.attr("spu_mask"))
        execution.attrs["spu_protocol"].number_value = int(self.attr("spu_protocol"))
        execution.attrs["spu_field"].number_value = int(self.attr("spu_field"))

        # Fire off execution on all nodes
        futures = []
        for rank in range(self.world_size):
            request = executor_pb2.CreateExecutionRequest(
                parent=SessionName(session_id).to_string(),
                execution=execution,
            )
            stub = self._stubs[rank]
            futures.append(stub.CreateExecution.future(request))

        # Wait for execution to complete
        _ = [future.result() for future in futures]

        # Create DriverVar objects for each output
        driver_vars = []
        for symbol_name, mptype in zip(output_symbols, expr.mptypes, strict=False):
            driver_var = DriverVar(
                self,
                symbol_name,
                mptype,
            )
            driver_vars.append(driver_var)

        return cast(list[MPObject], driver_vars)

    # override
    def fetch(self, obj: MPObject) -> list[Any]:
        """Fetch results from the distributed execution."""
        if not isinstance(obj, DriverVar):
            raise ValueError(f"Expected DriverVar, got {type(obj)}")

        # Fetch symbol by resource name from all executor nodes
        results = []
        for rank in range(self.world_size):
            request = executor_pb2.GetSymbolRequest(name=obj.symbol_name.to_string())
            response = self._stubs[rank].GetSymbol(request)
            results.append(pickle.loads(response.data))

        return results
