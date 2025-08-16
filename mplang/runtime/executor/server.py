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
Executor service server implementation.

This module contains the server-side implementation of the executor service,
including the gRPC service implementation and related server-side classes.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import time
from collections.abc import Callable
from typing import Any

import cloudpickle as pickle
import grpc
import spu.libspu as libspu
from google.protobuf import empty_pb2
from google.protobuf.timestamp_pb2 import Timestamp

import mplang.utils.mask_utils as mask_utils
from mplang.backend.builtin import BuiltinHandler
from mplang.backend.spu import SpuHandler
from mplang.backend.stablehlo import StablehloHandler
from mplang.core.base import Mask
from mplang.core.mpir import Reader
from mplang.expr.evaluator import Evaluator
from mplang.protos import executor_pb2, executor_pb2_grpc, mpir_pb2
from mplang.runtime.executor.resource import (
    ExecutionName,
    MessageName,
    SessionName,
    SymbolName,
)
from mplang.runtime.grpc_comm import LinkCommunicator
from mplang.runtime.mem_comm import CommunicatorBase as CommunicatorImpl


def datetime_to_timestamp(dt: datetime.datetime | None) -> Timestamp:
    """Convert a Python datetime to a Protobuf Timestamp"""
    timestamp = Timestamp()
    if dt is not None:
        timestamp.FromDatetime(dt)
    return timestamp


class Symbol:
    """Runtime representation of a symbol"""

    def __init__(self, type: str, data: Any):
        self.type = type
        self.data = data

    @classmethod
    def from_proto(cls, proto: executor_pb2.Symbol) -> Symbol:
        """Construct from protobuf executor_pb2.Symbol message"""
        return cls(proto.type, pickle.loads(proto.data))

    def to_proto(self, resource_name: str) -> executor_pb2.Symbol:
        """Convert to protobuf executor_pb2.Symbol message"""
        response = executor_pb2.Symbol()
        response.name = resource_name
        response.data = pickle.dumps(self.data)
        response.type = self.type
        return response


class GrpcCommunicator(CommunicatorImpl):
    """gRPC-based communicator for distributed execution."""

    def __init__(
        self,
        session_id: str,
        execution_id: str,
        rank: int,
        party_addrs: list[str],
        make_stub_func: Callable,
    ):
        super().__init__(rank, len(party_addrs))

        self.session_id = session_id
        self.execution_id = execution_id
        self.party_addrs = party_addrs
        self.make_stub_func = make_stub_func

        self._stubs = [
            make_stub_func(addr) if idx != rank else None
            for idx, addr in enumerate(party_addrs)
        ]

    # override
    def send(self, to: int, key: str, data: Any) -> None:
        if to == self.rank:
            self.onSent(self.rank, key, data)
            return

        assert 0 <= to < self.world_size
        stub = self._stubs[to]
        assert stub is not None

        ukey = MessageName(
            self.session_id,
            self.execution_id,
            key,
            self.rank,
        ).to_string()

        req = executor_pb2.CommXchgRequest(
            name=ukey,
            data=pickle.dumps(data),
        )

        # TODO: async send
        stub.CommXchg(req)


class LinkCommFactory:
    """Factory for creating and caching link communicators."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, tuple[str, ...]], LinkCommunicator] = {}

    def create_link(self, rank: int, addrs: list[str]) -> LinkCommunicator:
        key = (rank, tuple(addrs))
        val = self._cache.get(key, None)
        if val is not None:
            return val

        logging.info(f"LinkCommunicator created: {rank} {addrs}")
        new_link = LinkCommunicator(rank, addrs)
        self._cache[key] = new_link
        return new_link


# Global link factory instance
g_link_factory = LinkCommFactory()


class Execution:
    """Runtime representation of an execution"""

    def __init__(
        self,
        db: ExecutorState,
        session_id: str,
        execution_id: str,
        program: mpir_pb2.GraphProto,
        input_names: list[str],
        output_names: list[str],
        *,
        spu_mask: Mask,
        spu_protocol: libspu.ProtocolKind,
        spu_field: libspu.FieldType,
        make_stub_func: Callable,
    ):
        # basic attributes.
        self.db = db
        self.session_id = session_id
        self.execution_id = execution_id
        self.program = program
        self.input_names = input_names
        self.output_names = output_names

        self.spu_mask = spu_mask
        self.spu_protocol = spu_protocol
        self.spu_field = spu_field
        self.spu_comm: LinkCommunicator | None = None

        # runtime attributes.
        self.symbols: dict[str, Symbol] = {}
        self.state = executor_pb2.ExecutionState.UNSPECIFIED
        self.error: str | None = None
        self.create_time = datetime.datetime.now()
        self.start_time: datetime.datetime | None = None
        self.end_time: datetime.datetime | None = None

        # TODO: use per-session communicator
        session = self.db.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        self.comm = GrpcCommunicator(
            session_id,
            execution_id,
            session.rank,
            session.addrs,
            make_stub_func,
        )

        if (1 << session.rank) & self.spu_mask != 0:
            spu_addrs: list[str] = []
            for rank, addr in enumerate(session.addrs):
                if mask_utils.is_rank_in(rank, spu_mask):
                    ip, port = addr.split(":")
                    new_addr = f"{ip}:{int(port) + 100}"
                    spu_addrs.append(new_addr)
            spu_rank = mask_utils.global_to_relative_rank(session.rank, self.spu_mask)
            self.spu_comm = g_link_factory.create_link(spu_rank, spu_addrs)
        else:
            self.spu_comm = None

    def run(self) -> None:
        self.start_time = datetime.datetime.now()
        self.state = executor_pb2.ExecutionState.RUNNING

        # Program is now a GraphProto, deserialize it back to Expr
        reader = Reader()
        expr = reader.loads(self.program)
        if expr is None:
            raise ValueError("Failed to deserialize program from GraphProto")

        # Prepare input bindings
        bindings = {}
        for name in self.input_names:
            symbol = self.db.lookup_symbol(name)
            if not symbol:
                raise ValueError(f"Symbol {name} not found")
            bindings[name] = symbol.data

        # Get session info
        session = self.db.get_session(self.session_id)
        if session is None:
            raise ValueError(f"Session {self.session_id} not found")

        # Setup SPU configuration
        spu_config = libspu.RuntimeConfig(
            protocol=self.spu_protocol,
            field=self.spu_field,
            fxp_fraction_bits=18,
        )

        # Setup SPU handler
        spu_handler = SpuHandler(mask_utils.bit_count(self.spu_mask), spu_config)
        if self.spu_comm is not None:
            spu_handler.set_link_context(self.spu_comm)

        # Create evaluator with handlers
        evaluator = Evaluator(
            session.rank,
            {},  # empty environment, bindings will be provided during evaluation
            self.comm,
            [
                BuiltinHandler(),
                StablehloHandler(),
                spu_handler,
            ],
        )

        # Evaluate the expression
        forked_evaluator = evaluator.fork(bindings)
        results = expr.accept(forked_evaluator)

        # When the execution succeeds, update state and store results
        self.state = executor_pb2.ExecutionState.SUCCEEDED

        # Store results in symbols
        assert len(results) == len(self.output_names)
        for name, val in zip(self.output_names, results, strict=False):
            self.symbols[name] = Symbol(type="untyped", data=val)

        self.end_time = datetime.datetime.now()

    def to_proto(self) -> executor_pb2.Execution:
        """Convert to protobuf executor_pb2.Execution message"""
        proto = executor_pb2.Execution()
        proto.name = ExecutionName(self.session_id, self.execution_id).to_string()
        # Serialize the GraphProto to bytes for the protobuf message
        proto.program = self.program.SerializeToString()
        proto.input_names.extend(self.input_names)
        proto.output_names.extend(self.output_names)
        proto.state = self.state
        if self.error:
            proto.error = self.error
        proto.create_time.CopyFrom(datetime_to_timestamp(self.create_time))
        proto.start_time.CopyFrom(datetime_to_timestamp(self.start_time))
        proto.end_time.CopyFrom(datetime_to_timestamp(self.end_time))
        return proto


class Session:
    """Runtime representation of a session"""

    def __init__(
        self,
        party_id: str,
        session_id: str,
        party_addrs: dict[str, str],  # (party_id, address)
        metadata: dict[str, str],
    ):
        # basic attributes.
        self.session_id = session_id
        self.party_addrs = party_addrs
        self.metadata = metadata

        # derived attributes.
        sorted_parties = sorted(party_addrs.keys())
        self.rank = sorted_parties.index(party_id)
        self.addrs = [party_addrs[party_id] for party_id in sorted_parties]

        # runtime attributes.
        self.symbols: dict[str, Symbol] = {}

        self.create_time = datetime.datetime.now()
        self.update_time = self.create_time

    @property
    def name(self) -> str:
        return SessionName(self.session_id).to_string()

    def to_proto(self) -> executor_pb2.Session:
        """Convert to protobuf executor_pb2.Session message"""
        proto = executor_pb2.Session()
        proto.name = self.name
        for k, v in self.party_addrs.items():
            proto.party_addrs[k] = v
        for k, v in self.metadata.items():
            proto.metadata[k] = v

        proto.create_time.CopyFrom(datetime_to_timestamp(self.create_time))
        proto.update_time.CopyFrom(datetime_to_timestamp(self.update_time))

        return proto


class ExecutorState:
    """State management for executor service."""

    def __init__(self, party_id: str) -> None:
        self.party_id = party_id

        # Resource name to object mapping.
        self.symbols: dict[str, Symbol] = {}
        self.sessions: dict[str, Session] = {}
        self.executions: dict[str, Execution] = {}

    def get_execution(self, session_id: str, execution_id: str) -> Execution | None:
        name = ExecutionName(session_id, execution_id).to_string()
        return self.executions.get(name, None)

    def get_session(self, session_id: str) -> Session | None:
        name = SessionName(session_id).to_string()
        return self.sessions.get(name, None)

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        name = SymbolName.global_symbol(symbol_id).to_string()
        return self.symbols.get(name, None)

    def lookup_symbol(self, name: str) -> Symbol | None:
        symbol_resource = SymbolName.parse(name)
        if symbol_resource is None:
            return None

        if symbol_resource.is_global():
            return self.symbols.get(name, None)
        elif symbol_resource.is_session_scoped():
            assert symbol_resource.session_id is not None
            session = self.get_session(symbol_resource.session_id)
            if not session:
                return None
            return session.symbols.get(name, None)
        elif symbol_resource.is_execution_scoped():
            assert symbol_resource.session_id is not None
            assert symbol_resource.execution_id is not None
            execution = self.get_execution(
                symbol_resource.session_id, symbol_resource.execution_id
            )
            if not execution:
                return None
            return execution.symbols.get(name, None)
        return None


class ExecutorService(ExecutorState, executor_pb2_grpc.ExecutorServiceServicer):
    """gRPC service implementation for the executor."""

    def __init__(
        self, party_id: str, make_stub_func: Callable, debug_execution: bool = False
    ):
        super().__init__(party_id)
        self.make_stub_func = make_stub_func
        self._debug_execution = debug_execution

    # --- Message Methods ---
    def CommXchg(
        self, request: executor_pb2.CommXchgRequest, context: grpc.ServicerContext
    ) -> empty_pb2.Empty:
        logging.info(f"CommXchg: {request.name}")

        message_resource = MessageName.parse(request.name)
        if not message_resource:
            logging.warning(f"Invalid message name: {request.name}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid message name format")
            return empty_pb2.Empty()

        session_id = message_resource.session_id
        execution_id = message_resource.execution_id
        msg_id = message_resource.msg_id
        frm_rank = message_resource.frm_rank

        execution = self.get_execution(session_id, execution_id)
        if not execution:
            sleep_duration = 2  # second
            logging.info(
                f"Execution {execution_id} not found, sleeping for {sleep_duration} seconds"
            )
            time.sleep(sleep_duration)

            execution = self.get_execution(session_id, execution_id)
            if not execution:
                logging.warning(f"Execution {execution_id} not found after sleep")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Execution {execution_id} not found")
                return empty_pb2.Empty()

        data = pickle.loads(request.data)
        execution.comm.onSent(frm_rank, msg_id, data)
        return empty_pb2.Empty()

    # --- Symbol Methods ---

    def CreateSymbol(
        self, request: executor_pb2.CreateSymbolRequest, context: grpc.ServicerContext
    ) -> executor_pb2.Symbol:
        if request.parent != "":
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Parent should be empty for symbols")
            return executor_pb2.Symbol()

        name = SymbolName.global_symbol(request.symbol.name).to_string()
        if name in self.symbols:
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            context.set_details(f"Symbol {name} already exists")
            return executor_pb2.Symbol()

        self.symbols[name] = Symbol.from_proto(request.symbol)
        return self.symbols[name].to_proto(name)

    def GetSymbol(
        self, request: executor_pb2.GetSymbolRequest, context: grpc.ServicerContext
    ) -> executor_pb2.Symbol:
        logging.info(f"GetSymbol: {request.name}")

        name = request.name
        symbol_resource = SymbolName.parse(name)
        if symbol_resource is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid symbol name format")
            return executor_pb2.Symbol()

        if symbol_resource.is_global():
            symbol = self.symbols.get(name, None)
            if symbol:
                return symbol.to_proto(name)
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Global symbol {name} not found")
                return executor_pb2.Symbol()
        elif symbol_resource.is_session_scoped():
            assert symbol_resource.session_id is not None
            session = self.get_session(symbol_resource.session_id)
            if session:
                symbol = session.symbols.get(name, None)
                if symbol:
                    return symbol.to_proto(name)
                else:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details(f"Session symbol {name} not found")
                    return executor_pb2.Symbol()
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Session {symbol_resource.session_id} not found")
                return executor_pb2.Symbol()
        elif symbol_resource.is_execution_scoped():
            assert symbol_resource.session_id is not None
            assert symbol_resource.execution_id is not None
            execution = self.get_execution(
                symbol_resource.session_id, symbol_resource.execution_id
            )
            if execution:
                symbol = execution.symbols.get(name, None)
                if symbol:
                    return symbol.to_proto(name)
                else:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details(f"Execution symbol {name} not found")
                    return executor_pb2.Symbol()
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Execution {name} not found")
                return executor_pb2.Symbol()

        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("Invalid symbol name format")
        return executor_pb2.Symbol()

    def ListSymbols(
        self, request: executor_pb2.ListSymbolsRequest, context: grpc.ServicerContext
    ) -> executor_pb2.ListSymbolsResponse:
        parent = request.parent
        page_size = request.page_size
        page_token = request.page_token

        symbols = None
        if parent == "symbols":
            symbols = self.symbols
        else:
            # Try to parse as session
            session_resource = SessionName.parse(parent)
            if session_resource is not None:
                session = self.get_session(session_resource.session_id)
                if not session:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details(
                        f"Session {session_resource.session_id} not found"
                    )
                    return executor_pb2.ListSymbolsResponse()
                symbols = session.symbols
            else:
                # Try to parse as execution
                execution_resource = ExecutionName.parse(parent)
                if execution_resource is not None:
                    execution = self.get_execution(
                        execution_resource.session_id, execution_resource.execution_id
                    )
                    if not execution:
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details(f"Execution {parent} not found")
                        return executor_pb2.ListSymbolsResponse()
                    symbols = execution.symbols
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("Invalid parent format")
                    return executor_pb2.ListSymbolsResponse()

        matching_symbols = [s.to_proto(n) for n, s in symbols.items()]
        matching_symbols.sort(key=lambda s: s.name)

        # Pagination
        start_index = 0
        if page_token:
            start_index = next(
                (i for i, s in enumerate(matching_symbols) if s.name == page_token), -1
            )
            if start_index == -1:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid page token")
                return executor_pb2.ListSymbolsResponse()
            start_index += 1

        end_index = (
            (start_index + page_size) if page_size > 0 else len(matching_symbols)
        )
        page_symbols = matching_symbols[start_index:end_index]
        next_page_token = (
            page_symbols[-1].name if end_index < len(matching_symbols) else ""
        )

        return executor_pb2.ListSymbolsResponse(
            symbols=page_symbols, next_page_token=next_page_token
        )

    def UpdateSymbol(
        self, request: executor_pb2.UpdateSymbolRequest, context: grpc.ServicerContext
    ) -> executor_pb2.Symbol:
        name = request.symbol.name
        logging.info(f"UpdateSymbol: {name}")
        symbol = request.symbol
        update_mask = request.update_mask

        if name not in self.symbols:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Symbol {name} not found")
            return executor_pb2.Symbol()

        existing_symbol = self.symbols[name]
        for field in update_mask.paths:
            if field == "data":
                existing_symbol.data = symbol.data
            elif field == "type":
                existing_symbol.type = symbol.type
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Invalid field in update_mask: {field}")
                return executor_pb2.Symbol()
        return existing_symbol.to_proto(name)

    def DeleteSymbol(
        self, request: executor_pb2.DeleteSymbolRequest, context: grpc.ServicerContext
    ) -> empty_pb2.Empty:
        logging.info(f"DeleteSymbol: {request.name}")
        name = request.name
        if name not in self.symbols:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Symbol {name} not found")
            return empty_pb2.Empty()
        del self.symbols[name]
        return empty_pb2.Empty()

    # --- Session Methods ---

    def CreateSession(
        self, request: executor_pb2.CreateSessionRequest, context: grpc.ServicerContext
    ) -> executor_pb2.Session:
        logging.info(f"CreateSession: sessions/{request.session.name}")

        if request.parent != "":
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Parent should be empty for sessions")
            return executor_pb2.Session()

        session_id = request.session.name
        name = SessionName(session_id).to_string()
        if name in self.sessions:
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            context.set_details(f"Session {session_id} already exists")
            return executor_pb2.Session()

        if self.party_id not in request.session.party_addrs:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Peer addresses must include the party's address")
            return executor_pb2.Session()

        new_session = Session(
            party_id=self.party_id,
            session_id=session_id,
            party_addrs=dict(request.session.party_addrs),
            metadata=dict(request.session.metadata),
        )

        self.sessions[name] = new_session
        return new_session.to_proto()

    def GetSession(
        self, request: executor_pb2.GetSessionRequest, context: grpc.ServicerContext
    ) -> executor_pb2.Session:
        logging.info(f"GetSession: {request.name}")
        name = request.name
        if name not in self.sessions:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {name} not found")
            return executor_pb2.Session()
        return self.sessions[name].to_proto()

    def ListSessions(
        self, request: executor_pb2.ListSessionsRequest, context: grpc.ServicerContext
    ) -> executor_pb2.ListSessionsResponse:
        page_size = request.page_size
        page_token = request.page_token

        all_sessions = list(self.sessions.values())
        all_sessions.sort(key=lambda s: s.name)

        start_index = 0
        if page_token:
            start_index = next(
                (i for i, s in enumerate(all_sessions) if s.name == page_token), -1
            )
            if start_index == -1:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid page token")
                return executor_pb2.ListSessionsResponse()
            start_index += 1

        end_index = (start_index + page_size) if page_size > 0 else len(all_sessions)
        page_sessions = all_sessions[start_index:end_index]
        next_page_token = (
            page_sessions[-1].name if end_index < len(all_sessions) else ""
        )

        page_sessions_proto = [session.to_proto() for session in page_sessions]

        return executor_pb2.ListSessionsResponse(
            sessions=page_sessions_proto, next_page_token=next_page_token
        )

    def DeleteSession(
        self, request: executor_pb2.DeleteSessionRequest, context: grpc.ServicerContext
    ) -> empty_pb2.Empty:
        name = request.name
        if name not in self.sessions:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {name} not found")
            return empty_pb2.Empty()
        del self.sessions[name]
        return empty_pb2.Empty()

    # --- Execution Methods ---

    def CreateExecution(
        self,
        request: executor_pb2.CreateExecutionRequest,
        context: grpc.ServicerContext,
    ) -> executor_pb2.Execution:
        logging.info(
            f"CreateExecution: {request.parent}/executions/{request.execution.name}"
        )

        session_resource = SessionName.parse(request.parent)
        if not session_resource:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Parent must be in format 'sessions/{session_id}'")
            return executor_pb2.Execution()

        session_id = session_resource.session_id
        session = self.get_session(session_id)
        if not session:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {session_id} not found")
            return executor_pb2.Execution()

        execution_name = ExecutionName(session_id, request.execution.name).to_string()
        if execution_name in self.executions:
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            context.set_details(f"Execution {execution_name} already exists")
            return executor_pb2.Execution()

        for input_name in request.execution.input_names:
            symbol = self.lookup_symbol(input_name)
            if not symbol:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Symbol {input_name} not found")
                return executor_pb2.Execution()

        # Deserialize the program bytes to GraphProto
        program_proto = mpir_pb2.GraphProto()
        try:
            program_proto.ParseFromString(request.execution.program)
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Failed to parse program: {e}")
            return executor_pb2.Execution()

        execution = Execution(
            db=self,
            session_id=session_id,
            execution_id=request.execution.name,
            program=program_proto,
            input_names=list(request.execution.input_names),
            output_names=list(request.execution.output_names),
            spu_mask=int(request.execution.attrs["spu_mask"].number_value),
            spu_protocol=libspu.ProtocolKind(
                int(request.execution.attrs["spu_protocol"].number_value)
            ),
            spu_field=libspu.FieldType(
                int(request.execution.attrs["spu_field"].number_value)
            ),
            make_stub_func=self.make_stub_func,
        )
        self.executions[execution_name] = execution

        try:
            execution.run()
        except Exception as e:
            # Handle execution failure with optional debug information
            execution.state = executor_pb2.ExecutionState.FAILED
            execution.end_time = datetime.datetime.now()

            if self._debug_execution:
                import traceback

                execution.error = f"{e!s}\n\nStack trace:\n{traceback.format_exc()}"
            else:
                execution.error = str(e)

            logging.exception(f"An error occurred in execution: {e}")

        return execution.to_proto()

    def GetExecution(
        self, request: executor_pb2.GetExecutionRequest, context: grpc.ServicerContext
    ) -> executor_pb2.Execution:
        name = request.name
        if name not in self.executions:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Execution {name} not found")
            return executor_pb2.Execution()
        return self.executions[name].to_proto()

    def ListExecutions(
        self, request: executor_pb2.ListExecutionsRequest, context: grpc.ServicerContext
    ) -> executor_pb2.ListExecutionsResponse:
        parent = request.parent
        page_size = request.page_size
        page_token = request.page_token

        session_resource = SessionName.parse(parent)
        if not session_resource:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid parent format")
            return executor_pb2.ListExecutionsResponse()

        session_id = session_resource.session_id
        matching_executions = [
            execution.to_proto()
            for execution in self.executions.values()
            if execution.session_id == session_id
        ]
        matching_executions.sort(key=lambda e: e.name)

        start_index = 0
        if page_token:
            start_index = next(
                (i for i, e in enumerate(matching_executions) if e.name == page_token),
                -1,
            )
            if start_index == -1:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid page token")
                return executor_pb2.ListExecutionsResponse()
            start_index += 1

        end_index = (
            (start_index + page_size) if page_size > 0 else len(matching_executions)
        )
        page_executions = matching_executions[start_index:end_index]
        next_page_token = (
            page_executions[-1].name if end_index < len(matching_executions) else ""
        )

        return executor_pb2.ListExecutionsResponse(
            executions=page_executions, next_page_token=next_page_token
        )

    def DeleteExecution(
        self,
        request: executor_pb2.DeleteExecutionRequest,
        context: grpc.ServicerContext,
    ) -> empty_pb2.Empty:
        name = request.name
        if name not in self.executions:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Execution {name} not found")
            return empty_pb2.Empty()
        del self.executions[name]
        return empty_pb2.Empty()


def serve(
    party_id: str,
    addr: str,
    max_message_length: int = 1024 * 1024 * 1024,
    debug_execution: bool = False,
) -> None:
    """Start the executor service server."""

    def make_stub(addr: str) -> executor_pb2_grpc.ExecutorServiceStub:
        channel = grpc.insecure_channel(
            addr,
            options=[
                ("grpc.max_send_message_length", max_message_length),
                ("grpc.max_receive_message_length", max_message_length),
            ],
        )
        return executor_pb2_grpc.ExecutorServiceStub(channel)

    logging.basicConfig(
        level=logging.INFO,
        format=f"{party_id}: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", "a", "utf-8"),
        ],
    )

    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
        ],
    )
    executor_pb2_grpc.add_ExecutorServiceServicer_to_server(
        ExecutorService(party_id, make_stub, debug_execution), server
    )
    server.add_insecure_port(addr)
    print(f"Server started {addr}")
    server.start()
    server.wait_for_termination()


def start_cluster(party_addrs: dict[str, str], debug_execution: bool = False) -> None:
    """Start a cluster of executor services."""
    import multiprocessing as multiprocess
    import signal
    import sys

    Process = multiprocess.get_context("forkserver").Process

    # Setup logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Start the workers
    workers = []
    for pid, addr in party_addrs.items():
        worker = Process(
            target=serve, args=(pid, addr, 1024 * 1024 * 1024, debug_execution)
        )
        workers.append(worker)
        worker.start()

    def signal_handler(signum: int, frame: Any) -> None:
        """Handle signals and forcefully terminate all child processes."""
        logging.info(f"Received signal {signum}, terminating all child processes...")
        for worker in workers:
            if worker.is_alive():
                logging.info(f"Terminating process {worker.pid}")
                worker.terminate()

        # Wait a bit for graceful termination
        for worker in workers:
            worker.join(timeout=2)

        # Force kill any remaining processes
        for worker in workers:
            if worker.is_alive():
                logging.warning(f"Force killing process {worker.pid}")
                worker.kill()

        logging.info("All child processes terminated")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    finally:
        # Ensure all processes are terminated on exit
        for worker in workers:
            if worker.is_alive():
                logging.info(f"Cleaning up process {worker.pid}")
                worker.terminate()

        for worker in workers:
            worker.join(timeout=2)

        for worker in workers:
            if worker.is_alive():
                logging.warning(f"Force killing remaining process {worker.pid}")
                worker.kill()
