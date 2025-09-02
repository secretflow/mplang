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
This module provides the resource management for the HTTP backend.
It is a simplified, in-memory version of the original executor's resource manager.
"""

import base64
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import cloudpickle as pickle

from mplang.backend.builtin import BuiltinHandler
from mplang.backend.phe import PHEHandler
from mplang.backend.spu import SpuHandler
from mplang.backend.sql_duckdb import DuckDBHandler
from mplang.backend.stablehlo import StablehloHandler
from mplang.core.mask import Mask
from mplang.expr.ast import Expr
from mplang.expr.evaluator import Evaluator
from mplang.runtime.grpc_comm import LinkCommunicator
from mplang.runtime.http_backend.communicator import HttpCommunicator
from mplang.runtime.http_backend.exceptions import InvalidRequestError, ResourceNotFound


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


@dataclass
class Symbol:
    name: str
    mptype: Any  # More flexible type to handle dict or MPType
    data: Any  # More flexible data type


@dataclass
class Computation:
    name: str
    expr: Expr  # The computation expression
    symbols: dict[str, Symbol] = field(default_factory=dict)


@dataclass
class Session:
    name: str
    communicator: HttpCommunicator
    computations: dict[str, Computation] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)  # Session-level symbols

    # spu related
    spu_mask: int = -1
    spu_protocol: int = 0
    spu_field: int = 0


# Global session storage
_sessions: dict[str, Session] = {}


# Session Management
def create_session(
    name: str,
    rank: int,
    endpoints: list[str],
    # SPU related
    spu_mask: int = -1,
    spu_protocol: int = 0,
    spu_field: int = 0,
) -> Session:
    logging.info(f"Creating session: {name}, rank: {rank}, spu_mask: {spu_mask}")
    if name in _sessions:
        # Return existing session (idempotent operation)
        logging.info(f"Session {name} already exists, returning existing session")
        return _sessions[name]
    session = Session(
        name, HttpCommunicator(session_name=name, rank=rank, endpoints=endpoints)
    )

    session.spu_mask = spu_mask
    session.spu_protocol = spu_protocol
    session.spu_field = spu_field

    _sessions[name] = session
    logging.info(f"Session {name} created successfully")
    return session


def get_session(name: str) -> Session | None:
    return _sessions.get(name)


# Computation Management
def create_computation(
    session_name: str, computation_name: str, expr: Expr
) -> Computation:
    """Creates a computation resource within a session."""
    session = get_session(session_name)
    if not session:
        raise ResourceNotFound(f"Session '{session_name}' not found.")
    computation = Computation(computation_name, expr)
    session.computations[computation_name] = computation
    logging.info(f"Computation {computation_name} created for session {session_name}")
    return computation


def get_computation(session_name: str, comp_name: str) -> Computation | None:
    session = get_session(session_name)
    if session:
        return session.computations.get(comp_name)
    return None


def execute_computation(
    session_name: str, comp_name: str, input_names: list[str], output_names: list[str]
) -> None:
    """Execute a computation using the Evaluator."""
    session = get_session(session_name)
    if not session:
        raise ResourceNotFound(f"Session '{session_name}' not found.")

    computation = get_computation(session_name, comp_name)
    if not computation:
        raise ResourceNotFound(
            f"Computation '{comp_name}' not found in session '{session_name}'."
        )

    if not session.communicator:
        raise InvalidRequestError(
            f"Communicator not initialized for session '{session_name}'."
        )

    # Get rank from session communicator
    rank = session.communicator.rank

    # Prepare input bindings from session symbols
    bindings = {}
    for input_name in input_names:
        symbol = get_symbol(session_name, input_name)
        if not symbol:
            raise ResourceNotFound(
                f"Input symbol '{input_name}' not found in session '{session_name}'"
            )
        bindings[input_name] = symbol.data

    # Create handlers (similar to executor/server.py)
    import spu.libspu as libspu

    # This config is misleading, it configs the runtime as well as spu IO.
    spu_config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind(session.spu_protocol),
        field=libspu.FieldType(session.spu_field),
        fxp_fraction_bits=18,
    )

    spu_comm: LinkCommunicator | None = None
    spu_mask = (
        Mask(session.spu_mask)
        if session.spu_mask != -1
        else Mask.all(session.communicator.world_size)
    )

    if rank in spu_mask:
        spu_addrs: list[str] = []
        for r, addr in enumerate(session.communicator.endpoints):
            if r in spu_mask:
                # Robust URL parsing
                parsed = urlparse(addr)
                hostname = parsed.hostname or "localhost"
                if parsed.port is not None:
                    base_port = parsed.port
                else:
                    # Default ports: http 80, https 443; fallback to 80
                    if parsed.scheme == "https":
                        base_port = 443
                    else:
                        base_port = 80
                new_addr = f"{hostname}:{base_port + 100}"
                spu_addrs.append(new_addr)
        spu_rank = spu_mask.global_to_relative_rank(rank)
        spu_comm = g_link_factory.create_link(spu_rank, spu_addrs)

    # Use the world size from the communicator
    spu_handler = SpuHandler(
        spu_comm.world_size if spu_comm is not None else 0,
        spu_config,
    )
    if spu_comm is not None:
        spu_handler.set_link_context(spu_comm)

    # Instantiate the Evaluator with session symbols as environment
    evaluator = Evaluator(
        rank=rank,
        env={},  # Start with empty environment, will use fork with bindings
        comm=session.communicator,
        pfunc_handles=[
            BuiltinHandler(),
            StablehloHandler(),
            spu_handler,
            DuckDBHandler(),
            PHEHandler(),
        ],
    )

    # Execute with input bindings
    forked_evaluator = evaluator.fork(bindings)
    results = computation.expr.accept(forked_evaluator)

    # Store results in session symbols using output_names
    if results:
        if len(results) != len(output_names):
            raise RuntimeError(
                f"Expected {len(output_names)} results, got {len(results)}"
            )
        for name, val in zip(output_names, results, strict=True):
            session.symbols[name] = Symbol(name=name, mptype={}, data=val)


# Symbol Management
def create_symbol(session_name: str, name: str, mptype: Any, data: Any) -> Symbol:
    """Create a symbol in a session's symbol table.

    The `data` is expected to be a base64-encoded pickled Python object.
    """
    session = get_session(session_name)
    if not session:
        raise ResourceNotFound(f"Session '{session_name}' not found.")

    # Deserialize base64-encoded data to Python object
    try:
        data_bytes = base64.b64decode(data)
        obj = pickle.loads(data_bytes)
    except Exception as e:
        raise InvalidRequestError(f"Invalid symbol data encoding: {e!s}") from e

    symbol = Symbol(name, mptype, obj)
    session.symbols[name] = symbol
    return symbol


def create_computation_symbol(
    session_name: str, computation_name: str, symbol_name: str, mptype: Any, data: Any
) -> Symbol:
    """Create a symbol in a computation's symbol table."""
    session = get_session(session_name)
    if not session:
        raise ResourceNotFound(f"Session '{session_name}' not found.")

    computation = get_computation(session_name, computation_name)
    if not computation:
        raise ResourceNotFound(
            f"Computation '{computation_name}' not found in session '{session_name}'."
        )

    symbol = Symbol(symbol_name, mptype, data)
    computation.symbols[symbol_name] = symbol
    return symbol


def get_symbol(session_name: str, name: str) -> Symbol | None:
    """Get a symbol from a session's symbol table (session-level or computation-level)."""
    session = get_session(session_name)
    if not session:
        return None

    # First try session-level symbols
    if name in session.symbols:
        return session.symbols[name]

    # Then try computation-level symbols from all computations
    for computation in session.computations.values():
        if name in computation.symbols:
            return computation.symbols[name]

    return None


def list_symbols(session_name: str) -> list[str]:
    """List all symbols in a session's symbol table (both session-level and computation-level)."""
    session = get_session(session_name)
    if not session:
        raise ResourceNotFound(f"Session '{session_name}' not found.")

    symbols: list[str] = []
    # Add session-level symbols
    symbols.extend(session.symbols.keys())

    # Add computation-level symbols from all computations
    for computation in session.computations.values():
        symbols.extend(computation.symbols.keys())

    return symbols
