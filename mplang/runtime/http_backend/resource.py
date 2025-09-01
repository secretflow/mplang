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

import uuid
from dataclasses import dataclass, field
from typing import Any

from mplang.backend.builtin import BuiltinHandler
from mplang.backend.phe import PHEHandler
from mplang.backend.spu import SpuHandler
from mplang.backend.sql_duckdb import DuckDBHandler
from mplang.backend.stablehlo import StablehloHandler
from mplang.expr.ast import Expr
from mplang.expr.evaluator import Evaluator
from mplang.runtime.http_backend.communicator import HttpCommunicator


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
    communicator: HttpCommunicator | None = None
    computations: dict[str, Computation] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)  # Session-level symbols


# Global session storage
_sessions: dict[str, Session] = {}


def gen_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


# Session Management
def create_session(
    name: str | None = None, rank: int = 0, endpoints: dict[int, str] | None = None
) -> Session:
    name = name or gen_name("session")
    if name in _sessions:
        # Return existing session (idempotent operation)
        return _sessions[name]
    session = Session(name)
    if endpoints:
        # Each session has its own communicator
        session.communicator = HttpCommunicator(
            session_name=name, rank=rank, endpoints=endpoints
        )
    _sessions[name] = session
    return session


def get_session(name: str) -> Session | None:
    return _sessions.get(name)


# Computation Management
def create_computation(session_name: str, expr: Expr) -> Computation:
    session = get_session(session_name)
    if not session:
        raise ValueError(f"Session {session_name} not found.")
    comp_name = gen_name("comp")
    computation = Computation(comp_name, expr)
    session.computations[comp_name] = computation
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
        raise ValueError(f"Session {session_name} not found.")

    computation = get_computation(session_name, comp_name)
    if not computation:
        raise ValueError(
            f"Computation {comp_name} not found in session {session_name}."
        )

    if not session.communicator:
        raise ValueError(f"Communicator not initialized for session {session_name}.")

    # Get rank from session communicator
    rank = session.communicator.rank

    # Prepare input bindings from session symbols
    bindings = {}
    for input_name in input_names:
        symbol = get_symbol(session_name, input_name)
        if not symbol:
            raise ValueError(
                f"Input symbol {input_name} not found in session {session_name}"
            )
        bindings[input_name] = symbol.data

    # Create handlers (similar to executor/server.py)
    import spu.libspu as libspu

    spu_config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SEMI2K,
        field=libspu.FieldType.FM64,
        fxp_fraction_bits=18,
    )

    # Use the world size from the communicator
    world_size = session.communicator.world_size
    spu_handler = SpuHandler(world_size, spu_config)

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
        assert len(results) == len(
            output_names
        ), f"Expected {len(output_names)} results, got {len(results)}"
        for name, val in zip(output_names, results, strict=False):
            session.symbols[name] = Symbol(name=name, mptype={}, data=val)


# Symbol Management
def create_symbol(session_name: str, name: str, mptype: Any, data: Any) -> Symbol:
    """Create a symbol in a session's symbol table."""
    session = get_session(session_name)
    if not session:
        raise ValueError(f"Session {session_name} not found.")

    symbol = Symbol(name, mptype, data)
    session.symbols[name] = symbol
    return symbol


def create_computation_symbol(
    session_name: str, computation_name: str, symbol_name: str, mptype: Any, data: Any
) -> Symbol:
    """Create a symbol in a computation's symbol table."""
    session = get_session(session_name)
    if not session:
        raise ValueError(f"Session {session_name} not found.")

    computation = get_computation(session_name, computation_name)
    if not computation:
        raise ValueError(
            f"Computation {computation_name} not found in session {session_name}."
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
        raise ValueError(f"Session {session_name} not found.")

    symbols: list[str] = []
    # Add session-level symbols
    symbols.extend(session.symbols.keys())

    # Add computation-level symbols from all computations
    for computation in session.computations.values():
        symbols.extend(computation.symbols.keys())

    return symbols
