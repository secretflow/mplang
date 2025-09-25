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
import spu.libspu as libspu

# Import backends (side-effect: kernel registration)
from mplang.backend import (  # noqa: F401
    builtin,
    crypto,
    phe,
    spu,  # registers flat SPU kernels
    sql_duckdb,
    stablehlo,
    tee,
)
from mplang.backend.base import BackendRuntime, create_runtime
from mplang.backend.spu import PFunction  # type: ignore
from mplang.core.expr.ast import Expr
from mplang.core.expr.evaluator import IEvaluator, create_evaluator
from mplang.core.mask import Mask
from mplang.runtime.communicator import HttpCommunicator
from mplang.runtime.exceptions import InvalidRequestError, ResourceNotFound
from mplang.runtime.link_comm import LinkCommunicator
from mplang.utils.spu_utils import parse_field, parse_protocol


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


@dataclass
class Session:
    name: str
    communicator: HttpCommunicator
    computations: dict[str, Computation] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)  # Session-level symbols

    # spu related
    spu_mask: int = -1
    spu_protocol: str = "SEMI2K"
    spu_field: str = "FM64"


# Global session storage
_sessions: dict[str, Session] = {}

# Service-level global symbol table (process-local for this server)
_global_symbols: dict[str, Symbol] = {}


# Session Management
def create_session(
    name: str,
    rank: int,
    endpoints: list[str],
    # SPU related
    spu_mask: int = 0,
    spu_protocol: str = "SEMI2K",
    spu_field: str = "FM64",
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


def delete_session(name: str) -> bool:
    """Delete a session and all associated resources.

    Returns:
        True if session was deleted, False if session was not found.
    """
    if name in _sessions:
        del _sessions[name]
        logging.info(f"Session {name} deleted successfully")
        return True
    return False


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


def delete_computation(session_name: str, comp_name: str) -> bool:
    """Delete a computation from a session.

    Returns:
        True if computation was deleted, False if not found.
    """
    session = get_session(session_name)
    if not session:
        return False

    if comp_name in session.computations:
        del session.computations[comp_name]
        logging.info(f"Computation {comp_name} deleted from session {session_name}")
        return True
    return False


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

    spu_mask = (
        Mask(session.spu_mask)
        if session.spu_mask != -1
        else Mask.all(session.communicator.world_size)
    )

    # Build evaluator
    # Explicit per-rank backend runtime (deglobalized)
    runtime = create_runtime(rank, session.communicator.world_size)
    # Inject global symbol storage into backend runtime state so that
    # symbols:// provider can access it during builtin.read/write.
    if isinstance(runtime, BackendRuntime):
        pocket = runtime.state.setdefault("resource.providers", {})
        if "symbols" not in pocket:
            pocket["symbols"] = _global_symbols
        else:
            raise RuntimeError("resource.providers.symbols already exists")
    evaluator: IEvaluator = create_evaluator(
        rank=rank, env=bindings, comm=session.communicator, runtime=runtime
    )

    # Initialize SPU runtime state for flat kernels (once per evaluator invocation)
    if rank in spu_mask:
        # Build SPU address list (only once per rank; consistent ordering of participating ranks)
        spu_addrs: list[str] = []
        for r, addr in enumerate(session.communicator.endpoints):
            if r in spu_mask:
                if "://" not in addr:
                    addr = f"//{addr}"
                parsed = urlparse(addr)
                assert isinstance(parsed.port, int)
                new_addr = f"{parsed.hostname}:{parsed.port + 100}"
                spu_addrs.append(new_addr)
        # Determine this rank's relative index among participating ranks
        rel_index = sum(1 for r in range(rank) if r in spu_mask)
        link_ctx = g_link_factory.create_link(rel_index, spu_addrs)
    else:
        link_ctx = None
    # Always seed config/world; provide per-rank link (may be None if not participating)
    spu_config = libspu.RuntimeConfig(
        protocol=parse_protocol(session.spu_protocol),
        field=parse_field(session.spu_field),
        fxp_fraction_bits=18,
    )
    # Seed SPU env via backend kernel (inside evaluator's kernel context)
    seed_pfunc = PFunction(
        fn_type="spu.seed_env",
        ins_info=(),
        outs_info=(),
        config=spu_config,
        world=spu_mask.num_parties(),
        link=link_ctx,
    )
    # Run seeding kernel with evaluator (no inputs, no outputs)
    evaluator.runtime.run_kernel(seed_pfunc, [])

    results = evaluator.evaluate(computation.expr)

    # Store results in session symbols using output_names
    if results:
        if len(results) != len(output_names):
            raise RuntimeError(
                f"Expected {len(output_names)} results, got {len(results)}"
            )
        for name, val in zip(output_names, results, strict=True):
            session.symbols[name] = Symbol(name=name, mptype={}, data=val)


# Global symbol CRUD (service-level)
def create_global_symbol(name: str, mptype: Any, data: str) -> Symbol:
    """Create or update a global symbol in the service-level table.

    WARNING: Uses Python pickle for arbitrary object deserialization. Deploy
    only in trusted environments. Future work may replace this with a
    restricted / structured serialization.

    The `data` argument is a base64-encoded pickled Python object. Minimal
    validation of `mptype` is performed for tensor metadata (shape/dtype)
    when present to catch obvious mismatches.
    """
    try:
        data_bytes = base64.b64decode(data)
        obj = pickle.loads(data_bytes)
    except Exception as e:
        raise InvalidRequestError(f"Invalid global symbol data encoding: {e!s}") from e

    sym = Symbol(name, mptype, obj)
    _global_symbols[name] = sym
    return sym


def get_global_symbol(name: str) -> Symbol | None:
    return _global_symbols.get(name)


def list_global_symbols() -> list[str]:
    return list(_global_symbols.keys())


def delete_global_symbol(name: str) -> bool:
    if name in _global_symbols:
        del _global_symbols[name]
        logging.info(f"Global symbol {name} deleted")
        return True
    return False


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


def get_symbol(session_name: str, name: str) -> Symbol | None:
    """Get a symbol from a session's symbol table (session-level only)."""
    session = get_session(session_name)
    if not session:
        return None

    # Only session-level symbols are supported now
    return session.symbols.get(name)


def list_symbols(session_name: str) -> list[str]:
    """List all symbols in a session's symbol table."""
    session = get_session(session_name)
    if not session:
        raise ResourceNotFound(f"Session '{session_name}' not found.")

    # Only session-level symbols are supported now
    return list(session.symbols.keys())


def delete_symbol(session_name: str, symbol_name: str) -> bool:
    """Delete a symbol from a session.

    Returns:
        True if symbol was deleted, False if not found.
    """
    session = get_session(session_name)
    if not session:
        return False

    if symbol_name in session.symbols:
        del session.symbols[symbol_name]
        logging.info(f"Symbol {symbol_name} deleted from session {session_name}")
        return True
    return False


def list_all_sessions() -> list[str]:
    """List all session names."""
    return list(_sessions.keys())
