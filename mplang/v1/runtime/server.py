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
This module implements the HTTP server for the toy backend.
It uses FastAPI to provide a RESTful API for managing computations.
"""

import base64
import logging
import re
from typing import Any

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mplang.v1.core import IrReader, TableType, TensorType
from mplang.v1.core.cluster import ClusterSpec
from mplang.v1.kernels.base import KernelContext
from mplang.v1.kernels.value import Value, decode_value, encode_value
from mplang.v1.protos.v1alpha1 import mpir_pb2
from mplang.v1.runtime.data_providers import (
    DataProvider,
    ResolvedURI,
    register_provider,
)
from mplang.v1.runtime.exceptions import InvalidRequestError, ResourceNotFound
from mplang.v1.runtime.session import (
    Computation,
    Session,
    Symbol,
    create_session_from_spec,
)

logger = logging.getLogger(__name__)

app = FastAPI()

# per-server global state
_sessions: dict[str, Session] = {}
_global_symbols: dict[str, Symbol] = {}


def register_session(session: Session) -> Session:  # pragma: no cover - test helper
    existing = _sessions.get(session.name)
    if existing:
        return existing
    _sessions[session.name] = session
    return session


class _SymbolsProvider(DataProvider):
    """Server-local symbols provider backed by BackendRuntime.state."""

    @staticmethod
    def _symbol_name(uri: ResolvedURI) -> str:
        if uri.scheme != "symbols" or uri.parsed is None:
            raise InvalidRequestError(
                "symbols provider expects URI in the form symbols://{name}"
            )

        parsed = uri.parsed
        if parsed.query or parsed.params or parsed.fragment:
            raise InvalidRequestError(
                "symbols:// URI must not contain query or fragment"
            )

        if parsed.netloc:
            # e.g. symbols://foo -> name is carried in netloc (path may be empty or "/")
            if parsed.path not in ("", "/"):
                raise InvalidRequestError("symbols:// URIs cannot include subpaths")
            name = parsed.netloc
        else:
            # e.g. symbols:///foo -> netloc empty, single path segment is the symbol name
            path = parsed.path.lstrip("/")
            if not path or "/" in path:
                raise InvalidRequestError(
                    "symbols:// URI must specify a single symbol name"
                )
            name = path

        if not name:
            raise InvalidRequestError("symbols:// URI missing symbol name")
        return name

    def read(
        self,
        uri: ResolvedURI,
        out_spec: TensorType | TableType,
        *,
        ctx: KernelContext,
    ) -> Any:  # type: ignore[override]
        name = self._symbol_name(uri)
        sym = _global_symbols.get(name)
        if sym is None:
            raise ResourceNotFound(f"Global symbol '{name}' not found")
        return sym.data

    def write(
        self,
        uri: ResolvedURI,
        value: Any,
        *,
        ctx: KernelContext,
    ) -> None:  # type: ignore[override]
        name = self._symbol_name(uri)
        if not isinstance(value, Value):
            raise InvalidRequestError(
                f"symbols:// write expects Value instance, got {type(value)}"
            )
        _global_symbols[name] = Symbol(name=name, mptype={}, data=value)


# Register symbols provider explicitly for server runtime
register_provider("symbols", _SymbolsProvider())


@app.exception_handler(ResourceNotFound)
def resource_not_found_handler(request: Request, exc: ResourceNotFound) -> JSONResponse:
    """Handler for ResourceNotFound exceptions."""
    logger.warning(f"Resource not found at {request.url}: {exc}")
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)},
    )


@app.exception_handler(InvalidRequestError)
def invalid_request_handler(request: Request, exc: InvalidRequestError) -> JSONResponse:
    """Handler for InvalidRequestError exceptions."""
    logger.warning(f"Invalid request at {request.url}: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for better error reporting."""
    logger.error(f"Unhandled exception at {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {exc!s}",
            "error_type": type(exc).__name__,
            "path": str(request.url.path),
        },
    )


def validate_name(name: str, name_type: str) -> None:
    """Validate that a name is safe for use in URL paths.

    Args:
        name: The name to validate
        name_type: Type of name (for error messages, e.g., "session", "computation")

    Raises:
        HTTPException: If the name contains invalid characters
    """
    if not name:
        raise HTTPException(status_code=400, detail=f"{name_type} name cannot be empty")

    # Only allow alphanumeric, hyphens, underscores, and dots
    if not re.match(r"^[a-zA-Z0-9._-]+$", name):
        raise HTTPException(
            status_code=400,
            detail=f"{name_type} name can only contain letters, numbers, dots, hyphens, and underscores",
        )


# Request/Response Models
class CreateSessionRequest(BaseModel):
    rank: int
    cluster_spec: dict


class SessionResponse(BaseModel):
    name: str


class CreateComputationRequest(BaseModel):
    mpprogram: str  # Base64 encoded MPProgram proto
    input_names: list[str]  # Mandatory input symbol names
    output_names: list[str]  # Mandatory output symbol names


class ComputationResponse(BaseModel):
    name: str


class CreateSymbolRequest(BaseModel):
    mptype: dict
    data: str  # Base64 encoded Value data


class SymbolResponse(BaseModel):
    name: str
    mptype: dict
    data: str  # Base64 encoded Value data


class CommSendRequest(BaseModel):
    data: str  # Base64 encoded binary data


# Response Models for enhanced status
class SessionListResponse(BaseModel):
    sessions: list[str]


class ComputationListResponse(BaseModel):
    computations: list[str]


class GlobalSymbolResponse(BaseModel):
    name: str
    mptype: dict
    data: str  # Base64 encoded Value data


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# List all sessions
@app.get("/sessions", response_model=SessionListResponse)
def list_sessions() -> SessionListResponse:
    """List all session names."""
    return SessionListResponse(sessions=list(_sessions.keys()))


# List all computations in a session
@app.get(
    "/sessions/{session_name}/computations", response_model=ComputationListResponse
)
def list_session_computations(session_name: str) -> ComputationListResponse:
    """List all computation names in a session."""
    sess = _sessions.get(session_name)
    if not sess:
        raise ResourceNotFound(f"Session '{session_name}' not found")
    return ComputationListResponse(computations=sess.list_computations())


# Session endpoints
@app.put("/sessions/{session_name}", response_model=SessionResponse)
def create_session(session_name: str, request: CreateSessionRequest) -> SessionResponse:
    validate_name(session_name, "session")
    # Delegate cluster spec parsing & session construction to resource layer

    if session_name in _sessions:
        sess = _sessions[session_name]
    else:
        spec = ClusterSpec.from_dict(request.cluster_spec)
        sess = create_session_from_spec(name=session_name, rank=request.rank, spec=spec)
        _sessions[session_name] = sess
    return SessionResponse(name=sess.name)


@app.get("/sessions/{session_name}", response_model=SessionResponse)
def get_session(session_name: str) -> SessionResponse:
    sess = _sessions.get(session_name)
    if not sess:
        raise ResourceNotFound(f"Session '{session_name}' not found")
    return SessionResponse(name=sess.name)


@app.delete("/sessions/{session_name}")
def delete_session(session_name: str) -> dict[str, str]:
    """Delete a session and all its associated resources."""
    if session_name in _sessions:
        del _sessions[session_name]
        logging.info(f"Session {session_name} deleted successfully")
        return {"message": f"Session '{session_name}' deleted successfully"}
    else:
        raise ResourceNotFound(f"Session '{session_name}' not found")


# Computation endpoints
@app.put(
    "/sessions/{session_name}/computations/{computation_id}",
    response_model=ComputationResponse,
)
def create_and_execute_computation(
    session_name: str, computation_id: str, request: CreateComputationRequest
) -> ComputationResponse:
    graph_proto = mpir_pb2.GraphProto()
    try:
        graph_proto.ParseFromString(base64.b64decode(request.mpprogram))
    except Exception as e:
        raise InvalidRequestError(
            f"Invalid base64 or protobuf for mpprogram: {e!s}"
        ) from e

    reader = IrReader()
    expr = reader.loads(graph_proto)

    if expr is None:
        raise InvalidRequestError("Failed to parse expression from protobuf")

    # Create the computation resource
    sess = _sessions.get(session_name)
    if not sess:
        raise ResourceNotFound(f"Session '{session_name}' not found.")
    comp = sess.get_computation(computation_id)
    if not comp:
        comp = Computation(name=computation_id, expr=expr)
        sess.add_computation(comp)
    sess.execute(comp, request.input_names, request.output_names)
    return ComputationResponse(name=computation_id)


@app.delete("/sessions/{session_name}/computations/{computation_id}")
def delete_computation(session_name: str, computation_id: str) -> dict[str, str]:
    """Delete a specific computation."""
    sess = _sessions.get(session_name)
    if sess and sess.delete_computation(computation_id):
        logging.info(
            f"Computation {computation_id} deleted from session {session_name}"
        )
        return {"message": f"Computation '{computation_id}' deleted successfully"}
    else:
        raise ResourceNotFound(
            f"Computation '{computation_id}' not found in session '{session_name}'"
        )


# Symbol endpoints
@app.put(
    "/sessions/{session_name}/symbols/{symbol_name}", response_model=SymbolResponse
)
def create_session_symbol(
    session_name: str, symbol_name: str, request: CreateSymbolRequest
) -> SymbolResponse:
    """Create a symbol in a session."""
    sess = _sessions.get(session_name)
    if not sess:
        raise ResourceNotFound(f"Session '{session_name}' not found.")
    try:
        obj = decode_value(base64.b64decode(request.data))
    except Exception as e:
        raise InvalidRequestError(f"Invalid symbol data: {e!s}") from e
    symbol = Symbol(name=symbol_name, mptype=request.mptype, data=obj)
    sess.add_symbol(symbol)
    # Return the base64 data back to client; server stores Python object
    return SymbolResponse(
        name=symbol.name,
        mptype=symbol.mptype,
        data=base64.b64encode(encode_value(symbol.data)).decode("utf-8"),
    )


@app.get(
    "/sessions/{session_name}/symbols/{symbol_name}", response_model=SymbolResponse
)
def get_session_symbol(session_name: str, symbol_name: str) -> SymbolResponse:
    """Get a symbol from a session."""
    try:
        logger.debug(
            f"Looking for symbol: '{symbol_name}' in session: '{session_name}'"
        )
        sess = _sessions.get(session_name)
        symbol = sess.get_symbol(symbol_name) if sess else None
        if not symbol:
            raise HTTPException(
                status_code=404, detail=f"Symbol {symbol_name} not found"
            )

        # symbol data is None means this party does not participate the computation
        # that produced the symbol.
        if symbol.data is None:
            raise ResourceNotFound(f"Symbol '{symbol_name}' has no data on this party")

        # Serialize using Value envelope
        return SymbolResponse(
            name=symbol.name,
            mptype=symbol.mptype,
            data=base64.b64encode(encode_value(symbol.data)).decode("utf-8"),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/sessions/{session_name}/symbols")
def list_session_symbols(session_name: str) -> dict[str, list[str]]:
    """List all symbols in a session."""
    sess = _sessions.get(session_name)
    if not sess:
        raise ResourceNotFound(f"Session '{session_name}' not found.")
    symbols = sess.list_symbols()
    return {"symbols": symbols}


@app.delete("/sessions/{session_name}/symbols/{symbol_name}")
def delete_symbol(session_name: str, symbol_name: str) -> dict[str, str]:
    """Delete a specific symbol."""
    sess = _sessions.get(session_name)
    if sess and sess.delete_symbol(symbol_name):
        logging.info(f"Symbol {symbol_name} deleted from session {session_name}")
        return {"message": f"Symbol '{symbol_name}' deleted successfully"}
    else:
        raise ResourceNotFound(
            f"Symbol '{symbol_name}' not found in session '{session_name}'"
        )


# Global Symbols endpoints
@app.put("/api/v1/symbols/{symbol_name}", response_model=GlobalSymbolResponse)
def create_global_symbol(
    symbol_name: str, request: CreateSymbolRequest
) -> GlobalSymbolResponse:
    validate_name(symbol_name, "symbol")
    try:
        obj = decode_value(base64.b64decode(request.data))
    except Exception as e:
        raise InvalidRequestError(f"Invalid global symbol data: {e!s}") from e
    sym = Symbol(name=symbol_name, mptype=request.mptype, data=obj)
    _global_symbols[symbol_name] = sym
    return GlobalSymbolResponse(
        name=sym.name,
        mptype=sym.mptype,
        data=base64.b64encode(encode_value(sym.data)).decode("utf-8"),
    )


@app.get("/api/v1/symbols/{symbol_name}", response_model=GlobalSymbolResponse)
def get_global_symbol(symbol_name: str) -> GlobalSymbolResponse:  # route handler
    sym = _global_symbols.get(symbol_name)
    if not sym:
        raise ResourceNotFound(f"Global symbol '{symbol_name}' not found")
    # Serialize using Value envelope
    return GlobalSymbolResponse(
        name=sym.name,
        mptype=sym.mptype,
        data=base64.b64encode(encode_value(sym.data)).decode("utf-8"),
    )


@app.get("/api/v1/symbols")
def list_global_symbols() -> dict[str, list[str]]:
    return {"symbols": list(_global_symbols.keys())}


@app.delete("/api/v1/symbols/{symbol_name}")
def delete_global_symbol(symbol_name: str) -> dict[str, str]:  # route handler
    if symbol_name in _global_symbols:
        del _global_symbols[symbol_name]
        return {"message": f"Global symbol '{symbol_name}' deleted successfully"}
    else:
        raise ResourceNotFound(f"Global symbol '{symbol_name}' not found")


# Communication endpoints
# TODO(jint) this should be computation level, add multi computation parallel support.
@app.put("/sessions/{session_name}/comm/{key}/from/{from_rank}")
def comm_send(
    session_name: str, key: str, from_rank: int, request: CommSendRequest
) -> dict[str, str]:
    """
    Receive a message from another party and deliver it to the session's communicator.
    This endpoint runs on the receiver's server.
    """
    sess = _sessions.get(session_name)
    if not sess or not sess.communicator:
        logger.error(f"Session or communicator not found: session={session_name}")
        raise HTTPException(status_code=404, detail="Session or communicator not found")

    # The receiver rank should be the rank of the server hosting this endpoint
    # We don't need to validate to_rank since the request is coming to this server

    # Use the proper onSent mechanism from CommunicatorBase
    sess.communicator.onSent(from_rank, key, request.data)
    return {"status": "ok"}
