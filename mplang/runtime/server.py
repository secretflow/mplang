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

import cloudpickle as pickle
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mplang.backend.base import KernelContext
from mplang.core.mpir import Reader
from mplang.core.table import TableType
from mplang.core.tensor import TensorType
from mplang.protos.v1alpha1 import mpir_pb2
from mplang.runtime import resource
from mplang.runtime.data_providers import DataProvider, ResolvedURI, register_provider
from mplang.runtime.exceptions import InvalidRequestError, ResourceNotFound

logger = logging.getLogger(__name__)

app = FastAPI()


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
        sym = resource.get_global_symbol(name)
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
        try:
            data_b64 = base64.b64encode(pickle.dumps(value)).decode("utf-8")
        except Exception as e:  # pragma: no cover - defensive
            raise InvalidRequestError(
                f"Failed to encode value for symbols:// write: {e!s}"
            ) from e

        resource.create_global_symbol(name, {}, data_b64)


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
    endpoints: list[str]
    # SPU related
    spu_mask: int
    spu_protocol: str
    spu_field: str


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
    data: str  # Base64 encoded data


class SymbolResponse(BaseModel):
    name: str
    mptype: dict
    data: str


class CommSendRequest(BaseModel):
    data: str  # Base64 encoded data


# Response Models for enhanced status
class SessionListResponse(BaseModel):
    sessions: list[str]


class ComputationListResponse(BaseModel):
    computations: list[str]


class GlobalSymbolResponse(BaseModel):
    name: str
    mptype: dict
    data: str


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# List all sessions
@app.get("/sessions", response_model=SessionListResponse)
def list_sessions() -> SessionListResponse:
    """List all session names."""
    return SessionListResponse(sessions=resource.list_all_sessions())


# List all computations in a session
@app.get(
    "/sessions/{session_name}/computations", response_model=ComputationListResponse
)
def list_session_computations(session_name: str) -> ComputationListResponse:
    """List all computation names in a session."""
    session = resource.get_session(session_name)
    if not session:
        raise ResourceNotFound(f"Session '{session_name}' not found")
    return ComputationListResponse(computations=list(session.computations.keys()))


# Session endpoints
@app.put("/sessions/{session_name}", response_model=SessionResponse)
def create_session(session_name: str, request: CreateSessionRequest) -> SessionResponse:
    validate_name(session_name, "session")
    session = resource.create_session(
        name=session_name,
        rank=request.rank,
        endpoints=request.endpoints,
        spu_mask=request.spu_mask,
        spu_protocol=request.spu_protocol,
        spu_field=request.spu_field,
    )
    return SessionResponse(name=session.name)


@app.get("/sessions/{session_name}", response_model=SessionResponse)
def get_session(session_name: str) -> SessionResponse:
    session = resource.get_session(session_name)
    if not session:
        raise ResourceNotFound(f"Session '{session_name}' not found")
    return SessionResponse(name=session.name)


@app.delete("/sessions/{session_name}")
def delete_session(session_name: str) -> dict[str, str]:
    """Delete a session and all its associated resources."""
    if resource.delete_session(session_name):
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

    reader = Reader()
    expr = reader.loads(graph_proto)

    if expr is None:
        raise InvalidRequestError("Failed to parse expression from protobuf")

    # Create the computation resource
    computation = resource.create_computation(session_name, computation_id, expr)
    # Execute with input/output names
    resource.execute_computation(
        session_name, computation.name, request.input_names, request.output_names
    )
    return ComputationResponse(name=computation.name)


@app.delete("/sessions/{session_name}/computations/{computation_id}")
def delete_computation(session_name: str, computation_id: str) -> dict[str, str]:
    """Delete a specific computation."""
    if resource.delete_computation(session_name, computation_id):
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
    symbol = resource.create_symbol(
        session_name, symbol_name, request.mptype, request.data
    )
    # Return the base64 data back to client; server stores Python object
    return SymbolResponse(
        name=symbol.name,
        mptype=symbol.mptype,
        data=request.data,
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

        symbol = resource.get_symbol(session_name, symbol_name)
        if not symbol:
            raise HTTPException(
                status_code=404, detail=f"Symbol {symbol_name} not found"
            )

        data_bytes = pickle.dumps(symbol.data)
        data_b64 = base64.b64encode(data_bytes).decode("utf-8")

        return SymbolResponse(
            name=symbol.name,
            mptype=symbol.mptype,
            data=data_b64,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/sessions/{session_name}/symbols")
def list_session_symbols(session_name: str) -> dict[str, list[str]]:
    """List all symbols in a session."""
    symbols = resource.list_symbols(session_name)
    return {"symbols": symbols}


@app.delete("/sessions/{session_name}/symbols/{symbol_name}")
def delete_symbol(session_name: str, symbol_name: str) -> dict[str, str]:
    """Delete a specific symbol."""
    if resource.delete_symbol(session_name, symbol_name):
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
    sym = resource.create_global_symbol(symbol_name, request.mptype, request.data)
    return GlobalSymbolResponse(name=sym.name, mptype=sym.mptype, data=request.data)


@app.get("/api/v1/symbols/{symbol_name}", response_model=GlobalSymbolResponse)
def get_global_symbol(symbol_name: str) -> GlobalSymbolResponse:
    sym = resource.get_global_symbol(symbol_name)
    if not sym:
        raise ResourceNotFound(f"Global symbol '{symbol_name}' not found")
    data_bytes = pickle.dumps(sym.data)
    data_b64 = base64.b64encode(data_bytes).decode("utf-8")
    return GlobalSymbolResponse(name=sym.name, mptype=sym.mptype, data=data_b64)


@app.get("/api/v1/symbols")
def list_global_symbols() -> dict[str, list[str]]:
    return {"symbols": resource.list_global_symbols()}


@app.delete("/api/v1/symbols/{symbol_name}")
def delete_global_symbol(symbol_name: str) -> dict[str, str]:
    if resource.delete_global_symbol(symbol_name):
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
    session = resource.get_session(session_name)
    if not session or not session.communicator:
        logger.error(f"Session or communicator not found: session={session_name}")
        raise HTTPException(status_code=404, detail="Session or communicator not found")

    # The receiver rank should be the rank of the server hosting this endpoint
    # We don't need to validate to_rank since the request is coming to this server

    # Use the proper onSent mechanism from CommunicatorBase
    session.communicator.onSent(from_rank, key, request.data)
    return {"status": "ok"}
