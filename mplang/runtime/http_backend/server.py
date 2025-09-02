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

import cloudpickle as pickle
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mplang.core.mpir import Reader
from mplang.protos.v1alpha1 import mpir_pb2
from mplang.runtime.http_backend import resource
from mplang.runtime.http_backend.exceptions import InvalidRequestError, ResourceNotFound

logger = logging.getLogger(__name__)

app = FastAPI()


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
    name: str
    rank: int
    endpoints: list[str]
    # SPU related
    spu_mask: int = -1
    spu_protocol: int = 0
    spu_field: int = 0


class SessionResponse(BaseModel):
    name: str


class CreateComputationRequest(BaseModel):
    computation_id: str  # name of the computation
    mpprogram: str  # Base64 encoded MPProgram proto
    input_names: list[str]  # Mandatory input symbol names
    output_names: list[str]  # Mandatory output symbol names


class ComputationResponse(BaseModel):
    name: str


class CreateSymbolRequest(BaseModel):
    name: str
    mptype: dict
    data: str  # Base64 encoded data


class CreateComputationSymbolRequest(BaseModel):
    name: str
    mptype: dict
    data: str  # Base64 encoded data


class SymbolResponse(BaseModel):
    name: str
    mptype: dict
    data: str


class CommSendRequest(BaseModel):
    from_rank: int
    to_rank: int
    data: str  # Base64 encoded data
    key: str


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# Session endpoints
@app.post("/sessions", response_model=SessionResponse)
def create_session(request: CreateSessionRequest) -> SessionResponse:
    validate_name(request.name, "session")
    session = resource.create_session(
        name=request.name,
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


# Computation endpoints
@app.post("/sessions/{session_name}/computations", response_model=ComputationResponse)
def create_and_execute_computation(
    session_name: str, request: CreateComputationRequest
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
    computation = resource.create_computation(
        session_name, request.computation_id, expr
    )
    # Execute with input/output names
    resource.execute_computation(
        session_name, computation.name, request.input_names, request.output_names
    )
    return ComputationResponse(name=computation.name)


# Symbol endpoints
@app.post("/sessions/{session_name}/symbols", response_model=SymbolResponse)
def create_session_symbol(
    session_name: str, request: CreateSymbolRequest
) -> SymbolResponse:
    """Create a symbol in a session."""
    try:
        symbol = resource.create_symbol(
            session_name, request.name, request.mptype, request.data
        )
        return SymbolResponse(
            name=symbol.name,
            mptype=symbol.mptype,
            data=symbol.data,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post(
    "/sessions/{session_name}/computations/{computation_name}/symbols",
    response_model=SymbolResponse,
)
def create_computation_symbol(
    session_name: str, computation_name: str, request: CreateComputationSymbolRequest
) -> SymbolResponse:
    """Create a symbol in a computation."""
    try:
        symbol = resource.create_computation_symbol(
            session_name, computation_name, request.name, request.mptype, request.data
        )
        return SymbolResponse(
            name=symbol.name,
            mptype=symbol.mptype,
            data=symbol.data,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


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
    try:
        symbols = resource.list_symbols(session_name)
        return {"symbols": symbols}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# Communication endpoints
@app.post("/sessions/{session_name}/comm/send")
def comm_send(session_name: str, request: CommSendRequest) -> dict[str, str]:
    """
    Receive a message from another party and deliver it to the session's communicator.
    This endpoint runs on the receiver's server.
    """
    session = resource.get_session(session_name)
    if not session or not session.communicator:
        logger.error(f"Session or communicator not found: session={session_name}")
        raise HTTPException(status_code=404, detail="Session or communicator not found")

    # The 'to_rank' should be the rank of the server hosting this endpoint
    if session.communicator.rank != request.to_rank:
        logger.error(
            f"Rank mismatch: session.communicator.rank={session.communicator.rank}, request.to_rank={request.to_rank}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Mismatched rank. Receiver rank is {session.communicator.rank}, but request is for {request.to_rank}",
        )

    # Use the proper onSent mechanism from CommunicatorBase
    session.communicator.onSent(request.from_rank, request.key, request.data)
    return {"status": "ok"}
