# HTTP Backend Single-Process-Per-Party Architecture

## Overview

The HTTP backend implements a distributed multi-party computation architecture where each party runs as a single process containing both an HTTP server and communication logic. This design ensures proper state sharing and efficient inter-party communication.

## Core Architecture Principles

### Single-Process-Per-Party Model

Each party in the multi-party computation runs as **one process** that contains:
- An embedded HTTP server (FastAPI)
- A communicator instance (HttpCommunicator)  
- Session and resource management
- Party-specific computation logic

This differs from a multi-process approach where servers and communicators run in separate processes, which would prevent proper state sharing.

### Key Components

```
┌─────────────────────────────────────────┐
│             Party Process               │
│  ┌─────────────────────────────────────┐ │
│  │        HTTP Server (FastAPI)       │ │
│  │  - Health endpoints                 │ │
│  │  - Session management              │ │
│  │  - Communication endpoints         │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │      HttpCommunicator               │ │
│  │  - send() via HTTP POST            │ │
│  │  - recv() via onSent callbacks     │ │
│  │  - Inherits from CommunicatorBase  │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │      Resource Manager              │ │
│  │  - Session storage                 │ │
│  │  - Symbol management               │ │
│  │  - Global state (_sessions)        │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Communication Flow

### Message Sending Process

1. **Party A** calls `communicator.send(to=B, key="msg", data=payload)`
2. **HttpCommunicator** serializes data to Base64-encoded JSON
3. **HTTP POST** request sent to `http://party-b-host/sessions/{session}/comm/send`
4. **Party B's HTTP server** receives the request
5. **Server** calls `onSent(from_rank=A, key="msg", data=payload)` 
6. **onSent callback** triggers any waiting `recv()` calls in Party B

### Message Receiving Process

1. **Party B** calls `communicator.recv(frm=A, key="msg")`
2. **CommunicatorBase** waits for the key to be available
3. When **onSent** is called by the HTTP server, data becomes available
4. **recv()** returns the deserialized data

## Core Assumptions

### Process Architecture
- **One process per party**: Each party must run both server and communicator in the same process
- **Shared state**: The communicator instance and HTTP server must share the same memory space
- **Global variables**: `_sessions` and other global state are per-process, not per-thread

### Communication Model
- **Bidirectional HTTP**: All communication uses HTTP POST requests between parties
- **Asynchronous delivery**: Messages are delivered via callback mechanism (onSent)
- **Key-based matching**: Messages are matched by `(from_rank, to_rank, key)` tuples
- **Reliable delivery**: HTTP ensures message delivery and provides error handling

### Session Management
- **Session scope**: Each session creates its own communicator instance
- **Rank assignment**: Each party has a unique rank (0, 1, 2, ...) in the session
- **Endpoint mapping**: All parties must know all other parties' HTTP endpoints

## Implementation Details

### HttpCommunicator Class
```python
class HttpCommunicator(CommunicatorBase):
    def send(self, to: int, key: str, data: Any) -> None:
        # Serialize data and send HTTP POST to target party
        
    def recv(self, frm: int, key: str) -> Any:
        # Wait for onSent callback to provide data
```

### HTTP Server Endpoints
- `GET /health` - Health check
- `POST /sessions` - Create new session
- `GET /sessions/{session_name}` - Get session info
- `POST /sessions/{session_name}/comm/send` - Inter-party communication

### Resource Management
```python
# Global per-process storage
_sessions: dict[str, Session] = {}
_computations: dict[str, Computation] = {}
_symbols: dict[str, Symbol] = {}
```

## Usage Example

### Starting Parties
```bash
# Terminal 1: Start Party 0
python party_script.py --rank=0 --port=8000

# Terminal 2: Start Party 1  
python party_script.py --rank=1 --port=8001
```

### Basic Communication
```python
# In Party 0 process
communicator.send(to=1, key="hello", data={"message": "Hello Party 1"})
response = communicator.recv(frm=1, key="response")

# In Party 1 process  
message = communicator.recv(frm=0, key="hello")
communicator.send(to=0, key="response", data={"status": "received"})
```

## Testing Architecture

The test suite validates this architecture using:
- **Multiprocessing**: Each test party runs in its own process
- **Process isolation**: Ensures proper separation of global state
- **End-to-end validation**: Full communication round-trips are tested

See `tests/runtime/http_backend/test_communicator.py::test_end_to_end_communication` for the reference implementation.

## Benefits

1. **State Consistency**: Shared memory ensures communicator and server state alignment
2. **Simplified Debugging**: All party logic runs in one process, easier to trace
3. **Reliable Communication**: HTTP provides built-in reliability and error handling
4. **Scalable**: Each party can run on different machines with network connectivity
5. **Standard Protocols**: Uses REST APIs for inter-party communication

## Constraints

1. **Process Requirement**: Must use multiprocessing, not multithreading for proper isolation
2. **Network Dependency**: Requires network connectivity between all parties
3. **Port Management**: Each party needs its own HTTP port
4. **Synchronization**: Message ordering depends on application-level coordination

## Error Handling

- **HTTP Errors**: Network failures result in `OSError` exceptions
- **Rank Validation**: Server validates message ranks match session configuration
- **Timeout Handling**: Configurable timeouts for HTTP requests and recv operations
- **Resource Cleanup**: Proper cleanup of sessions, computations, and symbols
