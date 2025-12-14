# Simp Dialect Backend Design

## Overview

The `simp` (Simple Multi-Party) dialect implements SPMD (Single Program Multiple Data) distributed execution. A single program is written once and executed across multiple parties, with the runtime handling distribution, communication, and synchronization.

## Why Two Implementations?

The simp dialect requires **two separate backend implementations** because the same primitives (`pcall`, `shuffle`, `converge`) have fundamentally different semantics depending on where they execute:

| Primitive | Driver (Host) | Worker |
|-----------|---------------|--------|
| `pcall` | Dispatch work to workers | Execute locally |
| `shuffle` | Route data between workers | Send/Receive via communicator |
| `converge` | Merge HostVars | Pick non-null value |

This is the essence of SPMD: the Driver orchestrates, Workers execute.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      dialects/simp.py                          │
│                    (Primitive definitions)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│     simp_driver/          │   │     simp_worker/          │
│   (Host/Driver side)      │   │   (Worker side)           │
├───────────────────────────┤   ├───────────────────────────┤
│ base.py    SimpDriver     │   │ state.py   SimpWorker     │
│ ops.py     HOST_HANDLERS  │   │ ops.py     WORKER_HANDLERS│
│ values.py  HostVar        │   │                           │
│ mem.py     SimpMemDriver  │   │ mem.py     LocalMesh      │
│ http.py    SimpHttpDriver │   │ http.py    HTTP Server    │
└───────────────────────────┘   └───────────────────────────┘
```

## Directory Structure

```
backends/
├── simp_driver/                    # Driver/Host side
│   ├── __init__.py                 # Exports
│   ├── base.py                     # SimpDriver (abstract base)
│   ├── ops.py                      # HOST_HANDLERS
│   ├── values.py                   # HostVar
│   ├── mem.py                      # MemCluster + SimpMemDriver + make_simulator
│   └── http.py                     # SimpHttpDriver + make_driver
│
├── simp_worker/                    # Worker side
│   ├── __init__.py                 # Exports
│   ├── state.py                    # SimpWorker (DialectState)
│   ├── ops.py                      # WORKER_HANDLERS
│   ├── mem.py                      # LocalMesh + ThreadCommunicator
│   └── http.py                     # HTTP Worker Server
```

## Key Classes

### Driver Side

```python
class SimpDriver(DialectState, ABC):
    """Abstract interface for simp drivers."""
    dialect_name = "simp"
    world_size: int
    
    @abstractmethod
    def submit(self, rank, graph, inputs, job_id=None) -> Future: ...
    @abstractmethod
    def fetch(self, rank, uri) -> Future: ...
    @abstractmethod
    def collect(self, futures) -> list: ...

class SimpMemDriver(SimpDriver):
    """In-memory IPC via ThreadPoolExecutor."""

class SimpHttpDriver(SimpDriver):
    """HTTP IPC via httpx."""
```

### Worker Side

```python
class SimpWorker(DialectState):
    """Worker state with communicator and store."""
    dialect_name = "simp"
    rank: int
    world_size: int
    communicator: Any  # ThreadCommunicator or HTTP client
    store: ObjectStore
```

## IPC Symmetry

| IPC Type | Driver | Worker |
|----------|--------|--------|
| Memory | `simp_driver/mem.py` | `simp_worker/mem.py` |
| HTTP | `simp_driver/http.py` | `simp_worker/http.py` |

## Factory Functions

```python
# Create local simulator (memory IPC)
interp = simp.make_simulator(world_size=3)

# Create remote driver (HTTP IPC)
interp = simp.make_driver(["http://w1:8000", "http://w2:8000"])
```

## Data Flow

```
User Code
    │
    ▼
simp.pcall(parties=(0,1), fn, args)
    │
    ▼ (Driver Interpreter)
HOST_HANDLERS["simp.pcall"]
    │
    ├─► driver.submit(rank=0, graph, inputs)
    └─► driver.submit(rank=1, graph, inputs)
              │
              ▼ (IPC: Memory or HTTP)
         Worker Interpreters
              │
              ▼
    WORKER_HANDLERS["simp.pcall"]
              │
              ▼
         Local Execution
```
