# SIMP Backend Architecture

## Overview

SIMP (SPMD Multi-Party) is the execution layer for MPLang2's distributed multi-party computations. Unlike local dialects (e.g., `bfv`, `tensor`) that execute entirely on a single machine, SIMP splits execution between a **Host** (coordinator) and multiple **Workers** (party nodes).

## Design Principles

1. **Host/Worker Separation**: Host coordinates execution; Workers perform actual computation
2. **Driver Abstraction**: Different deployment modes share the same Worker logic
3. **Explicit Registration**: Operation implementations are registered in a dedicated module
4. **Transport Independence**: Worker logic is decoupled from communication transport

## Module Dependency Graph

```text
                    simp_host.py (base class)
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   simp_simulator.py              simp_http_driver.py
          │                             │
          │ (embedded)                  │ (HTTP calls)
          ▼                             ▼
   simp_worker.py  ◄───────────  simp_http_worker.py
          │                             │
          └──────────┬──────────────────┘
                     ▼
              simp_impl.py (op registration)
```

## Module Structure

```text
mplang2/backends/
├── simp_host.py           # Base classes: SimpHost, HostVar
├── simp_worker.py         # WorkerInterpreter class (no impl registration)
├── simp_impl.py           # All simp op implementations (def_impl)
│
├── simp_simulator.py      # SimpSimulator driver (local threading)
│
├── simp_http_driver.py    # SimpHttpDriver (HTTP-based distributed)
└── simp_http_worker.py    # HTTP worker entry point (FastAPI app)
```

## Component Responsibilities

### simp_host.py

- `SimpHost`: Abstract base class for all SIMP drivers
- `HostVar`: Runtime container holding values across all parties
- Defines `_submit()` / `_collect()` interface for driver implementations

### simp_worker.py

- `WorkerInterpreter`: Interpreter running on a single party
- Holds rank, world_size, and communicator reference
- Interprets Graph IR by dispatching to registered implementations

### simp_impl.py

- Registers all `simp.*` operation implementations via `@primitive.def_impl`
- Implementations assume they run inside `WorkerInterpreter`
- Imported by worker entry points to trigger registration

### simp_simulator.py

- `SimpSimulator`: Driver for local multi-threaded simulation
- `ThreadCommunicator`: In-memory inter-thread communication
- Creates `WorkerInterpreter` instances directly in threads

### simp_http_driver.py

- `SimpHttpDriver`: Driver that coordinates remote HTTP workers
- Sends serialized Graph + inputs to worker endpoints
- Collects results via HTTP responses

### simp_http_worker.py

- `HttpCommunicator`: HTTP-based inter-worker communication
- `create_worker_app()`: Factory for FastAPI application
- Standalone entry point for distributed deployment

## Execution Flow

### Simulation Mode

```text
User Code
    │
    ▼
SimpSimulator.evaluate_graph(graph, inputs)
    │
    ├── ThreadPool.submit(worker[0], graph, inputs[0])
    ├── ThreadPool.submit(worker[1], graph, inputs[1])
    └── ...
    │
    ▼ (in each thread)
WorkerInterpreter.evaluate_graph(graph, party_inputs)
    │
    ▼
simp_impl: dispatch based on opcode
    │
    ▼
HostVar(results)  ← collected from all workers
```

### Distributed Mode (HTTP)

```text
User Code
    │
    ▼
SimpHttpDriver.evaluate_graph(graph, inputs)
    │
    ├── HTTP POST /exec → Worker[0]
    ├── HTTP POST /exec → Worker[1]
    └── ...
    │
    ▼ (on each worker node)
FastAPI /exec endpoint
    │
    ▼
WorkerInterpreter.evaluate_graph(graph, party_inputs)
    │
    ▼
simp_impl: dispatch based on opcode
    │
    ▼
HTTP Response → collected by driver
```

## Key Operations

| Operation | Description | Worker Behavior |
|-----------|-------------|-----------------|
| `pcall_static` | Execute on explicit parties | Check rank ∈ parties, execute region or return None |
| `pcall_dynamic` | Execute on all parties | Always execute region |
| `shuffle_static` | Compile-time routing | Send/recv based on routing dict |
| `converge` | Merge disjoint partitions | Return first non-None input |
| `uniform_cond` | Uniform conditional | Evaluate predicate, execute selected branch |
| `while_loop` | SPMD while loop | Loop until condition is false |

## Communication Protocol

Workers communicate via the `communicator` interface:

- `send(to_rank, key, data)`: Send data to another worker
- `recv(from_rank, key)`: Receive data (blocking)

The `key` is derived from the operation name to ensure uniqueness within a graph execution.

## Adding New Drivers

To add a new transport (e.g., gRPC):

1. Create `simp_grpc_driver.py`:
   - Subclass `SimpHost`
   - Implement `_submit()` and `_collect()`

2. Create `simp_grpc_worker.py`:
   - Implement gRPC-compatible communicator
   - Create service that wraps `WorkerInterpreter`
   - Import `simp_impl` to register implementations

## Future Considerations

1. **Host-side Control Flow**: For `uniform_cond` and `while_loop`, the host may need to coordinate branch selection across workers
2. **Layered Registry**: Separate Host vs Worker implementation registries to prevent misuse
3. **Abstract Interpretation**: Host-side type/mask propagation for optimization
