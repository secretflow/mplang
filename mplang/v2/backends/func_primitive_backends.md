# Function Primitive Backends: Design & Implementation

## 1. Context & Motivation

The original `SimpBackend` implementation (specifically `SimpHost`) violated the "Kernel Backend" design principle by overriding `evaluate_graph`. This effectively created a specialized interpreter that hardcoded the execution strategy (distributed SPMD) at the graph level, bypassing the standard primitive dispatch mechanism.

**Issues with Legacy Approach:**
*   **Anti-Pattern**: `evaluate_graph` override prevents composition. Pass-through dispatch logic was mixed with execution logic.
*   **Rigidity**: Hard to mix local eager execution (e.g., debugging, preprocessing) with remote compiled execution.
*   **Maintenance**: Maintaining a separate interpreter loop in `SimpHost` duplicates logic from the standard `Interpreter`.

## 2. Design: "Unified Kernel Pattern"

We adopt a flexible **Kernel Dispatch** model that supports multiple execution strategies for the same IR, depending on which kernels are registered.

### Core Concept
*   **Driver / Interpreter**: The `SimpHost` acts as the coordinator's interpreter context.
*   **Primitives**: All operations, including `func.call` and `simp.pcall`, are primitives.
*   **Kernels**: The behavior is defined by the kernel registered for each primitive.

### Execution Scenarios

#### Scenario 1 & 2: Interactive / Standard Mode (Priority)
Used for debugging, REPL, and fine-grained control.
*   **`func.call`**: Routs to **`StandardFuncImpl`**. It recursively interprets the function's graph node-by-node on the Host.
*   **`simp.pcall`**: Routs to **`SimpHostPCallKernel`**. When the host interpreter encounters this, it sends a command (SPMD) to all workers to execute the body.
*   **`tensor.add`**: Routs to local host execution (e.g., for pre-processing).

#### Scenario 3: AOT / Launch Mode (Optimization)
Used for performance to reduce "chatty" communication.
*   **`func.call`**: Routs to **`SimpHostLaunchKernel`**. Instead of interpreting the graph, it serializes the **entire graph** and sends it to workers.
*   **Worker**: Receives the graph and runs its own local Loop to execute it.

### Decision
We will first implement **Standard Mode (Scenario 1 & 2)**. This ensures maximum flexibility and correctness. AOT Mode can be added later as an optimization by simply registering a different kernel for `func.call`.

```python
# Standard Mode Flow
mp.evaluate(sim, fn)
  -> INTERP: func.call(fn)
     -> KERNEL: recursive_eval(fn.graph)
        -> INTERP: op1 (local) -> executed locally
        -> INTERP: simp.pcall(body)
           -> KERNEL: broadcast(body); wait_results()
```

## 3. Implementation Status

The refactoring has been implemented and verified.

### 3.1. Components Implemented
*   **`mplang.v2.backends.simp_func_kernels.py`**:
    *   `simp_func_call_impl`: Implements the distributed execution logic. Checks if context is `SimpHost`.
    *   `simp_func_def_impl`: Basic implementation for `func.func` (returns the graph).
*   **`mplang.v2.backends.simp_host.py`**:
    *   Removed `evaluate_graph` override.
    *   Added import of `simp_func_kernels` to trigger registration.
*   **`mplang/v2/__init__.py` (`evaluate`)**:
    *   Refactored `evaluate` helper.
    *   Manually constructs a "Launcher Graph" containing a `func.call` op to wrap the user's traced function.
    *   Passes this launcher graph to the interpreter.

### 3.2. Verification Results
*   **`test_simple_guide.py`**: Passed. Confirms basic flow (`mp.compile` -> `mp.evaluate` -> `mp.fetch`) works.
*   **`test_rr22.py`**: Passed. Confirms complex multi-round MPC protocols work correctly under the new kernel-based architecture.

## 4. Future Roadmap

1.  **Refine `func` Dialect**:
    *   Currently, `func.call` tracing (`func.py`) enforces strict checks (e.g., must call a `TraceObject` originating from `func.func`).
    *   We bypassed this in `evaluate` by manually constructing the graph.
    *   **Goal**: Relax `func.py` constraints or introduce a `func.launch` primitive specifically for "simulating" a generic graph.

2.  **Closure Support**:
    *   The current "Launcher Graph" assumes inputs align perfectly.
    *   Better closure handling in `func.func` would allow more Pythonic function passing.

3.  **Clean Up `SimpHost`**:
    *   Continue moving any remaining logic (like `fetch`?) into kernels if possible, or verify if `fetch` is distinctly different (it is, mostly).

## 5. Summary
The "dirty" workspace state reflects the successful transition to this cleaner architecture. The code now adheres to the design principle: **"Backends only register kernels."**

## 6. Backends File Layout (Refactored)

*   **`simp_host.py`** (Host): **Coordinator Interpreter**: Inherits standard `Interpreter`, registers host kernels (like `func.call`).
*   **`simp_func_kernels.py`** (Host): **Host Kernels**: Implements `func.call` (SPMD launch) and `func.func` for `SimpHost`.
*   **`simp_worker.py`** (Worker): **Worker Interpreter**: Executes local kernels on each party.
*   **`simp_impl.py`** (Worker): **Worker Kernels**: Implements `simp.*` ops (e.g., `pcall`, `shuffle`) for `SimpWorker`.
*   **`simp_simulator.py`** (App): **Local Wrapper**: Thread-based simulation entry point (wraps `SimpHost`).
*   **`simp_http_driver.py`** (App): **Remote Wrapper**: HTTP-based driver entry point (wraps `SimpHost`).
*   **`simp_http_worker.py`** (App): **Worker Server**: HTTP server entry point for workers.
*   **`*_impl.py`** (Shared): **Dialect Kernels**: Standard dialect implementations (`tensor_impl`, `crypto_impl`, etc.). |
