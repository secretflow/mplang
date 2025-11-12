# MPLang Core2: New Core System

This directory contains the new core runtime system built on top of `mplang/edsl/`.

## Architecture Overview

```
mplang/edsl/              # EDSL Layer (Pure IR)
  ├── typing.py           # Type System (BaseType, Tensor, HE, SS, MP)
  ├── graph.py            # Graph IR (Value, Operation, Graph)
  └── builder.py          # GraphBuilder (High-level DSL)

mplang/core2/             # Runtime Layer (This directory)
  ├── object.py           # Object Hierarchy (TraceObject/InterpObject)
  ├── tracer.py           # Tracer (Python → Graph)
  ├── jit.py              # @jit decorator
  ├── interp.py           # GraphInterpreter + Interpreter (execute ops)
  └── context.py          # ExecutionContext + Context base class
```

## Key Concepts

### Object Hierarchy

The runtime uses an Object Hierarchy to distinguish between trace-time and interp-time execution:

- **`Object`**: Abstract base class for all runtime objects
- **`TraceObject`**: Trace-time object (holds `graph.Value`, operations recorded to Graph)
- **`InterpObject`**: Interp-time object (holds **any** runtime object, operations executed immediately)

**Important**: `InterpObject` is **backend-agnostic**. It holds different runtime objects depending on the backend:

- **FHE backend**: Local ciphertext objects (e.g., TenSEAL/SEAL ciphertexts)
- **JAX backend**: Local `jax.Array` objects
- **MP backend**: Backend handles (Driver-side handles to party-side data)
- **SQL backend**: DatabaseHandle objects
- **etc.**

### Tracer

The `Tracer` (inherits from `Context`) converts Python functions to Graph IR, handling:
- Function parameters (as graph inputs)
- Captured variables (external references promoted to graph inputs)
- Polymorphic handling of TraceObject/InterpObject
- Implements `execute_add()` by recording to Graph

### Interpreter

The `Interpreter` (inherits from `Context`) executes operations immediately:
- Implements `execute_add()` by executing on runtime objects
- Backend-agnostic execution (delegates to runtime_obj's operators)
- Used for eager execution mode

### JIT Compilation

The `@jit` decorator:
1. Traces the function to Graph IR on first call
2. Caches the Graph
3. Executes the cached Graph on subsequent calls

### Execution Modes

- **Eager Mode**: InterpObject operations execute immediately (RPC to parties)
- **Tracing Mode**: TraceObject operations are recorded into Graph IR

## Design Principles

1. **Separation of Concerns**: EDSL layer (typing, graph) is pure and isolated from runtime concerns
2. **Object Hierarchy for Polymorphism**: TraceObject/InterpObject provide uniform interfaces
3. **No Modification to Legacy Code**: `mplang/core/` remains untouched for gradual migration
4. **Clear Layering**: Type → IR → Runtime → Execution

## Example Usage

```python
from mplang.core2 import jit, trace
from mplang.core2.object import InterpObject

# Example 1: FHE backend (local execution)
import tenseal as ts  # Hypothetical FHE library

context = ts.context(...)
ciphertext1 = ts.encrypt(context, [1, 2, 3])  # Local ciphertext
ciphertext2 = ts.encrypt(context, [4, 5, 6])

x = InterpObject(ciphertext1, Tensor[f32, (3,)])  # Wrap in InterpObject
y = InterpObject(ciphertext2, Tensor[f32, (3,)])
z = x + y  # Executes locally on FHE backend

# Example 2: JAX backend (local array execution)
import jax.numpy as jnp

arr1 = jnp.array([1.0, 2.0, 3.0])
arr2 = jnp.array([4.0, 5.0, 6.0])

x = InterpObject(arr1, Tensor[f32, (3,)])
y = InterpObject(arr2, Tensor[f32, (3,)])
z = x + y  # Executes on JAX arrays

# JIT compilation (works for any backend)
@jit
def compute(a, b):
    return a + b  # Traced on first call

result = compute(x, y)  # First call: trace, subsequent: execute cached graph

# Manual tracing
graph = trace(lambda x, y: x + y, x, y)
print(graph)
```

## TODO

- [ ] Implement `GraphInterpreter.run()`
- [ ] Implement `ExecutionContext.executor`
- [ ] Add more operators to Object (`__mul__`, `__sub__`, `__matmul__`, etc.)
- [ ] Add support for control flow (cond, while_loop)
- [ ] Integrate with `mplang/runtime/` for actual RPC execution
- [ ] Add comprehensive tests
- [ ] Performance optimization (graph caching, operator fusion)

## Migration Path

1. **Phase 1**: Complete core2 implementation (object, tracer, jit, interp)
2. **Phase 2**: Add integration tests comparing core vs core2 behavior
3. **Phase 3**: Migrate ops layer to use core2
4. **Phase 4**: Update public API to use core2
5. **Phase 5**: Deprecate and remove core (legacy system)

## References

- [EDSL Architecture RFC](../edsl/README.md)
- [Type System Design](../edsl/typing.py)
- [Graph IR Design](../edsl/graph.py)
