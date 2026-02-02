# MPLang EDSL - Embedded Domain-Specific Language

The EDSL (Embedded Domain-Specific Language) module is the core infrastructure for MPLang's graph-based IR system. It provides the foundational components for tracing Python functions into SSA-based graphs, type checking, and program compilation.

## Overview

MPLang EDSL uses a modern **Operation List + SSA** approach for better analyzability and optimization, similar to PyTorch FX and JAX. The EDSL captures Python function execution into an explicit graph IR that can be analyzed, optimized, and executed across distributed parties and devices.

## Architecture

### Key Components

```
mplang/edsl/
├── typing.py         # Unified type system (MPType hierarchy)
├── graph.py          # IR: Operation List + SSA (Graph, Operation, Value)
├── primitive.py      # Primitive abstraction and registration
├── object.py         # Object hierarchy (TraceObject, runtime values)
├── context.py        # Context management (tracing vs execution)
├── tracer.py         # Explicit tracer for graph construction
├── jit.py            # @jit decorator for function compilation
├── program.py        # Compiled program representation
├── printer.py        # Graph visualization and debugging
├── registry.py       # Op registry and dialect management
└── serde.py          # Serialization/deserialization
```

### Type System

**Single source of truth**: `mplang.edsl.typing.MPType`

The EDSL provides a unified, extensible type system that supports:

- **ScalarType**: `i32`, `i64`, `f32`, `f64`, `bool`, etc.
- **TensorType**: Multi-dimensional arrays with shape and dtype
- **TableType**: Structured tabular data
- **VectorType**: Encrypted vectors (for FHE/MPC)
- **SSType**: Secret-shared values (for MPC)
- **CustomType**: Extensible type system for domain-specific types

```python
from mplang.edsl.typing import TensorType, VectorType, f32, i64

# Define types
plaintext: TensorType = TensorType(f32, (4096,))
ciphertext: VectorType = VectorType(f32, 4096)
counter: TensorType = TensorType(i64, ())  # scalar
```

### Graph IR

The Graph IR uses SSA (Static Single Assignment) form with explicit operations:

```python
# Example graph structure:
# %0 = input "x"
# %1 = input "y"
# %2 = add %0, %1
# return %2
```

Each operation has:
- **opcode**: Operation type (e.g., "add", "mul", "simp.pcall")
- **inputs**: List of input Values (SSA variables)
- **outputs**: List of output Values
- **attributes**: Operation-specific metadata

### Tracing and Compilation

```python
from mplang.edsl import jit, trace, Tracer

# Method 1: @jit decorator (automatic tracing and execution)
@jit
def my_program(x, y):
    return x + y

result = my_program(data_x, data_y)

# Method 2: Explicit tracing (returns TracedFunction with .graph)
traced_fn = trace(my_program, x_obj, y_obj)
graph = traced_fn.graph

# Method 3: Manual tracing with Tracer context
tracer = Tracer()
with tracer:
    result = my_program(x, y)
graph = tracer.finalize(result)
```

## API Layers

The EDSL provides two distinct API layers:

### 1. High-Level API (User-Facing)

- **Tracing**: `@jit`, `trace()`, `Tracer` context manager
- **Primitives**: `@primitive` decorator for defining new operations
- **Types**: Type annotations and type inference
- **Objects**: Automatic wrapping of Python values

This is the **primary interface for users**.

### 2. Low-Level API (Compiler/Backend)

- **Graph manipulation**: Direct `Graph.add_op()` calls
- **Op registry**: `register_impl()`, `get_impl()`
- **Serialization**: Graph to/from protobuf
- **Execution**: Direct graph interpretation via runtime

Used by compiler passes and backend implementations.

## Integration with Dialects

The EDSL is dialect-agnostic. Dialects provide domain-specific operations:

- **mplang.dialects.simp**: SPMD/MPI-style operations (pcall_static, pcall_dynamic)
- **mplang.dialects.tensor**: Tensor operations (run_jax for JAX-backed computation, structural ops)
- **mplang.dialects.table**: Table operations (run_sql, read/write, conversions)
- **mplang.dialects.spu**: Secure multi-party computation
- **mplang.dialects.tee**: Trusted execution environment
- **mplang.dialects.bfv**: Homomorphic encryption (BFV scheme)
- **mplang.dialects.phe**: Paillier homomorphic encryption

Each dialect registers its operations and type implementations with the EDSL.

## Examples

### Basic Tracing

```python
import mplang.edsl as el
from mplang.dialects.simp import pcall_static

@el.jit
def distribute_computation(x, y):
    # Execute computation on parties 0 and 1
    result = pcall_static((0, 1), lambda a, b: a + b, x, y)
    return result
```

### Custom Primitives

```python
from mplang.edsl import primitive
from mplang.edsl.typing import TensorType, f32

@primitive
def custom_op(x: TensorType, y: TensorType) -> TensorType:
    # Implementation for tracing/execution
    # This function is traced into the graph
    return x * 2 + y
```

### Graph Inspection

```python
from mplang.edsl import trace, format_graph
import mplang.edsl as el

def my_fn(x):
    return x * 2 + 1

# Trace the function
traced_fn = trace(my_fn, x_obj)

# Print graph IR
print(format_graph(traced_fn.graph))
```

## Testing

The EDSL has comprehensive test coverage:

```bash
# Run all EDSL tests
uv run pytest tests/edsl/

# Run specific test files
uv run pytest tests/edsl/test_tracer.py
uv run pytest tests/edsl/test_typing.py
uv run pytest tests/edsl/test_graph.py
```

Test files:
- `test_typing.py`: Type system tests
- `test_graph.py`: Graph IR tests
- `test_tracer.py`: Tracing functionality
- `test_primitive.py`: Primitive operations
- `test_context.py`: Context management
- `test_printer.py`: Graph visualization
- `test_serde.py`: Serialization/deserialization
- `test_compiled_program_artifact.py`: Program compilation

## Development

### Adding New Operations

1. Define the operation in the appropriate dialect (e.g., `mplang/dialects/tensor.py`)
2. Register the operation with `@primitive` or explicit registry
3. Implement backend execution in `mplang/backends/`
4. Add tests in `tests/dialects/` and `tests/backends/`

### Type System Extension

To add a new type:

1. Subclass `MPType` in `mplang/edsl/typing.py`
2. Implement required methods (`__repr__`, `__eq__`, `__hash__`)
3. Add serialization support in `typing.py` and `serde.py`
4. Add tests in `tests/edsl/test_typing.py`

### Code Style

```bash
# Format code
uv run ruff format mplang/edsl/

# Lint
uv run ruff check mplang/edsl/ --fix

# Type check
uv run mypy mplang/edsl/
```

## Relationship with Other Components

```
mplang/
├── edsl/              # Core IR and tracing (this module)
├── dialects/          # Domain-specific operations
├── backends/          # Execution implementations
├── runtime/           # Runtime execution engine
├── libs/              # High-level libraries (device, ml, mpc)
└── kernels/           # Low-level kernel implementations
```

**Design documents** are in the project root `design/` directory:
- [`design/architecture.md`](../../design/architecture.md): Overall architecture
- [`design/control_flow.md`](../../design/control_flow.md): Control flow handling
- [`design/compile_execute_decoupling.md`](../../design/compile_execute_decoupling.md): Compilation model

## References

- **MPLang Documentation**: See repository root README.md and AGENTS.md
- **Design Documents**: `/design/` directory
- **PyTorch FX**: https://pytorch.org/docs/stable/fx.html
- **JAX jaxpr**: https://jax.readthedocs.io/en/latest/jaxpr.html
- **MLIR**: https://mlir.llvm.org/

---

**Last Updated**: 2026-02-02
