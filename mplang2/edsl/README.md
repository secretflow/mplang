# MPLang EDSL - Experimental Architecture

**⚠️ Status**: Experimental / Work in Progress

This directory contains the next-generation EDSL (Embedded Domain-Specific Language) architecture for MPLang.

## Why a New Architecture?

The current `mplang.core` architecture (Expr Tree + @primitive) has served us well, but we're hitting limitations:

1. **Expr Tree** is hard to optimize (visitor pattern, nested structure)
2. **@primitive decorators** hide complexity and limit flexibility
3. **Type system** is split between `mptype.MPType` and `typing.BaseType`
4. **No clear separation** between IR, frontend, and backend

Modern EDSLs (torch.fx, JAX) use **Operation List + SSA** for better analyzability and optimization.

## Goals

### 1. Modern IR (Operation List)

**From** (Expr Tree):
```python
CallExpr(
    func=add,
    args=[VariableExpr("x"), VariableExpr("y")]
)
```

**To** (Operation List):
```python
%0 = input "x"
%1 = input "y"
%2 = add %0, %1
return %2
```

### 2. Unified Type System

**Single source of truth**: `mplang.edsl.typing.MPType`

```python
from mplang2.edsl.typing import Tensor, Vector, MPType, f32

# All types use BaseType
plaintext: MPType = Tensor[f32, (4096,)]
ciphertext: MPType = Vector[f32, 4096]
```

### 3. Explicit Tracing

**Clean context management**:
```python
from mplang2.edsl import Tracer

tracer = Tracer()
with tracer:  # Context manager protocol
    result = my_function(x, y)
graph = tracer.finalize(result)
```

### 4. Extensibility

Easy to add new backends:
- FHE (Fully Homomorphic Encryption)
- TEE (Trusted Execution Environment)
- Custom accelerators

### 5. Layered API Architecture

The EDSL provides two distinct API layers:

1.  **Low-Level API (Graph Manipulation)**:
    - Direct manipulation of the `Graph` IR.
    - Generic `add_op` method (pure graph API, no op semantics).
    - Analogous to MLIR's generic operation construction.
    - Used by compiler passes and backend implementations.

2.  **High-Level API (Tracing)**:
    - Uses `Tracer` + `Primitive` (with `abstract_eval`).
    - Pythonic interface (functions, operators).
    - Automatic type inference and graph construction.
    - The primary interface for users.

## Directory Structure

```
mplang/edsl/
├── __init__.py          # Public API
├── README.md            # This file
│
├── design/              # Design documents
│   ├── architecture.md  # Complete architecture overview
│   ├── type_system.md   # Type system design
│   └── migration.md     # Migration from mplang.core
│
├── typing.py            # ✅ Unified type system
├── graph.py             # ✅ IR: Operation List + SSA
├── primitive.py         # ✅ Primitive abstraction
├── object.py            # ✅ TraceObject/InterpObject
├── context.py           # ✅ Context management
├── tracer.py            # ✅ Explicit tracer
├── interpreter.py       # ✅ Interpreter + GraphInterpreter
└── jit.py               # ✅ @jit decorator
```

## Implementation Status

### ✅ Completed (Phase 1-4)

- [x] Type system (`typing.py`) - 649 lines
- [x] Graph IR (`graph.py`) - 388 lines
- [x] Primitive abstraction (`primitive.py`) - 338 lines
- [x] Object hierarchy (`object.py`) - 153 lines
- [x] Context system (`context.py`) - 117 lines
- [x] Tracer (`tracer.py`) - 201 lines
- [x] Interpreter (`interpreter.py`) - 66 lines
- [x] JIT decorator (`jit.py`) - 42 lines
- [x] Design documents
- [x] **153 tests passing** (140 edsl + 13 core2)

### 🚧 In Progress
- [ ] Integration with existing ops/kernels
- [ ] Migration utilities
- [ ] Performance benchmarks

### ❌ Dropped / Deprecated
- [x] Builder API (`builder.py`) - Integrated into `Tracer`

### 📋 Planned
- [ ] Advanced optimizations
- [ ] More backends (TEE, MPC)

## Quick Start

### Using the New Type System

```python
from mplang2.edsl.typing import Tensor, Vector, CustomType, f32

# Define types
PlaintextVec = Tensor[f32, (4096,)]
CiphertextVec = Vector[f32, 4096]
EncryptionKey = CustomType("EncryptionKey")

# Type annotations
def encrypt(data: PlaintextVec, key: EncryptionKey) -> CiphertextVec:
    ...
```

### Using the Tracer (Graph Construction)

```python
from mplang2.edsl import Tracer
from mplang2.dialects.simp import pcall_static

def my_program(x, y):
    # This function is traced into a Graph
    return pcall_static((0, 1), lambda a, b: a + b, x, y)

tracer = Tracer()
with tracer:
    # Inputs are automatically lifted to TraceObjects
    result = my_program(x, y)

# Finalize graph
graph = tracer.finalize(result)
```

## Design Documents

Detailed design documents are in the `design/` subdirectory:

### 1. [architecture.md](design/architecture.md)

Complete EDSL architecture overview covering:
- Core components (Tracer, Graph)
- Design principles (Closed-World, TracedFunction vs First-Class Functions)
- Control flow handling (Dialect-specific, e.g., `simp.uniform_cond`)
- Comparison with JAX, PyTorch, TensorFlow

### 2. [type_system.md](design/type_system.md)

New type system design:
- Three orthogonal dimensions (Layout, Encryption, Distribution)
- Type composition examples
- Ops writing guide
- Migration strategy

### 3. [migration.md](design/migration.md)

Migration path from `mplang.core` to `mplang.edsl`:
- 6-phase migration plan
- Backward compatibility strategy
- Type conversion utilities

## Relationship with mplang.core

```
mplang/
├── core/           # Stable API (current production)
│   ├── primitive.py
│   ├── tracer.py
│   └── expr/
│
├── edsl/           # Experimental (this directory)
│   ├── typing.py   # Can be used independently
│   ├── graph.py    # Future replacement for core.expr
│   └── tracer.py   # Future replacement for core.tracer
│
├── ops/            # Shared between core and edsl
├── kernels/        # Shared between core and edsl
└── runtime/        # Shared between core and edsl
```

**Migration Strategy**:
1. Develop `edsl` in parallel (no breaking changes to `core`)
2. Gradually move internal code to use `edsl.typing`
3. Add adapters between `core` and `edsl`
4. Deprecate `core` in future major version

## Contributing

We welcome contributions! Since this is experimental:

1. **Read the design docs first**: Understand the architecture
2. **Start small**: Pick a specific component (e.g., Graph IR)
3. **Discuss early**: Open an issue before implementing
4. **Test thoroughly**: Add unit tests for new code

### Development Workflow

```bash
# Install dev dependencies
uv sync --group dev

# Run tests (future)
uv run pytest mplang/edsl/

# Lint
uv run ruff check mplang/edsl/
uv run ruff format mplang/edsl/

# Type check
uv run mypy mplang/edsl/
```

## FAQ

### Q: Should I use `mplang.edsl` in production?

**A**: No, use `mplang.core`. `mplang.edsl` is experimental.

### Q: Can I use `mplang.edsl.typing` independently?

**A**: Yes! The type system is stable and can be used for type annotations.

### Q: When will `edsl` replace `core`?

**A**: No timeline yet. We need to:
1. Complete the implementation
2. Validate performance
3. Migrate all tests
4. Get community feedback

### Q: How can I help?

**A**: Check the implementation status above and pick an unimplemented component. Open an issue to discuss!

## References

- **torch.fx**: https://pytorch.org/docs/stable/fx.html
- **JAX jaxpr**: https://jax.readthedocs.io/en/latest/jaxpr.html
- **MLIR**: https://mlir.llvm.org/

---

**Last Updated**: 2025-01-11
**Maintainers**: MPLang Team
