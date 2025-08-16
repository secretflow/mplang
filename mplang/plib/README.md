# plib - Party Library

This module serves as the bridge between high-level mplang functions and
distributed execution environments. It provides a two-stage architecture:
**compilers** that transform functions into portable, serializable formats, and
**handlers** that execute these serialized functions on individual parties in a
multi-party computation system.

The design enables heterogeneous execution where different parties can use
different programming languages or runtime environments, as long as they can
interpret the standardized serialized formats.

## Module Structure

### Compiler Modules

- `spu_fe.py` - SpuFrontend class with compile method for transforming JAX
  functions to SPU executables.
- `jax_cc.py` - Compiler for transforming JAX functions to StableHLO MLIR.

### Handler Modules

- `spu_handler.py` - Execution handler for SPU executables in secure multi-party
  computation.
- `stablehlo_handler.py` - Execution handler for StableHLO MLIR in remote
  execution.

### I/O Modules

- `spu_fe.py` - SPU Frontend for input/output operations for share generation and
  reconstruction.

## Design Philosophy

This module follows a compilation-execution separation principle:

1. **Compilation Phase**: Transform high-level functions (e.g., JAX functions)
    into serializable intermediate representations.
    - Output formats include SPU protobuf and StableHLO MLIR.
    - Compilation results can be transmitted across languages and platforms.

2. **Execution Phase**: Execute serialized functions independently on each party.
    - Handlers are responsible for deserializing and executing compiled
      functions.
    - Support different execution environments (local, remote, secure
      computation, etc.).
    - Handler implementations can be multi-language, not limited to Python.

## Extensibility

In theory, handlers can be implemented in any language, as long as they can:

- Parse serialized function representations.
- Provide corresponding execution environments.
- Return results that conform to interface specifications.

This design enables mplang to support various execution backends and deployment
scenarios.
