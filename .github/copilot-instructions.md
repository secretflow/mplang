# Copilot instructions

This file provides guidance to GitHub Copilot when working with code in this repository.

## Project Overview

MPLang (Multi-Party Programming Language) is a single-controller programming library for secure multi-party computation (MPC) and multi-device workloads. It follows the SPMD (Single Program, Multiple Data) model where one Python program orchestrates multiple parties and devices with explicit security domains.

### Key Features
- Single-controller SPMD: One program orchestrates multiple parties in lockstep
- Explicit devices and security domains: Clear annotations for P0/P1/SPU
- Function-level compilation (@mplang.function): Enables graph optimizations and audit
- Pluggable frontends and backends: Supports JAX, Ibis frontends and StableHLO, SPU PPHLO IR backends

## Development Commands

### Installation and Setup
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install from source
uv pip install .

# Editable install for development
uv pip install -e .

# Install development dependencies
uv sync --group dev
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/core/test_primitive.py

# Run tests with coverage
uv run pytest --cov=mplang

# Run tests in parallel
uv run pytest -n auto
```

### Code Quality and Type Checking
```bash
# Run linter
uv run ruff check .

# Fix lint issues automatically
uv run ruff check . --fix

# Format code
uv run ruff format .

# Run type checker
uv run mypy mplang/
```

### Running Examples
```bash
# Run tutorials
uv run tutorials/0_basic.py

# Run other examples
uv run tutorials/1_condition.py
uv run tutorials/2_whileloop.py
```

### Protobuf (Buf)

```bash
# Install buf by following the official guide:
# https://buf.build/docs/installation

# Format .proto files (rewrites in place)
buf format -w

# Lint .proto files
buf lint

# Regenerate code from protos using buf.gen.yaml
buf generate

# Check for breaking API changes vs. main branch
buf breaking --against '.git#branch=main'

# Update dependencies in buf.lock
buf dep update
```

## Architecture Overview

### Core Components

1. **Core System (`mplang/core/`)**
   - `mpobject.py`: Base MPObject and MPContext classes
   - `primitive.py`: Fundamental primitive operations with @primitive decorator
   - `trace.py`: TraceContext and TraceVar for lazy evaluation and expression building
   - `interp.py`: InterpContext and InterpVar for runtime interpretation

2. **Expression System (`mplang/expr/`)**
   - AST representation of computation graphs
   - Expression evaluation and transformation

3. **Runtime System (`mplang/runtime/`)**
   - `simulation.py`: Simulator for local testing with multiple threads
   - Communication layers for multi-party execution

4. **Frontends and Backends**
   - Frontends (`mplang/frontend/`): JAX, Ibis integration
   - Backends (`mplang/backend/`): SPU, StableHLO, SQL/DuckDB implementations

5. **Devices (`mplang/device.py`)**
   - Device-oriented programming interface
   - Automatic data transformation between devices

### Key Design Patterns

1. **Context Management**: Uses context managers for switching between trace and interpretation contexts
2. **Lazy Evaluation**: Computations are traced into expressions rather than executed immediately
3. **SPMD Model**: Single Program, Multiple Data execution model for multi-party coordination
4. **Trace and Interpret**: Functions can be traced to build computation graphs or interpreted for immediate execution

### Important APIs

- `@mplang.function`: Decorator for tracing functions into computation graphs
- `mplang.compile()`: Compile functions into traced representations
- `mplang.evaluate()`: Execute traced functions in interpreter contexts
- `mplang.fetch()`: Retrieve results from interpreter contexts
- `mplang.Simulator()`: Create simulation environments for testing

### Testing Approach

Tests are organized by module and use pytest. Key patterns:
- Use TraceContext for testing primitive operations
- Trace functions to create TracedFunction objects
- Verify expression output using Printer for IR inspection
- Test both compile-time and runtime behavior
