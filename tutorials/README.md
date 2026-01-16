# MPLang Tutorials

Welcome to MPLang tutorials! This guide introduces the MPLang v2 API for multi-party computation.

## About MPLang v2

MPLang v2 is the next-generation EDSL (Embedded Domain-Specific Language) for multi-party computation that provides:

- **@mp.function decorator** for compiling Python functions to MPC protocols
- **Device API** with virtual devices (PPU, SPU, TEE) for safe computation
- **JAX integration** for numerical computing and neural networks
- **SQL support** for secure tabular data operations
- **Simulator mode** for development without external dependencies

## Tutorial Structure

The tutorials are organized by topic, starting with basics and advancing to complex applications:

### Getting Started

1. **[00_device_basics.py](00_device_basics.py)** - Device placement, masks, and auto device inference
2. **[01_function_decorator.py](01_function_decorator.py)** - Using @mp.function for compilation
3. **[02_simulation_and_driver.py](02_simulation_and_driver.py)** - Running programs with simulation and real drivers

### Intermediate Topics

4. **[03_run_jax.py](03_run_jax.py)** - JAX integration for numerical computing
5. **[04_ir_dump_and_analysis.py](04_ir_dump_and_analysis.py)** - IR inspection and analysis

### Advanced Topics

6. **[05_run_sql.py](05_run_sql.py)** - Secure SQL operations on tabular data
7. **[06_pipeline.py](06_pipeline.py)** - Building computation pipelines
8. **[07_stax_nn.py](07_stax_nn.py)** - Neural network training with Stax

## Quick Start

Install dependencies:

```bash
uv sync --group dev
uv pip install -e .
```

Run the first tutorial:

```bash
uv run tutorials/00_device_basics.py
```

All tutorials work in simulator mode - no external setup required!

## Core Concepts

- **PPU (Public Processing Unit)**: Plaintext computation on a single party
- **SPU (Secure Processing Unit)**: Secure multi-party computation with secret sharing
- **TEE (Trusted Execution Environment)**: Isolated computation environment
- **Auto device inference**: MPLang automatically infers device placement from context

## Migration from v1

If you're migrating from MPLang v1:

- Change `import mplang` to `import mplang as mp`
- Use `@mp.function` decorator instead of device-specific APIs
- All tutorials now use JAX natively for numerical operations

See [MIGRATION.md](MIGRATION.md) for detailed migration guide.

## Resources

- [Design Docs](../design/): Architecture and technical decisions
- [Examples](../examples/): Full applications (XGBoost, neural networks)
- [API Reference](../mplang/): Core modules and functions
- [Migration Guide](MIGRATION.md): Moving from v1 to v2

Questions? Open an issue on GitHub.
