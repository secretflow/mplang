# Project Overview

This project is `mplang`, a library for Multi-Party Programming Language. Its goal is to provide a single-controller programming model for
writing multi-party programs. `mplang` uses a SIMP (Single Instruction, Multiple Parties) execution model, analogous to SIMT (Single
Instruction, Multiple Threads), where each party executes identical code in an SPMD (Single Program, Multiple Data) fashion. While it is
a key component for secure computation, its programming model is more general.

## Folder Structure

- `mplang/`: Contains the core source code for the `mplang` library.
- `protos/`: Contains the protocol buffer definitions used for communication.
- `examples/`: Contains example usage of the library.
- `tests/`: Contains tests for the library.
- `tutorials/`: Contains tutorials for learning how to use `mplang`.

## Libraries and Frameworks

- `spu`: The underlying secure computation engine.
- `grpcio` & `protobuf`: For remote procedure calls and data serialization.
- `jax`: Used for numerical computation and transformation to StableHLO.
- `numpy`: For numerical operations.

## Coding Standards

- Code formatting and linting is enforced using `ruff`.
- Import statements are sorted using `ruff` (replaces isort).
- Static type checking is performed with `mypy`.
- Testing is done using the `pytest` framework.
- Employ inline comments to explain the intent and rationale ('the why') behind code, not to describe its mechanics ('the how').
  Prioritize writing self-documenting code.

## UI guidelines

- Not applicable, this is a library project.
