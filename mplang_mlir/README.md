# MPLANG (MPIR Dialect on MLIR)

An out-of-tree MLIR scaffold for MPLang. It hosts the mpir dialect, basic transforms (mpir-peval), and the `mplang-opt` tool. It is designed as a self-contained component that can be built and tested independently.

## Prerequisites

- A local LLVM/MLIR build (LLVM 18.x) with tools (FileCheck). The repo provides scripts to bootstrap a pinned build.
- `make` and `cmake`.

## Quick Start with Makefile

This project uses a `Makefile` to simplify the development workflow. From the `mplang_mlir` directory, you can run:

```bash
# Configure, build, and run tests in one go
make test

# Run a sample optimization
make run
```

For more commands, run `make help`.

## Development Workflow

### 1. Prepare Environment

Bootstrap LLVM/MLIR locally. This is a one-time setup.

```bash
# From the repository root
./mplang_mlir/scripts/setup_mlir.sh llvmorg-18.1.8
```

After success, MLIR CMake package is at `build/llvm/lib/cmake/mlir`.

### 2. Build

Build the `mplang-opt` tool using Make:

```bash
make build
```

### 3. Test

Run the test suite using Make:

```bash
make test
```

### 4. Run

Run a sample optimization pass:

```bash
make run
```

<details>
<summary><b>Manual Build and Test (Advanced)</b></summary>

If you prefer not to use Make, you can run the commands manually:

**Configure:**

```bash
# From repo root
cmake -S mplang_mlir -B build/mplang_mlir -G Ninja \
  -DMLIR_DIR="$PWD/build/llvm/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/build/llvm/lib/cmake/llvm"
```

**Build:**

```bash
cmake --build build/mplang_mlir --target mplang-opt -j
```

**Test:**

```bash
ctest --test-dir build/mplang_mlir --output-on-failure
```

**Run:**

```bash
build/mplang_mlir/tools/mplang-opt/mplang-opt --mpir-peval mplang_mlir/test/mlir/mpir/peval.mlir
```

</details>

## Troubleshooting

- **CMake error: MLIR_DIR is required**: Run `mplang_mlir/scripts/setup_mlir.sh` and ensure `build/llvm` exists, or manually specify `MLIR_DIR` and `LLVM_DIR` when running `make` or `cmake`.
- **FileCheck not found during tests**: Ensure LLVM tools were built by the setup script or add `build/llvm/bin` to your PATH.
- **Dialect ‘func’ not found**: Use the built `mplang-opt` from this repo; it registers `func` and `arith` dialects at startup.

## Makefile Targets

The `Makefile` provides several targets for convenience:

- `all`: Build the project (default).
- `configure`: Configure the project using CMake.
- `build`: Build the `mplang-opt` binary.
- `test`: Run tests using ctest.
- `run`: Run a sample peval test with `mplang-opt`.
- `clean`: Remove the build directory.
- `help`: Show this help message.

## Next steps

- Add verifiers for mpir ops (eval/tuple/pconv/shfl(_static)/if/while).
- Expand mpir-peval and add more transformation patterns.
- Lowerings from MPIR to PHE and additional backends.
