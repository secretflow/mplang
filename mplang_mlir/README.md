# MPLANG MLIR dialect (scaffold)

This directory contains a minimal MLIR dialect scaffold for MPLang's SPMD IR.
It is intentionally small to get parse/print working early, then we will
iterate with more ops, verifiers, canonicalizations, and Python bindings.

## Prerequisites

- LLVM/MLIR 18.x installed (with CMake config packages)
- CMake >= 3.20, Ninja (recommended)
- A C++17 toolchain

Set `MLIR_DIR` to the MLIR CMake package path, e.g. `<prefix>/lib/cmake/mlir`.

- Conda MLIR 18 example (recommended):
  - `conda activate mlir-env`
  - `MLIR_DIR="$CONDA_PREFIX/lib/cmake/mlir"`

If you built LLVM/MLIR from source with `LLVM_ENABLE_PROJECTS=mlir`, the CMake
package is under your build/install prefix.

## Build

```sh
# If using conda MLIR 18, activate your env first so $CONDA_PREFIX is set
# conda activate mlir-env

# Configure (example: conda MLIR package)
cmake -S . -B built -G Ninja -DMLIR_DIR="$CONDA_PREFIX/lib/cmake/mlir"

# Or configure with a custom install prefix
# cmake -S . -B built -G Ninja -DMLIR_DIR=/path/to/lib/cmake/mlir

# Build
cmake --build built -j
```

Artifacts:

- `built/lib/libMLIRMPLANG.*` — the dialect library
- `built/tools/mplang-opt/mplang-opt` — a tiny opt-like driver

## Try it (parse/print roundtrip)

Create a file `test.mlir` with a minimal module that uses mplang.eval and the
implicit mplang.yield (generic assembly form recommended for now):

```mlir
module {
  "mplang.eval"() ({
    "mplang.yield"() : () -> ()
  }) : () -> ()
}
```

Then run:

```sh
built/tools/mplang-opt/mplang-opt test.mlir -o -
```

Expected output (printer may elide the implicit yield):

```mlir
module {
  mplang.eval()({
  }) : () -> ()
}
```

Notes:

- The tool currently only registers the MPLANG dialect to keep linking small.
  If you include ops from other dialects (e.g., `func.func`), add
  `-allow-unregistered-dialect` or extend the tool to register additional dialects.

## Troubleshooting

- Configure fails with MLIR_DIR: ensure your conda env is active and
  `$CONDA_PREFIX/lib/cmake/mlir` exists. Example:

  ```sh
  echo $CONDA_PREFIX
  ls "$CONDA_PREFIX/lib/cmake/mlir"
  ```

- Switched build directory: if you previously configured `build/` with a wrong
  `MLIR_DIR`, prefer using the clean `built/` directory as shown above, or
  delete the stale `CMakeCache.txt` before re-configuring.

- Link errors referencing many MLIR passes/dialects: this scaffold’s tool only
  registers the MPLANG dialect. If you add `mlir/InitAllPasses.h` or register
  extra dialects, you must also link the corresponding MLIR libraries.

## Next steps (short-term)

- Flesh out `!mplang.tensor` / `!mplang.table` type params (dtype/shape/pmask)
- `mplang.eval` attributes (e.g. rmask/backend/attrs) and verifier
- Add ops: `mplang.cond`, `mplang.while`, `mplang.conv`, `mplang.shfl_s`, `mplang.shfl`
- Canonicalization: eval region <-> callee symref (outline/inline)
- Python bindings (pybind11 + MLIR C-API) to register/parse/print from Python

See also: `../../design/mlir_dialect_scaffold.md`.
