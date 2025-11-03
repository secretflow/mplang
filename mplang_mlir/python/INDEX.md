# Mplang MLIR Python Bindings - Documentation

## Quick Links

- **[README.md](README.md)** - Complete guide: building, usage, API reference
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start with examples

## What's This?

Python bindings for the Mpir MLIR dialect, enabling Python-based construction and manipulation of Mpir IR.

## Quick Example

```python
from mplang_mlir import ir
from mplang_mlir.dialects import mpir

ctx = ir.Context()
mpir.register_dialect(ctx)

# Mode 1: MLIR function
result = mpir.PEvalOp([result_type], [args], mask, callee="@func")

# Mode 2: External backend (PHE, SPU, TEE)
result = mpir.PEvalOp(
    [result_type], [args], mask,
    fn_type="phe",
    fn_name="encrypt",
    fn_attrs={"scheme": "paillier", "key_size": 2048}
)
```

## Build

```bash
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=mplang \
  -DLLVM_EXTERNAL_MPLANG_SOURCE_DIR=/path/to/mplang_mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

ninja MpirPythonModules
```

## Files

```
python/
├── README.md              - Main documentation
├── QUICKSTART.md          - Quick start guide
├── INDEX.md               - This file
├── test_bindings.py       - Test script
├── CMakeLists.txt         - Build configuration
├── MpirExtension.cpp    - C++ extension
└── mplang_mlir/
    ├── __init__.py
    └── dialects/
        ├── __init__.py
        ├── mplang.py          - Dialect wrapper
        └── _mpir_ops_ext.py - Op extensions (PEvalOp, etc.)
```

## Implementation

Follows MLIR's **Extension Pattern** (from `arith.py`):
- TableGen generates `_mpir_ops_gen.py` from ODS
- Extension classes use `@register_operation` to extend ops
- Override `__init__` for Pythonic API, call `super().__init__()`

See [README.md](README.md) for details.
