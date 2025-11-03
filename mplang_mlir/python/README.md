# Mpir Python Bindings

Python bindings for the Mpir MLIR dialect, enabling Python-based construction and manipulation of Mpir IR.

## Building

To build with Python bindings enabled:

```bash
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=mplang \
  -DLLVM_EXTERNAL_MPLANG_SOURCE_DIR=/path/to/mplang_mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3)

ninja check-mplang
```

## Installation

After building, the Python package will be located at:
```
${BUILD_DIR}/python_packages/mplang_mlir
```

You can install it with:
```bash
pip install ${BUILD_DIR}/python_packages/mplang_mlir
```

Or add to PYTHONPATH for development:
```bash
export PYTHONPATH=${BUILD_DIR}/python_packages/mplang_mlir:$PYTHONPATH
```

## Usage

### Basic Example

```python
from mplang_mlir import ir
from mplang_mlir.dialects import mpir

# Create context and register dialect
ctx = ir.Context()
mpir.register_dialect(ctx)

# Parse Mpir IR
module_str = """
module {
  func.func @example(%arg0: tensor<3xf32>, %mask: tensor<3xi1>) -> tensor<3xf32> {
    %result = mpir.peval @callee(%arg0, %mask) : (tensor<3xf32>, tensor<3xi1>) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }
}
"""

module = ir.Module.parse(module_str, ctx)
print(module)
```

### Building Operations Programmatically

```python
from mplang_mlir import ir
from mplang_mlir.dialects import func, mplang

ctx = ir.Context()
mpir.register_dialect(ctx)

with ir.Location.unknown(ctx):
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        # Create function
        f32_type = ir.F32Type.get()
        tensor_type = ir.RankedTensorType.get([3], f32_type)
        mask_type = ir.RankedTensorType.get([3], ir.IntegerType.get_signless(1))

        func_type = ir.FunctionType.get(
            inputs=[tensor_type, mask_type],
            results=[tensor_type]
        )

        func_op = func.FuncOp(name="example", type=func_type)

        with ir.InsertionPoint(func_op.add_entry_block()):
            arg0 = func_op.arguments[0]
            mask = func_op.arguments[1]

            # Build peval op (MLIR function mode)
            result = mpir.PEvalOp(
                [tensor_type],  # result_types
                [arg0],         # args
                mask,           # mask
                callee="@callee"
            )

            func.ReturnOp([result])

print(module)
```

### External Backend Mode

```python
# Build peval op with external backend (e.g., PHE)
result = mpir.PEvalOp(
    [tensor_type],      # result_types
    [pk, plaintext],    # args
    mask,               # mask
    fn_type="phe",
    fn_name="encrypt",
    fn_attrs={
        "scheme": "paillier",
        "key_size": 2048
    }
)
```

## Implementation Notes

### Extension Pattern (MLIR Best Practice)

This implementation follows MLIR's official extension pattern (see `llvm-project/mlir/python/mlir/dialects/arith.py`):

1. **TableGen** auto-generates `_mpir_ops_gen.py` from `MpirOps.td`
2. **Extension classes** use `@_cext.register_operation(_Dialect, replace=True)` to extend generated ops
3. **Override `__init__`** to provide Pythonic API, call `super().__init__()` to leverage generated builders

**Key principle:** Let TableGen generate boilerplate; we only customize what needs a better API.

### Structure

- `mplang_mlir/__init__.py` - Package entry point
- `mplang_mlir/dialects/mplang.py` - Dialect wrapper (imports generated + extension code)
- `mplang_mlir/dialects/_mpir_ops_ext.py` - Op extensions (PEvalOp, PEvalDynOp)
- `MpirExtension.cpp` - C++ CAPI bridge (pybind11)
- `lib/CAPI/` - C API for Python/C++ interop

## Operations

### PEvalOp (Private Evaluation)

Static private evaluation with execution mask.

**Modes:**
1. MLIR function: `mpir.peval @callee(%args, %mask)`
2. External backend: `mpir.peval fn_type="phe" fn_name="encrypt" fn_attrs={...}(%args, %mask)`

### PEvalDynOp (Dynamic Private Evaluation)

Dynamic execution without static mask.

### ShuffleStaticOp / ShuffleDynOp

Party-to-party shuffle operations.

### ConvOp (Convergence)

Multi-party convergence operation.

## Testing

Run the test script to verify the bindings:

```bash
python python/test_bindings.py
```

## Architecture

```
Python Code (user)
    ↓
PEvalOp(...) [_mpir_ops_ext.py]
    ↓
super().__init__(...) [_mpir_ops_gen.py - auto-generated]
    ↓
_mplang C++ extension [MpirExtension.cpp]
    ↓
CAPI [lib/CAPI/]
    ↓
MLIR Dialect [lib/Dialect/Mpir/]
```

## References

- [Quick Start Guide](QUICKSTART.md) - 5-minute examples
- [MLIR Python Bindings Docs](https://mlir.llvm.org/docs/Bindings/Python/)
- [Mplang Dialect Spec](../include/mplang/Dialect/Mpir/MpirOps.td)
