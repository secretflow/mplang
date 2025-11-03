# Mpir Dialect Python Bindings

Python bindings for the Mpir MLIR dialect, following MLIR best practices (StableHLO/HEIR patterns).

## Features

- ✅ **Pythonic API** - Clean, type-safe operation builders
- ✅ **MLIR Standard Pattern** - Uses `@register_operation` decorator pattern
- ✅ **Dual-Mode Support** - MLIR functions or external backends (PHE, SPU, TEE)
- ✅ **Auto Type Conversion** - Python dicts → MLIR attributes automatically
- ✅ **IDE Support** - Full type hints and docstrings

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

## Supported Operations

All Mpir dialect operations are available with Pythonic APIs:

### Core Operations

#### PEvalOp - Partial Evaluation

**Dual-mode operation:**

```python
# Mode 1: Call MLIR function
result = mpir.PEvalOp(
    [result_type],
    [arg0, arg1],
    mask,
    callee="@my_function"
)

# Mode 2: Call external backend
encrypted = mpir.PEvalOp(
    [encrypted_type],
    [plaintext],
    mask,
    fn_type="phe",
    fn_name="encrypt",
    fn_attrs={
        "scheme": "paillier",
        "key_size": 2048
    }
)
```

**Features:**
- ✅ Python dict automatically converted to MLIR DictionaryAttr
- ✅ Validation catches mode conflicts early
- ✅ Clear parameter names

#### UniformCondOp - Conditional Execution

```python
result = mpir.UniformCondOp(
    [result_type],
    condition  # MP<i1> scalar
)

# Populate regions after construction:
with ir.InsertionPoint(result.then_region.blocks[0]):
    then_value = ...
    mpir.YieldOp([then_value])

with ir.InsertionPoint(result.else_region.blocks[0]):
    else_value = ...
    mpir.YieldOp([else_value])
```

#### UniformWhileOp - While Loop

```python
result = mpir.UniformWhileOp(
    [result_type],
    [init_value]  # Initial loop-carried values
)

# Populate condition region:
with ir.InsertionPoint(result.condition_region.blocks[0]):
    arg = result.condition_region.blocks[0].arguments[0]
    cond = ...  # Compute MP<i1> condition
    mpir.ConditionOp(cond, [arg])

# Populate body region:
with ir.InsertionPoint(result.body_region.blocks[0]):
    arg = result.body_region.blocks[0].arguments[0]
    new_value = ...  # Update value
    mpir.YieldOp([new_value])
```

#### ConvOp - Party Mask Conversion

```python
# Convert data from party 0 to party 1
result = mpir.ConvOp(
    result_type,  # MP<tensor<10xi32>, pmask={1}>
    input_value,  # MP<tensor<10xi32>, pmask={0}>
    src_pmask="{0}",
    dst_pmask="{1}"
)
```

**Benefits:**
- ✅ String parameters instead of manual StringAttr construction
- ✅ Clear parameter names indicate purpose

#### ShflSOp - Static Shuffle

```python
# Shuffle: output party 0 gets data from input party 1, vice versa
result = mpir.ShflSOp(
    result_type,
    input_value,
    src_ranks=[1, 0]  # Python list!
)
```

**Benefits:**
- ✅ Python list automatically converted to DenseI64ArrayAttr
- ✅ Natural array syntax

### Other Operations

- **PEvalDynOp** - Dynamic evaluation without static mask
- **ShuffleDynOp** - Dynamic shuffle operations

## Design Principles

### MLIR Standard Pattern

Following **StableHLO** and **HEIR** projects:

1. **TableGen generates base bindings** (`_mpir_ops_gen.py`)
   - Automatic from ODS (Operation Definition Specification)
   - Creates basic Python classes for each operation

2. **Extensions provide Pythonic API** (`_mpir_ops_ext.py`)
   - Use `@_cext.register_operation(_Dialect, replace=True)` decorator
   - Override `__init__` with natural parameter names
   - Auto-convert Python types (dict → DictAttr, list → ArrayAttr)
   - Add validation in constructors

### Key Benefits

**Before (raw MLIR):**
```python
result = mpir.PEvalOp(
    result_type,
    [arg0, arg1, mask],
    callee=ir.FlatSymbolRefAttr.get("my_function"),
    fn_type=None,
    fn_name=None,
    fn_attrs=None
)
```

**After (Pythonic extension):**
```python
result = mpir.PEvalOp(
    [result_type],
    [arg0, arg1],
    mask,
    callee="@my_function"  # String automatically converted!
)
```

**Improvement:** ~50% less code, 200% more readable

## Quick Start Example

```python
from mlir import ir
from mlir.dialects import builtin
from mplang_mlir.dialects import mpir

# Create context and module
with ir.Context() as ctx:
    mpir.register_dialect(ctx)
    loc = ir.Location.unknown(ctx)
    module = ir.Module.create(loc)

    with ir.InsertionPoint(module.body):
        # Example: Conditional execution
        @builtin.FuncOp.from_py_func(
            mpir.MPType.get([ir.IntegerType.get_signless(1)], "{0,1}"),
            mpir.MPType.get([ir.IntegerType.get_signless(32)], "{0}")
        )
        def conditional_example(condition):
            i32_type = ir.IntegerType.get_signless(32)
            result_type = mpir.MPType.get([i32_type], "{0}")

            # Create conditional - Pythonic API!
            result = mpir.UniformCondOp([result_type], condition)

            # Populate then branch
            with ir.InsertionPoint(result.then_region.blocks[0]):
                const_10 = mpir.ConstantOp(result_type, 10)
                mpir.YieldOp([const_10])

            # Populate else branch
            with ir.InsertionPoint(result.else_region.blocks[0]):
                const_20 = mpir.ConstantOp(result_type, 20)
                mpir.YieldOp([const_20])

            return result

    print(module)
```

See `test_ops_ext.py` and `test_integration.py` for comprehensive examples.

## Testing

Run the test script to verify the bindings:

```bash
python python/test_bindings.py
```

## Architecture

**Binding Layers:**
```
Python Code (user)
    ↓
PEvalOp(...) [_mpir_ops_ext.py - Pythonic wrapper]
    ↓
super().__init__(...) [_mpir_ops_gen.py - TableGen auto-generated]
    ↓
_mplang C++ extension [MpirExtension.cpp - Pybind11]
    ↓
CAPI [lib/CAPI/ - C API layer]
    ↓
MLIR Dialect [lib/Dialect/Mpir/ - Core C++ implementation]
```

**File Structure:**
- **`_mpir_ops_gen.py`** - Auto-generated from TableGen (base bindings)
- **`_mpir_ops_ext.py`** - Python extensions with Pythonic API
- **`CMakeLists.txt`** - Build configuration linking to TableGen
- **`test_*.py`** - Usage examples and integration tests

## References

- [MLIR Python Bindings Docs](https://mlir.llvm.org/docs/Bindings/Python/)
- [Mplang Dialect Spec](../include/mplang/Dialect/Mpir/MpirOps.td)
- [StableHLO Python Bindings](https://github.com/openxla/stablehlo/tree/main/stablehlo/integrations/python) - Reference pattern
- [HEIR Python Bindings](https://github.com/google/heir/tree/main/heir/python) - Reference pattern
