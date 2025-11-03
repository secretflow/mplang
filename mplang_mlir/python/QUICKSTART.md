# Mplang Python Bindings - Quick Start

## 1-Minute Setup

### Build with Python Bindings
```bash
cd $LLVM_BUILD_DIR
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=mplang \
  -DLLVM_EXTERNAL_MPLANG_SOURCE_DIR=/path/to/mplang_mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

ninja MplangPythonModules
```

### Test Installation
```bash
export PYTHONPATH=$LLVM_BUILD_DIR/python_packages/mplang_mlir:$PYTHONPATH
python -c "from mplang_mlir.dialects import mplang; print('Success!')"
```

## 5-Minute Example

```python
#!/usr/bin/env python3
"""Example: Building Mplang IR from Python"""

from mplang_mlir import ir
from mplang_mlir.dialects import func, mplang

# Create context
ctx = ir.Context()
mplang.register_dialect(ctx)

# Build a simple module
with ir.Location.unknown(ctx):
    module = ir.Module.create()

    with ir.InsertionPoint(module.body):
        # Define types
        f32 = ir.F32Type.get()
        tensor_3xf32 = ir.RankedTensorType.get([3], f32)
        mask_3xi1 = ir.RankedTensorType.get([3], ir.IntegerType.get_signless(1))

        # Create function signature
        func_type = ir.FunctionType.get(
            inputs=[tensor_3xf32, mask_3xi1],
            results=[tensor_3xf32]
        )

        # Define function
        f = func.FuncOp(name="example", type=func_type)

        with ir.InsertionPoint(f.add_entry_block()):
            arg0, mask = f.arguments

            # Call external PHE backend
            result = mplang.PEvalOp(
                [tensor_3xf32],  # result_types
                [arg0],          # args
                mask,            # mask
                fn_type="phe",
                fn_name="encrypt",
                fn_attrs={"scheme": "paillier", "key_size": 2048}
            )

            func.ReturnOp([result])

print(module)
```

Expected output:
```mlir
module {
  func.func @example(%arg0: tensor<3xf32>, %arg1: tensor<3xi1>) -> tensor<3xf32> {
    %0 = mplang.peval fn_type = "phe" fn_name = "encrypt"
         fn_attrs = {scheme = "paillier", key_size = 2048 : i64}
         (%arg0, %arg1) : (tensor<3xf32>, tensor<3xi1>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
```

## Common Operations

### PEval (Static Private Evaluation)

**MLIR Function Mode:**
```python
result = mplang.PEvalOp(
    [result_type],           # result_types
    [input_tensors],         # args
    execution_mask,          # mask
    callee="@my_function"    # Reference to func.func @my_function
)
```

**External Backend Mode:**
```python
result = mplang.PEvalOp(
    [result_type],
    [inputs],
    mask,
    fn_type="spu",           # Backend: spu, phe, tee, etc.
    fn_name="add",           # Backend function name
    fn_attrs={               # Backend-specific attributes
        "protocol": "aby3",
        "field": "FM64"
    }
)
```

### PEval Dyn (Dynamic Execution)

No static mask, execution determined at runtime:
```python
result = mplang.PEvalDynOp(
    [result_type],
    [inputs],
    fn_type="phe",
    fn_name="decrypt"
)
```

### Shuffle (Party-to-Party Data Movement)

Static pattern:
```python
result = mplang.ShuffleStaticOp(
    result_types=[result_type],
    inputs=[input_val],
    mask=target_mask
)
```

Dynamic pattern:
```python
result = mplang.ShuffleDynOp(
    result_types=[result_type],
    inputs=[input_val],
    pmask=dynamic_pattern
)
```

### Conv (Convergence)

Multi-party convergence:
```python
result = mplang.ConvOp(
    result_types=[result_type],
    inputs=[val1, val2, val3]
)
```

## Parsing Existing IR

```python
from mplang_mlir import ir
from mplang_mlir.dialects import mplang

ctx = ir.Context()
mplang.register_dialect(ctx)

# Parse from string
mlir_code = """
module {
  func.func @test(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %mask = arith.constant dense<true> : tensor<3xi1>
    %0 = mplang.peval @compute(%arg0, %mask) : (tensor<3xf32>, tensor<3xi1>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
"""

module = ir.Module.parse(mlir_code, ctx)

# Walk operations
for op in module.body.operations:
    print(f"Operation: {op.OPERATION_NAME}")
```

## Attribute Types

Python values automatically convert to MLIR attributes:

| Python Type | MLIR Attribute | Example |
|-------------|----------------|---------|
| `bool` | `BoolAttr` | `True` → `true` |
| `int` | `IntegerAttr` (i64) | `2048` → `2048 : i64` |
| `float` | `FloatAttr` (f64) | `3.14` → `3.14 : f64` |
| `str` | `StringAttr` | `"paillier"` → `"paillier"` |
| `dict` | `DictionaryAttr` | `{"k": 1}` → `{k = 1 : i64}` |

## Troubleshooting

**Import Error:**
```
ImportError: No module named 'mplang_mlir'
```
→ Check `PYTHONPATH` includes `$BUILD_DIR/python_packages/mplang_mlir`

**Dialect Not Found:**
```
error: 'mplang.peval' op is not registered
```
→ Call `mplang.register_dialect(ctx)` or `mplang.load_dialect(ctx)`

**Build Error:**
```
ninja: error: 'MLIRMplangCAPI', needed by 'MplangPythonModules', missing
```
→ Build CAPI first: `ninja MLIRMplangCAPI`

## Next Steps

- **Read Full Docs**: [README.md](README.md)
- **MLIR Python Bindings**: <https://mlir.llvm.org/docs/Bindings/Python/>
- **Run Tests**: `python test_bindings.py`
