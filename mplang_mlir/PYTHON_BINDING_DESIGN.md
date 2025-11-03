# Python Binding Design for Mpir MLIR Dialect

## 概述

本文档描述如何将 MPLang Python 前端（`mplang.core.expr.ast`）与 MLIR Mpir Dialect 集成，实现 Python → MLIR IR 的转换和编译优化。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     MPLang Python 前端                           │
│  mplang.core.expr.ast (Expr AST)                               │
│  - EvalExpr, TupleExpr, CondExpr, WhileExpr, etc.             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ AST → MLIR Converter
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              MLIR Mpir Dialect (IR Layer)                       │
│  - mpir.peval, mpir.uniform_cond, mpir.uniform_while          │
│  - mpir.conv, mpir.shfl_s                                      │
│  - Type System: !mpir.mp<T, pmask>, !mpir.enc<T>, etc.        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ MLIR Passes (Optimization)
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                 MLIR Standard Dialects                          │
│  - func dialect, scf dialect, etc.                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ Lowering & Codegen
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              Runtime Execution (Python/C++)                     │
│  - mplang.runtime (Simulator, Driver, etc.)                    │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Python Bindings Layer (`mplang_mlir.dialects.mpir`)

#### 1.1 类型系统绑定

```python
# python/mplang_mlir/dialects/mpir.py

class MPType:
    """Wrapper for !mpir.mp<T, pmask> type."""

    @staticmethod
    def get(inner_type, pmask: int, context=None):
        """Create MP type with inner type and pmask."""
        pass

    @property
    def inner_type(self):
        """Get the inner type."""
        pass

    @property
    def pmask(self) -> int:
        """Get the pmask."""
        pass

class EncodedType:
    """Wrapper for !mpir.enc<T, schema> type."""

    @staticmethod
    def get(inner_type, schema: str, encoding_attrs=None, context=None):
        """Create Encoded type."""
        pass

class TableType:
    """Wrapper for !mpir.table<columns, types> type."""

    @staticmethod
    def get(column_names: list[str], column_types: list, context=None):
        """Create Table type."""
        pass
```

#### 1.2 操作绑定

```python
class PEvalOp:
    """Wrapper for mpir.peval operation."""

    @staticmethod
    def create(pfunc, args, rmask: int, result_types, loc=None, ip=None):
        """Create peval operation."""
        pass

class UniformCondOp:
    """Wrapper for mpir.uniform_cond operation."""

    @staticmethod
    def create(condition, then_fn, else_fn, args, result_types, loc=None, ip=None):
        """Create uniform_cond operation with regions."""
        pass

class UniformWhileOp:
    """Wrapper for mpir.uniform_while operation."""

    @staticmethod
    def create(cond_fn, body_fn, init_args, result_types, loc=None, ip=None):
        """Create uniform_while operation with regions."""
        pass

class ConvOp:
    """Wrapper for mpir.conv operation."""

    @staticmethod
    def create(inputs, result_type, loc=None, ip=None):
        """Create conv operation."""
        pass

class ShflSOp:
    """Wrapper for mpir.shfl_s operation."""

    @staticmethod
    def create(input, src_ranks: list[int], result_type, loc=None, ip=None):
        """Create shfl_s operation."""
        pass
```

### 2. AST to MLIR Converter (`mplang_mlir.converter`)

```python
# python/mplang_mlir/converter.py

from mplang.core.expr.ast import (
    Expr, EvalExpr, TupleExpr, AccessExpr, CondExpr, WhileExpr,
    ConvExpr, ShflSExpr, VariableExpr, FuncDefExpr, CallExpr
)
from mplang_mlir.dialects import mpir
from mlir import ir

class ASTToMLIRConverter:
    """Convert MPLang AST to MLIR Mpir dialect."""

    def __init__(self, context: ir.Context):
        self.context = context
        self.context.load_dialect("mpir")

        # Map from Expr to MLIR Value(s)
        self.value_map: dict[Expr, list[ir.Value]] = {}

        # Map from variable name to MLIR Value
        self.var_map: dict[str, ir.Value] = {}

        # Map from FuncDefExpr to MLIR func.func
        self.func_map: dict[FuncDefExpr, ir.FlatSymbolRefAttr] = {}

    def convert_expr(self, expr: Expr) -> list[ir.Value]:
        """Convert an expression to MLIR values (MIMO support)."""
        if expr in self.value_map:
            return self.value_map[expr]

        if isinstance(expr, EvalExpr):
            return self._convert_eval(expr)
        elif isinstance(expr, TupleExpr):
            return self._convert_tuple(expr)
        elif isinstance(expr, AccessExpr):
            return self._convert_access(expr)
        elif isinstance(expr, CondExpr):
            return self._convert_cond(expr)
        elif isinstance(expr, WhileExpr):
            return self._convert_while(expr)
        elif isinstance(expr, ConvExpr):
            return self._convert_conv(expr)
        elif isinstance(expr, ShflSExpr):
            return self._convert_shfl(expr)
        elif isinstance(expr, VariableExpr):
            return self._convert_variable(expr)
        elif isinstance(expr, CallExpr):
            return self._convert_call(expr)
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _convert_eval(self, expr: EvalExpr) -> list[ir.Value]:
        """Convert EvalExpr to mpir.peval operation."""
        # 1. Convert arguments
        arg_values = []
        for arg in expr.args:
            arg_values.extend(self.convert_expr(arg))

        # 2. Get or create PFunction reference
        pfunc_ref = self._get_pfunc_ref(expr.pfunc)

        # 3. Convert result types
        result_types = [self._convert_mptype(t) for t in expr.mptypes]

        # 4. Create peval operation
        peval_op = mpir.PEvalOp.create(
            pfunc=pfunc_ref,
            args=arg_values,
            rmask=expr.rmask.value if expr.rmask else 0,
            result_types=result_types
        )

        results = list(peval_op.results)
        self.value_map[expr] = results
        return results

    def _convert_cond(self, expr: CondExpr) -> list[ir.Value]:
        """Convert CondExpr to mpir.uniform_cond operation."""
        # 1. Convert predicate
        pred_values = self.convert_expr(expr.pred)
        if len(pred_values) != 1:
            raise ValueError("Condition must be single-output")
        pred = pred_values[0]

        # 2. Convert arguments
        arg_values = []
        for arg in expr.args:
            arg_values.extend(self.convert_expr(arg))

        # 3. Create function references for then/else branches
        then_fn_ref = self._get_funcdef_ref(expr.then_fn)
        else_fn_ref = self._get_funcdef_ref(expr.else_fn)

        # 4. Convert result types
        result_types = [self._convert_mptype(t) for t in expr.mptypes]

        # 5. Create uniform_cond operation
        cond_op = mpir.UniformCondOp.create(
            condition=pred,
            then_fn=then_fn_ref,
            else_fn=else_fn_ref,
            args=arg_values,
            result_types=result_types
        )

        results = list(cond_op.results)
        self.value_map[expr] = results
        return results

    def _convert_while(self, expr: WhileExpr) -> list[ir.Value]:
        """Convert WhileExpr to mpir.uniform_while operation."""
        # Similar to _convert_cond but for while loops
        pass

    def _convert_tuple(self, expr: TupleExpr) -> list[ir.Value]:
        """Convert TupleExpr - flatten outputs into a list."""
        # TupleExpr is just concatenating outputs from multiple expressions
        # No MLIR operation needed, just collect values
        result_values = []
        for arg in expr.args:
            result_values.extend(self.convert_expr(arg))

        self.value_map[expr] = result_values
        return result_values

    def _convert_access(self, expr: AccessExpr) -> list[ir.Value]:
        """Convert AccessExpr - select one output from multi-output expression."""
        # Get all outputs from the source expression
        src_values = self.convert_expr(expr.expr)

        # Select the indexed output
        if expr.index >= len(src_values):
            raise ValueError(f"Access index {expr.index} out of range (0-{len(src_values)-1})")

        result = [src_values[expr.index]]
        self.value_map[expr] = result
        return result

    def _convert_mptype(self, mptype: MPType) -> ir.Type:
        """Convert MPLang MPType to MLIR type."""
        from mplang.core.tensor import TensorType
        from mplang.core.table import TableType as PyTableType

        # Convert inner type (Tensor or Table)
        if isinstance(mptype._type, TensorType):
            inner = self._convert_tensor_type(mptype._type)
        elif isinstance(mptype._type, PyTableType):
            inner = self._convert_table_type(mptype._type)
        else:
            raise ValueError(f"Unsupported type: {type(mptype._type)}")

        # Wrap with encoding if needed
        # (检查 mptype._attrs 中的 encoding 信息)

        # Wrap with MP type
        if mptype._pmask:
            return mpir.MPType.get(inner, mptype._pmask.value, self.context)
        else:
            return inner

    def _convert_tensor_type(self, tensor_type: TensorType) -> ir.Type:
        """Convert TensorType to MLIR tensor type."""
        from mlir.ir import RankedTensorType

        # Convert dtype
        elem_type = self._convert_dtype(tensor_type.dtype)

        # Convert shape
        shape = list(tensor_type.shape)

        return RankedTensorType.get(shape, elem_type)

    def _convert_table_type(self, table_type: PyTableType) -> ir.Type:
        """Convert TableType to MLIR !mpir.table type."""
        column_names = list(table_type.schema.keys())
        column_types = [
            self._convert_dtype(dtype)
            for dtype in table_type.schema.values()
        ]

        return mpir.TableType.get(column_names, column_types, self.context)

    def _get_funcdef_ref(self, funcdef: FuncDefExpr) -> ir.FlatSymbolRefAttr:
        """Get or create function reference for FuncDefExpr."""
        if funcdef in self.func_map:
            return self.func_map[funcdef]

        # Create a func.func for this FuncDefExpr
        # (需要处理 region 和嵌套的 body)
        pass

    def _get_pfunc_ref(self, pfunc: PFunction) -> ir.FlatSymbolRefAttr:
        """Get or create function reference for PFunction."""
        # PFunction 应该映射到一个 func.func
        # 这需要额外的机制来管理 PFunction 到 MLIR func 的映射
        pass
```

### 3. 集成到 MPLang 编译流程

```python
# mplang/mlir/compiler.py

from mplang.core.expr.ast import FuncDefExpr
from mplang_mlir.converter import ASTToMLIRConverter
from mlir import ir, passmanager

class MLIRCompiler:
    """MLIR-based compiler for MPLang."""

    def __init__(self):
        self.context = ir.Context()
        self.context.load_dialect("mpir")
        self.context.load_dialect("func")
        self.context.load_dialect("scf")

    def compile(self, funcdef: FuncDefExpr, optimize: bool = True) -> ir.Module:
        """Compile FuncDefExpr to MLIR module."""
        # 1. Create module
        module = ir.Module.create(loc=ir.Location.unknown(self.context))

        # 2. Convert AST to MLIR
        with ir.InsertionPoint(module.body):
            converter = ASTToMLIRConverter(self.context)
            converter.convert_funcdef(funcdef)

        # 3. Run optimization passes
        if optimize:
            self._run_passes(module)

        # 4. Verify
        if not module.operation.verify():
            raise ValueError("Generated MLIR module failed verification")

        return module

    def _run_passes(self, module: ir.Module):
        """Run MLIR optimization passes."""
        pm = passmanager.PassManager.parse(
            "builtin.module("
            "  mpir-conv-elimination,"
            "  mpir-pmask-propagation,"
            "  canonicalize,"
            "  cse"
            ")",
            context=self.context
        )
        pm.run(module.operation)

    def compile_and_lower(self, funcdef: FuncDefExpr) -> str:
        """Compile to MLIR and lower to LLVM IR or executable."""
        module = self.compile(funcdef)

        # Lower to standard dialects
        # ... (需要实现 lowering passes)

        return str(module)
```

### 4. 使用示例

```python
# Example: Compile a simple MPLang program to MLIR

from mplang import function, compile
from mplang.mlir import MLIRCompiler

@function
def secure_add(x, y):
    return x + y

# Trace to AST
graph = compile(secure_add, x_shape=(10,), y_shape=(10,))

# Compile to MLIR
compiler = MLIRCompiler()
mlir_module = compiler.compile(graph)

print(mlir_module)
# Output:
# module {
#   func.func @secure_add(%arg0: !mpir.mp<tensor<10xf32>, 1>,
#                         %arg1: !mpir.mp<tensor<10xf32>, 1>)
#                         -> !mpir.mp<tensor<10xf32>, 1> {
#     %0 = mpir.peval @add_kernel(%arg0, %arg1) {rmask = 1}
#          : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xf32>, 1>)
#          -> !mpir.mp<tensor<10xf32>, 1>
#     return %0 : !mpir.mp<tensor<10xf32>, 1>
#   }
# }
```

## 实施计划

### Phase 1: 基础绑定 (1-2 天)

- [ ] 完善 `MpirExtension.cpp` 中的 Python bindings
- [ ] 实现类型系统绑定（MPType, EncodedType, TableType）
- [ ] 实现基本操作绑定（PEvalOp, ConvOp）
- [ ] 编写单元测试

### Phase 2: AST 转换器 (2-3 天)

- [ ] 实现 `ASTToMLIRConverter` 核心框架
- [ ] 实现基础表达式转换（EvalExpr, TupleExpr, AccessExpr）
- [ ] 实现控制流转换（CondExpr, WhileExpr）
- [ ] 处理函数嵌套和闭包

### Phase 3: 编译器集成 (1-2 天)

- [ ] 实现 `MLIRCompiler` 类
- [ ] 集成到 mplang.compile 流程
- [ ] 添加优化 pass pipeline
- [ ] 端到端测试

### Phase 4: Lowering 和执行 (3-5 天，可选)

- [ ] 实现 Mpir → Standard Dialects 的 lowering passes
- [ ] 集成 MLIR ExecutionEngine
- [ ] 或者：生成 Python callable 通过 runtime 执行

## 技术挑战

### 1. MIMO (Multi-Input Multi-Output) 处理

Python AST 的 `Expr` 有 `mptypes: list[MPType]`，支持多输出。MLIR 天然支持多结果操作，可以直接映射：

```mlir
%0, %1, %2 = mpir.peval @multi_output_func(...)
             : (...) -> (!mpir.mp<...>, !mpir.mp<...>, !mpir.mp<...>)
```

### 2. 嵌套函数和闭包

`CondExpr` 和 `WhileExpr` 的 `then_fn`/`else_fn`/`cond_fn`/`body_fn` 都是 `FuncDefExpr`，可能捕获外部变量。

解决方案：
- 将捕获的变量作为额外参数传入
- 使用 MLIR 的 nested functions 或 func.func with closure capture

### 3. PFunction 到 MLIR Function 的映射

`PFunction` 是 Python 端定义的后端函数，需要在 MLIR 中有对应的表示。

选项：
1. **符号引用**：PFunction 作为 `@symbol` 引用，在 runtime 解析
2. **内联**：将 PFunction 的实现直接转换为 MLIR ops（如果有 JAX/StableHLO）
3. **混合**：标准操作内联，自定义操作用符号引用

### 4. 类型转换的精确性

MPLang 的类型系统（TensorType, TableType, MPType, Mask）需要精确映射到 MLIR 类型：

- `TensorType(dtype, shape)` → `tensor<...x...xf32>`
- `TableType(schema)` → `!mpir.table<[...], [...]>`
- `MPType(type, pmask)` → `!mpir.mp<T, pmask>`
- Encoding 信息通过 attributes 传递

## 下一步行动

1. **立即开始**: 实现 Phase 1 的基础绑定
2. **优先级**: 先支持简单的 EvalExpr → peval 转换
3. **测试驱动**: 为每个转换功能编写测试用例
4. **迭代开发**: 逐步添加更复杂的表达式支持

## 参考资源

- MLIR Python Bindings: https://mlir.llvm.org/docs/Bindings/Python/
- Pybind11 Documentation: https://pybind11.readthedocs.io/
- MPLang AST: `mplang/core/expr/ast.py`
- Mpir Dialect Ops: `include/mplang/Dialect/Mpir/MpirOps.td`
- Mpir Dialect Types: `include/mplang/Dialect/Mpir/MpirTypes.td`
