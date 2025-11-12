# RFC: MPLang EDSL Architecture

**Status**: Design Document
**Author**: MPLang Team
**Created**: 2025-01-10
**Updated**: 2025-01-10

## 执行摘要

本 RFC 详细分析 MPLang 的 EDSL (Embedded Domain-Specific Language) 架构，阐明当前设计的核心原则、关键组件及其交互，并提出未来演进路径。

MPLang EDSL 的核心目标：
1. **多方计算抽象**：从单一 Python 程序生成多方协同执行的 IR
2. **类型安全**：通过类型系统表达数据分布、加密状态和隐私约束
3. **可组合性**：支持嵌套、高阶控制流和模块化 ops
4. **可分析性**：生成可优化、可验证的静态计算图

---

## 1. 架构概览

### 1.1 核心组件栈

```
┌─────────────────────────────────────────────────────────────┐
│  User Code (Python Functions)                              │
│  @function def encrypt(x): return smpc.seal(x)             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓ Tracing
┌─────────────────────────────────────────────────────────────┐
│  EDSL Layer (mplang.core)                                   │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ Primitives   │ Tracer       │ Interpreter  │            │
│  │ @function    │ TraceContext │ InterpContext│            │
│  │ @builtin     │ TraceVar     │ InterpVar    │            │
│  │ peval, cond  │ trace()      │ apply()      │            │
│  └──────────────┴──────────────┴──────────────┘            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓ IR Generation
┌─────────────────────────────────────────────────────────────┐
│  IR Layer (mplang.core.expr)                                │
│  Expr Tree (VariableExpr, CallExpr, CondExpr, ...)         │
│  TracedFunction(in_vars, out_vars, FuncDefExpr)            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓ Compilation
┌─────────────────────────────────────────────────────────────┐
│  Ops Layer (mplang.ops + mplang.kernels)                    │
│  Frontend: FeModule, simple_op, FeOperation                 │
│  Backend: KernelContext, evaluate_graph                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓ Execution
┌─────────────────────────────────────────────────────────────┐
│  Runtime (mplang.runtime)                                   │
│  Driver, Simulator, Server/Client, Communicator            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 数据流（Trace → IR → Execution）

```python
# 1. User Code
@mplang.function
def compute(x: MPObject, y: MPObject) -> MPObject:
    z = x + y
    return z * constant(2)

# 2. Tracing Phase (TraceContext)
tracer = TraceContext(cluster_spec)
tfn = trace(tracer, compute, x_var, y_var)
# 结果：TracedFunction 包含 FuncDefExpr

# 3. IR Representation (Expr Tree)
# FuncDefExpr(
#   params=["$0", "$1"],
#   body=TupleExpr([
#     CallExpr(add, [$0, $1]) → %tmp0
#     CallExpr(mul, [%tmp0, constant(2)]) → %result
#   ])
# )

# 4. Compilation (via Driver/Simulator)
compiled_fn = compile(compute, backend="simulation")

# 5. Execution
ctx = Simulator(cluster_spec)
result = evaluate(ctx, compiled_fn, x_data, y_data)
fetched = fetch(ctx, result)  # 获取各方数据
```

---

## 2. 核心设计原则

### 2.1 Closed-World Assumption（封闭世界假设）

**原则**：Traced 函数是数据转换管道，不能泄露控制流到外部。

**限制**：
- ✅ 允许：`MPObject` 输入/输出、立即值（`int`, `float`）
- ✅ 允许：结构化控制流（`cond`, `while_loop`）
- ❌ 禁止：返回捕获了 `TraceVar` 的 Python 函数
- ❌ 禁止：将任意 Python 函数作为第一类值传递

**原因**：
1. **静态性**：IR 必须在 trace 时完全确定（不能有运行时动态生成的函数）
2. **可分析性**：需要 CFG (Control Flow Graph) 在编译时可见
3. **多方同步**：所有方必须对控制流有相同理解

**设计对比**：

| 特性                | MPLang (当前)                   | JAX                          | PyTorch 2.0         |
|---------------------|--------------------------------|------------------------------|---------------------|
| 函数作为返回值      | ❌ 禁止                        | ❌ 禁止                       | ⚠️ 有限支持         |
| 高阶函数作为参数    | ⚠️ 仅控制流（`cond`, `while`）| ⚠️ 仅特定 primitives         | ✅ 广泛支持         |
| 静态 IR             | ✅ 完全静态                    | ✅ 完全静态                   | ❌ 动态 + 部分静态  |
| 理由                | 多方同步、CFG 必须确定          | XLA 编译需要静态图            | 灵活性优先          |

### 2.2 TracedFunction vs First-Class Functions

**关键决策**：使用专门的 `TracedFunction` 类而不是 `TraceVar(expr=FuncDefExpr)`

**原因**：

1. **类型安全**：
   ```python
   # Bad (如果允许)
   fn_var = TraceVar(ctx, FuncDefExpr(...))  # fn_var 是 MPObject？
   result = fn_var(x, y)  # 如何类型检查？

   # Good (当前设计)
   tfn = TracedFunction(...)  # 明确是 Callable
   result = apply(ctx, tfn, x, y)  # 类型安全
   ```

2. **元数据保留**：
   ```python
   class TracedFunction:
       in_struct: MorphStruct   # PyTree 结构（如何 flatten/unflatten）
       out_struct: MorphStruct
       capture_map: dict        # 捕获的外部变量
   ```
   这些信息对于正确 marshalling 参数至关重要，`Expr` 无法携带。

3. **避免复杂性**：
   - 不需要在 IR 中支持动态调度
   - 不需要 Y-combinator 或递归
   - 简化编译器实现

**借鉴 JAX**：
```python
# JAX 也不支持真正的一等函数
def bad():
    if condition:
        fn = lambda x: x + 1
    else:
        fn = lambda x: x - 1
    return jax.jit(fn)  # ❌ 无法静态确定 fn

# 正确方式
def good(condition):
    return jax.lax.cond(
        condition,
        lambda x: x + 1,
        lambda x: x - 1,
        x
    )  # ✅ 两个分支都被 trace
```

### 2.3 双上下文架构（TraceContext vs InterpContext）

**核心理念**：相同的 Python 代码在不同上下文下有不同行为

```python
# Context-Aware Execution
def add(x, y):
    ctx = cur_ctx()
    if isinstance(ctx, TraceContext):
        # Build IR
        return TraceVar(ctx, CallExpr(...))
    elif isinstance(ctx, InterpContext):
        # Execute eagerly
        return ctx.evaluate(add_kernel, {**bindings})
```

**TraceContext** (Lazy / Graph Building)：
- 目的：构建计算图（Expr Tree）
- `TraceVar`：存储 `Expr`，不执行计算
- 输出：`TracedFunction` + `FuncDefExpr`

**InterpContext** (Eager / Execution)：
- 目的：立即执行并返回结果
- `InterpVar`：引用已计算的值
- 输出：实际的 `MPObject` 结果

**上下文切换**：
```python
# primitive.py::_switch_ctx
def _switch_ctx(ctx: MPContext, obj: MPObject) -> MPObject:
    if obj.ctx is ctx:
        return obj  # Same context, no-op

    if isinstance(ctx, TraceContext):
        # Capture into trace context
        return ctx.capture(obj)  # Create VariableExpr

    elif isinstance(ctx, InterpContext):
        # Cannot import InterpVar from another context
        raise ValueError(...)
```

**自动上下文推断**：
```python
@function
def my_func(x):
    # x 可能是 TraceVar 或 InterpVar
    # primitive 装饰器自动处理
    return x + constant(1)

# 场景 1：在 TraceContext 中调用
tracer = TraceContext(...)
with with_ctx(tracer):
    result = my_func(x)  # x 是 TraceVar → 构建 IR

# 场景 2：在 InterpContext 中调用
interp = Simulator(...)
with with_ctx(interp):
    result = my_func(x)  # x 是 InterpVar → 立即执行
```

---

## 3. 关键组件详解

### 3.1 Primitive 装饰器系统

#### 3.1.1 `@function` (Inline Tracing)

**定义**：
```python
def function(fn: Callable[P, R]) -> Callable[P, R]:
    """Inline the function body into caller's trace."""
    return trace_before_apply(fn, make_call=False)
```

**行为**：
- **TraceContext**：展开函数体，直接插入 caller 的图
- **InterpContext**：先 trace 成 `TracedFunction`，然后 `apply()`

**示例**：
```python
@function
def add_twice(x, y):
    z = x + y
    return z + y

# 在 TraceContext 中
result = add_twice(a, b)
# 生成的 IR：
# %0 = add(a, b)
# %1 = add(%0, b)
# return %1
```

**用途**：
- 用户级函数组合
- 纯 Python 逻辑（无需跨语言边界）

#### 3.1.2 `@builtin_function` (Opaque Call)

**定义**：
```python
def builtin_function(fn: Callable[P, R]) -> Callable[P, R]:
    """Trace as an opaque CallExpr node."""
    return trace_before_apply(fn, make_call=True)
```

**行为**：
- **TraceContext**：
  1. Fork 一个新的 sub-context
  2. Trace `fn` → 生成 `TracedFunction`
  3. 创建 `CallExpr(tfn.make_expr(), args)`
  4. 在 caller 的图中插入单个 `CallExpr` 节点
- **InterpContext**：同 `@function`

**示例**：
```python
@builtin_function
def complex_crypto(x, key):
    # 内部逻辑被封装成单个 CallExpr
    ct = encrypt(x, key)
    mac = compute_mac(ct, key)
    return (ct, mac)

# 在 TraceContext 中
result = complex_crypto(data, k)
# 生成的 IR：
# %0 = call complex_crypto(data, k)  ← 单个节点
```

**用途**：
- 封装复杂逻辑（避免 IR 膨胀）
- 第三方库集成
- 优化边界（子图可独立优化）

#### 3.1.3 Context Switching Logic

```python
def trace_before_apply(fn, make_call: bool):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        current_ctx = cur_ctx()

        if isinstance(current_ctx, TraceContext):
            if make_call:
                # @builtin_function: 创建 CallExpr
                tracer = current_ctx
                tfn = trace(tracer.fork(), fn, *args, **kwargs)

                # 提取参数和捕获变量
                arg_exprs = [arg.expr for arg in args]
                cap_exprs = [tracer.capture(v).expr for v in tfn.capture_map]

                # 构建 CallExpr
                call_expr = CallExpr(
                    fn.__name__,
                    tfn.make_expr(),
                    arg_exprs + cap_exprs
                )

                # 返回 TraceVar
                return [TraceVar(tracer, AccessExpr(call_expr, i))
                        for i in range(call_expr.num_outputs)]
            else:
                # @function: 直接展开
                args, kwargs = tree_map(
                    partial(_switch_ctx, current_ctx),
                    (args, kwargs)
                )
                return fn(*args, **kwargs)

        elif isinstance(current_ctx, InterpContext):
            # 统一：先 trace 后 apply
            trace_ctx = TraceContext(current_ctx.cluster_spec, parent=current_ctx)
            tfn = trace(trace_ctx, fn, *args, **kwargs)
            return apply(current_ctx, tfn, *args, **kwargs)

    return wrapped
```

**关键点**：
1. **自动上下文感知**：用户无需关心当前在 trace 还是 execute
2. **统一接口**：`@function` 和 `@builtin_function` 在用户侧调用方式相同
3. **捕获变量处理**：自动 capture 外部作用域的变量

### 3.2 控制流 Primitives

#### 3.2.1 `uniform_cond` (Global Conditional)

**语义**：所有方执行**相同的分支**（predicate 必须 uniform）

**关键设计点**：

1. **Predicate Uniformity**：
   ```python
   # ✅ Good: predicate 对所有方相同
   pred = reveal(secret_flag)  # 公开后所有方一致
   result = uniform_cond(pred, then_fn, else_fn, x)

   # ❌ Bad: predicate 可能各方不同
   pred = x > local_threshold  # 各方有不同的 x
   result = uniform_cond(pred, ...)  # RuntimeError!
   ```

2. **Static Pmask Check**：
   ```python
   # uniform_cond 内部
   pred_pmask = pred.mptype.pmask
   if pred_pmask is not None and pred_pmask != cur_tracer.mask:
       raise ValueError(
           f"Predicate pmask {pred_pmask} != trace mask {cur_tracer.mask}"
       )
   ```

   **原因**：如果 pred 只在部分方可见，其他方无法判断走哪个分支。

3. **Branch Type Matching**：
   ```python
   # 两个分支必须返回相同类型（包括 pmask）
   for tv, ev in zip(then_tfn.out_vars, else_tfn.out_vars):
       if tv.mptype != ev.mptype:
           raise TypeError(f"Branch type mismatch: {tv.mptype} != {ev.mptype}")
   ```

4. **Capture Handling**：
   ```python
   # 问题：分支可能捕获外部变量
   outer_var = ...
   def then_fn(x):
       return x + outer_var  # 捕获 outer_var

   # 解决：将捕获变量作为隐式参数传递
   all_captures = then_tfn.capture_map | else_tfn.capture_map
   capture_vars = [cur_tracer.capture(v) for v in all_captures]

   # CondExpr 的参数 = [regular_args] + [captured_vars]
   fn_expr = CondExpr(pred, then_fn_expr, else_fn_expr,
                      arg_exprs + capture_exprs)
   ```

**生成的 IR**：
```python
CondExpr(
    pred=VariableExpr("%pred"),
    then_branch=FuncDefExpr(
        params=["x", "outer_var"],  # 包含 captures
        body=...
    ),
    else_branch=FuncDefExpr(
        params=["x", "outer_var"],
        body=...
    ),
    inputs=[%x, %outer_var],
    verify_uniform=True
)
```

#### 3.2.2 `while_loop` (Global Loop)

**语义**：所有方同步迭代（predicate 必须 uniform）

**关键设计点**：

1. **Loop State Invariant**：
   ```python
   # body 的输出类型必须等于输入类型
   for out_v, in_v in zip(body_tfn.out_vars, cond_tfn.in_vars):
       if out_v.mptype != in_v.mptype:
           raise TypeError(f"Loop state type mismatch")
   ```

2. **Predicate Static Check**（同 `uniform_cond`）：
   ```python
   pred_pmask = cond_out_var.mptype.pmask
   if pred_pmask is not None and pred_pmask != cur_tracer.mask:
       raise ValueError("Predicate pmask must match trace mask")
   ```

3. **State + Captures**：
   ```python
   # WhileExpr 参数 = [loop_state_vars] + [captured_vars]
   state_exprs = [v.expr for v in init_vars]
   capture_exprs = [cur_tracer.capture(v).expr for v in all_captures]

   WhileExpr(
       cond_fn_expr,
       body_fn_expr,
       all_args=state_exprs + capture_exprs
   )
   ```

**使用场景对比**：

| 场景                          | 推荐方案                          | 原因                              |
|-------------------------------|----------------------------------|-----------------------------------|
| 纯 local 计算（无 MPC）       | `peval(jax.lax.while_loop, ...)`  | 更高效，JAX 编译优化              |
| 各方独立停止（非同步）        | `peval(jax.lax.while_loop, ...)`  | 不需要全局同步                    |
| 全局同步停止（predicate uniform）| `while_loop(cond_fn, body_fn, init)` | 需要 MPC 语义保证                |

**示例**：
```python
# 全局同步的梯度下降
def cond_fn(state):
    loss = compute_loss(state)
    return loss > threshold  # uniform predicate

def body_fn(state):
    grad = compute_gradient(state)
    return state - learning_rate * grad

final_state = while_loop(cond_fn, body_fn, init_state)
```

#### 3.2.3 控制流的 Trace Time vs Runtime 检查

**Static Checks (Trace Time)**：
- Predicate 的 pmask 必须匹配 trace context mask
- Branch/Body 的输入输出类型必须匹配
- 不允许动态生成分支函数

**Dynamic Checks (Runtime)**：
- `verify_uniform=True` 时检查 predicate 各方是否一致
- 检查 pmask=None 的变量在运行时是否满足约束

**设计权衡**：
- ✅ **优点**：早期错误检测（编译时 > 运行时）
- ⚠️ **限制**：某些动态场景需要 workaround（如先 reveal 再 cond）

### 3.3 peval (Primitive Evaluation)

**核心作用**：调用 backend 的 kernel 并构建 IR

**签名**：
```python
def peval(
    pfunc: PFunction,
    args: list[MPObject],
    rmask: Mask | None = None,
) -> list[MPObject]:
    """Execute primitive function and create IR."""
```

**Rmask 语义**（重要！）：

| 场景                     | args pmask | rmask      | 结果 pmask          | 说明                          |
|--------------------------|------------|------------|---------------------|-------------------------------|
| 所有 args pmask 已知     | `Mask(0b11)` | `None`     | `Mask(0b11)`        | 自动推导                      |
| 所有 args pmask 已知     | `Mask(0b11)` | `Mask(0b01)` | `Mask(0b01)`        | 强制子集（验证合法性）        |
| 有 args pmask=None       | `None`     | `Mask(0b01)` | `Mask(0b01)`        | 强制使用 rmask                |
| 有 args pmask=None       | `None`     | `None`     | `None`              | 延迟到 runtime                |

**示例**：
```python
# 场景 1：自动推导
x = ...  # pmask=Mask(0b11)
y = ...  # pmask=Mask(0b11)
z = peval(add_kernel, [x, y])  # z.pmask = Mask(0b11)

# 场景 2：强制子集
z = peval(add_kernel, [x, y], rmask=Mask(0b01))
# ✅ 只有 P0 执行，z.pmask = Mask(0b01)

# 场景 3：动态 pmask
x = ...  # pmask=None (运行时确定)
z = peval(add_kernel, [x], rmask=Mask(0b11))
# ✅ 强制 z.pmask = Mask(0b11)，runtime 检查 x
```

**在 TraceContext 中的行为**：
```python
ctx = _tracer()

# 1. 构建 EvalExpr
arg_exprs = [arg.expr for arg in args]
fn_expr = EvalExpr(pfunc, arg_exprs, rmask)

# 2. 为每个输出创建 AccessExpr
ret_exprs = [AccessExpr(fn_expr, idx)
             for idx in range(fn_expr.num_outputs)]

# 3. 返回 TraceVar 列表
return [TraceVar(ctx, res) for res in ret_exprs]
```

### 3.4 Ops 层 (FeModule + FeOperation)

**架构模式**：Frontend (Ops) ↔ Backend (Kernels)

#### 3.4.1 FeModule (Frontend Module)

**作用**：组织相关的 operations（类似 Python module）

```python
# 定义 module
_CRYPTO_MOD = StatelessFeModule("crypto")

# 注册 operation
@_CRYPTO_MOD.simple_op(pfunc_name="crypto.encrypt")
def encrypt(x: TensorType, key: CustomType) -> SIMDHEType:
    """Type-only kernel: 不执行计算，只推导类型"""
    return SIMD_HE[x.element_type, x.shape]
```

**使用**：
```python
# 用户调用
from mplang.ops import crypto

ciphertext = crypto.encrypt(plaintext, key)
# 返回：MPObject with _type = SIMDHEType
```

#### 3.4.2 SimpleFeOperation (Typed Op)

**核心理念**：Kernel 只负责**类型推导**，不执行计算

**Triad ABI**：
```python
Triad = tuple[PFunction, list[MPObject], PyTreeDef]
#              ^           ^               ^
#              |           |               └─ Output structure
#              |           └─────────────────── Input MPObjects
#              └─────────────────────────────── Routing + Types + Attrs
```

**simple_op 的魔法转换**：
```python
# 1. 用户调用
result = crypto.encrypt(plaintext_mpobj, key_mpobj, key_size=2048)
#                       ^^^^^^^^^^^^^^^  ^^^^^^^^  ^^^^^^^^^^^^^^
#                       MPObject         MPObject  Python attr

# 2. SimpleFeOperation.trace() 内部
pos_mp_inputs = [plaintext_mpobj, key_mpobj]  # MPObjects
attr_kwargs = {"key_size": 2048}              # Attributes

# 3. 调用 kernel（传入类型，不是 MPObject）
call_pos_types = [plaintext_mpobj.mptype._type, key_mpobj.mptype._type]
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  TensorType/CustomType
result_type = encrypt_kernel(*call_pos_types, **attr_kwargs)
#             ^^^^^^^^^^^^^^ 返回 SIMDHEType

# 4. 构建 PFunction
pfunc = PFunction(
    fn_type="crypto.encrypt",
    ins_info=(TensorType(...), CustomType(...)),
    outs_info=(SIMDHEType(...),),
    key_size=2048  # 属性存在 PFunction 中
)

# 5. 返回 Triad
return (pfunc, pos_mp_inputs, out_tree)
```

**在 Primitive 中使用**：
```python
# mplang/ops/crypto.py
@_CRYPTO_MOD.simple_op()
def encrypt(x: TensorType, key: CustomType, *, key_size: int) -> SIMDHEType:
    if key.kind != "EncryptionKey":
        raise TypeError("Expected EncryptionKey")
    return SIMD_HE[x.element_type, x.shape]

# mplang/simp/api.py (用户级封装)
@function
def seal(data: MPObject) -> MPObject:
    key = get_encryption_key()
    pfunc, mp_inputs, out_tree = crypto.encrypt(data, key, key_size=4096)
    result_vars = peval(pfunc, mp_inputs)
    return tree_unflatten(out_tree, result_vars)
```

**关键约束**：
1. **Positional = MPObjects**：所有 MPObject 必须位置传参
2. **Keyword = Attributes**：所有 Python 值必须 keyword 传参
3. **Kernel = Types**：Kernel 接收 `TensorType`/`TableType`，不是 `MPObject`
4. **No *args/**kwargs**：签名必须显式（编译时验证）

#### 3.4.3 Backend Kernel 实现

**示例**：PHE (Paillier Homomorphic Encryption)

```python
# mplang/kernels/phe.py
class PHEKernelContext(KernelContext):
    def evaluate_pfunction(self, pfunc: PFunction, inputs: list[Value]) -> list[Value]:
        if pfunc.fn_type == "crypto.encrypt":
            # 实际执行加密
            plaintext = inputs[0].get_data()
            key = inputs[1].get_data()
            ciphertext = self.phe_lib.encrypt(plaintext, key)
            return [Value(ciphertext, pfunc.outs_info[0])]
        ...
```

**关键点**：
- Backend 不依赖 Ops 层（解耦）
- Backend 通过 `pfunc.fn_type` 路由到对应 kernel
- Backend 负责实际执行（调用底层库）

### 3.5 Data Shuffle Primitives

#### 3.5.1 `pshfl` (Dynamic Shuffle)

**语义**：根据运行时 `index` 重新分布数据

```python
def pshfl(src: MPObject, index: MPObject) -> MPObject:
    """
    src: 源数据 (pmask 指示谁持有数据)
    index: 目标方的索引 (各方本地值，指示从哪个方拉取)
    返回：按 index 重新分布的数据
    """
```

**示例**：
```
     P0   P1   P2
src: x0   -    x2   (pmask=0b101)
idx: -    0    -    (P1 想从 P0 拉取)
-----------------------
out: -    x0   -    (pmask=0b010)
```

**实现**：
```python
src_expr = src.expr
index_expr = index.expr
shfl_expr = ShflExpr(src_expr, index_expr)
return TraceVar(_tracer(), shfl_expr)
```

#### 3.5.2 `pshfl_s` (Static Shuffle)

**语义**：编译时确定的 shuffle 映射

```python
def pshfl_s(
    src_val: MPObject,
    pmask: Mask,           # 哪些方产生输出
    src_ranks: list[Rank]  # 每个输出方从哪个源 rank 拉取
) -> MPObject:
```

**示例**：
```
     P0   P1   P2
src: x0   x1   x2
pmask=[0,1,2], src_ranks=[2,0,1]
-----------------------
out: x2   x0   x1   (cross shuffle)
```

**优势**：
- ✅ 编译时优化（静态分析数据流）
- ✅ 更清晰的类型推导（输出 pmask 已知）

#### 3.5.3 `pconv` (Convergence)

**语义**：合并多个 disjoint 的 MPObject

```python
def pconv(vars: list[MPObject]) -> MPObject:
    """
    要求：vars 的 pmask 互不重叠
    返回：合并后的 MPObject (pmask = union of all pmasks)
    """
```

**示例**：
```
     P0   P1   P2
x0:  a0   -    -    (pmask=0b001)
x1:  -    b1   -    (pmask=0b010)
x2:  -    -    c2   (pmask=0b100)
-----------------------
out: a0   b1   c2   (pmask=0b111)
```

**使用场景**：
- 分布式计算后汇总结果
- 从不同方收集数据

**错误检查**：
```python
# 编译时/运行时检查
for i, var1 in enumerate(vars):
    for var2 in vars[i+1:]:
        if var1.pmask & var2.pmask != 0:
            raise ValueError("Overlapping pmasks")
```

---

## 4. 类型系统集成

### 4.1 MPObject 的类型接口

**当前架构**：
```python
class MPObject(ABC):
    @property
    @abstractmethod
    def _type(self) -> MPType:  # typing.MPType (新)
        """Multi-party distributed type."""

    @property
    @abstractmethod
    def pmask(self) -> Mask | None:
        """Party mask (from _type.pmask)."""

    @property
    @abstractmethod
    def mptype(self) -> MPType:  # mptype.MPType (旧，废弃)
        """DEPRECATED: Use _type instead."""
```

**MPType 定义**（`mplang.core.typing`）：
```python
class MPType(BaseType):
    """Multi-party distributed type = BaseType + Pmask"""
    def __init__(self, base_type: BaseType, pmask: Mask | None = None):
        self.base_type = base_type  # Tensor, HE, SIMD_HE, etc.
        self.pmask = pmask
```

**TraceVar 的类型**：
```python
class TraceVar(MPObject):
    @property
    def _type(self) -> MPType:
        # 从 Expr 提取
        return self._expr._type  # Expr 也需要提供 _type

    @property
    def pmask(self) -> Mask | None:
        return self._type.pmask
```

### 4.2 Expr 的类型系统

**当前**（使用旧 MPType）：
```python
class Expr:
    @property
    def mptypes(self) -> list[MPType]:  # mptype.MPType
        if self._mptypes is None:
            self._mptypes = self._compute_mptypes()
        return self._mptypes
```

**目标**（迁移到 typing.MPType）：
```python
class Expr:
    @property
    def output_types(self) -> list[MPType]:  # typing.MPType
        if self._output_types is None:
            self._output_types = self._compute_output_types()
        return self._output_types

    @property
    def output_type(self) -> MPType:
        """Single output convenience."""
        types = self.output_types
        if len(types) != 1:
            raise ValueError(f"Expected 1 output, got {len(types)}")
        return types[0]

    # 向后兼容
    @property
    def mptypes(self) -> list[mptype.MPType]:
        return [convert_to_legacy(t) for t in self.output_types]
```

### 4.3 类型转换函数

```python
# mplang/core/typing.py

def scalar_to_dtype(scalar: ScalarType) -> DType:
    """typing.ScalarType → dtypes.DType"""
    return {f32: DType.FLOAT32, i32: DType.INT32, ...}[scalar]

def dtype_to_scalar(dtype: DType) -> ScalarType:
    """dtypes.DType → typing.ScalarType"""
    return {DType.FLOAT32: f32, DType.INT32: i32, ...}[dtype]

def to_legacy_tensor_type(t: TensorType) -> tensor.TensorType:
    """typing.TensorType → tensor.TensorType (legacy)"""
    return tensor.TensorType(
        dtype=scalar_to_dtype(t.element_type),
        shape=t.shape
    )

def from_legacy_mptype(old: mptype.MPType) -> typing.MPType:
    """mptype.MPType → typing.MPType"""
    if isinstance(old._type, tensor.TensorType):
        base = Tensor[
            dtype_to_scalar(old._type.dtype),
            old._type.shape
        ]
    # ... handle other types
    return typing.MPType(base, old.pmask)
```

---

## 5. 当前架构的优势与限制

### 5.1 优势

#### ✅ 1. 上下文感知的自动化
- 用户代码在 Trace/Interp 模式下自动切换
- 无需手动管理 IR 构建

#### ✅ 2. 类型安全
- `MPType` 携带 pmask 和 encryption 信息
- 编译时验证类型兼容性

#### ✅ 3. 控制流的显式性
- `uniform_cond`, `while_loop` 明确语义
- 静态检查 predicate uniformity

#### ✅ 4. 模块化 Ops 系统
- Frontend (Ops) 和 Backend (Kernels) 解耦
- `simple_op` 简化类型推导逻辑

#### ✅ 5. PyTree 支持
- 自动 flatten/unflatten 嵌套结构
- 用户可使用 dict, list, dataclass 等

### 5.2 限制与改进方向

#### ⚠️ 1. Expr Tree 的局限性

**当前问题**：
- 树状结构难以做 CFG 分析
- Visitor 模式冗长
- 无法直接表示 DAG（共享子表达式需特殊处理）

**改进方向**（未来）：
```python
# 迁移到 Operation List (类似 torch.fx, JAX)
class Graph:
    operations: list[Operation]  # 扁平化
    values: dict[str, Value]     # SSA values

class Operation:
    opcode: str
    inputs: list[Value]
    outputs: list[Value]
```

**优势**：
- ✅ 扁平化：易于遍历和变换
- ✅ SSA 形式：use-def 链清晰
- ✅ 易于优化：fusion, dead code elimination
- ✅ 对接 MLIR/XLA：更自然

#### ⚠️ 2. Primitive 装饰器的冗余

**当前问题**：
- `@function` 和 `@builtin_function` 本质是上下文切换逻辑
- 可以用显式 Tracer 对象替代

**未来方向**（参考 torch.fx）：
```python
class Tracer:
    def trace(self, fn, *args):
        with set_tracer(self):
            return fn(*args)

    def call(self, op_name, inputs, output_types):
        # Record operation
        ...

# 用户代码（不需要 @primitive）
def add(x, y):
    tracer = get_current_tracer()
    if tracer:
        return tracer.call("add", [x, y], [...])
    else:
        return x + y  # Eager mode
```

#### ⚠️ 3. 控制流的啰嗦性

**当前**：
```python
result = uniform_cond(
    pred,
    lambda x: x + 1,
    lambda x: x - 1,
    x
)
```

**理想**（需要 AST transformation）：
```python
@auto_trace
def compute(x):
    if x > 0:  # 自动转换为 uniform_cond
        return x + 1
    else:
        return x - 1
```

**权衡**：
- AST transformation 复杂度高
- 当前 lambda 方式简单、可预测

---

## 6. 未来演进路径

### Phase 1: 完成类型系统迁移 (当前分支)

**目标**：统一到 `typing.MPType`

**任务**：
1. 更新 `Expr.output_types` → `list[typing.MPType]`
2. `TraceVar._type` → `typing.MPType`
3. `InterpVar._type` → `typing.MPType`
4. 添加类型转换辅助函数
5. 废弃 `mptype.MPType`（保留向后兼容）

**不改变**：
- Expr Tree 架构
- Primitive 装饰器
- Visitor 模式

### Phase 2: Operation List IR (未来大重构)

**目标**：迁移到扁平化 IR

**参考**：
- torch.fx: `Graph` + `Node`
- JAX: `jaxpr` (Jaxpr equations)
- MLIR: `Operation` + `Block` + `Region`

**关键改动**：
1. **新 IR 表示**：
   ```python
   class MPGraph:
       values: dict[str, Value]       # SSA values
       operations: list[Operation]    # 扁平化操作列表

   class Operation:
       opcode: str                    # "add", "cond", "while", ...
       inputs: list[Value]
       outputs: list[Value]
       attrs: dict[str, Any]
       regions: list[MPGraph] = []    # For cond/while (nested graphs)
   ```

2. **Builder API**：
   ```python
   builder = GraphBuilder()
   v0 = builder.constant(1.0, MPType(...))
   v1 = builder.add(x, v0)
   v2 = builder.mul(v1, y)
   ```

3. **控制流**：
   ```python
   # 方案 A: 结构化（类似 JAX）
   class CondOp(Operation):
       opcode = "cond"
       inputs: [pred, *args]
       regions: [then_graph, else_graph]

   # 方案 B: Basic Blocks（类似 MLIR）
   entry_block = Block()
   then_block = Block()
   else_block = Block()
   entry_block.terminator = CondBranchOp(pred, then_block, else_block)
   ```

### Phase 3: 简化 Primitive 系统 (依赖 Phase 2)

**目标**：移除或简化 `@primitive` 装饰器

**方案**：
1. **显式 Tracer**：
   ```python
   tracer = Tracer(cluster_spec)
   graph, result = tracer.trace(my_function, x, y)
   ```

2. **操作符重载**（类似 JAX）：
   ```python
   class TracedArray(MPObject):
       def __add__(self, other):
           return get_current_tracer().call("add", [self, other], [...])
   ```

3. **保留控制流 helpers**：
   ```python
   # 仍然需要显式控制流（Python if 无法自动 trace）
   result = mplang.cond(pred, then_fn, else_fn, x)
   result = mplang.while_loop(cond_fn, body_fn, init)
   ```

### Phase 4: 类型推导与检查 (长期)

**目标**：在 `peval` 中添加可选的类型检查

```python
def peval(pfunc, *args, rmask=None, *, type_check=False):
    if type_check:
        # 从函数签名提取类型注解
        hints = get_type_hints(pfunc, include_extras=True)
        for arg, expected_type in zip(args, param_types):
            if not type_matches(arg._type, expected_type):
                raise TypeError(f"Type mismatch: {arg._type} != {expected_type}")
```

---

## 7. 设计决策总结

### 7.1 核心决策

| 决策                  | 选择                        | 原因                              |
|-----------------------|----------------------------|-----------------------------------|
| 函数作为值            | ❌ 不支持                   | 保持 IR 静态、简化编译器          |
| TracedFunction 类型   | 专门类（非 TraceVar）       | 类型安全、保留元数据              |
| 控制流                | 显式 primitives             | 语义清晰、可静态分析              |
| 上下文切换            | 自动（via `@primitive`）    | 用户友好                          |
| Ops 层                | Frontend + Backend 解耦     | 可扩展性                          |
| IR 表示               | Expr Tree (当前)            | 简单、够用；未来可迁移到 Op List  |

### 7.2 与其他框架对比

| 特性                  | MPLang          | JAX             | PyTorch 2.0     | TensorFlow 2    |
|-----------------------|-----------------|-----------------|-----------------|-----------------|
| 函数式                | ✅ 纯函数        | ✅ 纯函数        | ⚠️ 部分         | ❌ 命令式       |
| 静态 IR               | ✅ 完全静态      | ✅ 完全静态      | ⚠️ 混合         | ⚠️ 混合         |
| 控制流                | 显式 primitives  | 显式 lax.* | 自动 trace      | tf.cond/while   |
| 类型系统              | ✅ 强类型 + pmask| ⚠️ 弱类型       | ⚠️ 动态         | ⚠️ 动态         |
| 多方语义              | ✅ 第一类        | ❌              | ❌              | ❌              |
| 上下文管理            | Trace/Interp     | JIT/Eager       | compile/eager   | Graph/Eager     |

### 7.3 最佳实践

#### ✅ 推荐

1. **使用 `@function` 组合逻辑**：
   ```python
   @function
   def my_pipeline(x):
       x = preprocess(x)
       x = model(x)
       return postprocess(x)
   ```

2. **用 `@builtin_function` 封装复杂 ops**：
   ```python
   @builtin_function
   def secure_aggregation(values):
       # 复杂的 MPC 协议
       sealed = [seal(v) for v in values]
       aggregated = sum(sealed)
       return reveal(aggregated)
   ```

3. **显式控制流**：
   ```python
   # ✅ Good
   result = uniform_cond(pred, then_fn, else_fn, x)

   # ❌ Bad (无法 trace)
   if pred:
       result = then_fn(x)
   else:
       result = else_fn(x)
   ```

4. **Ops 定义时分离类型和执行**：
   ```python
   # Ops layer: 只做类型推导
   @_MOD.simple_op()
   def my_op(x: TensorType) -> TensorType:
       return TensorType(x.dtype, x.shape)

   # Kernel layer: 实际执行
   class MyKernel(KernelContext):
       def evaluate(self, pfunc, inputs):
           return compute_result(inputs)
   ```

#### ❌ 避免

1. **不要在 `@function` 中使用 Python 控制流**：
   ```python
   @function
   def bad(x):
       if x > 0:  # ❌ 无法 trace
           return x
       return -x
   ```

2. **不要混淆 Trace/Runtime 边界**：
   ```python
   # ❌ Bad
   def compute(x):
       print(x.shape)  # shape 在 trace 时可能未知

   # ✅ Good
   @function
   def compute(x):
       # 在 peval 中访问 mptype
       return peval(kernel, [x])
   ```

3. **不要返回捕获了 TraceVar 的函数**：
   ```python
   # ❌ Bad
   @function
   def make_adder(x):
       return lambda y: x + y  # 捕获 x (TraceVar)

   # ✅ Good
   @function
   def add_with_context(x, y):
       return x + y
   ```

---

## 8. 结论

MPLang 的 EDSL 架构基于以下核心理念：

1. **封闭世界**：函数是数据转换管道，控制流必须显式
2. **类型驱动**：MPType 携带分布、加密等多方语义
3. **双上下文**：Trace 构建 IR，Interp 执行计算
4. **模块化**：Frontend (Ops) 专注类型，Backend (Kernels) 专注执行

**当前架构**：成熟、可用，适合多方计算场景

**未来方向**：
- **Phase 1**（当前）：完成类型系统统一
- **Phase 2**（未来）：迁移到 Operation List IR
- **Phase 3**（长期）：简化 Primitive 系统

**关键洞察**：
- Expr Tree → Operation List 是现代 EDSL 的趋势（torch.fx, JAX）
- Primitive 装饰器可以用显式 Tracer 替代
- 控制流的显式性是多方计算的必要约束

---

## 附录 A：术语表

| 术语                | 定义                                          |
|---------------------|-----------------------------------------------|
| EDSL                | Embedded Domain-Specific Language             |
| TraceContext        | 构建计算图的上下文                            |
| InterpContext       | 执行计算的上下文                              |
| TraceVar            | 存储 Expr 的变量（lazy）                      |
| InterpVar           | 引用已计算值的变量（eager）                   |
| TracedFunction      | 包含 FuncDefExpr 和元数据的可调用对象         |
| Expr                | IR 节点（VariableExpr, CallExpr, etc.）      |
| Primitive           | 带 `@function` 或 `@builtin_function` 的函数  |
| PFunction           | 描述 backend 操作的元数据（类型、属性）       |
| FeOperation         | Frontend 操作（Ops 层）                       |
| KernelContext       | Backend 执行引擎（Kernels 层）                |
| Triad               | `(PFunction, list[MPObject], PyTreeDef)`      |
| MPType              | 多方分布式类型（BaseType + Pmask）            |
| Pmask               | Party mask（哪些方持有数据）                  |
| Rmask               | Runtime mask（哪些方执行计算）                |
| SSA                 | Static Single Assignment                      |
| CFG                 | Control Flow Graph                            |

## 附录 B：代码示例

### 完整的 E2E 示例

```python
import mplang
from mplang.simp import constant, seal, reveal

# 1. 定义计算逻辑
@mplang.function
def secure_sum(x: mplang.MPObject, y: mplang.MPObject) -> mplang.MPObject:
    # Step 1: Seal inputs
    x_sealed = seal(x)
    y_sealed = seal(y)

    # Step 2: Compute on sealed data
    sum_sealed = x_sealed + y_sealed

    # Step 3: Reveal result
    result = reveal(sum_sealed)

    return result

# 2. 创建集群配置
cluster_spec = mplang.ClusterSpec([
    mplang.Node(rank=0, address="localhost:50051"),
    mplang.Node(rank=1, address="localhost:50052"),
])

# 3. Trace (构建计算图)
tracer = mplang.TraceContext(cluster_spec)
x_var = constant(tracer, [1, 2, 3])
y_var = constant(tracer, [4, 5, 6])

with mplang.with_ctx(tracer):
    tfn = mplang.trace(tracer, secure_sum, x_var, y_var)
    print(tfn.compiler_ir())  # 打印 IR

# 4. Compile
compiled_fn = mplang.compile(
    secure_sum,
    cluster_spec=cluster_spec,
    backend="simulation"
)

# 5. Execute
simulator = mplang.Simulator(cluster_spec)
result = mplang.evaluate(simulator, compiled_fn, [1,2,3], [4,5,6])
output = mplang.fetch(simulator, result)

print(f"Result: {output}")  # [[5, 7, 9], [5, 7, 9]]
```

---

**审阅者请重点关注**：
1. 是否有遗漏的核心组件？
2. 未来演进路径是否合理？
3. 是否有更好的设计替代方案？
