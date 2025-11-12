# Promote Timing: Design Rationale

## 问题背景

在 `Tracer` 中，`InterpObject → TraceObject` 的 promote 时机非常关键。之前的设计在 `execute_add()` 中自动 promote，这会导致 **captured variable 处理问题**。

## 为什么 trace(fn) 中的 promote 不行？

### ❌ 错误场景：在 execute_add() 中 promote

```python
# 之前的实现（已修复）
class Tracer:
    def execute_add(self, left: Object, right: Object) -> TraceObject:
        # 自动 promote InterpObject
        if isinstance(left, InterpObject):
            left = self.promote(left)  # ❌ 问题！
        ...
```

**问题**：当 `fn` 有 captured variables 时会出错：

```python
def outer():
    y = some_interp_object  # captured variable

    def inner(x):
        return x + y  # y 是 captured，在这里触发 execute_add

    return trace(inner, x_arg)
```

在 `x + y` 时：
1. `x` 是 `TraceObject`（trace 入口转换的参数）
2. `y` 是 `InterpObject`（captured variable）
3. `TraceObject.__add__(y)` → `execute_add(x_trace, y_interp)`
4. 旧实现会在 `execute_add` 中 promote `y`，但此时 **无法区分 y 是 captured variable 还是正常参数**

### ✅ 正确方案：在 primitive call 边界 promote

```python
class Primitive:
    def _bind_trace(self, tracer, args, kwargs):
        # 在这里 promote
        trace_args = []
        for arg in args:
            if isinstance(arg, TraceObject):
                trace_args.append(arg)
            else:
                # InterpObject → TraceObject (promote)
                trace_args.append(tracer.promote(arg))
        ...
```

**为什么这里可以？**

1. **Primitive 内部不会有 captured variables** - primitive 是原子操作，不会嵌套定义函数
2. **Context 明确** - 在 trace context 中调用 primitive，所有 InterpObject 都是显式传入的参数
3. **边界清晰** - primitive call 是 Python → Graph IR 的明确边界点

## 修复后的设计

### 1. `Tracer.execute_add()` - 不再自动 promote

```python
def execute_add(self, left: Object, right: Object) -> TraceObject:
    """现在要求所有操作数都是 TraceObject。

    如果收到 InterpObject，说明是 captured variable，
    应该在 primitive call 边界处理，而不是在这里。
    """
    if not isinstance(left, TraceObject) or not isinstance(right, TraceObject):
        raise TypeError(
            "execute_add expects TraceObject operands. "
            "InterpObject promotion should happen at primitive call boundaries."
        )
    ...
```

### 2. `Primitive._bind_trace()` - 在这里 promote

```python
def _bind_trace(self, tracer, args, kwargs):
    """在 primitive call 边界 promote InterpObject。"""
    trace_args = []
    for arg in args:
        if isinstance(arg, TraceObject):
            trace_args.append(arg)
        else:
            # InterpObject → TraceObject (作为 captured variable)
            trace_args.append(tracer.promote(arg))
    ...
```

### 3. `trace()` 支持 `Callable | Primitive`

```python
def trace(fn: Callable | Primitive, *args) -> Graph:
    """支持直接 trace Primitive。

    - 如果 fn 是 Primitive: 直接调用 bind()，让 primitive 处理 promote
    - 如果 fn 是 Callable: 转换参数为 TraceObject，然后执行
    """
    tracer = Tracer()

    if isinstance(fn, Primitive):
        # Primitive 会在 _bind_trace 中处理 promote
        ctx.enter_context(tracer)
        result = fn.bind(*args)
        ctx.exit_context()
        ...
    else:
        # Callable: 转换参数，然后执行
        trace_args = [convert_to_trace(arg) for arg in args]
        ctx.enter_context(tracer)
        result = fn(*trace_args)
        ctx.exit_context()
        ...
```

## 测试验证

新增测试确保正确性：

1. **`test_trace_primitive_directly`**: 验证 `trace(primitive, args)` 工作正常
2. **`test_execute_add_rejects_interp_object`**: 验证 `execute_add()` 拒绝 InterpObject
3. **`test_execute_add_rejects_interp_object`**: 验证 `primitive.bind()` 正确处理 promote

## 总结

| 场景 | Promote 位置 | 原因 |
|-----|------------|------|
| **trace(lambda x: x + y, ...)** | ❌ 不在 `execute_add` | captured variable 无法正确处理 |
| **primitive.bind(x, y)** | ✅ 在 `_bind_trace` | primitive 无 capture，边界清晰 |
| **trace(primitive, ...)** | ✅ 在 `_bind_trace` | primitive 处理 promote |

**关键原则**: Promote 应该发生在 **明确的、无 captured variable 的边界点**，即 **primitive call 时**。
