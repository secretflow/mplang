# Type System Migration: MPType → BaseType

## 目标

将 MPLang 的类型系统从双轨制（`MPType` + `BaseType`）统一到单一的 `BaseType` 系统。

## 当前架构问题

### 1. 双轨类型系统

```python
# typing.py - 类型表达式系统 (新)
Tensor[f32, (10,)]  # typing.TensorType
SIMD_HE[f32, (4096,)]  # typing.SIMDHEType
CustomType('EncryptionKey')  # typing.CustomType

# mptype.py + tensor.py - 运行时类型系统 (旧)
MPType(
    _type=TensorType(DType.FLOAT32, (10,)),  # tensor.TensorType
    pmask=Mask.from_int(0b11),
    attrs={}
)
```

**问题**：
- 两个 `TensorType`：`typing.TensorType` 和 `tensor.TensorType`
- 类型信息重复：`typing.Tensor[f32, (10,)]` vs `MPType(..., shape=(10,), dtype=f32)`
- 类型注解（`x: MPObject[Tensor[...]]`）和运行时类型（`x.mptype`）完全脱节

### 2. MPObject 的类型接口混乱

```python
class MPObject:
    @property
    def mptype(self) -> MPType:  # 旧接口
        ...

    # 以下都是从 mptype 派生的便利属性
    @property
    def dtype(self) -> DType:
        return self.mptype.dtype

    @property
    def shape(self) -> Shape:
        return self.mptype.shape

    @property
    def pmask(self) -> Mask | None:
        return self.mptype.pmask

    @property
    def attrs(self) -> dict[str, Any]:
        return self.mptype.attrs
```

**问题**：
- `MPType` 本质上只是 `(_type, pmask, attrs)` 的容器
- 多一层间接：`obj.mptype._type` 而不是 `obj._type`
- `MPType` 在 `typing.py` 中已经有对应物：`MP[T]`

## 目标架构

### 1. 统一的类型系统

```python
# mplang.core.typing - 唯一的类型系统
from mplang.core.typing import BaseType, Tensor, SIMD_HE, MP, f32

# 类型表达式
pt_type = Tensor[f32, (4096,)]                    # 明文
ct_type = SIMD_HE[f32, (4096,)]                   # 密文
distributed = MP[Tensor[f32, (4096,)]]            # 分布式

# CustomType 表示不透明对象
key_type = CustomType('EncryptionKey')
```

### 2. 简化的 MPObject 接口

```python
class MPObject(ABC):
    @property
    @abstractmethod
    def _type(self) -> BaseType:
        """The type expression from mplang.core.typing."""

    @property
    @abstractmethod
    def pmask(self) -> Mask | None:
        """Which parties hold this object."""

    @property
    def attrs(self) -> dict[str, Any]:
        """Additional attributes."""
        return {}

    @property
    def ctx(self) -> MPContext:
        """The execution context."""

    # 便利属性（从 _type 推导）
    @property
    def dtype(self) -> DType:
        """Extract dtype from _type."""
        if isinstance(self._type, Tensor):
            return scalar_to_dtype(self._type.element_type)
        # ... handle other types

    @property
    def shape(self) -> Shape:
        """Extract shape from _type."""
        if isinstance(self._type, Tensor):
            return self._type.shape
        # ...
```

### 3. 废弃 MPType

```python
# mptype.py - 标记为废弃
class MPType:
    """DEPRECATED: Use MPObject._type, .pmask, .attrs instead.

    This class is being phased out. Use BaseType from mplang.core.typing.
    """
    ...

# typing.py - MP[T] 替代 MPType
class MPType(BaseType):
    """Multi-party distributed type.

    Represents data distributed across multiple parties.

    Examples:
        MP[Tensor[f32, (10,)]]  # Distributed tensor
        MP[SIMD_HE[f32, (4096,)]]  # Distributed ciphertext
    """
    def __init__(self, base_type: BaseType):
        self.base_type = base_type
```

## 迁移步骤

### Phase 1: 添加 `_type` 属性 ✅

**目标**：保持向后兼容，同时添加新接口

```python
class MPObject(ABC):
    @property
    @abstractmethod
    def _type(self) -> BaseType:  # 新接口
        """The unified type expression."""

    @property
    @abstractmethod
    def pmask(self) -> Mask | None:  # 提升为顶层属性
        """Party mask."""

    @property
    @abstractmethod
    def mptype(self) -> MPType:  # 保留旧接口（标记为废弃）
        """DEPRECATED: Use _type, pmask, attrs instead."""
```

**实现**：
- [x] 修改 `MPObject` 基类，添加 `_type` 和 `pmask` 抽象属性
- [x] 标记 `mptype` 为废弃（但保留）
- [x] 更新文档字符串

### Phase 2: 更新 TraceVar 和 InterpVar

**目标**：实现新接口

```python
class TraceVar(MPObject):
    @property
    def _type(self) -> BaseType:
        return self._expr._type  # Expr 需要提供 _type

    @property
    def pmask(self) -> Mask | None:
        return self._expr.pmask

    @property
    def attrs(self) -> dict[str, Any]:
        return self._expr.attrs

    @property
    def mptype(self) -> MPType:  # 向后兼容
        # 从 _type 构建 MPType
        return MPType.from_base_type(self._type, self.pmask, self.attrs)

class InterpVar(MPObject):
    def __init__(self, ctx, _type: BaseType, pmask=None, attrs=None):
        self._ctx = ctx
        self.__type = _type
        self._pmask = pmask
        self._attrs = attrs or {}

    @property
    def _type(self) -> BaseType:
        return self.__type

    @property
    def pmask(self) -> Mask | None:
        return self._pmask

    @property
    def attrs(self) -> dict[str, Any]:
        return self._attrs

    @property
    def mptype(self) -> MPType:  # 向后兼容
        return MPType.from_base_type(self._type, self.pmask, self.attrs)
```

**实现**：
- [ ] 更新 Expr 系统，添加 `_type`, `pmask`, `attrs` 属性
- [ ] 更新 TraceVar 实现新接口
- [ ] 更新 InterpVar 实现新接口
- [ ] 添加 `MPType.from_base_type()` 辅助方法
- [ ] 运行测试确保兼容性

### Phase 3: 类型转换辅助函数

**目标**：实现 `typing.BaseType` ↔ `tensor.TensorType`/`DType` 转换

```python
# mplang/core/typing.py

def scalar_to_dtype(scalar: ScalarType) -> DType:
    """Convert typing.ScalarType to DType."""
    mapping = {
        "f32": DType.FLOAT32,
        "f64": DType.FLOAT64,
        "i32": DType.INT32,
        "i64": DType.INT64,
    }
    return mapping[scalar._name]

def dtype_to_scalar(dtype: DType) -> ScalarType:
    """Convert DType to typing.ScalarType."""
    mapping = {
        DType.FLOAT32: f32,
        DType.FLOAT64: f64,
        DType.INT32: i32,
        DType.INT64: i64,
    }
    return mapping[dtype]

def to_runtime_tensor_type(base_type: BaseType) -> TensorType:
    """Convert typing.TensorType to tensor.TensorType (legacy)."""
    if isinstance(base_type, Tensor):
        return TensorType(
            dtype=scalar_to_dtype(base_type.element_type),
            shape=base_type.shape
        )
    raise TypeError(f"Cannot convert {type(base_type)} to TensorType")

def from_runtime_tensor_type(tensor_type: TensorType) -> BaseType:
    """Convert tensor.TensorType to typing.TensorType."""
    return Tensor[
        dtype_to_scalar(tensor_type.dtype),
        tensor_type.shape
    ]
```

**实现**：
- [ ] 实现 `scalar_to_dtype` / `dtype_to_scalar`
- [ ] 实现 `to_runtime_tensor_type` / `from_runtime_tensor_type`
- [ ] 添加单元测试

### Phase 4: 更新 Expr 系统

**目标**：Expr 直接存储 `BaseType`

```python
# mplang/core/expr/__init__.py

@dataclass
class Expr:
    _types: tuple[BaseType, ...]  # 替代 mptypes
    pmasks: tuple[Mask | None, ...]
    attrs: tuple[dict[str, Any], ...]

    # 向后兼容
    @property
    def mptypes(self) -> tuple[MPType, ...]:
        """DEPRECATED: Returns legacy MPType objects."""
        return tuple(
            MPType.from_base_type(t, p, a)
            for t, p, a in zip(self._types, self.pmasks, self.attrs)
        )
```

**实现**：
- [ ] 更新 Expr 基类
- [ ] 更新所有 Expr 子类（VariableExpr, CallExpr, etc.）
- [ ] 运行所有测试

### Phase 5: 更新 peval 和 simple_op

**目标**：在 peval 中提取类型注解并做类型推导

```python
from typing import get_type_hints

def extract_type_expr(type_hint) -> BaseType | None:
    """从类型注解提取 BaseType 表达式."""
    if type_hint is MPObject or type_hint is None:
        return None
    if hasattr(type_hint, '__metadata__'):  # Annotated[MPObject, TypeExpr]
        return type_hint.__metadata__[0]
    return None

def peval(pfunc, *args, rmask=None, *, type_check=False):
    """Execute primitive function and create IR.

    Args:
        pfunc: The primitive function to call
        args: MPObject arguments (TraceVar or InterpVar)
        rmask: Runtime mask
        type_check: If True, validate args against function type annotations
    """
    # 1. 提取类型注解
    if type_check:
        hints = get_type_hints(pfunc, include_extras=True)
        param_names = list(inspect.signature(pfunc).parameters.keys())
        param_types = [extract_type_expr(hints.get(name)) for name in param_names]

        # 2. 验证参数类型
        for arg, expected_type in zip(args, param_types):
            if expected_type is not None:
                if not type_matches(arg._type, expected_type):
                    raise TypeError(
                        f"Argument type mismatch: "
                        f"expected {expected_type}, got {arg._type}"
                    )

    # 3. 调用 pfunc（可能推导返回类型）
    ...
```

**实现**：
- [ ] 实现 `extract_type_expr`
- [ ] 实现 `type_matches` (类型匹配/统一)
- [ ] 在 `peval` 中添加可选的 `type_check`
- [ ] 测试类型检查功能

### Phase 6: 渐进式废弃 MPType

**目标**：逐步移除 `MPType` 使用

1. [ ] 更新所有内部代码使用 `_type` 而不是 `mptype`
2. [ ] 更新文档和示例
3. [ ] 添加废弃警告：`warnings.warn("mptype is deprecated, use _type")`
4. [ ] 在主要版本更新时移除 `MPType` 类

## 最终目标

```python
# 清晰的类型系统
from mplang import MPObject, function
from mplang.core.typing import Tensor, SIMD_HE, CustomType, f32

# 类型别名
PlaintextVec = MPObject[Tensor[f32, (4096,)]]
CiphertextVec = MPObject[SIMD_HE[f32, (4096,)]]
EncryptionKey = MPObject[CustomType('EncryptionKey')]

# 类型化函数
@function
def encrypt(
    data: PlaintextVec,
    key: EncryptionKey,
) -> CiphertextVec:
    ...

# 运行时
x = trace_var  # x._type == Tensor[f32, (4096,)]
result = encrypt(x, key)  # result._type == SIMD_HE[f32, (4096,)]

# 类型检查（在 peval 中）
peval(encrypt, x, key, type_check=True)  # ✓ 验证类型匹配
```

## 优势

1. **单一真相来源**：只有 `BaseType` 类型系统
2. **类型注解有意义**：`x: MPObject[Tensor[...]]` 的类型表达式就是 `x._type`
3. **简化接口**：`obj._type` 而不是 `obj.mptype._type`
4. **更强的类型安全**：peval 可以验证类型匹配
5. **更好的表达力**：支持 HE、SS、MP、CustomType 等高级类型
6. **向后兼容**：渐进式迁移，不破坏现有代码

## 当前进度

- [x] Phase 1: 添加 `_type` 属性到 MPObject
- [ ] Phase 2: 更新 TraceVar 和 InterpVar
- [ ] Phase 3: 类型转换辅助函数
- [ ] Phase 4: 更新 Expr 系统
- [ ] Phase 5: peval 类型推导
- [ ] Phase 6: 废弃 MPType
