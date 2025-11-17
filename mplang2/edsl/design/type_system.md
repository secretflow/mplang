# RFC: New Type System for MPLang

**Status**: Proposal (Implementation in Progress)
**Author**: MPLang Team
**Created**: 2025-01-10
**Updated**: 2025-01-10

## 执行摘要

本 RFC 提出用新的参数化类型系统（`mplang.core.typing`）替换当前的 `MPType` 类型系统，以支持：

1. **加密类型**: HE (同态加密)、SIMD_HE (SIMD 同态加密)、SS (秘密分享)
2. **不透明类型**: CustomType (密钥、句柄等)
3. **类型安全的函数签名**: `MPObject[Tensor[f32, (10,)]]`
4. **更好的 IDE 支持和静态检查**

## 当前类型系统的限制

### 1. MPType 的表达能力不足

**当前设计** (`mplang/core/mptype.py`):

```python
class MPType:
    _type: TensorType | TableType  # ← 只支持这两种
    _pmask: Mask | None
    _attrs: dict[str, Any]
```

**问题**:

- ❌ 无法表达加密类型 (HE、SS、SIMD_HE)
- ❌ 无法表达不透明类型 (密钥、句柄)
- ❌ 无法表达参数化类型 (如 `Tensor[HE[f32], (10,)]`)
- ❌ 类型信息只能靠 `attrs` 携带，不是 first-class

### 2. 缺乏类型注解支持

**当前状况**:

```python
# ❌ 无法表达具体类型
def encrypt(x: MPObject) -> MPObject:
    # x 是什么类型？返回值是什么类型？不知道
    pass

# ❌ 即使用 Annotated，也无法在 runtime 使用
from typing import Annotated
def encrypt(x: Annotated[MPObject, "Tensor[f32, (4096,)]"]) -> MPObject:
    # x.mptype 仍然是 MPType(TensorType(...))
    # annotation 只是文档，无法参与类型推导
    pass
```

### 3. Ops 实现复杂

**当前加密 ops 的实现困境**:

```python
# mplang/ops/crypto.py (假设)

# ❌ 问题：无法表达返回类型是密文
@_CRYPTO_MOD.simple_op()
def encrypt(x: TensorType, key: ???) -> ???:
    # key 是什么类型？无法表达
    # 返回值是密文，但只能返回 TensorType + attrs
    return TensorType(x.dtype, x.shape)  # 丢失了"这是密文"的信息
```

**Workaround (使用 attrs)**:

```python
# 当前的临时方案
result_mptype = MPType(
    TensorType(x.dtype, x.shape),
    pmask=x.pmask,
    attrs={"encrypted": True, "scheme": "SIMD_HE"}  # ← 类型信息降级为 attrs
)
```

**问题**:

- ❌ 类型信息散落在 attrs，难以维护
- ❌ 无法做类型检查（attrs 是 `dict[str, Any]`）
- ❌ 容易出错（拼写错误、值类型不一致）

## 新类型系统设计

### 核心原则

基于 **三个正交维度** 的组合：

1. **布局类型** (Layout): 描述数据的物理结构
   - `Scalar`: 标量 (f32, i64)
   - `Tensor`: 多维数组
   - `Table`: 表格结构
   - `CustomType`: 不透明类型

2. **加密类型** (Encryption): 描述隐私保护方式
   - `HE[Scalar]`: 元素级同态加密
   - `SIMD_HE[Scalar, Shape]`: SIMD 打包同态加密
   - `SS[BaseType]`: 秘密分享

3. **分布类型** (Distribution): 描述多方分布
   - `MP[BaseType]`: 多方分布式数据

### 类型组合示例

```python
from mplang.core.typing import Tensor, HE, SIMD_HE, SS, MP, CustomType, f32

# 明文 tensor
plaintext = Tensor[f32, (4096,)]

# 元素级加密 tensor (World 2)
he_tensor = Tensor[HE[f32], (4096,)]

# SIMD 打包加密 (World 3 - 不是 Tensor)
simd_ciphertext = SIMD_HE[f32, (4096,)]

# 秘密分享的 tensor
ss_tensor = SS[Tensor[f32, (4096,)]]

# 多方分布式数据
distributed = MP[Tensor[f32, (4096,)]]

# 不透明类型
encryption_key = CustomType("EncryptionKey")
```

### 统一接口

```python
class MPObject(ABC):
    @property
    @abstractmethod
    def _type(self) -> BaseType:  # 新主接口
        """The type expression from mplang.core.typing."""

    @classmethod
    def __class_getitem__(cls, type_expr):
        """Enable MPObject[TypeExpr] syntax."""
        return Annotated[MPObject, type_expr]
```

## 使用场景对比

### 场景 1: 加密计算

#### Before (当前方案)

```python
# ❌ 类型信息丢失
def my_computation(x, y):
    # x, y 是什么类型？不知道
    key = keygen()  # key 是什么？不知道
    x_enc = encrypt(x, key)  # x_enc 是密文，但类型系统不知道
    y_enc = encrypt(y, key)
    z_enc = add(x_enc, y_enc)  # 能加吗？运行时才知道
    z = decrypt(z_enc, key)
    return z
```

#### After (新方案)

```python
from mplang import MPObject
from mplang.core.typing import Tensor, SIMD_HE, CustomType, f32

# Type aliases
PlaintextVec = MPObject[Tensor[f32, (4096,)]]
CiphertextVec = MPObject[SIMD_HE[f32, (4096,)]]
EncryptionKey = MPObject[CustomType("EncryptionKey")]

def my_computation(
    x: PlaintextVec,
    y: PlaintextVec,
) -> PlaintextVec:
    key: EncryptionKey = keygen(key_size=4096)
    x_enc: CiphertextVec = encrypt(x, key)
    y_enc: CiphertextVec = encrypt(y, key)
    z_enc: CiphertextVec = add(x_enc, y_enc)
    z: PlaintextVec = decrypt(z_enc, key)
    return z
```

**优势**:

- ✅ 类型清晰：一眼看出哪些是明文、哪些是密文
- ✅ IDE 支持：自动补全、类型提示
- ✅ 可选的 runtime 验证：`peval(..., type_check=True)`

### 场景 2: Ops 定义

#### Before (当前方案)

```python
# ❌ 无法表达密文类型
@_CRYPTO_MOD.simple_op()
def encrypt(x: TensorType, key: TensorType) -> TensorType:
    # 返回的明明是密文，但只能返回 TensorType
    # 类型信息靠 attrs 携带（脆弱）
    return TensorType(x.dtype, x.shape)
```

#### After (新方案)

```python
from mplang.core.typing import TensorType, SIMDHEType, CustomType, SIMD_HE

@_CRYPTO_MOD.simple_op()
def encrypt(x: TensorType, key: CustomType) -> SIMDHEType:
    """Encrypt plaintext tensor to SIMD_HE ciphertext.

    Args:
        x: Plaintext tensor (TensorType instance)
        key: Encryption key (CustomType instance)

    Returns:
        SIMD_HE ciphertext type
    """
    # Type checking
    if key.kind != "EncryptionKey":
        raise TypeError(f"Expected EncryptionKey, got {key.kind}")

    # Return proper ciphertext type
    return SIMD_HE[x.scalar_type, x.shape]
```

**优势**:

- ✅ 类型签名准确：`CustomType` → `SIMDHEType`
- ✅ 类型推导：自动构建正确的 IR 类型
- ✅ 类型检查：验证 key 的 kind

### 场景 3: 用户体验

#### Before (当前方案)

```python
# ❌ 用户无法知道类型
x = mplang.constant(np.array([1, 2, 3]))
# x 是什么类型？只能打印看：
print(x.mptype)  # MPType(TensorType(FLOAT32, (3,)), pmask=None)

y = encrypt(x, key)
# y 是密文吗？无法从类型看出来
print(y.mptype)  # MPType(TensorType(FLOAT32, (3,)), pmask=None, attrs={"encrypted": True})
# 需要检查 attrs 才知道
```

#### After (新方案)

```python
# ✅ 类型注解清晰
x: MPObject[Tensor[f32, (3,)]] = mplang.constant(np.array([1, 2, 3]))
print(x._type)  # Tensor[f32, (3,)]

y: MPObject[SIMD_HE[f32, (3,)]] = encrypt(x, key)
print(y._type)  # SIMD_HE[f32, (3,)]

# IDE 提示：
# - x 是 MPObject[Tensor[f32, (3,)]]
# - y 是 MPObject[SIMD_HE[f32, (3,)]]
# - 一眼看出 y 是密文
```

## Ops 编写指南

### simple_op Kernel 的类型注解

**关键理解**: Kernel 接收的是 **类型实例**，不是 MPObject。

```python
from mplang.core.typing import (
    TensorType,      # typing.TensorType (新类型系统)
    SIMDHEType,
    CustomType,
    SIMD_HE,
)

@_CRYPTO_MOD.simple_op()
def encrypt(x: TensorType, key: CustomType) -> SIMDHEType:
    """
    Args:
        x: TensorType 实例 (不是 MPObject)
        key: CustomType 实例

    Returns:
        SIMDHEType 实例 (描述返回值的类型)
    """
    # simple_op 自动从 MPObject 提取 obj._type 传入
    # 所以这里收到的是 BaseType 实例

    # 类型推导
    return SIMD_HE[x.element_type, x.shape]
```

### simple_op 的魔法转换

```python
# 用户调用
x_mpobject = mplang.constant(...)  # x: MPObject[Tensor[f32, (10,)]]
key_mpobject = keygen()             # key: MPObject[CustomType("EncryptionKey")]

y_mpobject = encrypt(x_mpobject, key_mpobject)
#                    ^^^^^^^^^^^  ^^^^^^^^^^^
#                    MPObject     MPObject

# simple_op 内部 (ops/base.py:SimpleFeOperation.trace())
pos_mp_inputs = [x_mpobject, key_mpobject]
call_pos_types = tuple(obj._type for obj in pos_mp_inputs)
#                      ^^^^^^^^
#                      提取 BaseType 实例

# 调用 kernel
result_type = encrypt_kernel(*call_pos_types)
#                            ^^^^^^^^^^^^^^^^^
#                            (TensorType(...), CustomType(...))
#             返回 SIMDHEType(...)

# 构建输出 MPObject
output = TraceVar(ctx, _type=result_type, ...)
```

### 完整示例：加密 ops

```python
# mplang/ops/crypto.py

from mplang.core.typing import (
    BaseType,
    TensorType,
    SIMDHEType,
    CustomType,
    SIMD_HE,
    f32,
)
from mplang.ops.base import stateless_mod

_CRYPTO_MOD = stateless_mod("crypto")

@_CRYPTO_MOD.simple_op()
def keygen(*, key_size: int) -> BaseType:
    """Generate encryption key.

    Args:
        key_size: Key size (keyword-only, goes to attrs)

    Returns:
        CustomType for encryption key
    """
    return CustomType("EncryptionKey")

@_CRYPTO_MOD.simple_op()
def encrypt(x: TensorType, key: CustomType) -> BaseType:
    """Encrypt plaintext to SIMD_HE ciphertext.

    Args:
        x: Plaintext tensor type
        key: Encryption key type

    Returns:
        SIMD_HE ciphertext type
    """
    # Validate key type
    if key.kind != "EncryptionKey":
        raise TypeError(f"Expected EncryptionKey, got {key.kind}")

    # Type inference
    return SIMD_HE[x.element_type, x.shape]

@_CRYPTO_MOD.simple_op()
def add(x: SIMDHEType, y: SIMDHEType) -> BaseType:
    """Add two SIMD_HE ciphertexts.

    Args:
        x, y: SIMD_HE ciphertext types

    Returns:
        SIMD_HE ciphertext type
    """
    # Type checking
    if x.scalar_type != y.scalar_type or x.packing_shape != y.packing_shape:
        raise TypeError(f"Type mismatch: {x} vs {y}")

    return x  # Same type

@_CRYPTO_MOD.simple_op()
def decrypt(x: SIMDHEType, key: CustomType) -> BaseType:
    """Decrypt SIMD_HE ciphertext to plaintext.

    Args:
        x: SIMD_HE ciphertext type
        key: Decryption key type

    Returns:
        Plaintext tensor type
    """
    # Extract plaintext type from ciphertext
    return TensorType(x.scalar_type, x.packing_shape)
```

## 迁移策略

遵循 **渐进式兼容迁移** 原则，分 6 个 Phase 逐步替换 MPType。

### 6 Phase 路径（概要）

1. **Phase 1** ✅: 添加 `_type` 属性，保持向后兼容
2. **Phase 2**: 更新 TraceVar/InterpVar 实现新接口
3. **Phase 3**: 类型转换辅助函数
4. **Phase 4**: 更新 Expr 系统
5. **Phase 5**: peval 类型推导和检查
6. **Phase 6**: 移除 MPType

### 关键兼容性保证

- ✅ 双接口共存：`obj._type` (新) 和 `obj.mptype` (旧)
- ✅ 渐进式迁移：每个 Phase 都保持代码可运行
- ✅ 测试覆盖：每个 Phase 完成后运行完整测试套件

**详细实施计划见** [`design/type_system_migration.md`](./type_system_migration.md)。

---

## 总结

### 为什么需要新类型系统

当前 MPType 面临的根本问题：

1. **无法表达加密类型** (HE, SS, SIMD_HE)
2. **类型信息分散** (tensor.TensorType + attrs)
3. **缺乏类型推导** (runtime 才知道类型)

### 新类型系统的优势

1. **类型表达力**: 支持 Layout × Encryption × Distribution 的正交组合
2. **开发体验**: 清晰的类型注解，IDE 支持
3. **类型安全**: simple_op 自动类型检查和推导
4. **可扩展性**: CustomType 支持不透明类型

---

**相关文件**:

- 核心实现: `mplang/core/mpobject.py`, `mplang/core/typing.py`, `mplang/core/mptype.py`
- 详细实施计划: [`design/type_system_migration.md`](./type_system_migration.md)
