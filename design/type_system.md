# MPLang 类型系统（提案）

一句话目标：以一棵简单的 TypeExprTree 表达“值的结构与表示”，把“分布（多方/pmask）”与“表示（明文/编码/加密/秘密分享）”解耦；在 Python 侧提供最小而有用的判定与校验。
## TL;DR

- 纯 Expr AST 设计：所有类型直接继承 Type（ABC）
- 值类型（Type）= 内建结构 + 表示域
  - 内建：TensorType、TableType
  - 表示：EncodedType / EncryptedType / SecretSharedType（可叠加，直接继承 Type）
- 多方类型（MpType）：直接继承 Type，描述多方分布（pmask + 可选 quals）
  - 参与方数量在 MpType.pmask 中体现
- 分层契约：ops（单方）只用 Type；primitives（多方）用 MpType
- 最小谓词：is_tensor/is_table/is_mp、unwrap_repr、same_dtype/shape/schema、same_security

## 设计目标
- 表达力：类型直接表达张量/表与 Enc/HE/SS 的表示域；分布与表示正交可叠加。
- 可组合：类型是小而统一的 AST；构造子组合保持不变式；相等/哈希/打印稳定。
- 可扩展：核心只给出基类与最小工具；各领域模块（crypto/smpc/encode）自由扩展包装类型。
- 实用校验：Python 侧做“明显错误”的前置校验；强约束交给 primitives（以及未来 MLIR）。

非目标（本稿不讨论）：迁移计划、运行时数据协议（numpy/pandas）、Union/Product 类型（若控制流 join 需要再考虑）。
## 核心抽象

- Type（抽象）：类型层 AST 的不可变、可哈希节点；所有构造子均继承自它。
- 内建值类型（直接继承 Type）：
  - TensorType(dtype: DType, shape: tuple[int,...])
    - dtype 为 DType；shape 全为 int（可含 -1 动态维）。
  - TableType(columns: tuple[tuple[str, DType], ...])
    - 非空；列名唯一且非空；列类型为 DType。
- 表示域类型（直接继承 Type）：
  - EncodedType(inner: Type, codec: str, params: Mapping[str, object])
  - EncryptedType(inner: Type, scheme: str, key_ref: str|None, params: Mapping[str, object])
  - SecretSharedType(inner: Type, scheme: str, field_bits: int|None, params: Mapping[str, object])
  - 不变式：不可变；params 规范化为排序后的 (k,v) 元组，保证相等/哈希稳定；可递归组合。
  - 参与方数量由外层 MpType 的 pmask 表达。
- 多方类型（直接继承 Type）：
  - MpType(inner: Type, pmask: Mask|None, quals: dict[str, object]|None = None)
  - 描述"哪些参与方持有/执行同构的 inner 类型"；pmask=None 表示运行期决定；quals 预留设备/放置等限定。
  - 典型：MpType(SecretSharedType(Tensor(...), scheme='aby3', field_bits=64), pmask=0b0111)
  - MpType 是独立的类型类，不是"包装器"；它描述多方持有同构数据的分布式类型。

## 最小谓词与工具

- 头判定（不解包）：
  - is_tensor(t: Type) -> bool
  - is_table(t: Type) -> bool
  - is_mp(t: Type) -> bool
- 解包：
  - unwrap_repr(t: Type) -> Type    // 去表示包装（Enc/HE/SS/...），循环剥离表示类型直到非表示类型
- 基类等价（自动 unwrap MpType + 表示包装）：
  - same_dtype(a, b) / same_shape(a, b) / same_schema(a, b)
  - 这些函数内部会先 unwrap MpType（如果存在），再 unwrap_repr，最后比较基类
- 安全域等价：
  - same_security(a, b)：比较从外到内的表示包装链（种类/标签/参数）是否完全一致（跳过 MpType）

## 签名与校验（ops vs primitives）
- ops（单方、纯 Type）：
  - crypto.enc: X: Type -> EncryptedType(X, scheme=..., params=...)
    - 前置：调用方需显式检查 X 是否为 Tensor/Table（或 unwrap_repr(X) 后检查）
  - smpc.share: X: Type -> SecretSharedType(X, scheme='aby3', field_bits=...)
    - 前置：调用方需显式检查 X 类型
  - basic.pack/unpack：默认建议不直接处理 Enc/HE/SS（需显式转换），保持边界清晰。
- primitives（多方、Mp<Type>）：
  - crypto.enc: Mp(X, pm) -> Mp(EncryptedType(X,...), pm)
  - smpc.share: Mp(X, pm_in) -> Mp(SS(X,...), pm_out)
  - conv/shfl（只改分布）：Mp(X, pm_in) -> Mp(X, pm_out)

备注：参与方数量/HE 参数兼容性等强约束属于 primitives（或 ops）的 verifier，不在类型构造子中硬编码。
## 相等、哈希与打印（可读）

- 所有类型不可变；params 归一化；相等/哈希稳定。
- 建议打印格式（示例）：
  - f32[3, 4]
  - Tbl(id:i64, name:str)
  - HE{n=16384}(f64[2048]; ckks)
  - SS{field_bits=64, n=3}(i64[10, 10]; aby3)
  - Mp<pm=0b0111>(SS{...}(i64[...]; aby3))   // 具体格式可微调

## 示例
- 纯张量：TensorType(i64, (1024,))
- 固定点编码：EncodedType(TensorType(i64,(1024,)), codec='fixed', params={'scale':16})
- HE-CKKS：EncryptedType(TensorType(f64,(2048,)), scheme='ckks', params={'n':16384})
- SS-ABY3：SecretSharedType(TensorType(i64,(10,10)), scheme='aby3', field_bits=64)
- 多方 SS：MpType(SecretSharedType(TensorType(i64,(10,10)), scheme='aby3', field_bits=64), pmask=0b0111)
  - pmask=0b0111 表示 3 方（P0/P1/P2）持有
- 非法：TensorType(dtype=SecretSharedType(...), shape=...)   // 构造器应报错

## 可插拔扩展（轻量注册表，可选）

为避免 core 硬编码所有包装类型名，可提供轻量注册表：
```python
# 伪代码
class WrapperRegistry:
  repr_wrappers: set[type] = set()
  dist_wrappers: set[type] = set()

  @classmethod
  def register_repr(cls, t: type): cls.repr_wrappers.add(t)
  @classmethod
  def register_dist(cls, t: type): cls.dist_wrappers.add(t)

# 各模块初始化时注册
encrypt.register(); smpc.register(); encode.register()

# unwrap_repr/unwrap_dist 内部参考该注册表决定是否剥层
```

当第三方扩展增多或需要通用 pass/映射时，可将注册表升级为 ABC（RepresentationWrapper/DistributionWrapper）。

## 未来工作
- 结构化参数对象（替代 dict）：如 CKKSParams、FixedPointParams —— 在构造类型时前置验证参数合法性；到 MLIR 的映射更直接。
- 小型“签名-约束”DSL：把 SameShape/HasBaseTensor/SameSecurity 等约束声明化，统一被 ops/primitives 的 verifier 复用，并可自动生成文档/映射。
- MLIR 映射：将 TypeExprTree 机械化映射到 dialect types/attributes。

## 可读性约定（文档风格）

- 先结论后细节：每节以 3-5 条要点开头（像 TL;DR）。
- 多用示例与对照：每个概念给 1-2 个正/反例。
- 层次扁平：一级标题控制在 6-8 个，避免深层嵌套。
- 名词与缩写：首次出现定义一句话，后续统一使用。
- 打印规范：固定几种打印风格，示例保持一致，降低阅读成本。
# MPLang Type System Design

This document defines the new type system for MPLang. It focuses on design goals, core abstractions, well-formedness rules, and how the system composes. It intentionally does not cover migration steps or current implementation constraints.

## Goals

- Expressive value types
  - Model data structure (tensor/table) and representation domains (encoded, encrypted, secret-shared) directly in the type system.
  - Keep distribution (which parties hold/execute) orthogonal from representation.
- Compositional and predictable
  - Types form a small, uniform AST; constructors compose and preserve invariants.
  - Deterministic equality/hash and readable string representation.
- Extensible by modules
  - Core provides base types and minimal contracts.
  - Domain modules (crypto, smpc, encode, etc.) can introduce new wrappers without touching core.
- Practical verification at Python level
  - Provide minimal, useful predicates/utilities for operator/primitives verifiers.
  - Avoid over-constraining; authoritative checks can live in primitives or (later) MLIR.

## Non-goals (for now)

- Migration plans from the existing system.
- Runtime data protocols (numpy/pandas) — this is strictly type-level.
- Union/Product types — defer until concrete need. Control flow expects equal types across branches.

## Core abstractions

- Type (abstract)
  - Immutable, hashable node in a TypeExprTree (type-level AST).
  - All type constructors (builtins and wrappers) derive from Type.
- Built-in value types
  - TensorType(dtype: DType, shape: tuple[int, ...])
    - shape supports dynamic dims (e.g., -1); all dims are ints.
    - dtype is a DType (existing enumeration/record).
  - TableType(columns: tuple[tuple[str, DType], ...])
    - Non-empty; unique, non-empty column names; each column has a DType.
- WrapperType (abstract)
  - inner: Type (the wrapped type)
  - category: Literal['repr','dist']
    - 'repr': representation wrappers (encoded, encrypted, secret-shared, etc.)
    - 'dist': distribution wrapper (multi-party/pmask)
  - Provides a uniform surface to core utilities without knowing each module’s wrapper name.

## Representation wrappers (category='repr')

Represent how values are represented, independent of who holds them.

- EncodedType(inner: Type, codec: str, params: Mapping[str, object])
- EncryptedType(inner: Type, scheme: str, key_ref: str|None, params: Mapping[str, object])
- SecretSharedType(inner: Type, scheme: str, field_bits: int|None, parties: int|None, params: Mapping[str, object])

Notes:
- All wrappers are immutable; params are normalized to a sorted tuple of key-value pairs for stable equality/hash.
- Wrappers can compose (e.g., Encoded(Encrypted(Tensor(...)))) if a domain requires it.

## Distribution wrapper (category='dist')

Express which parties hold/execute a value; separate from representation.

- MpType(inner: Type, pmask: Mask | None, quals: dict[str, object] = {})
  - pmask describes ownership/execution parties; None means runtime-decided.
  - quals extensibly stores extra execution qualifiers (e.g., device/placement) if needed.
  - Typical secret-sharing scenario: MpType(SecretSharedType(Tensor(...), scheme=aby3, ...), pmask=<party set>)
  - Single-party shares are allowed (degenerate scenario); primitives can enforce participation constraints.

Important: the type system itself does not enforce “MpType must be the outermost wrapper.” Primitives accept MpType explicitly, which is sufficient to keep multi-party boundaries clean without coupling this constraint into type constructors.

## Well-formedness & invariants

- TensorType: dtype is a DType; shape is a tuple of ints.
- TableType: non-empty; unique, non-empty column names; column dtypes are DType.
- Wrapper types: immutable; params normalized; inner is a Type.
- MpType: represents distribution; nesting MpType(MpType(...)) should be normalized or rejected by utilities/primitives (not by the Type class itself).
- No implicit cross-domain conversions (e.g., decrypt/reveal/decode) — must be explicit via ops/primitives.

## Minimal predicates and utilities

Keep the surface small; add more only when needed.

- Strict head checks (no unwrapping):
  - is_tensor(t: Type) -> bool  // isinstance(t, TensorType)
  - is_table(t: Type) -> bool   // isinstance(t, TableType)
  - is_mp(t: Type) -> bool      // isinstance(t, MpType)
- Unwrapping:
  - unwrap_repr(t: Type) -> Type  // remove all representation types (Enc/HE/SS/…)
- Equivalence on base (auto unwraps MpType + repr):
  - same_dtype(a: Type, b: Type) -> bool
  - same_shape(a: Type, b: Type) -> bool
  - same_schema(a: Type, b: Type) -> bool
- Security-chain equivalence:
  - same_security(a: Type, b: Type) -> bool
    - Compare the outer-to-inner sequence of representation wrappers by kind/tag/params

## Signatures and verifiers pattern

- ops (single-party; pure Type signatures):
  - crypto.enc: X: Type -> EncryptedType(X, scheme=..., params=...)
    - Pre: callers explicitly check X is Tensor/Table (or unwrap_repr(X) then check)
  - smpc.share: X: Type -> SecretSharedType(X, scheme='aby3', field_bits=...)
    - Pre: callers explicitly check X type
  - basic.pack/unpack: prefer explicit conversions; by default, do not pack Enc/HE/SS directly.
- primitives (multi-party; MpType signatures):
  - crypto.enc: MpType(X, pm) -> MpType(EncryptedType(X,...), pm)
  - smpc.share: MpType(X, pm_in) -> MpType(SecretSharedType(X,...), pm_out)
  - conv/shfl (distribution only): MpType(X, pm_in) -> MpType(X, pm_out)

Constraints like party-count for SS, scheme-compatibility for HE, etc., belong to primitives (or ops) verifiers, not to type constructors.

## Equality, hashing, and repr/parse

- All types are immutable dataclasses with stable equality/hash.
- params are normalized (sorted tuple) to keep equality/hash deterministic.
- Provide a readable repr, e.g.:
  - f32[3, 4]
  - Tbl(id:i64, name:str)
  - HE{n=16384}(f64[2048]; ckks)
  - SS{field_bits=64, n=3}(i64[10, 10]; aby3)
  - Mp<pm=0b0111>(SS{...}(i64[...]; aby3))  // repr example; actual formatting TBD

## Extensibility model

- Core defines: Type (ABC), TensorType, TableType, MpType, EncodedType, EncryptedType, SecretSharedType, and minimal utilities.
- Modules may add new representation types by directly subclassing Type and implementing inner: Type field.
- Utilities like unwrap_repr can be extended to handle new representation types by checking isinstance.

## Examples

- Plain tensor: TensorType(i64, (1024,))
- Encoded fixed-point tensor: EncodedType(TensorType(i64, (1024,)), codec='fixed', params={'scale': 16})
- HE CKKS vector: EncryptedType(TensorType(f64, (2048,)), scheme='ckks', params={'n': 16384})
- SS ABY3 matrix: SecretSharedType(TensorType(i64, (10, 10)), scheme='aby3', field_bits=64)
- Multi-party SS value: MpType(SecretSharedType(TensorType(i64, (10, 10)), scheme='aby3', field_bits=64), pmask=0b0111)
  - pmask=0b0111 indicates 3 parties (P0/P1/P2) hold the value
- Invalid: TensorType(dtype=SecretSharedType(...), shape=...)  // constructors must reject

## Open questions

- Should pack/unpack allow Enc/HE/SS by default, or require explicit conversions? (recommend explicit conversions)
- Table encryption policy: forbid non-numeric columns or require prior encoding?
- Do we need a canonical order between multiple representation wrappers if composable? (default: preserve construction order)

## Future work

- Map this TypeExprTree to MLIR dialect types/attributes in a mechanical way.
- Consider adding sum/product (Union/Product) types if control-flow/type-joins require them.
- Introduce a small signature-DSL to express operator constraints uniformly (using the minimal predicates).
