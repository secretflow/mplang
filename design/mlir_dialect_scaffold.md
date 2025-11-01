# MPLang MLIR 方言脚手架落地计划（C++/ODS + Python 绑定）

目的：尽早把上层 SPMD IR 在 MLIR 上“形式化”定义出来（方言/类型/算子/验证/注册），可本地构建、跑简单验证与文本往返；并给出 Python 侧的最小绑定方案，便于在现有 Python 框架中调用 Parser/Printer/注册方言。

本计划分三期推进，期内保持可用性与依赖可控。

---

## 期望成果（Phase 1 → Phase 3）

- Phase 1（1-2 周）：最小可编译的方言骨架
  - C++/ODS 定义 `mplang` 方言与最小子集算子/类型：
    - 类型：`!mplang.tensor`、`!mplang.table`（先以属性包/参数承载 dtype/shape/pmask，必要时简化为字符串/整数属性；后续细化）
    - 算子：`mplang.eval`（region 形态）与 `mplang.yield`（terminator）
  - Dialect/Op/Type 注册、Verifier 雏形（参数个数/region 产出数量检查等）
  - CMake 构建（依赖外部 LLVM/MLIR 安装，传入 `LLVM_DIR`）
  - 最小 `mplang-opt`（或 `mplang-translate`）命令行工具，能 parse/print 我们方言
  - 文档：本文件 + README（快速构建指南）

- Phase 2（2-3 周）：扩充与 Python 绑定
  - 扩充上层 SPMD 原语：`mplang.cond`、`mplang.while`、`mplang.conv`、`mplang.shfl_s`、`mplang.shfl`
  - Verifier 规则：
    - `eval`: rmask ⊆ deduced(arg pmask)（静态可得时）
    - `cond`: then/else 出参完全同型；`verify_uniform` 语义属性
    - `while`: cond/体出入参同型校验等
  - Python 绑定（首选）：`pybind11` + MLIR C-API（`libMLIR-CAPI*`）
    - 能在 Python 中：注册 `mplang` 方言、parse/print 模块、（可选）跑简单 verifier
    - 打包：`scikit-build-core` 生成 `mplang-mlir` 轮子
  - CI：Ubuntu 22.04，使用预编译 LLVM/MLIR 18（或缓存编译产物）

- Phase 3（后续）：与外部子方言联动与降级
  - 给 `mplang.eval` 提供 symref 形态（`callee=@foo`）与 region 形态互转的 canonicalization
  - 示例：把 region 内部改写/外提为 `func.func @foo`（主体可用 stablehlo/HEIR/lingo-db/sql 等）
  - 预留 pass 管线挂载位（例如 cond/while 向 scf 降级的实验性 rewrite）

---

## 目录结构建议（位于 `mplang_mlir/`）

```
mplang_mlir/
  CMakeLists.txt
  cmake/                      # 可选：LLVM/MLIR 查找与工具宏
  include/
    mplang/Dialect/MPLANG/
      MPLANGDialect.td        # Dialect/Attr/Type ODS 声明
      MPLANGOps.td            # Ops ODS（eval/yield 起步）
      MPLANGTypes.td          # Types ODS（tensor/table 参数化）
  lib/
    Dialect/MPLANG/
      CMakeLists.txt
      MPLANGDialect.cpp       # 方言注册、generated includes
      MPLANGOps.cpp           # op/class 生成 glue
      MPLANGTypes.cpp         # type storage/printer/parser
      Generated/              # （构建时生成到 build tree）
  tools/
    mplang-opt/
      CMakeLists.txt
      mplang-opt.cpp          # 注册 dialect + 驱动（仿照 mlir-opt）
  python/                     # Phase 2：pybind11 绑定工程（scikit-build-core）
    CMakeLists.txt
    mplang_mlir/
      __init__.py
      _mlir_capi.py           # 轻量包装（parse/print/register）
  README.md                   # 构建与使用说明
```

> 备注：也可以选择将 `include/` 安装到前缀，`Generated` 放到 build 目录并通过 `add_mlir_dialect` 宏管理（见 LLVM 官方样例）。

---

## 依赖与版本建议

- LLVM/MLIR：18.x（统一在 CI 和本地），最低 C++17
- CMake：>= 3.20
- Python 绑定：
  - 编译时依赖 `pybind11`, `scikit-build-core`, `setuptools_scm`
  - 运行时依赖 Python 3.10+（与项目一致）

---

## CMake 要点（摘要）

顶层 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.20)
project(MPLANG_Dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG) # 依赖外部已安装的 LLVM/MLIR，传入 -DMLIR_DIR
message(STATUS "Using MLIRConfig at: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)

add_subdirectory(lib/Dialect/MPLANG)
add_subdirectory(tools/mplang-opt)
```

`lib/Dialect/MPLANG/CMakeLists.txt`（示例）：

```cmake
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/mplang/Dialect/MPLANG/MPLANGOps.td)
mlir_tablegen(MPLANGOps.h.inc -gen-op-decls)
mlir_tablegen(MPLANGOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/mplang/Dialect/MPLANG/MPLANGDialect.td)
mlir_tablegen(MPLANGDialect.h.inc -gen-dialect-decls -dialect=mplang)
mlir_tablegen(MPLANGDialect.cpp.inc -gen-dialect-defs -dialect=mplang)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/mplang/Dialect/MPLANG/MPLANGTypes.td)
mlir_tablegen(MPLANGTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MPLANGTypes.cpp.inc -gen-typedef-defs)

add_mlir_dialect(MPLANG mplang)

add_mlir_library(MLIRMPLANG
  MPLANGDialect.cpp
  MPLANGOps.cpp
  MPLANGTypes.cpp

  DEPENDS
    MLIRMPLANGIncGen

  LINK_LIBS PUBLIC
    MLIRIR
)
```

`tools/mplang-opt/mplang-opt.cpp`（骨架思路）：

```c++
// 注册 MLIR core + 我们的方言，调用 mlir::MlirOptMain
```

---

## ODS 最小草案（示意）

`MPLANGDialect.td`

```tablegen
include "mlir/IR/DialectSpecification.td"

def MPLANG_Dialect : Dialect {
  let name = "mplang";
  let cppNamespace = "mplang";
}
```

`MPLANGTypes.td`

```tablegen
include "mlir/IR/OpBase.td"

def MPLANG_TensorType : TypeDef<MPLANG_Dialect, "Tensor"> {
  let mnemonic = "tensor"; // !mplang.tensor
  let summary = "Distributed tensor type with pmask & attrs";
}

def MPLANG_TableType : TypeDef<MPLANG_Dialect, "Table"> {
  let mnemonic = "table";  // !mplang.table
  let summary = "Distributed table type";
}
```

`MPLANGOps.td`

```tablegen
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

def MPLANG_YieldOp : Op<MPLANG_Dialect, "yield", [Terminator]> {
  let summary = "Yield from peval region";
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict";
}

def MPLANG_EvalOp : Op<MPLANG_Dialect, "eval", [SingleBlockImplicitTerminator<"MPLANG_YieldOp">]> {
  let summary = "SPMD per-party execution (region form)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "`(` $inputs `)` attr-dict `(` $body `)` `:` functional-type($inputs, $outs)";
}
```

> 说明：类型参数/属性细节、Verifier（rmask/pmask 规则）将分阶段补全。此处先能 parse/print，并支撑最小 roundtrip。

---

## Python 绑定（Phase 2）

目标：在 Python 中完成：

- `import mplang_mlir as mpm`
- `mpm.register_dialect()`：注册方言到 MLIR Context
- `mpm.parse(text)` / `mpm.print(module)`：解析/打印

技术路线：

- 依赖 MLIR C-API：`libMLIR-CAPI*`（官方提供 C 层稳定接口）
- 用 `pybind11` 包一层薄封装，暴露 parse/print/register（参考 mlir-python 的做法，但我们更轻量）
- 构建：`scikit-build-core` + `pyproject.toml`；wheel 名称建议 `mplang-mlir`，在主包中可选依赖或延迟导入

---

## CI 方案

- GitHub Actions（linux/amd64, Ubuntu 22.04）：
  - 步骤：缓存或下载预编译的 LLVM/MLIR 18；`cmake .. -DMLIR_DIR=... && ninja`
  - 构建 `MLIRMPLANG` 库与 `mplang-opt` 可执行
  - 运行最小单元测试（parse/print 的 smoke tests）
  - Phase 2：构建 Python wheel（CPython 3.10/3.11 矩阵）并做 import smoke test

---

## 与现有 Python 端的集成

- 短期：仍以 Python AST（Expr）为主，文本 MLIR 仅用于 dump 与实验互操作。
- 引入方言后：
  - 可在 Python 中通过绑定注册方言并 parse MLIR 文本（来自外部工具/生成器），再回到现有执行路径（或仅用于可视化/分析）。
  - 逐步将 Printer/Parser 与 C++ 方言保持一致（语法对齐、属性命名稳定）。

---

## 里程碑与验收标准

- M1（Phase 1 完成）
  - `mplang_mlir` 编译通过；`mplang-opt` 能 parse/print 我们的最小 IR（eval/yield + 基础类型）
  - README 提供构建步骤；在本机 Docker/CI 上验证

- M2（Phase 2 完成）
  - cond/while/conv/shfl[_s] ODS 定义与 Verifier 初步到位
  - Python 轮子在 CI 产出，可 `import mplang_mlir` 并 `register_dialect()` 与 parse/print 成功

- M3（Phase 3 完成）
  - 支持 eval 的 region/symref 互转（canonicalization）
  - 提供示例将 region 外提为 `func.func @foo`（stablehlo/HEIR/SQL 任一），并从 peval 以 callee 调用

---

## 风险与对策

- LLVM/MLIR 版本漂移：统一锁定 18.x，必要时提供脚本构建本地安装；CI 缓存工件
- Python 绑定复杂度：先提供最小功能（register/parse/print），避免一开始就承接 pass/pipeline
- 属性/类型序列化复杂：先以简单参数/字典承载，后续逐步变为结构化 Attribute/TypeParam

---

如需，我可以在 `mplang_mlir/` 下提交最小骨架（CMake + ODS 空壳 + 空 cpp），作为 Phase 1 的起步提交。
