# MPLang MLIR 开发指南

## 快速开始

### 开发环境设置（PYTHONPATH 方式）

```bash
# 1. 构建 MLIR bindings（一次性，30分钟）
cd mplang_mlir/build
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON
ninja

# 2. 设置开发环境（每次新终端）
cd /path/to/mplang
source mplang_mlir/scripts/dev-env.sh

# 3. 开始开发
python your_code.py  # Python 修改立即生效！
```

---

## 构建产物位置

```
/path/to/llvm-project/build/python_packages/mplang_mlir/
└── mplang_mlir/
    ├── __init__.py
    ├── dialects/
    │   ├── mpir.py
    │   ├── _mpir_ops_gen.py      # TableGen 生成
    │   └── _mpir_ops_ext.py      # 手写扩展
    └── _mlir_libs/
        └── _mplang.*.so           # C++ 扩展
```

---

## 开发模式

### Python First（推荐）

尽可能多的逻辑用 Python 实现，最小化 C++ 修改：

```
mplang/mlir/           # 纯 Python（修改立即生效）
├── converter.py       # AST → MLIR
└── compiler.py        # 编译集成

mplang_mlir/           # C++ + bindings（很少修改）
└── lib/Dialect/Mpir/  # 核心 dialect 定义
```

### 开发频率统计

- 修改 Python 代码：90%+ 的时间，**0秒**构建
- 修改 C++ 代码：< 10% 的时间，**1-5分钟**构建

---

## 发布方案

### 当前阶段（内部使用）

使用 PYTHONPATH 方式开发，无需打包：

```bash
source mplang_mlir/scripts/dev-env.sh
python your_code.py
```

### 未来发布（可选）

#### 方案 1: 打包到 mplang 主包

```bash
# 复制构建产物
cp -r /llvm/build/python_packages/mplang_mlir/mplang_mlir mplang/_mlir_bindings/

# pyproject.toml 添加
[tool.hatch.build.targets.wheel.force-include]
"mplang/_mlir_bindings/mplang_mlir" = "mplang_mlir"

# 构建
python -m build --wheel
```

#### 方案 2: 独立 mplang-mlir 包

```bash
# 在独立 repo 中发布 mplang-mlir
# 用户: pip install mplang[mlir]
```

---

## 架构决策

### 当前（0-3个月）：PYTHONPATH 方式
- ✅ 零配置，立即可用
- ✅ 修改 Python 代码立即生效
- ✅ 最快的开发速度

### 未来（3个月+）：考虑独立包
- 当 MLIR 后端 API 稳定
- 准备公开发布时
- 提供预编译 wheels

---

## 关键原则

1. **Python 优先** - 尽可能用 Python 写逻辑
2. **快速反馈** - 修改 → 测试 < 10 秒
3. **延迟优化** - 先做出来，再优化架构

---

## 工具脚本

- `dev-env.sh` - 设置开发环境
- `quick-test.py` - 验证环境配置

```bash
source mplang_mlir/scripts/dev-env.sh
python mplang_mlir/scripts/quick-test.py
```

---

## 参考文档

- Python Bindings 设计: [PYTHON_BINDING_DESIGN.md](PYTHON_BINDING_DESIGN.md)
- 主 README: [README.md](python/README.md)
