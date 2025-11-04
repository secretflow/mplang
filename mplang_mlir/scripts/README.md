# 快速迭代开发指南

## TL;DR - 立即开始

```bash
# 1. 设置开发环境（每次新终端）
source mplang_mlir/scripts/dev-env.sh

# 2. 快速测试
python mplang_mlir/scripts/quick-test.py

# 3. 开始开发！
python your_code.py
```

---

## 🎯 核心思路

在 **同一个 repo** 内快速迭代 MLIR 后端，无需发布到 PyPI：

```
mplang/                      # 主项目 (已存在)
├── pyproject.toml          # ← 保持不变
├── mplang/                 # ← 核心 Python 代码（修改立即生效）
└── mplang_mlir/            # ← MLIR 后端
    ├── lib/                # C++ dialect（需要编译）
    └── python/             # Python bindings（需要编译）
```

**关键：** 使用 `PYTHONPATH` 让 Python 直接导入两个目录，无需安装！

---

## 📦 方案对比

| 方案 | 何时使用 | 优点 | 缺点 |
|------|---------|------|------|
| **PYTHONPATH** (推荐) | 日常开发 | ✅ 零配置<br>✅ 即改即用 | 需设置环境变量 |
| **uv workspace** | CI/发布准备 | ✅ 正式<br>✅ 依赖管理 | 配置复杂 |
| **editable install** | 过渡阶段 | ✅ pip 标准 | 每次需 pip install |

---

## 🚀 推荐工作流（PYTHONPATH）

### 初始设置（一次性）

```bash
cd /path/to/mplang

# 如果需要 MLIR bindings，先构建 C++
cd mplang_mlir
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON
ninja
cd ../..
```

### 日常开发

```bash
# 终端 1: 启动开发环境
source mplang_mlir/scripts/dev-env.sh
# ✓ mplang
# ✓ mplang_mlir (如果已构建)

# 修改 Python 代码 - 立即生效！
vim mplang/core/primitive.py
python test.py  # 无需重新安装

# 修改 MLIR Python 代码 - 立即生效！
vim mplang_mlir/python/mplang_mlir/converter.py
python test.py  # 无需重新安装

# 修改 C++ 代码 - 需要重新编译
vim mplang_mlir/lib/Dialect/Mpir/MpirOps.cpp
cd mplang_mlir/build && ninja && cd -
python test.py
```

### 快速测试

```bash
# 测试导入是否正常
python mplang_mlir/scripts/quick-test.py

# 运行项目测试
pytest tests/

# 运行 tutorials
python tutorials/0_basic.py
```

---

## 🧪 示例：开发 AST-to-MLIR 转换器

```bash
# 1. 启动环境
source mplang_mlir/scripts/dev-env.sh

# 2. 创建转换器（纯 Python，立即生效）
cat > mplang/mlir/converter.py << 'EOF'
"""Convert MPLang AST to MLIR IR."""

def convert_ast_to_mlir(ast):
    """Convert AST to MLIR."""
    from mlir import ir
    from mplang_mlir.dialects import mpir

    with ir.Context() as ctx:
        mpir.register_dialect(ctx)
        # ... conversion logic ...

    return module
EOF

# 3. 测试（修改立即生效，无需重新安装！）
cat > test_converter.py << 'EOF'
from mplang import function
from mplang.mlir.converter import convert_ast_to_mlir

@function
def my_func(x, y):
    return x + y

# 获取 AST
ast = my_func._ast  # 假设有这个属性

# 转换到 MLIR
mlir_module = convert_ast_to_mlir(ast)
print(mlir_module)
EOF

python test_converter.py
```

---

## 💡 最佳实践

### 1. 分离关注点

```
mplang/mlir/              # 纯 Python 集成代码
├── __init__.py
├── converter.py          # AST → MLIR converter
├── compiler.py           # MLIR compiler integration
└── runtime.py            # MLIR runtime

mplang_mlir/python/       # C++ 生成的 bindings（最小化）
└── mplang_mlir/
    ├── __init__.py
    └── dialects/
        └── mpir.py       # 自动生成 + 扩展
```

**原则：尽可能多用 Python，最小化 C++ 部分**

### 2. 快速反馈循环

```bash
# 修改 → 测试 应该 < 10 秒
vim mplang/mlir/converter.py
python test.py  # 立即看到结果
```

### 3. C++ 最小化

```python
# ❌ 不要在 C++ 中做业务逻辑
# lib/Dialect/Mpir/MpirConverter.cpp - 1000 行 C++

# ✅ 在 Python 中做业务逻辑
# mplang/mlir/converter.py - 300 行 Python
# 使用 mplang_mlir Python bindings 生成 IR
```

---

## 🔧 工具脚本

### dev-env.sh
设置 `PYTHONPATH`，让 Python 找到两个包

### quick-test.py
验证环境配置正确

### 使用方式

```bash
# 方式 1: 每次手动 source
source mplang_mlir/scripts/dev-env.sh

# 方式 2: 添加 alias 到 .zshrc
echo "alias mplang-dev='source ~/github/mplang/mplang_mlir/scripts/dev-env.sh'" >> ~/.zshrc

# 然后每次开发
mplang-dev
```

---

## 📊 何时切换到其他方案

### 当前阶段（0-3个月）: PYTHONPATH ✅
- 快速迭代原型
- 频繁修改代码
- 团队内部使用

### 过渡阶段（3-6个月）: uv workspace
- 代码趋于稳定
- 准备对外发布
- 需要正式的依赖管理

### 发布阶段（6个月+）: 独立 PyPI 包
- 稳定 API
- 用户需要 `pip install`
- 多平台支持

---

## ❓ 常见问题

### Q: 每次都要 source dev-env.sh 吗？
A: 是的，或者添加到 `.zshrc`:
```bash
# ~/.zshrc
alias mplang-dev='source ~/path/to/mplang/mplang_mlir/scripts/dev-env.sh'
```

### Q: 修改 Python 代码后需要重新安装吗？
A: 不需要！这就是 PYTHONPATH 的优势。

### Q: 修改 C++ 代码后呢？
A: 需要重新编译：`cd mplang_mlir/build && ninja`

### Q: VS Code 能自动补全吗？
A: 可以，添加到 `.vscode/settings.json`:
```json
{
  "python.analysis.extraPaths": [
    "${workspaceFolder}",
    "${workspaceFolder}/mplang_mlir/build/python_packages/mplang_mlir"
  ]
}
```

### Q: 为什么不用 `pip install -e .`？
A: 可以，但 PYTHONPATH 更简单直接，不需要修改 `pyproject.toml`。

---

## 📚 更多信息

- 详细分析：[DISTRIBUTION_STRATEGY.md](DISTRIBUTION_STRATEGY.md)
- 完整指南：[RAPID_ITERATION_GUIDE.md](RAPID_ITERATION_GUIDE.md)
