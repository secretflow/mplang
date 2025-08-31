# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Bootstrap LLVM/MLIR build locally under third_party/llvm-project and build/llvm
# Usage: scripts/mlir/setup_mlir.sh [llvmorg-18.1.8]

BRANCH="${1:-llvmorg-18.1.8}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TP_DIR="$ROOT_DIR/third_party"
SRC_DIR="$TP_DIR/llvm-project"
BUILD_DIR="$ROOT_DIR/build/llvm"

mkdir -p "$TP_DIR" "$BUILD_DIR"

if [ ! -d "$SRC_DIR/.git" ]; then
  git clone --depth=1 --branch "$BRANCH" https://mirrors.tuna.tsinghua.edu.cn/git/llvm-project.git "$SRC_DIR"
else
  echo "llvm-project already present at $SRC_DIR"
fi

cmake -S "$SRC_DIR/llvm" -B "$BUILD_DIR" -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

cmake --build "$BUILD_DIR" --target mlir-tblgen mlir-headers mlir-cmake-exports -- -k 0
# Build FileCheck (useful for lit/FileCheck tests)
cmake --build "$BUILD_DIR" --target FileCheck -- -k 0
# Build core libraries (default 'all' target)
cmake --build "$BUILD_DIR" -- -k 0

cat <<EOF

MLIR build prepared.
Set MLIR_DIR to: $BUILD_DIR/lib/cmake/mlir
Example:
  export MLIR_DIR=$BUILD_DIR/lib/cmake/mlir
  export LLVM_DIR=$BUILD_DIR/lib/cmake/llvm
  export PATH=$BUILD_DIR/bin:$PATH  # for FileCheck
Then configure mplang_mlir with CMake.
EOF
