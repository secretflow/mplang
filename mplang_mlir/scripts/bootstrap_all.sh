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

# One-shot bootstrap: build LLVM/MLIR locally and build mplang components.
# Usage: scripts/mlir/bootstrap_all.sh [llvm_tag]

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LLVM_TAG="${1:-llvmorg-18.1.8}"

"$ROOT_DIR/mplang_mlir/scripts/mlir/setup_mlir.sh" "$LLVM_TAG"

export MLIR_DIR="$ROOT_DIR/build/llvm/lib/cmake/mlir"
export LLVM_DIR="$ROOT_DIR/build/llvm/lib/cmake/llvm"
export PATH="$ROOT_DIR/build/llvm/bin:$PATH"

cmake -S "$ROOT_DIR/mplang_mlir" -B "$ROOT_DIR/build/mplang_mlir" -G Ninja -DMLIR_DIR="$MLIR_DIR"
cmake --build "$ROOT_DIR/build/mplang_mlir" --target mplang-opt -j

cat <<EOF

Bootstrap complete.
- MLIR_DIR: $MLIR_DIR
- LLVM_DIR: $LLVM_DIR
- mplang-opt: $ROOT_DIR/build/mplang_mlir/tools/mplang-opt/mplang-opt

Try:
  $ROOT_DIR/build/mplang_mlir/tools/mplang-opt/mplang-opt --pass-pipeline="mpir-peval" \
    $ROOT_DIR/mplang_mlir/test/mlir/mpir/peval.mlir | FileCheck \
    $ROOT_DIR/mplang_mlir/test/mlir/mpir/peval.mlir
EOF
