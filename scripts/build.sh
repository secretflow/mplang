#!/bin/bash
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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/../protos"
OUTPUT_DIR="${SCRIPT_DIR}/../mplang/protos"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# FIXME(jint): I can not pass build without manually setting the site-packages path
SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
python -m grpc_tools.protoc -I"${PROTO_DIR}" \
  -I"${SITE_PACKAGES_PATH}" \
  --python_out="${OUTPUT_DIR}"  \
  --mypy_out="${OUTPUT_DIR}" \
  --grpc_python_out="${OUTPUT_DIR}"  \
  --mypy_grpc_out="${OUTPUT_DIR}" \
  "${PROTO_DIR}"/executor.proto

python -m grpc_tools.protoc -I"${PROTO_DIR}" \
  -I"${SITE_PACKAGES_PATH}" \
  --python_out="${OUTPUT_DIR}"  \
  --mypy_out="${OUTPUT_DIR}" \
  "${PROTO_DIR}"/mpir.proto