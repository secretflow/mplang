# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool-layer APIs for MPLang.

This package contains utilities that are intentionally *not* part of the core
EDSL execution surface. In particular, compile/execute decoupling lives here:
- build a portable `CompiledProgram`
- pack/unpack to a container format

These helpers must not depend on user source code being available at execution.
"""

from __future__ import annotations

from mplang.edsl.program import CompiledProgram, FlatIOSignature
from mplang.tool.program import (
    compile_program,
    inspect_artifact,
    pack,
    pack_to_path,
    unpack,
    unpack_path,
)

__all__ = [
    "CompiledProgram",
    "FlatIOSignature",
    "compile_program",
    "inspect_artifact",
    "pack",
    "pack_to_path",
    "unpack",
    "unpack_path",
]
