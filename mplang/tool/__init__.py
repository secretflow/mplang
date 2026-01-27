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
