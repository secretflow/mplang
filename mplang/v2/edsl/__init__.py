# Copyright 2025 Ant Group Co., Ltd.
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

"""Public entrypoint for the MPLang EDSL.

This module keeps the surface area intentionally small so downstream code can
simply write::

    import mplang.v2.edsl as el
    import mplang.v2.edsl.typing as elt

The `el` namespace re-exports the commonly used building blocks (context,
graph, tracer, primitives, etc.), while the full type system lives under
``mplang.edsl.typing``.
"""

from __future__ import annotations

# Re-export the typing module so callers can `import mplang.v2.edsl.typing as elt`
from . import typing as typing

# Context management
from .context import (
    Context,
    find_context,
    find_context_with_state,
    find_interpreter,
    get_current_context,
    get_default_context,
    is_tracing,
    pop_context,
    push_context,
    register_default_context_factory,
    set_root_context,
)

# Graph IR
from .graph import Graph, Operation, Value

# High-level helpers
from .jit import jit
from .object import Object
from .primitive import Primitive, primitive
from .printer import GraphPrinter, format_graph
from .tracer import TracedFunction, TraceObject, Tracer, trace
from .typing import MPType, ScalarType, SSType, TableType, TensorType, VectorType

# Type Aliases for strong typing
MPObject = Object[MPType]
ScalarObject = Object[ScalarType]
SSObject = Object[SSType]
TableObject = Object[TableType]
TensorObject = Object[TensorType]
VectorObject = Object[VectorType]

__all__ = [
    "Context",
    "Graph",
    "GraphPrinter",
    "MPObject",
    "Object",
    "Operation",
    "Primitive",
    "SSObject",
    "ScalarObject",
    "TableObject",
    "TensorObject",
    "TraceObject",
    "TracedFunction",
    "Tracer",
    "Value",
    "VectorObject",
    "find_context",
    "find_context_with_state",
    "find_interpreter",
    "format_graph",
    "get_current_context",
    "get_default_context",
    "is_tracing",
    "jit",
    "pop_context",
    "primitive",
    "push_context",
    "register_default_context_factory",
    "set_root_context",
    "trace",
    "typing",
]
