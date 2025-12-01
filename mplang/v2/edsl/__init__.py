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
    get_current_context,
    get_default_context,
    pop_context,
    push_context,
    register_default_context_factory,
)

# Graph IR
from .graph import Graph, Operation, Value

# Interpreter + execution helpers
from .interpreter import InterpObject, Interpreter, interpret

# High-level helpers
from .jit import jit
from .object import Object
from .primitive import Primitive, primitive
from .printer import GraphPrinter, format_graph
from .tracer import TracedFunction, TraceObject, Tracer, trace
from .typing import (
    MPType,
    ScalarType,
    SSType,
    TableType,
    TensorType,
    VectorType,
)

# Register default context factory
register_default_context_factory(Interpreter)

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
    "InterpObject",
    "Interpreter",
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
    "format_graph",
    "get_current_context",
    "get_default_context",
    "interpret",
    "jit",
    "pop_context",
    "primitive",
    "push_context",
    "register_default_context_factory",
    "trace",
    "typing",
]
