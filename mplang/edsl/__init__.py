"""Public entrypoint for the MPLang EDSL.

This module keeps the surface area intentionally small so downstream code can
simply write::

    import mplang.edsl as el
    import mplang.edsl.typing as elt

The `el` namespace re-exports the commonly used building blocks (context,
graph, tracer, primitives, etc.), while the full type system lives under
``mplang.edsl.typing``.
"""

from __future__ import annotations

# Re-export the typing module so callers can `import mplang.edsl.typing as elt`
from . import typing as typing

# Context management
from .context import (
    Context,
    get_current_context,
    get_default_interpreter,
    pop_context,
    push_context,
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

__all__ = [
    # modules
    "typing",
    # context
    "Context",
    "get_current_context",
    "get_default_interpreter",
    "pop_context",
    "push_context",
    # graph
    "Graph",
    "Operation",
    "Value",
    # interpreter
    "InterpObject",
    "Interpreter",
    "interpret",
    # primitives / helpers
    "Primitive",
    "primitive",
    "GraphPrinter",
    "format_graph",
    "TraceObject",
    "TracedFunction",
    "Tracer",
    "trace",
    "jit",
    "Object",
]
