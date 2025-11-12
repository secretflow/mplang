"""MPLang Core2: New core system built on EDSL.

This module provides the runtime layer built on top of mplang.edsl, supporting:
- Object Hierarchy (TraceObject/InterpObject)
- Python → Graph IR tracing
- JIT compilation and execution
- Graph interpretation

Isolated from the legacy system (mplang.core) for gradual migration.
"""

from mplang.core2.context import ExecutionContext, get_context
from mplang.core2.interp import GraphInterpreter, interpret
from mplang.core2.jit import jit
from mplang.core2.object import InterpObject, Object, TraceObject
from mplang.core2.tracer import Tracer, trace

__all__ = [
    # Context
    "ExecutionContext",
    # Interpretation
    "GraphInterpreter",
    "InterpObject",
    # Object Hierarchy
    "Object",
    "TraceObject",
    # Tracing
    "Tracer",
    "get_context",
    "interpret",
    # JIT
    "jit",
    "trace",
]
