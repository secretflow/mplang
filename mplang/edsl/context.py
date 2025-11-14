"""Context: EDSL Execution Context Abstraction.

This module defines the Context hierarchy:
- Context: Base class for EDSL execution contexts (with bind_primitive method)
- Tracer: Tracing context (records operations to Graph IR)
- Interpreter: Execution context (executes operations immediately)

Contexts can be used directly with Python's 'with' statement:

    from mplang.edsl import Tracer

    tracer = Tracer()
    with tracer:
        # Operations run under tracer context
        result = primitive.bind(x, y)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mplang.edsl.interpreter import Interpreter
    from mplang.edsl.object import Object
    from mplang.edsl.primitive import Primitive


class Context(ABC):
    """Base class for EDSL execution contexts.

    A Context represents an environment where primitives are executed.
    There are two types of contexts:
    - Tracer: Records operations to Graph IR (compile-time)
    - Interpreter: Executes operations immediately (runtime)

    Each context decides how to handle primitive operations by implementing
    the bind_primitive() method.

    This abstraction provides:
    1. Clear responsibility: Context knows how to execute primitives
    2. Context management: enter/exit context for operation tracing/execution
    3. Extensibility: Easy to add new context types (Profiler, Debugger, etc.)

    Usage:
        >>> tracer = Tracer()
        >>> with tracer:  # Context manager protocol
        ...     result = primitive.bind(x, y)
    """

    @abstractmethod
    def bind_primitive(
        self, primitive: Primitive, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Execute a primitive in this context.

        Args:
            primitive: The primitive to execute
            args: Positional arguments (Objects)
            kwargs: Keyword arguments (plain values)

        Returns:
            Result Object (TraceObject in Tracer, InterpObject in Interpreter)
        """

    @abstractmethod
    def lift(self, obj: Any) -> Object:
        """Lift an object to this context's native Object type.

        Converts objects to the appropriate type for this context:
        - Tracer: InterpObject → TraceObject (via promote), constants → TraceObject
        - Interpreter: keeps InterpObject as-is, may convert constants

        Args:
            obj: Object to lift (Object, constant, etc.)

        Returns:
            Object in the context's native type (TraceObject or InterpObject)
        """

    def __enter__(self):
        """Enter context manager (push context onto stack)."""
        push_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager (pop context from stack)."""
        pop_context()


# ============================================================================
# Global Context Stack Management
# ============================================================================

_context_stack: list[Context] = []
_default_interpreter: Interpreter | None = None


def get_current_context() -> Context | None:
    """Get the current active context.

    Returns None if no context is active (will use default interpreter).
    """
    return _context_stack[-1] if _context_stack else None


def push_context(context: Context):
    """Push a context onto the stack (enter context)."""
    _context_stack.append(context)


def pop_context():
    """Pop a context from the stack (exit context)."""
    if _context_stack:
        _context_stack.pop()


def get_default_interpreter() -> Interpreter:
    """Get the default interpreter for eager execution."""
    global _default_interpreter
    if _default_interpreter is None:
        from mplang.edsl.interpreter import Interpreter

        _default_interpreter = Interpreter()
    return _default_interpreter
