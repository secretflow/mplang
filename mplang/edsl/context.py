"""Context: EDSL Execution Context Abstraction.

This module defines the Context hierarchy:
- Context: Abstract base class for EDSL execution contexts
- Tracer: Tracing context (records operations to Graph IR)
- Interpreter: Execution context (executes operations immediately)

The context stack is managed globally via ExecutionContext.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mplang.edsl.interpreter import Interpreter
    from mplang.edsl.object import Object
    from mplang.edsl.tracer import Tracer


class Context(ABC):
    """Abstract base class for EDSL execution contexts.

    A Context represents an environment where operations are executed.
    There are two types of contexts:
    - Tracer: Records operations to Graph IR (compile-time)
    - Interpreter: Executes operations immediately (runtime)

    This abstraction provides:
    1. Conceptual symmetry: Both Tracer and Interpreter are Context types
    2. Unified operation interface: execute_add(), execute_mul(), etc.
    3. Extensibility: Easy to add new context types (Profiler, Debugger, etc.)
    """

    @abstractmethod
    def execute_add(self, left: Object, right: Object) -> Object:
        """Execute addition operation.

        In Tracer: Records to Graph IR
        In Interpreter: Executes immediately
        """

    # TODO: Add more operations
    # @abstractmethod
    # def execute_mul(self, left: Object, right: Object) -> Object: ...
    # @abstractmethod
    # def execute_sub(self, left: Object, right: Object) -> Object: ...


class ExecutionContext:
    """Global execution context manager.

    Manages:
    - Current mode (tracing vs eager execution)
    - Context stack (Tracer or Interpreter contexts)
    - Default interpreter (for eager execution)
    """

    def __init__(self):
        self._mode: Literal["eager", "tracing"] = "eager"
        self._context_stack: list[Context] = []
        self._default_interpreter: Interpreter | None = None

    @property
    def is_tracing(self) -> bool:
        """Check if currently in tracing mode."""
        return self._mode == "tracing"

    @property
    def current_context(self) -> Context | None:
        """Get current context (Tracer or Interpreter).

        Returns None if no context is active (should use default interpreter).
        """
        return self._context_stack[-1] if self._context_stack else None

    @property
    def current_tracer(self) -> Tracer | None:
        """Get current tracer (None if not tracing)."""
        from mplang.edsl.tracer import Tracer

        ctx = self.current_context
        return ctx if isinstance(ctx, Tracer) else None

    @property
    def default_interpreter(self) -> Interpreter:
        """Get the default interpreter for eager execution."""
        if self._default_interpreter is None:
            from mplang.edsl.interpreter import Interpreter

            self._default_interpreter = Interpreter()
        return self._default_interpreter

    def enter_context(self, context: Context):
        """Enter a context (Tracer or Interpreter)."""
        from mplang.edsl.tracer import Tracer

        self._context_stack.append(context)
        if isinstance(context, Tracer):
            self._mode = "tracing"

    def exit_context(self):
        """Exit current context."""
        self._context_stack.pop()
        if not self._context_stack or not self.is_tracing:
            self._mode = "eager"


# Global context (singleton)
_global_context = ExecutionContext()


def get_context() -> ExecutionContext:
    """Get the global ExecutionContext."""
    return _global_context
