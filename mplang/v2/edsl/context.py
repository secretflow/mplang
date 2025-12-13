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

"""Context: EDSL Execution Context Abstraction.

This module defines the Context hierarchy:
- Context: Base class for EDSL execution contexts (with bind_primitive method)
- Tracer: Tracing context (records operations to Graph IR)
- Interpreter: Execution context (executes operations immediately)

Contexts can be used directly with Python's 'with' statement:

    from mplang.v2.edsl import Tracer

    tracer = Tracer()
    with tracer:
        # Operations run under tracer context
        result = primitive.bind(x, y)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from mplang.v2.edsl.graph import Graph
    from mplang.v2.edsl.object import Object
    from mplang.v2.edsl.primitive import Primitive


class Context(ABC):
    """Base class for EDSL execution contexts.

    A Context represents an environment where primitives are executed.
    There are two types of contexts:
    - Tracer: Records operations to Graph IR (compile-time)
    - Interpreter: Execution context (executes operations immediately)
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

    def __enter__(self) -> Self:
        """Enter context manager (push context onto stack)."""
        push_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Exit context manager (pop context from stack)."""
        pop_context()


# ============================================================================
# Global Context Stack Management
# ============================================================================

_context_stack: list[Context] = []
_default_context: Context | None = None
_default_context_factory: Callable[[], Context] | None = None


def get_current_context() -> Context | None:
    """Get the current active context.

    Returns None if no context is active (will use default context).
    """

    return _context_stack[-1] if _context_stack else None


def get_root_context() -> Context | None:
    """Get the root context (bottom of the stack).

    This context typically holds the global environment state (e.g. ClusterSpec).
    """
    return _context_stack[0] if _context_stack else None


def push_context(context: Context) -> None:
    """Push a context onto the stack (enter context)."""
    _context_stack.append(context)


def pop_context() -> None:
    """Pop a context from the stack (exit context)."""
    if _context_stack:
        _context_stack.pop()


def register_default_context_factory(factory: Callable[[], Context]) -> None:
    """Register a factory function to create the default context."""
    global _default_context_factory
    _default_context_factory = factory


def get_default_context() -> Context:
    """Get the default context for eager execution."""
    global _default_context
    if _default_context is None:
        if _default_context_factory is None:
            raise RuntimeError(
                "No default context factory registered. "
                "Ensure mplang.v2.edsl is imported or register a factory manually."
            )
        _default_context = _default_context_factory()
    return _default_context


class AbstractInterpreter(Context):
    """Abstract interface for Interpreters.

    This allows EDSL components (like JIT) to depend on the Interpreter interface
    without depending on the concrete Runtime implementation (which may depend on
    ObjectStore, Backends, etc.).
    """

    @abstractmethod
    def evaluate_graph(self, graph: Graph, inputs: list[Any]) -> Any:
        """Execute a Graph IR with given inputs."""

    @abstractmethod
    def lift(self, obj: Any) -> Any:
        """Lift a python object to an interpreter object."""
