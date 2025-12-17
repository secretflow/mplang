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

State Management:
    Contexts can carry arbitrary named state via set_state/get_state.
    This allows different layers (device, ml, analytics) to attach their
    own state without the EDSL layer knowing about specific state types.

    State key conventions:
    - "dialect.{name}": Dialect runtime state (e.g., "dialect.simp")
    - "device.cluster": Device/cluster configuration
    - "ml.{component}": ML pipeline components
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
    """Base class for EDSL execution contexts with extensible state slots.

    A Context represents an environment where primitives are executed.
    There are two types of contexts:
    - Tracer: Records operations to Graph IR (compile-time)
    - Interpreter: Execution context (executes operations immediately)

    State Management:
        Contexts can carry arbitrary named state. Different layers can attach
        their own state without the EDSL layer knowing specifics:

        >>> ctx.set_state("device.cluster", cluster_spec)
        >>> ctx.set_state("dialect.simp", simp_driver)
        >>> cluster = ctx.get_state("device.cluster")
    """

    def __init__(self) -> None:
        self._states: dict[str, Any] = {}

    # =========================================================================
    # State Management
    # =========================================================================

    def set_state(self, key: str, value: Any) -> None:
        """Attach state to this context.

        Args:
            key: State key (e.g., "dialect.simp", "device.cluster")
            value: State value
        """
        self._states[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get attached state by key.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value or default
        """
        return self._states.get(key, default)

    def has_state(self, key: str) -> bool:
        """Check if state exists.

        Args:
            key: State key

        Returns:
            True if state exists
        """
        return key in self._states

    # =========================================================================
    # Abstract Methods
    # =========================================================================

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

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> Self:
        """Enter context manager (push context onto stack)."""
        push_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Exit context manager (pop context from stack)."""
        pop_context()


# =============================================================================
# Abstract Interpreter Interface
# =============================================================================


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


# =============================================================================
# Global Context Stack Management
# =============================================================================

_context_stack: list[Context] = []
_default_context: Context | None = None
_default_context_factory: Callable[[], Context] | None = None


def get_current_context() -> Context | None:
    """Get the current active context (top of stack).

    Returns None if no context is active.
    """
    return _context_stack[-1] if _context_stack else None


def push_context(context: Context) -> None:
    """Push a context onto the stack (enter context)."""
    _context_stack.append(context)


def pop_context() -> Context | None:
    """Pop a context from the stack (exit context).

    Returns:
        The popped context, or None if stack was empty.
    """
    return _context_stack.pop() if _context_stack else None


def find_context(predicate: Callable[[Context], bool]) -> Context | None:
    """Find a context in the stack that satisfies the predicate.

    Traverses from top (most recent) to bottom of the context stack,
    returning the first context for which predicate(ctx) returns True.

    Args:
        predicate: A callable that takes a Context and returns True if it matches.

    Returns:
        The first matching Context, or None if no match found.

    Example:
        >>> # Find context with simp dialect state
        >>> ctx = find_context(lambda c: c.has_state("dialect.simp"))
    """
    for ctx in reversed(_context_stack):
        if predicate(ctx):
            return ctx
    return None


def find_context_with_state(key: str) -> Context | None:
    """Find first context that has the specified state.

    Args:
        key: State key to look for

    Returns:
        First context with the state, or None
    """
    return find_context(lambda c: c.has_state(key))


def find_interpreter() -> Context | None:
    """Find first Interpreter in the context stack.

    Returns:
        First Interpreter context, or None if not found.
    """
    return find_context(lambda c: isinstance(c, AbstractInterpreter))


def is_tracing() -> bool:
    """Check if current context is a Tracer.

    Returns:
        True if the top of the context stack is a Tracer.
    """
    from mplang.v2.edsl.tracer import Tracer

    return isinstance(get_current_context(), Tracer)


# =============================================================================
# Default Context Management
# =============================================================================


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


def set_root_context(context: Context, force: bool = False) -> None:
    """Set the root/default execution context.

    This sets the provided context as the base of the context stack.
    All subsequent operations will use this context as the default environment.

    Args:
        context: Context to set as root.
        force: If True, clears the existing context stack before setting.
               If False (default), raises error if stack is not empty.
    """
    if force:
        _context_stack.clear()
        _context_stack.append(context)
        return

    if get_current_context() is not None:
        raise RuntimeError(
            "Cannot set root context: Context stack is not empty. "
            "Use force=True to overwrite the existing context."
        )

    push_context(context)
