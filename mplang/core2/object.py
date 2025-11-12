"""Object Hierarchy: TraceObject vs InterpObject.

Runtime object abstractions for distinguishing trace-time and interp-time execution.

TraceObject and InterpObject now hold their Context (Tracer or Interpreter).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mplang.edsl.typing import BaseType

if TYPE_CHECKING:
    from mplang.core2.interp import Interpreter
    from mplang.core2.tracer import Tracer
    from mplang.edsl.graph import Value as GraphValue


class Object(ABC):
    """Base class for MPLang runtime objects.

    This is a Driver-side abstraction used for:
    1. Distinguishing between trace-time and interp-time objects
    2. Providing uniform operation interfaces (arithmetic, attribute access, etc.)
    3. Enabling polymorphic handling by the Tracer

    Subclasses:
    - TraceObject: Trace-time object (holds a Value in Graph IR)
    - InterpObject: Interp-time object (holds backend-specific runtime data)
    """

    @property
    @abstractmethod
    def type(self) -> BaseType:
        """Type of the object (available in both trace and interp modes)."""

    @abstractmethod
    def __add__(self, other: Object) -> Object:
        """Arithmetic addition (intercepted by Tracer or executed immediately)."""

    # TODO: Other operators (__mul__, __sub__, __matmul__, etc.)


class TraceObject(Object):
    """Trace-time object (during JIT tracing).

    Holds a Value in the Graph IR and a reference to the Tracer (Context).
    All operations delegate to Tracer.execute_add() which records into Graph.

    Example:
        >>> from mplang.core2 import trace
        >>> def compute(x, y):
        ...     z = x + y  # TraceObject.__add__ → Tracer.execute_add()
        ...     return z
        >>> graph = trace(compute, x_interp, y_interp)
    """

    def __init__(self, graph_value: GraphValue, tracer: Tracer):
        self._graph_value = graph_value
        self._context = tracer  # TraceObject holds its Tracer (Context)

    @property
    def type(self) -> BaseType:
        return self._graph_value.type

    @property
    def _tracer(self) -> Tracer:
        """Backward compatibility: access Tracer via _context."""
        return self._context

    def __add__(self, other: Object) -> TraceObject:
        """Delegate addition to Tracer.execute_add()."""
        return self._context.execute_add(self, other)

    def __repr__(self) -> str:
        return f"TraceObject({self._graph_value.name}: {self.type})"


class InterpObject(Object):
    """Interp-time object (during eager execution).

    Holds a runtime object (the actual data/handle owned by the backend executor)
    and a reference to the Interpreter (Context).
    Operations delegate to Interpreter.execute_add() which executes immediately.

    The runtime object can be:
    - FHE backend: Local TenSEAL/SEAL ciphertext
    - JAX backend: Local jax.Array
    - MP backend: Backend handle (pointer to party-side data)
    - SQL backend: DatabaseHandle
    - etc.

    Example:
        >>> # FHE backend (local execution)
        >>> x = fhe.encrypt([1, 2, 3])  # InterpObject with local ciphertext
        >>> y = fhe.encrypt([4, 5, 6])
        >>> z = x + y  # InterpObject.__add__ → Interpreter.execute_add()

        >>> # MP backend (distributed execution)
        >>> x = mp.random.uniform(shape=(10,))  # InterpObject with backend handle
        >>> y = mp.random.uniform(shape=(10,))
        >>> z = x + y  # InterpObject.__add__ → Interpreter.execute_add()
    """

    def __init__(
        self,
        runtime_obj: Any,
        obj_type: BaseType,
        interpreter: Interpreter | None = None,
    ):
        """Initialize InterpObject.

        Args:
            runtime_obj: Backend-specific runtime object (ciphertext, array, handle, etc.)
            obj_type: Type of the object (BaseType from edsl.typing)
            interpreter: Interpreter context (if None, uses default from ExecutionContext)
        """
        self._runtime_obj = runtime_obj
        self._type = obj_type
        self._context = interpreter  # InterpObject holds its Interpreter (Context)

    @property
    def type(self) -> BaseType:
        return self._type

    @property
    def runtime_obj(self) -> Any:
        """Get the underlying runtime object (backend-specific)."""
        return self._runtime_obj

    def __add__(self, other: Object) -> InterpObject:
        """Delegate addition to Interpreter.execute_add()."""
        from mplang.core2.context import get_context

        # Get the interpreter context
        interpreter = self._context
        if interpreter is None:
            # Use default interpreter from ExecutionContext
            ctx = get_context()
            interpreter = ctx.default_interpreter

        # Delegate to Interpreter.execute_add()
        return interpreter.execute_add(self, other)

    def __repr__(self) -> str:
        runtime_repr = repr(self._runtime_obj)
        # Truncate long representations
        if len(runtime_repr) > 50:
            runtime_repr = runtime_repr[:47] + "..."
        return f"InterpObject({runtime_repr}, type={self.type})"
