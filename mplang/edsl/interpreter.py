"""Interpreter: Execute Graph IR and Eager Operations.

Interpreter is a Context that executes operations immediately.
It can execute both:
1. Graph IR (via GraphInterpreter)
2. Eager operations on InterpObject (via backend executors)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mplang.edsl.context import Context
from mplang.edsl.graph import Graph
from mplang.edsl.object import Object
from mplang.edsl.typing import BaseType

if TYPE_CHECKING:
    from mplang.edsl.primitive import Primitive


class InterpObject(Object):
    """Interp-time object (during eager execution).

    Holds a runtime object (the actual data/handle owned by the backend executor)
    and a reference to the Interpreter (Context).
    Operations delegate to primitives which execute immediately.

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
        >>> z = x + y  # InterpObject.__add__ → add_p.bind(x, y)

        >>> # MP backend (distributed execution)
        >>> x = mp.random.uniform(shape=(10,))  # InterpObject with backend handle
        >>> y = mp.random.uniform(shape=(10,))
        >>> z = x + y  # InterpObject.__add__ → add_p.bind(x, y)
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
            interpreter: Interpreter context (if None, uses default interpreter)
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

    def __repr__(self) -> str:
        runtime_repr = repr(self._runtime_obj)
        # Truncate long representations
        if len(runtime_repr) > 50:
            runtime_repr = runtime_repr[:47] + "..."
        return f"InterpObject({runtime_repr}, type={self.type})"


class Interpreter(Context):
    """Execution context for eager execution.

    Inherits from Context and implements bind_primitive() by executing immediately.

    Responsibilities:
    1. Execute primitives on InterpObject immediately
    2. Delegate to backend-specific executors
    3. Execute Graph IR (via GraphInterpreter)

    Example:
        >>> interp = Interpreter()
        >>> x = InterpObject(np.array([1, 2, 3]), Tensor[f32, (3,)])
        >>> y = InterpObject(np.array([4, 5, 6]), Tensor[f32, (3,)])
        >>> z = x + y  # InterpObject.__add__ → add_p.bind(x, y)
    """

    def __init__(self):
        # TODO: Backend executor registry
        self._executors = {}

    def bind_primitive(
        self, primitive: Primitive, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> InterpObject | list[InterpObject] | Any:
        """Execute primitive by building and executing Graph IR.

        All primitives (both def_abstract_eval and def_trace) produce Graph IR
        which is then executed by the backend.

        Args:
            primitive: The primitive to execute
            args: Positional arguments (can be Objects, opaques like callables, or constants)
            kwargs: Keyword arguments (can be Objects, opaques, or constants)

        Returns:
            InterpObject, list[InterpObject], or PyTree containing InterpObjects

        Raises:
            RuntimeError: If primitive has neither trace nor abstract_eval defined
        """
        # For now, primitives with def_trace are called directly
        # In the future, all primitives will build Graph IR that backends execute
        if primitive._trace is not None:
            # Execute the trace function with actual Objects
            # This allows code like run_jax to work in eager mode
            result = primitive._trace(*args, **kwargs)
            return result

        # For primitives with only def_abstract_eval, we need to build a simple Graph
        # This will be implemented when we have backend execution infrastructure
        if primitive._abstract_eval is not None:
            # TODO: Build Graph IR and execute via backend
            # For now, raise NotImplementedError
            raise NotImplementedError(
                f"Primitive '{primitive.name}' execution via Graph IR not yet implemented. "
                f"Use def_trace() to provide custom execution logic."
            )

        # No implementation
        raise RuntimeError(
            f"Primitive '{primitive.name}' has neither trace nor abstract_eval defined. "
            f"Define one using @{primitive.name}_p.def_trace or @{primitive.name}_p.def_abstract_eval"
        )

    def lift(self, obj: Any) -> InterpObject:
        """Lift an object to InterpObject.

        For Interpreter, most objects are already InterpObject.
        Non-Object values are kept as-is (will be handled by primitives).

        Args:
            obj: Object to lift

        Returns:
            The object (InterpObject or constant)
        """
        # InterpObject: already correct type
        if isinstance(obj, InterpObject):
            return obj
        # TraceObject: should not happen in eager mode
        elif isinstance(obj, Object):
            raise TypeError(
                f"Cannot lift {type(obj).__name__} to InterpObject in Interpreter context"
            )
        # Constants: return as-is (primitives will handle)
        else:
            return obj


def interpret(graph: Graph, args: tuple) -> Any:
    """Convenience function: Interpret and execute a Graph.

    Args:
        graph: Graph IR
        args: Input arguments

    Returns:
        Execution result
    """
    raise NotImplementedError("interpret() not yet implemented")
