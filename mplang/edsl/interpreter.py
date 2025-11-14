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
        # Object registry: id(InterpObject) -> InterpObject
        # Used to resolve graph.inputs back to runtime objects
        self._objects: dict[int, InterpObject] = {}
        # TraceObject -> InterpObject cache: id(TraceObject) -> InterpObject
        # Used to avoid re-evaluating when the same TraceObject is lifted multiple times
        # Key is id(TraceObject), not id(Graph), because different TraceObjects
        # from the same graph represent different Values
        self._trace_obj_cache: dict[int, InterpObject] = {}

    def bind_primitive(
        self, primitive: Primitive, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> InterpObject | list[InterpObject] | Any:
        """Execute primitive by tracing and interpreting.

        Implements the unified trace → interpret flow:
        1. All InterpObject arguments already registered via lift()
        2. Create a Tracer and push it as context
        3. Call primitive.bind() to build Graph IR (uses obj id in value names)
        4. Execute the graph via interpret() (resolves inputs via registry)

        Args:
            primitive: The primitive to execute
            args: Positional arguments (already lifted by Primitive.bind)
            kwargs: Keyword arguments (already lifted by Primitive.bind)

        Returns:
            Execution result (InterpObject or list of InterpObject)
        """
        from mplang.edsl.tracer import Tracer

        # Create tracer and build graph
        with Tracer() as tracer:
            # Finalize graph by setting outputs
            result_traced = primitive.bind(*args, **kwargs)
            graph = tracer.finalize(result_traced)

        # Execute graph (uses self._objects to resolve inputs)
        result_runtime = interpret(graph, self)

        # Wrap result back to InterpObject
        # TODO: Get result type from graph outputs
        if isinstance(result_traced, list):
            # Multiple outputs
            if not isinstance(result_runtime, (list, tuple)):
                raise RuntimeError(
                    f"Graph returned {len(result_traced)} outputs but interpret() returned single value"
                )
            return [
                InterpObject(rt, tr.type, self)
                for rt, tr in zip(result_runtime, result_traced, strict=True)
            ]
        else:
            # Single output
            return InterpObject(result_runtime, result_traced.type, self)

    def lift(self, obj: Any) -> InterpObject | Any:
        """Lift an object to the Interpreter's native representation.

        This is THE central method that manages the boundary between
        InterpObject and TraceObject:

        1. **InterpObject → TraceObject** (during nested tracing):
           - Register the InterpObject in self._objects for later resolution
           - The InterpObject must belong to this Interpreter
           - When the object flows into Tracer.lift() during bind_primitive,
             it will be captured as input with a clean SSA name like "%arg0"

        2. **TraceObject → InterpObject** (evaluate traced computation):
           - Extract the graph from the TraceObject's context (Tracer)
           - Execute the graph via interpret() to get runtime result
           - Wrap result as InterpObject and register it

        3. **Constants**: Pass through unchanged

        Args:
            obj: Object to lift (InterpObject, TraceObject, or constant)

        Returns:
            InterpObject (if Object input) or constant (pass-through)

        Example:
            >>> # InterpObject case
            >>> x = InterpObject(np.array([1, 2]), Tensor[f32, (2,)])
            >>> x_lifted = interp.lift(x)  # registers in _objects, returns x
            >>>
            >>> # TraceObject case
            >>> tracer = Tracer()
            >>> push_context(tracer)
            >>> z_trace = some_primitive.bind(x, y)  # TraceObject
            >>> pop_context()
            >>> interp = Interpreter()
            >>> z_interp = interp.lift(z_trace)  # evaluate graph → InterpObject
        """
        from mplang.edsl.tracer import TraceObject

        if isinstance(obj, InterpObject):
            # InterpObject must belong to this interpreter
            # (In future: verify obj._context is self or None)

            # Register for later resolution when graph inputs are processed
            obj_id = id(obj)
            if obj_id not in self._objects:
                self._objects[obj_id] = obj

            # Return as-is (will be captured by nested Tracer.lift if tracing)
            return obj

        elif isinstance(obj, TraceObject):
            # Check cache: have we already lifted this exact TraceObject?
            trace_obj_id = id(obj)
            if trace_obj_id in self._trace_obj_cache:
                return self._trace_obj_cache[trace_obj_id]

            # First time seeing this TraceObject
            # TODO (MIMO optimization): When lifting multiple TraceObjects from
            # the same graph, we currently execute the graph multiple times.
            # Future: execute once and cache all intermediate results.

            # Get the graph and value from TraceObject
            tracer = obj._context
            graph = tracer.graph
            # value = obj._graph_value  # TODO: use this for MIMO optimization

            # Execute the graph to get runtime result for this specific Value
            # TODO: For now, interpret() returns single value
            # MIMO optimization: make interpret() return dict[value_name -> runtime_obj]
            # so we can cache all intermediate results from one execution
            result_runtime = interpret(graph, self)

            # Wrap as InterpObject, register, and cache
            result_obj = InterpObject(result_runtime, obj.type, self)
            self._objects[id(result_obj)] = result_obj
            self._trace_obj_cache[trace_obj_id] = result_obj

            return result_obj

        else:
            # Constants: pass through unchanged
            return obj


def interpret(graph: Graph, interpreter: Interpreter) -> Any:
    """Execute a Graph IR with runtime data from Interpreter.

    The graph must be finalized (graph.outputs must be set) before interpretation.
    This function executes all operations in the graph and returns the values
    corresponding to graph.outputs.

    Args:
        graph: Finalized Graph IR to execute
               - graph.inputs: runtime data references with clean SSA names (e.g., "%arg0")
               - graph.outputs: Values to compute and return
               - graph.operations: computation steps
        interpreter: Interpreter context that owns the runtime objects
                    - interpreter._objects[obj_id] provides runtime data for inputs

    Returns:
        Runtime execution results corresponding to graph.outputs:
        - Single value if graph.outputs has 1 element
        - Tuple of values if graph.outputs has multiple elements

    Raises:
        AssertionError: If graph.outputs is empty (graph not finalized)

    Note:
        This function should be implemented by concrete interpreters:
        - Simulator: Local multi-threaded execution
        - Driver: Distributed execution coordinator
        - BackendInterpreter: Single-backend execution (JAX, FHE, etc.)

    Example:
        >>> tracer = Tracer()
        >>> push_context(tracer)
        >>> result = some_primitive.bind(x, y)
        >>> pop_context()
        >>> graph = tracer.finalize(result)  # Sets graph.outputs
        >>> runtime_result = interpret(graph, interpreter)
        >>> # Internally:
        >>> # 1. Get graph.inputs[i].name -> "%arg0"
        >>> # 2. Lookup: interpreter._objects[123] -> InterpObject
        >>> # 3. Extract: InterpObject.runtime_obj -> actual data
        >>> # 4. Execute operations
        >>> # 5. Return values for graph.outputs
    """
    assert len(graph.outputs) > 0, "Graph must be finalized (outputs must be set)"
    raise NotImplementedError("interpret() not yet implemented")
