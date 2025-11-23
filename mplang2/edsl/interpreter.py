"""Interpreter: Execute Graph IR and Eager Operations.

Interpreter is a Context that executes operations immediately.
It can execute both:
1. Graph IR (via GraphInterpreter)
2. Eager operations on InterpObject (via backend executors)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mplang2.edsl.context import Context
from mplang2.edsl.graph import Graph
from mplang2.edsl.object import Object
from mplang2.edsl.registry import get_impl
from mplang2.edsl.typing import BaseType

if TYPE_CHECKING:
    from mplang2.edsl.primitive import Primitive


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

    def __init__(self) -> None:
        # GraphValue -> InterpObject cache
        # Maps a GraphValue (IR node) to its computed InterpObject (Runtime result).
        # This serves two purposes:
        # 1. Caching: Avoid re-evaluating the same graph node multiple times.
        # 2. MIMO Optimization: When one output of a multi-output op is computed,
        #    all sibling outputs are cached here to avoid re-execution.
        self._execution_cache: dict[Any, InterpObject] = {}

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
        from mplang2.edsl.tracer import Tracer

        # Create tracer and build graph
        with Tracer() as ctx:
            # Finalize graph by setting outputs
            result_traced = primitive.bind(*args, **kwargs)
            graph = ctx.finalize(result_traced)

        # Build inputs map for interpret
        # We need to map the Graph Inputs (Values) to the Runtime Objects (InterpObjects)
        # The Tracer creates inputs for free variables (which are our args/kwargs)
        # We can recover this mapping from the Tracer's _freevars
        inputs_map = {}
        for _, (obj, graph_value) in ctx._freevars.items():
            if isinstance(obj, InterpObject):
                inputs_map[graph_value] = obj.runtime_obj
            else:
                # For constants/immediates, we might not need to do anything if they are
                # inlined into the graph, but if they are captured as inputs:
                inputs_map[graph_value] = obj

        # Execute graph
        result_runtime = self.evaluate_graph(graph, inputs_map)

        # Wrap result back to InterpObject
        # TODO: rebuild output structure from traced function
        if isinstance(result_traced, (list, tuple)):
            # Multiple outputs
            if not isinstance(result_runtime, (list, tuple)):
                raise RuntimeError(
                    f"Graph returned {len(result_traced)} outputs but interpret() returned single value"
                )
            results = [
                InterpObject(rt, tr.type, self)
                for rt, tr in zip(result_runtime, result_traced, strict=True)
            ]
            return type(result_traced)(results)
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
        from mplang2.edsl.tracer import TraceObject

        if isinstance(obj, InterpObject):
            # InterpObject must belong to this interpreter
            if obj._context is not None and obj._context is not self:
                raise ValueError(
                    f"InterpObject belongs to a different Interpreter. "
                    f"Object context: {obj._context}, Current interpreter: {self}"
                )
            return obj

        elif isinstance(obj, TraceObject):
            # Check execution cache
            # If this value was computed as part of a previous execution (e.g. sibling output)
            # we can return it immediately without re-execution.
            graph_value = obj._graph_value
            if graph_value in self._execution_cache:
                return self._execution_cache[graph_value]

            # First time seeing this Value.
            # We need to execute the graph to compute it.
            # MIMO Optimization:
            # Instead of just asking for this single value, we ask for ALL outputs
            # of the operation that produced this value. This ensures that if we
            # later ask for a sibling output, it will be in the cache.

            tracer = obj._context
            graph = tracer.graph
            defining_op = graph_value.defining_op

            if defining_op is None:
                # Value is likely a constant or input (no defining op in graph)
                # Just execute graph for this single value
                target_outputs = [graph_value]
            else:
                # Fetch all outputs of the defining op
                target_outputs = defining_op.outputs

            # Temporarily set graph outputs to the target outputs
            # We must save/restore original outputs to avoid side effects
            original_outputs = graph.outputs
            graph.outputs = target_outputs

            try:
                # Resolve inputs from Tracer's freevars
                # This maps the graph's input Values to runtime objects
                inputs_map = {}
                for _, (captured_obj, val) in tracer._freevars.items():
                    # Recursively lift captured objects to ensure they are ready
                    lifted = self.lift(captured_obj)
                    if isinstance(lifted, InterpObject):
                        inputs_map[val] = lifted.runtime_obj
                    else:
                        inputs_map[val] = lifted

                # Execute graph
                results_runtime = self.evaluate_graph(graph, inputs_map)

                # If single output, wrap in list for uniform handling
                if len(target_outputs) == 1:
                    results_runtime = [results_runtime]

                # Cache all results
                for val, res in zip(target_outputs, results_runtime, strict=True):
                    # Wrap as InterpObject and cache
                    # Note: We use obj.type for the requested value, but for siblings
                    # we should ideally use their types. However, we don't have TraceObjects
                    # for siblings here, only GraphValues.
                    # InterpObject needs a type. GraphValue has a type.
                    self._execution_cache[val] = InterpObject(res, val.type, self)

            finally:
                # Restore original outputs
                graph.outputs = original_outputs

            # Now the result for our requested object should be in the cache
            if graph_value not in self._execution_cache:
                raise RuntimeError(
                    f"Failed to compute value for {obj} even after graph execution"
                )

            return self._execution_cache[graph_value]

        else:
            # Constants: pass through unchanged
            return obj

    def evaluate_graph(self, graph: Graph, inputs: dict[Any, Any]) -> Any:
        """Execute a Graph IR with runtime data.

        Can be overridden by subclasses to implement remote execution or compilation.

        Args:
            graph: Finalized Graph IR to execute
            inputs: Mapping from Graph Value (input) to Runtime Object

        Returns:
            Runtime execution results corresponding to graph.outputs
        """
        assert len(graph.outputs) > 0, "Graph must be finalized (outputs must be set)"

        # Local environment: Value -> Runtime Object
        env = inputs.copy()

        for op in graph.operations:
            # Resolve inputs
            try:
                args = [env[val] for val in op.inputs]
            except KeyError as e:
                missing_keys = [str(k) for k in op.inputs if k not in env]
                # Limit available keys output to avoid flooding logs if env is huge
                available_keys = [str(k) for k in list(env.keys())[:20]]
                if len(env) > 20:
                    available_keys.append("...")

                raise RuntimeError(
                    f"Failed to resolve inputs for op '{op.opcode}'.\n"
                    f"Missing values: {missing_keys}\n"
                    f"Available values (partial): {available_keys}"
                ) from e

            # Dispatch
            handler = get_impl(op.opcode)
            if handler:
                # Pass interpreter to support recursive execution (HOFs)
                # Pass op to access attributes and regions
                # Pass args as runtime values
                results = handler(self, op, *args)
            else:
                raise NotImplementedError(
                    f"No implementation registered for opcode: {op.opcode}"
                )

            # Update environment with outputs
            # Handler should return a single value or a tuple/list of values
            if len(op.outputs) == 1:
                env[op.outputs[0]] = results
            else:
                if len(results) != len(op.outputs):
                    raise RuntimeError(
                        f"Op {op.opcode} returned {len(results)} values, expected {len(op.outputs)}"
                    )
                for out_val, res in zip(op.outputs, results, strict=True):
                    env[out_val] = res

        # Return outputs
        if len(graph.outputs) == 1:
            return env[graph.outputs[0]]
        else:
            return [env[out] for out in graph.outputs]


def interpret(graph: Graph, inputs: dict[Any, Any], interpreter: Interpreter) -> Any:
    """Execute a Graph IR with runtime data from Interpreter.

    This is a functional helper that delegates to interpreter.evaluate_graph().
    The graph must be finalized (graph.outputs must be set) before interpretation.
    This function executes all operations in the graph and returns the values
    corresponding to graph.outputs.

    Args:
        graph: Finalized Graph IR to execute
        inputs: Mapping from Graph Value (input) to Runtime Object
        interpreter: Interpreter context (for recursive execution)

    Returns:
        Runtime execution results corresponding to graph.outputs:
        - Single value if graph.outputs has 1 element
        - Tuple of values if graph.outputs has multiple elements

    Raises:
        AssertionError: If graph.outputs is empty (graph not finalized)
        NotImplementedError: If opcode has no registered implementation
    """
    return interpreter.evaluate_graph(graph, inputs)
