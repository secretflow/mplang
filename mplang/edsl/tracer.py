"""Tracer: Python Function → Graph IR.

Responsible for converting Python functions to Graph IR, handling:
- Function parameters
- Captured variables (external references)
- Polymorphic handling of TraceObject/InterpObject

Tracer is a Context (inherits from Context abstract base class).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from jax import tree_flatten

from mplang.edsl.context import Context
from mplang.edsl.graph import Graph
from mplang.edsl.graph import Value as GraphValue
from mplang.edsl.object import Object
from mplang.edsl.typing import BaseType

if TYPE_CHECKING:
    from mplang.edsl.primitive import Primitive


class TraceObject(Object):
    """Trace-time object (during JIT tracing).

    Holds a Value in the Graph IR and a reference to the Tracer (Context).
    All operations delegate to primitives which record into Graph.

    Example:
        >>> from mplang.edsl import trace
        >>> def compute(x, y):
        ...     z = x + y  # TraceObject.__add__ → add_p.bind(x, y)
        ...     return z
        >>> graph = trace(compute, x_interp, y_interp)
    """

    def __init__(self, graph_value: GraphValue, tracer: Tracer):
        self._graph_value = graph_value
        self._context = tracer

    @property
    def type(self) -> BaseType:
        return self._graph_value.type

    @property
    def _tracer(self) -> Tracer:
        """Backward compatibility: access Tracer via _context."""
        return self._context

    def __repr__(self) -> str:
        return f"TraceObject({self._graph_value.name}: {self.type})"


class Tracer(Context):
    """Converter from Python Function to Graph IR.

    Inherits from Context and implements bind_primitive() by recording to Graph.

    Responsibilities:
    1. Convert Python functions to Graph IR
    2. Manage captured variables (external references)
    3. Manage function parameters
    4. Handle Object Hierarchy (TraceObject/InterpObject)
    5. Promote InterpObject → TraceObject
    6. Implement Context.bind_primitive() by recording to Graph

    Example:
        >>> tracer = Tracer()
        >>> graph = tracer.trace(lambda x, y: x + y, x_interp, y_interp)
        >>> print(graph)
    """

    def __init__(self):
        self.graph = Graph()
        self._captured_vars: dict[int, GraphValue] = {}
        self._params: list[GraphValue] = []
        self.captures: dict[str, Any] = {}
        self._arg_counter = 0

    def bind_primitive(
        self, primitive: Primitive, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> TraceObject | list[TraceObject] | Any:
        """Execute primitive by recording to Graph IR (trace mode).

        Handles two modes:
        1. def_trace: Primitive has full control - builds graph via other primitives
        2. def_abstract_eval: Tracer controls - infers types and builds operation

        Args:
            primitive: The primitive to trace
            args: Positional arguments (can be Objects, opaques like callables, or constants)
            kwargs: Keyword arguments (can be Objects, opaques, or constants)

        Returns:
            TraceObject, list[TraceObject], or PyTree containing TraceObjects

        Raises:
            RuntimeError: If primitive has neither trace nor abstract_eval defined
        """
        if primitive._trace is not None:
            return primitive._trace(*args, **kwargs)

        if primitive._abstract_eval is not None:
            trace_args = list(args)
            input_objects = [arg for arg in trace_args if isinstance(arg, TraceObject)]
            input_types = [obj.type for obj in input_objects]

            sig = inspect.signature(primitive._abstract_eval)
            params = list(sig.parameters.values())
            is_flat_style = (
                len(params) >= 2
                and params[0].name == "in_types"
                and params[1].name == "attrs"
            )

            if is_flat_style:
                output_types = primitive._abstract_eval(input_types, kwargs)
            else:
                output_types = primitive._abstract_eval(*input_types, **kwargs)

            if not isinstance(output_types, list):
                output_types = [output_types]

            input_values = [obj._graph_value for obj in input_objects]
            result_values = self.graph.add_op(
                opcode=primitive.name,
                inputs=input_values,
                output_types=output_types,
                attrs=kwargs,
            )

            if not isinstance(result_values, list):
                result_values = [result_values]

            if len(result_values) == 1:
                return TraceObject(result_values[0], self)
            else:
                return [TraceObject(v, self) for v in result_values]

        raise RuntimeError(
            f"Primitive '{primitive.name}' has neither trace nor abstract_eval defined. "
            f"Define one using @{primitive.name}_p.def_trace or @{primitive.name}_p.def_abstract_eval"
        )

    def _capture_as_input(self, obj: Object) -> TraceObject:
        """Capture an object as graph input.

        Helper to avoid code duplication in lift(). Creates a new graph input
        for the object and maintains the obj_id -> Value and name -> obj mappings.

        Args:
            obj: Object to capture (InterpObject or TraceObject from different context)

        Returns:
            TraceObject wrapping the newly created input Value
        """
        obj_id = id(obj)
        if obj_id in self._captured_vars:
            graph_value = self._captured_vars[obj_id]
        else:
            name = f"%arg{self._arg_counter}"
            self._arg_counter += 1
            graph_value = self.graph.add_input(
                name=name,
                type=obj.type,
            )
            self._captured_vars[obj_id] = graph_value
            self.captures[name] = obj
        return TraceObject(graph_value, self)

    def lift(self, obj: Any) -> Any:
        """Lift an object to TraceObject.

        Converts objects to TraceObject for use in tracing:
        - TraceObject (same context): return as-is (idempotent)
        - TraceObject (different context): capture as input (cross-context reference)
        - InterpObject: promote to TraceObject (uses obj id as value name)
        - Non-Object types: return as-is (handled by primitives or opaque)

        Note on non-Object types:
            This method is only called on Objects (via Primitive.bind's lift_if_object).
            Non-Object types (int, float, np.ndarray, callables, etc.) are passed
            directly to primitives without lifting.

        Note on cross-context references:
            When inner tracer (e.g., in run_jax, cond) captures outer TraceObject,
            we need to create a captured input in the inner graph. This handles
            nested tracing scenarios correctly.

        Args:
            obj: Object to lift (should always be an Object subclass in practice)

        Returns:
            TraceObject, or original value for non-Object types (though these
            shouldn't be passed in via normal Primitive.bind flow)
        """
        if isinstance(obj, TraceObject):
            if obj._context is self:
                return obj
            else:
                return self._capture_as_input(obj)
        elif isinstance(obj, Object):
            # Check class name to avoid circular import
            if type(obj).__name__ == "InterpObject":
                return self._capture_as_input(obj)
            else:
                raise TypeError(f"Unknown Object type: {type(obj)}")
        else:
            return obj

    def finalize(self, result: Any) -> Graph:
        """Finalize the graph by setting outputs.

        This marks the traced result as the outputs of the graph,
        completing the graph construction. After this, the graph
        is ready for interpretation or transformation.

        Args:
            result: Traced result - can be:
                - Single TraceObject
                - List/tuple of TraceObjects
                - PyTree containing TraceObjects (TODO)

        Returns:
            The finalized graph (self.graph with outputs set)

        Example:
            >>> tracer = Tracer()
            >>> push_context(tracer)
            >>> result = some_primitive.bind(x, y)
            >>> pop_context()
            >>> graph = tracer.finalize(result)
        """
        out_flat, _out_tree = tree_flatten(result)
        for out in out_flat:
            if not isinstance(out, TraceObject) or out._context is not self:
                raise TypeError(
                    f"Graph output must be TraceObject from this Tracer context, got: {type(out)}"
                )
            self.graph.add_output(out._graph_value)

        return self.graph


def trace(fn: Callable | Primitive, *args, **kwargs) -> Graph:
    """Convenience function: Trace a Python function or Primitive into Graph IR.

    Args:
        fn: Function or Primitive to trace
        *args: Function arguments

    Returns:
        Graph IR

    Example:
        >>> # Trace a lambda function
        >>> graph = trace(lambda x, y: x + y, x_interp, y_interp)

        >>> # Trace a primitive directly
        >>> graph = trace(add_p, x_interp, y_interp)
    """
    tracer = Tracer()
    with tracer:
        result = fn(*args, **kwargs)
    return tracer.finalize(result)
