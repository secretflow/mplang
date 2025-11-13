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

import numpy as np

from mplang.edsl.context import Context, pop_context, push_context
from mplang.edsl.graph import Graph
from mplang.edsl.graph import Value as GraphValue
from mplang.edsl.object import Object
from mplang.edsl.typing import BaseType

if TYPE_CHECKING:
    from mplang.edsl.primitive import Primitive
    from mplang.edsl.typing import Tensor, f32


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
        self._context = tracer  # TraceObject holds its Tracer (Context)

    @property
    def type(self) -> BaseType:
        return self._graph_value.type

    @property
    def _tracer(self) -> Tracer:
        """Backward compatibility: access Tracer via _context."""
        return self._context

    def __add__(self, other: Object) -> Object:
        """Delegate addition to add primitive."""
        from mplang.edsl.primitive import add_p

        return add_p.bind(self, other)

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
        # Mode 1: def_trace - Primitive has full control
        # The primitive builds the graph by calling other primitives.
        # Tracer simply returns the result without adding extra operations.
        if primitive._trace is not None:
            return primitive._trace(*args, **kwargs)

        # Mode 2: def_abstract_eval - Tracer controls graph construction
        if primitive._abstract_eval is not None:
            # All Objects should already be TraceObjects (lifted by Primitive.bind)
            trace_args = list(args)  # args already lifted

            # Extract Object inputs and their types
            input_objects = [arg for arg in trace_args if isinstance(arg, TraceObject)]
            input_types = [obj.type for obj in input_objects]

            # Detect signature style (positional vs flat)
            sig = inspect.signature(primitive._abstract_eval)
            params = list(sig.parameters.values())

            # Check if it's flat style: (in_types: list[BaseType], attrs: dict)
            is_flat_style = (
                len(params) >= 2
                and params[0].name == "in_types"
                and params[1].name == "attrs"
            )

            # Call abstract_eval to get output types
            if is_flat_style:
                output_types = primitive._abstract_eval(input_types, kwargs)
            else:
                output_types = primitive._abstract_eval(*input_types, **kwargs)

            # Normalize output_types to list
            if not isinstance(output_types, list):
                output_types = [output_types]

            # Add operation to graph
            input_values = [obj._graph_value for obj in input_objects]
            result_values = self.graph.add_op(
                opcode=primitive.name,
                inputs=input_values,
                output_types=output_types,
                attrs=kwargs,
            )

            # Ensure result_values is a list
            if not isinstance(result_values, list):
                result_values = [result_values]

            # Return TraceObject(s)
            if len(result_values) == 1:
                return TraceObject(result_values[0], self)
            else:
                return [TraceObject(v, self) for v in result_values]

        # No trace or abstract_eval defined
        raise RuntimeError(
            f"Primitive '{primitive.name}' has neither trace nor abstract_eval defined. "
            f"Define one using @{primitive.name}_p.def_trace or @{primitive.name}_p.def_abstract_eval"
        )

    def lift(self, obj: Any) -> Any:
        """Lift an object to TraceObject.

        Converts objects to TraceObject for use in tracing:
        - TraceObject (same context): return as-is (idempotent)
        - TraceObject (different context): capture as input (cross-context reference)
        - InterpObject: promote to TraceObject (as captured input)
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
            # Check if TraceObject belongs to this context
            if obj._context is self:
                # Same context: idempotent
                return obj
            else:
                # Different context: capture as input (cross-context reference)
                # This handles nested tracers (e.g., run_jax, cond, while_loop)
                obj_id = id(obj)
                if obj_id in self._captured_vars:
                    # Already captured
                    graph_value = self._captured_vars[obj_id]
                else:
                    # Create new input node for captured TraceObject
                    graph_value = self.graph.add_input(
                        name=f"captured_{len(self._captured_vars)}",
                        type=obj.type,
                    )
                    self._captured_vars[obj_id] = graph_value
                return TraceObject(graph_value, self)
        elif isinstance(obj, Object):
            # InterpObject → TraceObject (promote)
            from mplang.edsl.interpreter import InterpObject

            if isinstance(obj, InterpObject):
                # Promote: introduce eager value as captured variable in Graph
                obj_id = id(obj)
                if obj_id in self._captured_vars:
                    # Already promoted
                    graph_value = self._captured_vars[obj_id]
                else:
                    # Create new input node (representing captured variable)
                    graph_value = self.graph.add_input(
                        name=f"captured_{len(self._captured_vars)}",
                        type=obj.type,
                    )
                    self._captured_vars[obj_id] = graph_value
                return TraceObject(graph_value, self)
            else:
                raise TypeError(f"Unknown Object type: {type(obj)}")
        else:
            # Non-Object types: pass through as-is (should be rare in normal flow)
            # Primitives.bind filters Objects before calling lift, so constants,
            # callables, etc. are passed directly to primitives
            return obj

    def trace(self, fn: Callable | Primitive, *args) -> Graph:
        """Trace a Python function or Primitive into Graph IR.

        Args:
            fn: Function or Primitive to trace
            *args: Function arguments (can be InterpObject, TraceObject, or constants)

        Returns:
            Constructed Graph IR

        Note:
            When tracing a Callable, captured variables cannot be properly handled
            in execute_add(). Use Primitives for operations that need InterpObject
            promotion.
        """

        # Handle Primitive directly
        from mplang.edsl.primitive import Primitive

        if isinstance(fn, Primitive):
            # Enter trace context
            push_context(self)
            try:
                result = fn.bind(*args)
            finally:
                pop_context()

            # Mark output
            if isinstance(result, TraceObject):
                self.graph.add_output(result._graph_value)
            elif isinstance(result, (tuple, list)):
                for r in result:
                    if isinstance(r, TraceObject):
                        self.graph.add_output(r._graph_value)

            return self.graph

        # Handle Callable
        # 1. Convert arguments to TraceObject
        from mplang.edsl.interpreter import InterpObject

        trace_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, InterpObject):
                # Promote to TraceObject (as graph input)
                graph_value = self.graph.add_input(
                    name=f"arg{i}",
                    type=arg.type,
                )
                trace_arg = TraceObject(graph_value, self)
                self._params.append(graph_value)
            elif isinstance(arg, TraceObject):
                trace_arg = arg
                self._params.append(arg._graph_value)
            else:
                # Constant
                trace_arg = self.make_constant(arg)

            trace_args.append(trace_arg)

        # 2. Execute function in tracing context
        push_context(self)
        try:
            result = fn(*trace_args)
        finally:
            pop_context()

        # 3. Mark outputs
        if isinstance(result, TraceObject):
            self.graph.add_output(result._graph_value)
        elif isinstance(result, (tuple, list)):
            for r in result:
                if isinstance(r, TraceObject):
                    self.graph.add_output(r._graph_value)

        return self.graph

    def make_constant(self, value: Any) -> TraceObject:
        """Convert Python constant to TraceObject."""
        if isinstance(value, (int, float)):
            # Scalar constant
            graph_value = self.graph.add_constant(
                value,
                type=Tensor[f32, ()],
            )
            return TraceObject(graph_value, self)
        elif isinstance(value, np.ndarray):
            # Array constant
            graph_value = self.graph.add_constant(
                value,
                type=Tensor[f32, value.shape],
            )
            return TraceObject(graph_value, self)
        else:
            raise NotImplementedError(f"Unsupported constant type: {type(value)}")


def trace(fn: Callable | Primitive, *args) -> Graph:
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
    return tracer.trace(fn, *args)
