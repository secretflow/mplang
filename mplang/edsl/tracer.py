"""Tracer: Python Function → Graph IR.

Responsible for converting Python functions to Graph IR, handling:
- Function parameters
- Captured variables (external references)
- Polymorphic handling of TraceObject/InterpObject

Tracer is a Context (inherits from Context abstract base class).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mplang.edsl.context import Context, pop_context, push_context
from mplang.edsl.graph import Graph
from mplang.edsl.graph import Value as GraphValue
from mplang.edsl.object import Object
from mplang.edsl.typing import BaseType

if TYPE_CHECKING:
    from mplang.edsl.interpreter import InterpObject
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
        self, primitive: Primitive, args: tuple[Object, ...], kwargs: dict[str, Any]
    ) -> TraceObject:
        """Execute primitive by recording to Graph IR (trace mode).

        Args:
            primitive: The primitive to trace
            args: Positional arguments (Objects)
            kwargs: Keyword arguments (plain values)

        Returns:
            TraceObject wrapping the result Value

        Raises:
            RuntimeError: If primitive has no abstract_eval rule
        """
        if primitive._abstract_eval is None:
            raise RuntimeError(
                f"Primitive '{primitive.name}' has no abstract_eval rule. "
                f"Define it using @{primitive.name}_p.def_abstract_eval"
            )

        # Promote InterpObjects to TraceObjects if needed
        trace_args = []
        for arg in args:
            if isinstance(arg, TraceObject):
                trace_args.append(arg)
            else:
                # InterpObject → TraceObject (promote to graph)
                trace_args.append(self.promote(arg))

        # Get input types
        input_types = [arg.type for arg in trace_args]

        # Infer output type using abstract_eval
        output_type = primitive._abstract_eval(*input_types, **kwargs)

        # Add operation to graph using self.graph.add_op()
        input_values = [arg._graph_value for arg in trace_args]

        # Use Graph.add_op() which handles Value creation and Operation registration
        result_value = self.graph.add_op(
            opcode=primitive.name,
            inputs=input_values,
            output_types=[output_type],
            attrs=kwargs,
        )

        # add_op returns Value or list[Value], we know it's Value for single output
        if isinstance(result_value, list):
            result_value = result_value[0]

        # Return TraceObject wrapping the result Value
        return TraceObject(result_value, self)

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

    def promote(self, obj: InterpObject) -> TraceObject:
        """Promote InterpObject → TraceObject.

        Introduces the eager execution value as a captured variable into the Graph.

        Args:
            obj: InterpObject (eager execution object)

        Returns:
            TraceObject (containing corresponding Graph input)
        """
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

    def make_constant(self, value: Any) -> TraceObject:
        """Convert Python constant to TraceObject."""
        import numpy as np

        from mplang.edsl.typing import Tensor, f32

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
