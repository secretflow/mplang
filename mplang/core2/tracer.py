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

from mplang.core2.context import Context
from mplang.edsl.graph import Graph
from mplang.edsl.graph import Value as GraphValue

if TYPE_CHECKING:
    from mplang.core2.object import InterpObject, Object, TraceObject


class Tracer(Context):
    """Converter from Python Function to Graph IR.

    Inherits from Context and implements execute_add() by recording to Graph.

    Responsibilities:
    1. Convert Python functions to Graph IR
    2. Manage captured variables (external references)
    3. Manage function parameters
    4. Handle Object Hierarchy (TraceObject/InterpObject)
    5. Promote InterpObject → TraceObject
    6. Implement Context.execute_add() by recording to Graph

    Example:
        >>> tracer = Tracer()
        >>> graph = tracer.trace(lambda x, y: x + y, x_interp, y_interp)
        >>> print(graph)
    """

    def __init__(self):
        self.graph = Graph()
        self._captured_vars: dict[int, GraphValue] = {}
        self._params: list[GraphValue] = []

    def execute_add(self, left: Object, right: Object) -> TraceObject:
        """Execute addition by recording to Graph IR.

        This is the Context.execute_add() implementation for Tracer.
        Called by TraceObject.__add__().

        Args:
            left: Left operand (must be TraceObject or will be promoted)
            right: Right operand (must be TraceObject or will be promoted)

        Returns:
            TraceObject containing the result Value in Graph
        """
        from mplang.core2.object import InterpObject, TraceObject

        # Promote InterpObject to TraceObject if needed
        if isinstance(left, InterpObject):
            left = self.promote(left)
        if isinstance(right, InterpObject):
            right = self.promote(right)

        if not isinstance(left, TraceObject) or not isinstance(right, TraceObject):
            raise TypeError("Both operands must be TraceObject or InterpObject")

        # Add operation to Graph
        result = self.graph.add_op(
            "add",
            inputs=[left._graph_value, right._graph_value],
            output_types=[left.type],
        )

        # add_op returns Value | list[Value], extract single value
        result_value = result if isinstance(result, GraphValue) else result[0]
        return TraceObject(result_value, self)

    def trace(self, fn: Callable, *args) -> Graph:
        """Trace a Python function into Graph IR.

        Args:
            fn: Function to trace
            *args: Function arguments (can be InterpObject, TraceObject, or constants)

        Returns:
            Constructed Graph IR
        """
        from mplang.core2.context import get_context

        # 1. Convert arguments to TraceObject
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
        ctx = get_context()
        ctx.enter_context(self)
        try:
            result = fn(*trace_args)
        finally:
            ctx.exit_context()

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
        from mplang.core2.object import TraceObject

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

        from mplang.core2.object import TraceObject
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


def trace(fn: Callable, *args) -> Graph:
    """Convenience function: Trace a Python function into Graph IR.

    Args:
        fn: Function to trace
        *args: Function arguments

    Returns:
        Graph IR

    Example:
        >>> graph = trace(lambda x, y: x + y, x_interp, y_interp)
    """
    tracer = Tracer()
    return tracer.trace(fn, *args)
