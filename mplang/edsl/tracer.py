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
    ) -> TraceObject | list[TraceObject] | Any:
        """Execute primitive by recording to Graph IR (trace mode).

        Handles two modes:
        1. def_trace: Custom trace logic with PyTree support
        2. def_abstract_eval: Automatic type inference with input/output separation

        Args:
            primitive: The primitive to trace
            args: Positional arguments (Objects or plain values)
            kwargs: Keyword arguments (Objects or plain values)

        Returns:
            TraceObject, list[TraceObject], or PyTree containing TraceObjects

        Raises:
            RuntimeError: If primitive has neither trace nor abstract_eval defined
        """
        # Mode 1: Custom trace (full control)
        if primitive._trace is not None:
            from jax.tree_util import tree_flatten, tree_unflatten

            from mplang.utils.func_utils import var_morph

            # Call the custom trace function
            # It can access other primitives and construct arbitrary output structure
            result = primitive._trace(*args, **kwargs)

            # Extract Objects from inputs (for IR recording)
            in_vars, _, in_morph = var_morph(
                (args, kwargs), is_variable=lambda x: isinstance(x, Object)
            )

            # All Objects should already be TraceObjects (lifted by Primitive.bind)
            trace_in_vars = []
            for var in in_vars:
                if isinstance(var, TraceObject):
                    trace_in_vars.append(var)
                else:
                    raise TypeError(
                        f"Expected TraceObject in bind_primitive, got {type(var)}. "
                        f"Objects should be lifted by Primitive.bind() before reaching here."
                    )

            # Flatten output to extract Objects
            result_flat, out_tree = tree_flatten(result)
            out_vars = [v for v in result_flat if isinstance(v, Object)]
            out_types = [v.type for v in out_vars]

            # Record operation with morph info in attrs
            input_values = [v._graph_value for v in trace_in_vars]
            result_values = self.graph.add_op(
                opcode=primitive.name,
                inputs=input_values,
                output_types=out_types,
                attrs={
                    "_in_morph": in_morph,  # Input structure (for reconstruction)
                    "_out_tree": out_tree,  # Output structure
                },
            )

            # Ensure result_values is a list
            if not isinstance(result_values, list):
                result_values = [result_values]

            # Replace Objects in result_flat with TraceObjects
            trace_objects = [TraceObject(v, self) for v in result_values]
            obj_idx = 0
            reconstructed_flat = []
            for item in result_flat:
                if isinstance(item, Object):
                    reconstructed_flat.append(trace_objects[obj_idx])
                    obj_idx += 1
                else:
                    # Keep non-Object values (constants)
                    reconstructed_flat.append(item)

            # Reconstruct output PyTree
            return tree_unflatten(out_tree, reconstructed_flat)

        # Mode 2: Abstract eval (automatic)
        if primitive._abstract_eval is not None:
            import inspect

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
        - TraceObject: return as-is
        - InterpObject: promote to TraceObject (as captured input)
        - Numeric constants (int, float, np.ndarray): convert to TraceObject
        - Other types (callable, etc.): return as-is (opaque)

        Args:
            obj: Object to lift

        Returns:
            TraceObject for traceable values, or original value for opaque types
        """
        if isinstance(obj, TraceObject):
            return obj
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
        elif callable(obj):
            # Functions/callables are opaque - don't trace them
            return obj
        elif isinstance(obj, (int, float)):
            # Numeric constants → TraceObject
            return self.make_constant(obj)
        else:
            # For other types (e.g., np.ndarray), try to make constant
            import numpy as np

            if isinstance(obj, np.ndarray):
                return self.make_constant(obj)
            else:
                # Unknown type - pass through as opaque
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
