"""Tracer: Python Function → Graph IR.

Responsible for converting Python functions to Graph IR, handling:
- Function parameters
- Free variables (external references including captures)
- Polymorphic handling of TraceObject/InterpObject

Tracer is a Context (inherits from Context abstract base class).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jax.tree_util import PyTreeDef, tree_flatten, tree_map

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
    2. Manage free variables (function params and captured external references)
    3. Handle Object Hierarchy (TraceObject/InterpObject)
    4. Promote InterpObject → TraceObject
    5. Implement Context.bind_primitive() by recording to Graph

    Example:
        >>> tracer = Tracer()
        >>> graph = tracer.trace(lambda x, y: x + y, x_interp, y_interp)
        >>> print(graph)
    """

    def __init__(self):
        self.graph = Graph()
        self._freevars: dict[int, tuple[Object, GraphValue]] = {}
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
            outs = [TraceObject(v, self) for v in result_values]
            return outs[0] if len(outs) == 1 else outs

        raise RuntimeError(
            f"Primitive '{primitive.name}' has neither trace nor abstract_eval defined. "
            f"Define one using @{primitive.name}_p.def_trace or @{primitive.name}_p.def_abstract_eval"
        )

    def _capture_as_input(self, obj: Object) -> TraceObject:
        """Capture an object as graph input (free variable).

        Helper to avoid code duplication in lift(). Creates a new graph input
        for the object and maintains the obj_id → Value mapping for deduplication.

        Free variables include:
        - Function parameters (from outer context)
        - Captured external variables (from outer context)

        Args:
            obj: Object to capture (InterpObject or TraceObject from different context)

        Returns:
            TraceObject wrapping the newly created input Value
        """
        obj_id = id(obj)
        entry = self._freevars.get(obj_id)
        if entry is not None:
            _, graph_value = entry
        else:
            name = f"%arg{self._arg_counter}"
            self._arg_counter += 1
            graph_value = self.graph.add_input(
                name=name,
                type=obj.type,
            )
            self._freevars[obj_id] = (obj, graph_value)
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

    def reconstruct_outputs(
        self,
        out_var_pos: list[int],
        out_imms: list[Any],
        out_tree: PyTreeDef,
        result_values: list[GraphValue],
    ) -> Any:
        """Rebuild PyTree outputs from recorded metadata."""

        var_iter = iter([TraceObject(val, self) for val in result_values])
        var_pos_iter = iter(out_var_pos)
        next_var_pos = next(var_pos_iter, None)
        imm_idx = 0
        total_len = len(out_imms) + len(out_var_pos)
        flat_out: list[Any] = []
        for idx in range(total_len):
            if next_var_pos is not None and idx == next_var_pos:
                flat_out.append(next(var_iter))
                next_var_pos = next(var_pos_iter, None)
            else:
                flat_out.append(out_imms[imm_idx])
                imm_idx += 1
        return out_tree.unflatten(flat_out)



def _separate_vars_and_imms(
    flat_values: list[Any],
) -> tuple[list[Any], list[int], list[Any]]:
    """Separate a flattened list into variables (Objects) and immediates (constants).

    Args:
        flat_values: Flattened list of values (mix of Objects and constants)

    Returns:
        Tuple of (imms, var_pos, vars) where:
            - imms: List of immediate values (constants) in order
            - var_pos: List of positions where variables appear in flat_values
            - vars: List of variable values (Objects) in order
    """
    imms = []
    var_pos = []
    vars_list = []

    for i, val in enumerate(flat_values):
        if isinstance(val, Object):
            var_pos.append(i)
            vars_list.append(val)
        else:
            imms.append(val)

    return imms, var_pos, vars_list


@dataclass
class TracedFunction:
    """Result of tracing a Python function into Graph IR.

    Represents a fully Pythonic function captured as a graph, distinguishing
    between constants (immediates) and traced values (graph inputs/outputs).

    Attributes:
        name: Function name (from fn.__name__)
        graph: The finalized Graph IR containing traced computations
        in_imms: Input immediates (constants) in flattened order
        in_var_pos: Positions of graph.inputs in the flattened input list
        in_tree: PyTreeDef to reconstruct (args, kwargs) from flattened inputs
        out_imms: Output immediates (constants) in flattened order
        out_var_pos: Positions of graph.outputs in the flattened output list
        out_tree: PyTreeDef to reconstruct result from flattened outputs

    Reconstruction:
        To reconstruct *args, **kwargs from graph.inputs:
        1. Create flattened list: [in_imms[i] if i not in in_var_pos else graph.inputs[...]]
        2. Use in_tree.unflatten() to get (args, kwargs)

        To reconstruct result from graph.outputs:
        1. Create flattened list: [out_imms[i] if i not in out_var_pos else graph.outputs[...]]
        2. Use out_tree.unflatten() to get result

    Example:
        >>> def fn(x, y, *, scale=2.0):
        ...     return x + y, scale
        >>> traced = make_graph(fn, x_obj, y_obj, scale=2.0)
        >>> # in_imms = [2.0], in_var_pos = [0, 1] (x, y are vars)
        >>> # out_imms = [2.0], out_var_pos = [0] (x+y is var, scale is constant)
    """

    name: str
    graph: Graph
    in_imms: list[Any]
    in_var_pos: list[int]
    in_tree: PyTreeDef
    out_imms: list[Any]
    out_var_pos: list[int]
    out_tree: PyTreeDef
    captured: list[Object]

    def is_input_signature_match(self, other: TracedFunction) -> bool:
        """Check if this TracedFunction has the same input signature as another.

        Args:
            other: Another TracedFunction to compare against

        Returns:
            True if input counts and types match, False otherwise
        """
        if len(self.graph.inputs) != len(other.graph.inputs):
            return False
        return all(
            self_in.type == other_in.type
            for self_in, other_in in zip(
                self.graph.inputs, other.graph.inputs, strict=True
            )
        )

    def is_output_signature_match(self, other: TracedFunction) -> bool:
        """Check if this TracedFunction has the same output signature as another.

        Args:
            other: Another TracedFunction to compare against

        Returns:
            True if output counts and types match, False otherwise
        """
        if len(self.graph.outputs) != len(other.graph.outputs):
            return False
        return all(
            self_out.type == other_out.type
            for self_out, other_out in zip(
                self.graph.outputs, other.graph.outputs, strict=True
            )
        )


def trace(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> TracedFunction:
    """Trace a Python function with PyTree inputs/outputs into a Graph.

    This utility function traces an arbitrary Python function into Graph IR,
    distinguishing between constants (immediates) and traced values (variables).

    Args:
        fn: Python callable to trace. Can accept/return arbitrary PyTrees.
        *args: Positional arguments (arbitrary PyTree structure).
               Can mix Objects (traced as variables) and constants (immediates).
        **kwargs: Keyword arguments (arbitrary PyTree structure).

    Returns:
        TracedFunction containing:
            - name: Function name
            - graph: Graph IR with only variable inputs/outputs
            - in_imms: Input constants in order
            - in_var_pos: Positions of graph.inputs in flattened input
            - in_tree: PyTreeDef to reconstruct (args, kwargs)
            - out_imms: Output constants in order
            - out_var_pos: Positions of graph.outputs in flattened output
            - out_tree: PyTreeDef to reconstruct result

    Example:
        >>> # Function with mixed inputs
        >>> def scale_add(x, y, scale=2.0):
        ...     return x + y, scale
        >>> traced = make_graph(scale_add, x_obj, y_obj, scale=2.0)
        >>> # traced.in_imms = [2.0]
        >>> # traced.in_var_pos = [0, 1]  # x, y are variables
        >>> # traced.out_imms = [2.0]
        >>> # traced.out_var_pos = [0]  # x+y is variable, scale is constant
        >>>
        >>> # Reconstruct inputs from graph.inputs
        >>> flat_in = [None] * (len(traced.in_imms) + len(traced.in_var_pos))
        >>> imm_idx = 0
        >>> for i in range(len(flat_in)):
        ...     if i in traced.in_var_pos:
        ...         var_idx = traced.in_var_pos.index(i)
        ...         flat_in[i] = graph.inputs[var_idx]
        ...     else:
        ...         flat_in[i] = traced.in_imms[imm_idx]
        ...         imm_idx += 1
        >>> args, kwargs = traced.in_tree.unflatten(flat_in)

    Design Note:
        - Constants (non-Objects) are stored as immediates
        - Only Objects are traced into graph.inputs
        - Output constants are similarly separated from graph.outputs
        - This enables efficient serialization and execution
    """

    # Validate fn is callable
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn)}")

    # Get function name
    fn_name = getattr(fn, "__name__", "anonymous")

    # Step 1: Flatten input PyTree (args, kwargs)
    # We treat (args, kwargs) as a tuple to get a single tree structure
    in_flat, in_treedef = tree_flatten((args, kwargs))

    # Step 2: Separate input constants from variables, and record positions
    in_imms, in_var_pos, in_vars = _separate_vars_and_imms(in_flat)
    param_obj_ids = {id(obj) for obj in in_vars}

    # Step 3: Create a new tracer for the function body
    func_tracer = Tracer()

    # Step 4: Lift inputs and trace the function body
    # We MUST lift here, not rely on auto-lift during fn() execution, because:
    # - fn() might directly return inputs without calling primitives (e.g., return x, y)
    # - Auto-lift only happens when Objects flow into primitive.bind()
    # - We need to ensure all input Objects become TraceObjects in func_tracer
    def lift_if_object(x: Any) -> Any:
        return func_tracer.lift(x) if isinstance(x, Object) else x

    with func_tracer:
        args_traced, kwargs_traced = tree_map(lift_if_object, (args, kwargs))
        result = fn(*args_traced, **kwargs_traced)
        # Branches may return captured InterpObjects without touching primitives,
        # so force every Object in the result to live in func_tracer as well.
        result = tree_map(lift_if_object, result)

    # Step 5: Flatten outputs and separate TraceObjects from constants
    output_flat, output_treedef = tree_flatten(result)
    out_imms, out_var_pos, out_vars = _separate_vars_and_imms(output_flat)

    # Step 6: Finalize the graph with variable outputs only
    if out_vars:
        graph = func_tracer.finalize(out_vars)
    else:
        # No traced outputs (all constants)
        graph = func_tracer.graph
        graph.outputs = []

    captured_objects: list[Object] = []
    for obj_id, (obj, _) in func_tracer._freevars.items():
        if obj_id not in param_obj_ids:
            captured_objects.append(obj)

    # Step 7: Return TracedFunction
    return TracedFunction(
        name=fn_name,
        graph=graph,
        in_imms=in_imms,
        in_var_pos=in_var_pos,
        in_tree=in_treedef,
        out_imms=out_imms,
        out_var_pos=out_var_pos,
        out_tree=output_treedef,
        captured=captured_objects,
    )
