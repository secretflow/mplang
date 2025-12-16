# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from typing import TYPE_CHECKING, Any, cast

from jax.tree_util import PyTreeDef, tree_flatten, tree_map

from mplang.v2.edsl.context import Context
from mplang.v2.edsl.graph import Graph
from mplang.v2.edsl.graph import Value as GraphValue
from mplang.v2.edsl.object import Object
from mplang.v2.edsl.typing import BaseType

if TYPE_CHECKING:
    from mplang.v2.edsl.primitive import Primitive


class TraceObject(Object):
    """Trace-time object (during JIT tracing).

    Holds a Value in the Graph IR and a reference to the Tracer (Context).
    All operations delegate to primitives which record into Graph.

    Example:
        >>> from mplang.v2.edsl import trace
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

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset graph state so a tracer instance can be reused."""
        self.graph = Graph()
        # Cache for captured variables (closures), keyed by id(obj)
        # Does NOT include function parameters - those are created per-position
        self._captured_vars: dict[int, tuple[Object, GraphValue]] = {}
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
            # Detect flat style: first param is list-annotated "in_types"
            is_flat_style = len(params) >= 1 and params[0].name == "in_types"

            if is_flat_style:
                output_types = primitive._abstract_eval(input_types, **kwargs)
            else:
                output_types = primitive._abstract_eval(*input_types, **kwargs)

            # Normalize to list: single type or sequence → list
            if isinstance(output_types, BaseType):
                output_types = [output_types]
            else:
                output_types = list(output_types)

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

    def lift(self, obj: Any, *, is_param: bool = False) -> Any:
        """Lift an object to TraceObject.

        Converts objects to TraceObject for use in tracing:
        - Non-Object types: return as-is (int, float, np.ndarray, callables, etc.)
        - TraceObject (same context): return as-is (idempotent)
        - TraceObject (different context): create graph input
        - InterpObject: promote to TraceObject

        Args:
            obj: Value to lift (Object or non-Object constant)
            is_param: If True, create independent graph input (no caching).
                      If False, cache by id() for captures (same object → same input).

        Returns:
            TraceObject for Objects, or original value for non-Objects

        Note:
            - Parameters (is_param=True): Each position gets independent input,
              so `trace(fn, x, x)` creates two separate graph inputs.
            - Captures (is_param=False): Cached by id(), so the same captured
              object always maps to the same graph input.

        Subclass extension:
            Override _lift_type() to customize type transformation
            (e.g., unwrap MPType → value_type, TensorType → element_type).
        """
        # Early return for non-Object types (constants, callables, etc.)
        if not isinstance(obj, Object):
            return obj

        # Same-context TraceObject → return as-is (idempotent)
        if isinstance(obj, TraceObject) and obj._context is self:
            return obj

        # Parameters: always create fresh input (no caching)
        if is_param:
            return self._new_arg(self._lift_type(obj))

        # Captures: cache by id()
        obj_id = id(obj)
        if obj_id in self._captured_vars:
            _, graph_value = self._captured_vars[obj_id]
            return TraceObject(graph_value, self)

        lifted = self._new_arg(self._lift_type(obj))
        self._captured_vars[obj_id] = (obj, lifted._graph_value)
        return lifted

    def _lift_type(self, obj: Object) -> BaseType:
        """Get the graph input type for an object.

        Subclasses override this to customize type transformation:
        - _LocalMPTracer: unwrap MPType → value_type
        - _ElementwiseTracer: unwrap TensorType → element_type

        The base class preserves the object's type unchanged.

        Args:
            obj: Object being lifted to a graph input

        Returns:
            The type to use for the graph input
        """
        return cast(BaseType, obj.type)

    def _new_arg(self, arg_type: BaseType) -> TraceObject:
        """Create a new graph input for the given type.

        Internal method - prefer using lift() which handles caching logic.
        Use this for function parameters where each position should be independent.

        Args:
            arg_type: The type of the argument

        Returns:
            TraceObject wrapping a new graph input Value
        """
        name = f"%arg{self._arg_counter}"
        self._arg_counter += 1
        graph_value = self.graph.add_input(
            name=name,
            type=arg_type,
        )
        return TraceObject(graph_value, self)

    def finalize(self, result: Any) -> Graph:
        """Finalize the graph by setting outputs.

        This marks the traced result as the outputs of the graph,
        completing the graph construction. After this, the graph
        is ready for interpretation or transformation.

        Args:
            result: Traced result, PyTree containing TraceObjects

        Returns:
            The finalized graph (self.graph with outputs set)

        Example:
            >>> tracer = Tracer()
            >>> push_context(tracer)
            >>> result = do_something(x, y)
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

        return self.graph  # type: ignore[return-value]

    def run(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> TracedFunction:
        """Trace `fn` using this tracer instance.

        Parameter handling:
            Each parameter position gets an independent graph input via new_arg(),
            even if the same Python object is passed multiple times. This ensures
            correct semantics: `trace(fn, x, x)` creates two separate inputs.

        Capture handling:
            Variables captured from closures are cached by id() via lift(),
            so the same captured object always maps to the same graph input.
        """
        self.reset()
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn)}")

        fn_name = getattr(fn, "__name__", "anonymous")
        in_flat, in_treedef = tree_flatten((args, kwargs))
        in_imms, in_var_pos, in_vars = _separate_vars_and_imms(in_flat)

        with self:
            # Helper to lift params, allowing BaseType as placeholders
            def lift_param(obj: Any) -> Any:
                if isinstance(obj, Object):
                    return self.lift(obj, is_param=True)
                return obj

            # Lift parameters with is_param=True (each position gets independent input)
            args_traced, kwargs_traced = tree_map(lift_param, (args, kwargs))

            result = fn(*args_traced, **kwargs_traced)
            # Lift any Objects in result (captures use default is_param=False)
            result = tree_map(self.lift, result)

        output_flat, output_treedef = tree_flatten(result)
        out_imms, out_var_pos, out_vars = _separate_vars_and_imms(output_flat)

        if out_vars:
            graph = self.finalize(out_vars)
        else:
            graph = self.graph
            graph.outputs = []

        # Captured objects are those in _captured_vars (excludes parameters)
        captured_objects: list[Object] = [
            obj for obj, _ in self._captured_vars.values()
        ]

        return TracedFunction(
            name=fn_name,
            graph=graph,
            in_imms=in_imms,
            in_var_pos=in_var_pos,
            in_tree=in_treedef,
            out_imms=out_imms,
            out_var_pos=out_var_pos,
            out_tree=output_treedef,
            params=in_vars,  # Original parameter objects
            captured=captured_objects,
        )

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

    Graph Inputs Order Convention:
        graph.inputs = [*params_inputs, *captured_inputs]
        - First len(params) inputs correspond to function parameters
        - Remaining inputs correspond to captured variables (closures)

    Attributes:
        name: Function name (from fn.__name__)
        graph: The finalized Graph IR containing traced computations
        in_imms: Input immediates (constants) in flattened order
        in_var_pos: Positions of graph.inputs in the flattened input list
        in_tree: PyTreeDef to reconstruct (args, kwargs) from flattened inputs
        out_imms: Output immediates (constants) in flattened order
        out_var_pos: Positions of graph.outputs in the flattened output list
        out_tree: PyTreeDef to reconstruct result from flattened outputs
        params: Original parameter Objects (in order matching graph.inputs[:len(params)])
        captured: Captured Objects from closures (in order matching graph.inputs[len(params):])

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
        >>> # params = [x_obj, y_obj], captured = []
    """

    name: str
    graph: Graph
    in_imms: list[Any]
    in_var_pos: list[int]
    in_tree: PyTreeDef
    out_imms: list[Any]
    out_var_pos: list[int]
    out_tree: PyTreeDef
    params: list[Object]  # Original parameter objects
    captured: list[Object]  # Captured objects from closures

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

    def compiler_ir(self, verbose: bool = False) -> str:
        """Get human-readable IR representation of the traced function.

        This is useful for debugging, auditing, and understanding what
        operations were captured during tracing.

        Args:
            verbose: If True, include type annotations in the output

        Returns:
            String representation of the Graph IR

        Example:
            >>> traced = compile(lambda x, y: x + y, x_obj, y_obj)
            >>> print(traced.compiler_ir())
            %arg0 = input
            %arg1 = input
            %0 = add(%arg0, %arg1)
            return %0
        """
        return self.graph.to_string(verbose=verbose)

    def align_region_inputs(
        self, leading_count: int, capture_order: list[Object]
    ) -> None:
        """Align region graph inputs as [leading_values..., captures...] sequence.

        Reorders the graph inputs to have a standardized structure:
        - First `leading_count` inputs: explicit function parameters
        - Remaining inputs: captured variables in the specified order

        This is essential for multi-region primitives (e.g., uniform_cond, while_loop)
        where different regions need to share the same capture ordering.

        Args:
            leading_count: Number of explicit function parameters (non-captured)
            capture_order: Desired order of captured variables

        Example:
            >>> # Align two branches to have same capture order
            >>> all_captures = merge_captures(then_fn.captured, else_fn.captured)
            >>> then_fn.align_region_inputs(num_args, all_captures)
            >>> else_fn.align_region_inputs(num_args, all_captures)
        """
        assert len(self.graph.inputs) >= leading_count

        leading_inputs = self.graph.inputs[:leading_count]
        capture_inputs = self.graph.inputs[leading_count:]
        capture_map = (
            dict(zip(self.captured, capture_inputs, strict=True))
            if self.captured
            else {}
        )

        new_capture_inputs = []
        for capture_obj in capture_order:
            value = capture_map.get(capture_obj)
            if value is None:
                value = self.graph.add_input(
                    name=f"%capture{len(self.graph.inputs)}",
                    type=capture_obj.type,
                )
            new_capture_inputs.append(value)

        self.graph.inputs = leading_inputs + new_capture_inputs
        self.captured = list(capture_order)

    def prepare_inputs(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Flatten arguments and map them to graph inputs.

        Used by the runtime to prepare inputs for graph execution.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            List of values corresponding to graph.inputs (may include InterpObject).
            The caller is responsible for unwrapping InterpObject at execution boundary.
        """
        flat_args, _ = tree_flatten((args, kwargs))

        # Map to graph inputs
        # fn.in_var_pos contains indices in flat_args that correspond to graph inputs
        # Note: graph.inputs = [explicit_inputs...] + [captured_inputs...]
        explicit_inputs = [flat_args[i] for i in self.in_var_pos]
        all_inputs = explicit_inputs + list(self.captured)
        return all_inputs

    def reconstruct_outputs(self, execution_result: list[Any]) -> Any:
        """Reconstruct structured output from execution result.

        Used by the runtime to format the result of graph execution.

        Args:
            execution_result: List of results from interpreter.evaluate_graph().

        Returns:
            Structured output matching the original function's return signature.
        """
        # execution_result is always a list (now that evaluate_graph returns list)
        results = execution_result

        # Reconstruct
        total_len = len(self.out_imms) + len(self.out_var_pos)
        flat_out = [None] * total_len

        var_indices = set(self.out_var_pos)
        imm_iter = iter(self.out_imms)
        res_iter = iter(results)

        for i in range(total_len):
            if i in var_indices:
                flat_out[i] = next(res_iter)
            else:
                flat_out[i] = next(imm_iter)

        return self.out_tree.unflatten(flat_out)


def trace(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> TracedFunction:
    """Trace a Python function with the default Tracer."""

    return Tracer().run(fn, *args, **kwargs)
