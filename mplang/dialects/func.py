"""Func dialect: Function definition and call primitives for EDSL.

Provides function-related operations:
- make_graph: Trace a Python function into a Graph (not a Primitive - direct utility)
- call: Call a graph with arguments (Primitive for use in EDSL programs)

See individual function docstrings for detailed documentation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jax.tree_util import PyTreeDef, tree_flatten, tree_map

from mplang.edsl.graph import Graph
from mplang.edsl.object import Object
from mplang.edsl.primitive import Primitive
from mplang.edsl.tracer import Tracer
from mplang.edsl.typing import BaseType

# ---------------------------------------------------------------------------
# Function definition (utility, not a Primitive)
# ---------------------------------------------------------------------------


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


def _separate_traced_and_imms(
    flat_values: list[Any], tracer: Tracer
) -> tuple[list[Any], list[int], list[Any]]:
    """Separate a flattened list into traced values and immediates.

    Args:
        flat_values: Flattened list of values (mix of TraceObjects and constants)
        tracer: The tracer context to check TraceObjects against

    Returns:
        Tuple of (imms, var_pos, vars) where:
            - imms: List of immediate values (non-TraceObjects) in order
            - var_pos: List of positions where TraceObjects appear in flat_values
            - vars: List of TraceObject values in order
    """
    from mplang.edsl.tracer import TraceObject

    imms = []
    var_pos = []
    vars_list = []

    for i, val in enumerate(flat_values):
        if isinstance(val, TraceObject) and val._context is tracer:
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


def make_graph(
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
    in_imms, in_var_pos, _ = _separate_vars_and_imms(in_flat)

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

    # Step 5: Flatten outputs and separate TraceObjects from constants
    output_flat, output_treedef = tree_flatten(result)
    out_imms, out_var_pos, out_vars = _separate_traced_and_imms(
        output_flat, func_tracer
    )

    # Step 6: Finalize the graph with variable outputs only
    if out_vars:
        graph = func_tracer.finalize(out_vars)
    else:
        # No traced outputs (all constants)
        graph = func_tracer.graph
        graph.outputs = []

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
    )


# ---------------------------------------------------------------------------
# Function call (func.call)
# ---------------------------------------------------------------------------

call_p = Primitive("func.call")


@call_p.def_abstract_eval
def _call_ae(func_ref_t: BaseType, *args_t: BaseType) -> list[BaseType]:
    """Abstract evaluation for func.call.

    Args:
        func_ref_t: Type of the function reference (to be designed)
        *args_t: Types of arguments to pass to the function

    Returns:
        List of output types (inferred from function signature)

    Note: This is a scaffold. Full implementation needs:
    - Function reference type design
    - Function signature lookup mechanism
    - Type checking of arguments against signature
    """
    # TODO: Implement type inference based on function signature
    raise NotImplementedError(
        "func.call abstract_eval is a scaffold. Needs:\n"
        "1. Function reference type (FunctionType?)\n"
        "2. Function registry/lookup mechanism\n"
        "3. Signature matching and type inference"
    )


# No eager implementation yet - these are IR-level operations
@call_p.def_trace
def _call_trace(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("func.call execution is not implemented yet")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TracedFunction",
    "call_p",
    "make_graph",
]
