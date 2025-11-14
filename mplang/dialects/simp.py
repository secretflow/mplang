"""SIMP dialect: SPMD multi-party primitives for EDSL.

Provides control flow and communication primitives:
- uniform_cond: Uniform conditional (eager mode)
- while_loop: While loop (eager mode)
- pshfl, pshfl_s, pconv: Communication ops (scaffolded)

Primitive definition guideline:
- Simple ops (add, mul) → use def_abstract_eval
- Complex ops (control flow, fork tracer) → use def_trace

See individual primitive docstrings for detailed documentation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jax.tree_util import tree_flatten

from mplang.edsl.context import get_current_context
from mplang.edsl.object import Object
from mplang.edsl.primitive import Primitive
from mplang.edsl.tracer import TraceObject, Tracer, trace
from mplang.edsl.typing import BaseType

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Whether to verify predicate uniformity at runtime in uniform_cond
# Set to False to disable runtime checks (useful for testing or when uniformity is guaranteed)
VERIFY_UNIFORM_DEFAULT = True

# ---------------------------------------------------------------------------
# Control flow (scaffold)
# ---------------------------------------------------------------------------

uniform_cond = Primitive("simp.uniform_cond")


@uniform_cond.def_trace
def _uniform_cond_trace(
    pred: Object,
    then_fn: Callable[..., Any],
    else_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Implementation for uniform_cond in trace mode.

    Uses def_trace (not def_abstract_eval) because uniform_cond is a complex
    control flow primitive that requires forking tracers for both branches.

    Args:
        pred: Boolean scalar TraceObject (must be uniform across all parties)
        then_fn: Callable accepting (*args, **kwargs) to execute when pred is True
        else_fn: Callable accepting (*args, **kwargs) to execute when pred is False
        *args: Positional arguments to pass to branch functions
        **kwargs: Keyword arguments to pass to branch functions

    Returns:
        Result from tracing both branches (TraceObject or tuple of TraceObjects)

    Raises:
        TypeError: If pred is not TraceObject, branches are not callable,
                   or branch outputs have mismatched types/counts

    Note:
        The verify_uniform flag is controlled by the global VERIFY_UNIFORM_DEFAULT config.
        To change it, set mplang.dialects.simp.VERIFY_UNIFORM_DEFAULT = False

    Example:
        >>> def then_fn(x, y):
        ...     return x + y
        >>> def else_fn(x, y):
        ...     return x - y
        >>> result = uniform_cond(pred, then_fn, else_fn, x, y)
    """
    cur_ctx = get_current_context()
    assert isinstance(cur_ctx, Tracer), f"Expected Tracer context, got {type(cur_ctx)}"

    # Validate inputs
    if not isinstance(pred, TraceObject):
        raise TypeError(f"predicate must be TraceObject, got {type(pred)}")
    if not callable(then_fn) or not callable(else_fn):
        raise TypeError("In trace mode, both branches must be callable functions")

    # Trace both branches using the new trace() function
    # trace() handles lifting arguments automatically
    then_traced = trace(then_fn, *args, **kwargs)
    else_traced = trace(else_fn, *args, **kwargs)

    # Validate branch output signatures match
    if not then_traced.is_output_signature_match(else_traced):
        raise TypeError(
            f"Branch output signature mismatch between then_fn and else_fn. "
            f"then={[v.type for v in then_traced.graph.outputs]}, "
            f"else={[v.type for v in else_traced.graph.outputs]}"
        )

    # Collect input variables from traced functions
    # in_var_pos tells us which positions in the flattened input are variables
    # We need to map these back to the current context's TraceObjects
    input_flat, _ = tree_flatten((args, kwargs))

    # Get variables (Objects) from inputs in their original order
    input_vars = [val for val in input_flat if isinstance(val, Object)]
    input_values = [
        obj._graph_value for obj in input_vars if isinstance(obj, TraceObject)
    ]

    # Build cond operation with regions
    output_types = [v.type for v in then_traced.graph.outputs]
    all_input_values = [pred._graph_value, *input_values]

    result_values = cur_ctx.graph.add_op(
        opcode="simp.uniform_cond",
        inputs=all_input_values,
        output_types=output_types,
        attrs={
            "verify_uniform": VERIFY_UNIFORM_DEFAULT,
            "num_args": len(input_values),  # Number of variable args (excludes pred)
        },
        regions=[then_traced.graph, else_traced.graph],
    )

    # Return TraceObject(s)
    if not isinstance(result_values, list):
        result_values = [result_values]

    if len(result_values) == 1:
        return TraceObject(result_values[0], cur_ctx)
    else:
        return [TraceObject(v, cur_ctx) for v in result_values]


# ---------------------------------------------------------------------------
# While loop (scaffold)
# ---------------------------------------------------------------------------

while_loop_p = Primitive("simp.while_loop")


@while_loop_p.def_trace
def _while_loop_trace(
    cond_fn: Callable[[Object], Any],
    body_fn: Callable[[Object], Any],
    init: Object,
) -> Any:
    """Implementation for while_loop (def_trace mode for complex control flow).

    Uses def_trace (not def_abstract_eval) because while_loop is a complex
    control flow primitive that needs full control over execution:
    - Eager mode: Direct Python while loop
    - Trace mode: Fork tracer for cond/body, build WhileExpr (requires region ops)

    Current implementation: Eager mode only (Interpreter context).
    Future: Trace mode when EDSL Graph supports region-based operations.

    Args:
        cond_fn: Function that takes loop state (type T) and returns boolean scalar.
                Must return True to continue looping, False to terminate.
        body_fn: Function that takes loop state (type T) and returns updated state (type T).
                Must preserve the type of the loop state across iterations.
        init: Initial loop state value of type T.

    Returns:
        Final loop state (type T) after cond_fn returns False.

    Design Note:
        This uses def_trace (not def_abstract_eval) because:
        - Complex operation requiring tracer fork (in future trace mode)
        - Type inference depends on tracing cond_fn/body_fn
        - Needs full control over graph construction

        Simple primitives (add, mul) should use def_abstract_eval.
        Complex primitives (control flow, custom integrations) use def_trace.
    """
    state = init
    while cond_fn(state):
        state = body_fn(state)
    return state


# Public API: Expose .bind method as a callable function
# Users can call: while_loop(cond_fn, body_fn, init)
while_loop = while_loop_p.bind


# ---------------------------------------------------------------------------
# Communication primitives (scaffold)
# ---------------------------------------------------------------------------

pshfl_p = Primitive("simp.pshfl")
pshfl_s_p = Primitive("simp.pshfl_s")
pconv_p = Primitive("simp.pconv")


@pshfl_p.def_abstract_eval
def _pshfl_ae(src_t: BaseType, index_t: BaseType) -> BaseType:
    # TODO: validate index_t is scalar; output type matches src_t (shape/dtype)
    return src_t


@pshfl_s_p.def_abstract_eval
def _pshfl_s_ae(src_t: BaseType, pmask: Any, src_ranks: Any) -> BaseType:
    # pmask/src_ranks are attributes until edsl.typing models them; passthrough type
    return src_t


@pconv_p.def_abstract_eval
def _pconv_ae(*vars_t: BaseType) -> BaseType:
    # TODO: ensure non-empty, identical dtype/shape, disjoint pmasks (when available)
    # For now, return the type of the first input
    if not vars_t:
        raise TypeError("pconv requires at least one input type")
    return vars_t[0]


# No eager impls for communication yet; they depend on runtime/party environment
# Use def_trace to make that explicit.
@pshfl_p.def_trace
def _pshfl_trace(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("simp.pshfl execution is not implemented yet")


@pshfl_s_p.def_trace
def _pshfl_s_trace(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("simp.pshfl_s execution is not implemented yet")


@pconv_p.def_trace
def _pconv_trace(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("simp.pconv execution is not implemented yet")


__all__ = [
    # Communication
    "pconv_p",
    "pshfl_p",
    "pshfl_s_p",
    # Control flow
    "uniform_cond",
    "while_loop",
    "while_loop_p",
]
