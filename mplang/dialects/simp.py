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

import numpy as np

from mplang.edsl.object import Object
from mplang.edsl.primitive import Primitive
from mplang.edsl.typing import BaseType

# ---------------------------------------------------------------------------
# Control flow (scaffold)
# ---------------------------------------------------------------------------

uniform_cond_p = Primitive("simp.uniform_cond")


@uniform_cond_p.def_trace
def _uniform_cond_trace(
    pred: Object,
    then_fn: Callable[..., Any],
    else_fn: Callable[..., Any],
    *args: Any,
    verify_uniform: bool = True,
) -> Any:
    """Implementation for uniform_cond (def_trace mode for complex control flow).

    Uses def_trace (not def_abstract_eval) because uniform_cond is a complex
    control flow primitive that needs full control over execution:
    - Eager mode: Direct branch execution
    - Trace mode: Fork tracer, build CondExpr (requires region-based ops)

    Current implementation: Eager mode only (Interpreter context).
    Future: Trace mode when EDSL Graph supports region-based operations.

    Args:
        pred: Boolean scalar value (Python bool or NumPy bool).
             Must be uniform across all parties in multi-party context.
        then_fn: Either a callable (args -> T) to execute when pred is True,
                or a pre-computed value of type T to return.
        else_fn: Either a callable (args -> T) to execute when pred is False,
                or a pre-computed value of type T to return.
        *args: Arguments to pass to the selected branch function if it's callable.
        verify_uniform: Whether to verify predicate uniformity at runtime.
                       (Currently no-op in eager mode; future: enforce in trace mode)

    Returns:
        Result from executing the selected branch (if callable) or the pre-computed value.
        Type is inferred from then_fn/else_fn return type or value type.

    Raises:
        TypeError: If pred is not a boolean scalar (bool or np.bool_).

    Design Note:
        This uses def_trace (not def_abstract_eval) because:
        - Complex operation requiring tracer fork (in future trace mode)
        - Type inference depends on tracing branch functions
        - Needs full control over graph construction

        Simple primitives (add, mul) should use def_abstract_eval.
        Complex primitives (control flow, custom integrations) use def_trace.
    """
    # Convert pred to boolean
    if isinstance(pred, (bool, np.bool_)):
        pred_val = bool(pred)
    else:
        raise TypeError(
            f"uniform_cond predicate must be boolean scalar, got {type(pred)}"
        )

    # Select the branch
    selected = then_fn if pred_val else else_fn

    # If selected is callable, execute it with args and return the result
    if callable(selected):
        return selected(*args)

    # Otherwise return the pre-computed value
    return selected


# Public API: Expose .bind method as a callable function
# Users can call: uniform_cond(pred, then_fn, else_fn, *args, verify_uniform=True)
uniform_cond = uniform_cond_p.bind


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
    "uniform_cond_p",
    "while_loop",
    "while_loop_p",
]
