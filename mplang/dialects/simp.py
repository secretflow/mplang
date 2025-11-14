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

from mplang.edsl.context import get_current_context
from mplang.edsl.object import Object
from mplang.edsl.primitive import Primitive
from mplang.edsl.tracer import TraceObject, Tracer
from mplang.edsl.typing import BaseType

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
    verify_uniform: bool = True,
) -> Any:
    """Implementation for uniform_cond (def_trace mode for complex control flow).

    Uses def_trace (not def_abstract_eval) because uniform_cond is a complex
    control flow primitive that needs full control over execution.

    Context behavior:
    - def_trace is always called in Tracer context
    - When called from Interpreter, the Interpreter.bind_primitive creates
      a Tracer context first, then calls primitive.bind (which invokes def_trace)
    - So this function always sees ctx as Tracer and builds Graph IR

    Args:
        pred: Boolean scalar value (Python bool, NumPy bool, or TraceObject).
             Must be uniform across all parties in multi-party context.
        then_fn: Either a callable (args -> T) to execute when pred is True,
                or a pre-computed value of type T to return.
        else_fn: Either a callable (args -> T) to execute when pred is False,
                or a pre-computed value of type T to return.
        *args: Arguments to pass to the selected branch function if it's callable.
        verify_uniform: Whether to verify predicate uniformity at runtime.

    Returns:
        Result from executing the selected branch (if callable) or the pre-computed value.
        Type is inferred from then_fn/else_fn return type or value type.

    Raises:
        TypeError: If pred is not a boolean scalar or branches have incompatible types.

    Design Note:
        This uses def_trace (not def_abstract_eval) because:
        - Complex operation requiring tracer fork (in trace mode)
        - Type inference depends on tracing branch functions
        - Needs full control over graph construction

        Simple primitives (add, mul) should use def_abstract_eval.
        Complex primitives (control flow, custom integrations) use def_trace.
    """
    ctx = get_current_context()

    # def_trace is always called in Tracer context
    # (Interpreter.bind_primitive creates Tracer before calling primitive.bind)
    assert isinstance(ctx, Tracer), f"Expected Tracer context, got {type(ctx)}"

    # Validate pred is TraceObject
    if not isinstance(pred, TraceObject):
        raise TypeError(f"predicate must be TraceObject, got {type(pred)}")

    # Both branches must be callable in trace mode
    if not callable(then_fn) or not callable(else_fn):
        raise TypeError("In trace mode, both branches must be callable functions")

    # Step 1: Fork tracers for both branches
    then_tracer = Tracer()
    else_tracer = Tracer()

    # Step 2: Trace both branches
    # Note: We don't need to manually lift args here.
    # When primitives are called inside then_fn/else_fn, Primitive.bind()
    # will automatically lift args to the current context (then_tracer/else_tracer)
    with then_tracer:
        then_result = then_fn(*args)
    then_graph = then_tracer.finalize(then_result)

    with else_tracer:
        else_result = else_fn(*args)
    else_graph = else_tracer.finalize(else_result)

    # Step 3: Validate branch outputs match
    if len(then_graph.outputs) != len(else_graph.outputs):
        raise TypeError(
            f"Branch output count mismatch: then={len(then_graph.outputs)} "
            f"vs else={len(else_graph.outputs)}"
        )

    for i, (then_out, else_out) in enumerate(
        zip(then_graph.outputs, else_graph.outputs, strict=True)
    ):
        if then_out.type != else_out.type:
            raise TypeError(
                f"Branch output type mismatch at index {i}: "
                f"{then_out.type} vs {else_out.type}"
            )

    # Step 4: Collect input arguments (TraceObjects from current context)
    input_objects = [arg for arg in args if isinstance(arg, TraceObject)]
    input_values = [obj._graph_value for obj in input_objects]

    # Step 5: Handle free variables
    # Merge free vars from both branches (union, preserving order)
    # Free vars are objects from outer context lifted as inputs in sub-tracers
    all_freevar_obj_ids = list(
        dict.fromkeys(
            list(then_tracer._freevars.keys()) + list(else_tracer._freevars.keys())
        )
    )

    # Map free variable object IDs back to original TraceObjects and their values
    # The _freevars dict maps id(original_obj) -> graph_value_in_sub_tracer
    # We need to find the original objects from the current context
    freevar_values = []
    for obj_id in all_freevar_obj_ids:
        # Find the original object - it should be in args
        original_obj = None
        for arg in args:
            if isinstance(arg, TraceObject) and id(arg) == obj_id:
                original_obj = arg
                break

        if original_obj is None:
            raise RuntimeError(
                f"Could not find original object for free var id {obj_id}"
            )

        freevar_values.append(original_obj._graph_value)

    # Step 6: Build cond operation with regions
    # Input order: [pred, regular_args, freevars]
    output_types = [v.type for v in then_graph.outputs]

    all_input_values = [pred._graph_value, *input_values, *freevar_values]

    result_values = ctx.graph.add_op(
        opcode="simp.uniform_cond",
        inputs=all_input_values,
        output_types=output_types,
        attrs={
            "verify_uniform": verify_uniform,
            "num_args": len(
                input_values
            ),  # Number of regular args (excludes pred and freevars)
        },
        regions=[then_graph, else_graph],
    )

    # Step 7: Return TraceObject(s)
    if not isinstance(result_values, list):
        result_values = [result_values]

    if len(result_values) == 1:
        return TraceObject(result_values[0], ctx)
    else:
        return [TraceObject(v, ctx) for v in result_values]


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
