# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Primitive operations for the new expr-based implementation.

This module defines the fundamental primitive operations that form the building
blocks of multi-party computations. All primitives are designed to work in
TraceContext by default, automatically switching contexts as needed.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, wraps
from typing import Any, ParamSpec, TypeVar, cast

from jax.tree_util import tree_map

from mplang.v1.core.context_mgr import cur_ctx
from mplang.v1.core.dtypes import BOOL
from mplang.v1.core.expr.ast import (
    AccessExpr,
    CallExpr,
    CondExpr,
    ConvExpr,
    EvalExpr,
    ShflExpr,
    ShflSExpr,
    WhileExpr,
)
from mplang.v1.core.interp import InterpContext, InterpVar, apply
from mplang.v1.core.mask import Mask
from mplang.v1.core.mpobject import MPContext, MPObject
from mplang.v1.core.mptype import Rank
from mplang.v1.core.pfunc import PFunction
from mplang.v1.core.tracer import TraceContext, TraceVar, trace
from mplang.v1.utils.func_utils import var_demorph, var_morph


def _switch_ctx(ctx: MPContext, obj: MPObject | Any) -> MPObject | Any:
    assert isinstance(ctx, MPContext), f"Expect MPContext, got {ctx}"

    if not isinstance(obj, MPObject):
        # If obj is not an MPObject, return it as is
        return obj

    if ctx is obj.ctx:
        # If the object is already in the correct context, return it directly
        return obj

    if obj.ctx.world_size() != ctx.world_size():
        # TODO(jint): strict check if source and target context are compatible.
        raise ValueError(f"{obj} world_size mismatch, expect {ctx.world_size()}.")

    if isinstance(ctx, TraceContext):
        # Capture the object (as a variable) into current TraceContext
        return ctx.capture(obj)
    elif isinstance(ctx, InterpContext):
        if isinstance(obj, InterpVar):
            raise ValueError(f"Cannot import InterpVar {obj} from {obj.ctx} to {ctx}")
        elif isinstance(obj, TraceVar):
            assert isinstance(obj.ctx, TraceContext), obj
            # TODO: implement eval method in InterpContext
            raise NotImplementedError("InterpContext.eval not implemented yet")
        else:
            raise ValueError(f"Import from {obj.ctx} to {ctx} not supported")
    else:
        raise ValueError(f"Unsupported context type: {type(ctx)}")


# Define type variables for preserving function signatures
P = ParamSpec("P")
R = TypeVar("R")


def trace_before_apply(fn: Callable[P, R], make_call: bool) -> Callable[P, R]:
    """A decorator to make all primitive call in trace context."""

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        current_ctx = cur_ctx()
        if isinstance(current_ctx, TraceContext):
            # If we are already in a tracer context
            if make_call:
                # make a primitive call
                tracer = current_ctx
                tfn = trace(tracer.fork(), fn, *args, **kwargs)
                is_mpobj = lambda x: isinstance(x, MPObject)
                in_vars, in_imms, in_struct = var_morph((args, kwargs), is_mpobj)
                assert in_struct == tfn.in_struct and in_imms == tfn.in_imms
                arg_exprs = [arg.expr for arg in in_vars]
                # re-capture all captured variables into current context if needed.
                cap_exprs = [tracer.capture(var).expr for var in tfn.capture_map.keys()]
                caller_expr = CallExpr(
                    name=fn.__name__, fn=tfn.make_expr(), args=arg_exprs + cap_exprs
                )
                out_vars = [
                    TraceVar(tracer, AccessExpr(caller_expr, idx))
                    for idx in range(caller_expr.num_outputs)
                ]
                return cast(R, var_demorph(out_vars, tfn.out_imms, tfn.out_struct))
            else:
                # embed the function call in the current tracer context
                # Note: switch_ctx will do the capture if needed.
                args, kwargs = tree_map(
                    partial(_switch_ctx, current_ctx), (args, kwargs)
                )
                return fn(*args, **kwargs)
        elif isinstance(current_ctx, InterpContext):
            trace_ctx = TraceContext(current_ctx.cluster_spec, parent=current_ctx)
            # TODO(jint): should we add trace_and_apply to improve the performance?
            tfn = trace(trace_ctx, fn, *args, **kwargs)
            # Return back to the original context.
            return cast(R, apply(current_ctx, tfn, *args, **kwargs))
        else:
            raise ValueError(f"Unsupported context type: {type(current_ctx)}")

    return wrapped


def builtin_function(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator to trace a Python function as an opaque primitive call (`CallExpr`).

    When a function decorated with `@builtin_function` is called within a `TraceContext`, it is
    not inlined. Instead, it is traced separately in a forked context, and a `CallExpr`
    node is inserted into the main graph. This is useful for encapsulating complex
    operations or third-party library calls as single, opaque nodes.

    **Implementation Note:**
    A `CallExpr` represents a call to a single inline lambda (non-recursive, as we don't
    have Y-combinator support). This single lambda call can be treated as a "primitive call"
    by the printer/visualizer - hence the name "primitive". The function body is captured
    once during tracing and represented as an opaque callable unit in the expression graph,
    maintaining a clear boundary between the caller and callee contexts.

    Args:
        fn: The function to be traced as a primitive operation.

    Returns:
        A wrapped function that creates a `CallExpr` node when called in a trace context.

    Example:
        ```python
        @builtin_function
        def my_op(x: MPObject) -> MPObject:
            # Complex logic traced as a single CallExpr node
            return x + 1
        ```
    """
    return trace_before_apply(fn, make_call=True)


def function(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator to trace a Python function by inlining its body.

    When a function decorated with `@function` is called within a `TraceContext`, its
    underlying primitive operations are expanded and inserted directly into the caller's
    graph. This is the default tracing behavior and is suitable for most pure-Python
    multi-party functions.

    Args:
        fn: The function to be traced and inlined.

    Returns:
        A wrapped function that inlines its operations into the caller's trace context.

    Example:
        ```python
        @function
        def my_func(x: MPObject, y: MPObject) -> MPObject:
            # Operations are inlined into the caller's trace
            return x + y * constant(2)
        ```
    """
    return trace_before_apply(fn, make_call=False)


# ============================================================================
# Basic Primitive Operations
# ============================================================================


def _tracer() -> TraceContext:
    """Get the current context and ensure it's a Tracer."""
    ctx = cur_ctx()
    if not isinstance(ctx, TraceContext):
        raise ValueError(f"Expect tracer, got {ctx}")
    return ctx


def psize() -> int:
    """Get the size of the current party world.

    Returns:
        int: The total number of parties in the current multi-party computation context.
    """
    return cur_ctx().world_size()


def pmask() -> Mask:
    """Get the current party mask in this computation context.

    Returns:
        Mask: The current party mask indicating which parties are active
              in the current computation context.
    """
    return _tracer().mask


@function
def peval(
    pfunc: PFunction,
    args: list[MPObject],
    rmask: Mask | None = None,
) -> list[MPObject]:
    """Multi-party evaluate a function in a SPMD (Single Program, Multiple Data) way.

    This function evaluates a PFunction (primitive function) across multiple parties
    in a coordinated manner. All parties execute the same function logic but operate
    on their own local data portions according to their party masks.

    Args:
        pfunc: The function to be evaluated in multi-party computation.
               This should be a compiled primitive function that supports
               multi-party execution.
        args: Input arguments as a list of MPObject variables.
              Each argument represents data distributed across parties
              according to their respective party masks.
        rmask: Execution enforcement mask that forces the
            runtime to evaluate the function with the specified party mask.

            **Important**: This rmask is different from MPObject.pmask:
            - MPObject.pmask: Compile-time type information indicating data distribution
            - This rmask: Runtime execution constraint specifying which parties execute

            If None, the runtime automatically determines the execution mask based
            on the current context. If provided, the runtime will attempt to execute
            with this exact mask. Defaults to None.

    Returns:
        list[MPObject]: A list of output variables from the evaluation.

    Raises:
        ValueError: Raised at compile-time when all input arguments have known
            pmasks but they are incompatible with the required rmask constraint.
            This is a static validation error detected during graph construction.
        RuntimeError: Raised at runtime when the rmask constraint cannot be
            satisfied. This occurs when some input arguments have unknown pmasks
            (determined at runtime) and the actual runtime pmasks don't meet
            the rmask requirement.

    Note:
        The function body operates in SPMD fashion where all parties execute the
        same program logic but on their respective data partitions.
    """
    ctx = _tracer()

    if rmask is None and len(args) == 0:
        # Zero-arg call: default to current context mask (do not implicitly widen)
        rmask = ctx.mask
    if rmask is not None and not Mask(rmask).is_subset(ctx.mask):
        # Keep error wording for backward-compatibility with existing tests/docs
        raise ValueError(
            f"Specified rmask {rmask} is not a subset of deduced pmask {ctx.mask}"
        )

    arg_exprs = [arg.expr for arg in cast(list[TraceVar], args)]
    fn_expr = EvalExpr(pfunc, arg_exprs, rmask)
    ret_exprs = [AccessExpr(fn_expr, idx) for idx in range(fn_expr.num_outputs)]

    return [TraceVar(ctx, res) for res in ret_exprs]


@function
def uniform_cond(
    pred: MPObject,
    then_fn: Callable[..., Any],
    else_fn: Callable[..., Any],
    *args: Any,
    verify_uniform: bool = True,
) -> Any:
    """Global (uniform) multi-party conditional.

    Exactly one branch (``then_fn`` or ``else_fn``) is executed *globally* across
    all active parties. Use this primitive when:

    1. ``pred`` is a boolean scalar whose runtime value is identical for every enabled party.
    2. At least one branch contains multi-party primitives (``seal`` / ``reveal`` /
       ``srun_jax`` / ``pshfl`` / mask transformations) whose cost or side-effects you
       want to avoid if the branch is not taken.
    3. You require the semantic guarantee that the *non-selected* branch does **not**
       perform communication, allocate intermediate buffers, or leak timing/side-effects.

    DO NOT use this when:
    * Predicate differs per party (use party-local selection or ``jax.where``).
    * You only need elementwise / per-entry selection (use ``jax.where`` / ``peval(jax.where)``).
    * Predicate is still secret-shared and you cannot reveal it (future: oblivious branch).

    Choosing between primitives (decision guide):

    1. Use ``jax.where`` (elementwise select) WHEN:
       - You already have both candidate tensors computed (cheap or unavoidable), AND
       - You want per-element blending, OR
       - Predicate may differ per party / per element.

       Example::
           y = peval(jax.where, [mask, a, b])  # both a and b computed

    2. Use ``uniform_cond`` (this primitive) WHEN:
       - Exactly one expensive or MPC-effectful branch should run, AND
       - Predicate is (or must be) globally uniform, AND
       - You want to avoid executing the non-selected branch entirely.

       Example::
           def heavy_then(x):
               sealed = smpc.seal(x)
               return smpc.reveal(sealed) + constant(1)


           def light_else(x):
               return x - constant(1)


           pred = reveal(global_flag)  # uniform bool
           y = uniform_cond(pred, heavy_then, light_else, x)

    3. Use ``jax.lax.cond`` (inside peval) WHEN:
       - Both branches are purely local numeric compute (no MPC comms), AND
       - You accept both branches being traced & (possibly) device-compiled, OR
       - You operate fully in JAX world without multi-party side-effects.

       Example::
           # Branches are pure JAX functions
           y = peval(jax.lax.cond, [pred, fn_a, fn_b, x])

    Args:
        pred: Boolean scalar ``MPObject``; must have shape ``()`` and dtype bool. Intended to be
              *uniform* (same logical value) across parties. If ``verify_uniform`` is True,
              runtime will assert uniformity.
        then_fn: Multi-party function executed when ``pred`` is True.
        else_fn: Multi-party function executed when ``pred`` is False.
        *args: MPObject arguments passed to the selected branch.
        verify_uniform: Whether to perform a runtime uniformity assertion. Disable only if
              the caller can guarantee (by construction) uniformity; disabling removes a
              safety check and may lead to undefined behavior if predicate diverges.

    Returns:
        A PyTree of MPObjects whose structure and per-leaf MPType matches the outputs of both
        branches (branches must agree exactly on MPType including pmask).

    Raises:
        TypeError: If ``pred`` is not a bool scalar; or branch output types mismatch.
        ValueError: If ``verify_uniform=True`` and runtime detects non-uniform predicate.

    Security:
        ``pred`` must be public (revealed) – using a secret, non-revealed boolean would create
        a data-dependent control path (timing / communication pattern leak). Reveal first, or
        use an oblivious selection (``jax.where``) if you cannot reveal.

    Example (common):
        >>> pred = simp.reveal(secret_flag)  # bool scalar, now public + uniform
        >>> out = uniform_cond(pred, branch_a, branch_b, x, y)

    """
    assert all(isinstance(x, MPObject) for x in args), args

    cur_tracer = _tracer()

    # Predicate static shape/dtype check
    pred_ty = pred.mptype
    if len(pred_ty.shape) != 0:
        raise TypeError(
            f"uniform_cond predicate must be scalar, got shape {pred_ty.shape}"
        )
    # dtype naming depends on dtype system; assume name property or eq compare
    if pred_ty.dtype != BOOL:
        raise TypeError(f"uniform_cond predicate must be boolean, got {pred_ty.dtype}")

    # Static pmask rule:
    # If predicate has a static pmask (not None), it must equal the current trace
    # context mask. Otherwise some parties would execute a branch without a
    # defined predicate value (unsafe). To run on a subset either:
    #   1. Trace the entire uniform_cond under a subset TraceContext (ctx.fork(mask=...))
    #   2. Broadcast / lift predicate to full mask (e.g. pshfl_s)
    # Pred pmask None => dynamic: defer to runtime uniformity (if verify_uniform=True).
    pred_pmask = pred_ty.pmask
    if pred_pmask is not None and pred_pmask != cur_tracer.mask:
        raise ValueError(
            "uniform_cond predicate static pmask mismatch: predicate pmask="
            f"{pred_pmask} trace mask={cur_tracer.mask}. Trace under a subset "
            "context (ctx.fork(mask=...)) or broadcast predicate (pshfl_s) to all parties."
        )
    # Step 1: Trace both branches in separate contexts
    then_tracer = cur_tracer.fork()
    then_tfn = trace(then_tracer, then_fn, *args)

    else_tracer = cur_tracer.fork()
    else_tfn = trace(else_tracer, else_fn, *args)

    if not then_tfn.is_signature_match(else_tfn, check_captures=False):
        # Branch outputs (structure, MPType, shape) must match exactly; treat mismatch as a
        # type error per uniform_cond contract (docstring promises TypeError for output mismatch).
        raise TypeError(
            f"uniform_cond branch output/signature mismatch: {then_tfn} vs {else_tfn}"
        )

    # Enforce identical output MPTypes (including pmask). Then/else already have out_vars.
    if len(then_tfn.out_vars) != len(else_tfn.out_vars):
        raise TypeError(
            "uniform_cond branches must return same number of outputs: "
            f"{len(then_tfn.out_vars)} vs {len(else_tfn.out_vars)}"
        )
    for i, (tv, ev) in enumerate(
        zip(then_tfn.out_vars, else_tfn.out_vars, strict=True)
    ):
        if tv.mptype != ev.mptype:
            raise TypeError(
                "uniform_cond branch output MPType mismatch at index "
                f"{i}: {tv.mptype} vs {ev.mptype}"
            )

    # Step 2: Handle variable captures from outer scopes

    # Collect all variables captured by either branch function
    # Example: then_fn captures (a, b), else_fn captures (a, c)
    # Result: all_captures = [a, b, c] (union, order preserved)
    all_captures = list((then_tfn.capture_map | else_tfn.capture_map).keys())

    # Problem: Branch functions may capture variables from outer scopes, but
    # expr only permits parameter passing from current scope.
    #
    # Scope diagram:
    #   outer_scope    [var_a, var_b]
    #        |
    #   cur_tracer     [pred, x]  ← we are here
    #        |
    #   ┌────┴────┐
    #   then_fn   else_fn   ← both may capture var_a, var_b
    #                         but expr needs them in cur_tracer!
    #
    # Solution: Re-capture all outer variables into current scope
    # Before: var_a lives in outer_scope, branches reference it
    # After:  var_a is re-captured into cur_tracer, expr can use it
    capture_vars = [
        var if var.ctx is cur_tracer else cur_tracer.capture(var)
        for var in all_captures
    ]

    assert all(isinstance(var, TraceVar) for var in capture_vars), capture_vars
    capture_exprs = [cast(TraceVar, var).expr for var in capture_vars]

    # Step 3: Build the conditional expression
    pred_expr = cast(TraceVar, pred).expr
    arg_exprs = [arg.expr for arg in cast(list[TraceVar], args)]

    # Input order: [regular_args, captured_vars]
    in_exprs = arg_exprs + capture_exprs

    # Generate branch functions with correct parameter mapping:
    # Parameter list = [args_params, capture_params]
    then_fn_expr = then_tfn.make_expr(
        then_tfn.in_names() + then_tfn.capture_names(all_captures)
    )
    else_fn_expr = else_tfn.make_expr(
        else_tfn.in_names() + else_tfn.capture_names(all_captures)
    )

    # Step 4: Create final conditional and return values
    assert then_fn_expr is not None and else_fn_expr is not None
    fn_expr = CondExpr(
        pred_expr,
        then_fn_expr,
        else_fn_expr,
        in_exprs,
        verify_uniform=verify_uniform,
    )

    rets_expr = [AccessExpr(fn_expr, idx) for idx in range(fn_expr.num_outputs)]
    out_vars = [TraceVar(cur_tracer, res) for res in rets_expr]

    return var_demorph(out_vars, then_tfn.out_imms, then_tfn.out_struct)  # type: ignore[no-any-return]


@function
def while_loop(
    cond_fn: Callable[[Any], MPObject],
    body_fn: Callable[[Any], Any],
    init: Any,
) -> Any:
    """Multi-party while loop with condition and body functions.

    This function implements iterative computation in multi-party settings using
    a while loop construct. The loop continues executing as long as the condition
    function returns true, with all parties maintaining synchronization throughout
    the iteration process.

    The condition function must return a scalar boolean value, and the body function
    must have the same input and output signature to enable proper iteration. Both
    functions operate on the loop variable, which is updated in each iteration.

    Args:
        cond_fn: A multi-party function that evaluates
                 the loop condition. Must take the same
                 input type as body_fn and return a single
                 scalar boolean output.
        body_fn: A multi-party function that represents
                 the loop body. Must take the same input
                 type as cond_fn and return a single output
                 with the same type as its input (for state update).
        init: The initial value for the loop variable. This value is passed
              to both cond_fn and body_fn in the first iteration.

    Returns:
        MPObject: The final value of the loop variable after the while loop terminates.
                 The output type is inferred from the body function and initial value,
                 with conservative pmask if they change during iteration.

    Raises:
        ValueError: If cond_fn or body_fn don't have exactly one output,
                   if cond_fn output is not scalar, or if input signatures
                   are incompatible, or if body function output type doesn't
                   match initial state type.

    Examples:
        **Scenario 1 – Local (non-synchronized) predicate**

        Each party decides *independently* when to leave the loop.

        cond_fn: ``lambda x: x < 10``
        body_fn: ``lambda x: x + constant(1)``
        init: party-local values ``[0, 5]``

            Iterations              P0   P1
            --------------------------------
            start                   0    5
            after 1st iter          1    6
            after 5th iter          5   10   ← P1 is done
            after 10th iter        10   10   ← P0 is done

        The parties stop at different iterations yet converge to the same final
        value ``[10, 10]``.  Such patterns are usually implemented more
        efficiently via ``peval(jax.while_loop, …)``.

        **Scenario 2 – Globally synchronized predicate**

        All parties evaluate *exactly* the same boolean each round (e.g. via a
        secret-shared reduction).

        cond_fn::
            sealed_sum = smpc.reveal(smpc.srun_jax(lambda x: jnp.sum(x), smpc.seal(x)))
            return sealed_sum < constant(10)

        body_fn::
            return x + prank()  # every party adds its own rank

        Iterations (rank 0 & rank 1 example):

            Iteration        P0 (rank 0)   P1 (rank 1)   sealed_sum   predicate
            -------------------------------------------------------------------
            start                0              5             5        True
            after 1st iter       0              6             6        True
            after 2nd iter       0              7             7        True
            after 3rd iter       0              8             8        True
            after 4th iter       0              9             9        True
            after 5th iter       0             10            10        False ← loop exits *simultaneously*

        Because the predicate is identical for every party at every step, they
        enter and exit the loop together.  Supporting such globally
        synchronized control flow is the primary reason this primitive exists
        (plain ``jax.while_loop`` cannot express it).

    Note:
        Control-flow execution domain (who runs cond/body) follows the outer context's
        mask; we do not shrink the tracer at trace time based on state pmasks. Value
        visibility and real participation are enforced per-op by argument pmask
        intersection (and optional rmask). The loop state MPType (including pmask)
        must remain identical across iterations. Both functions can capture variables
        from outer scopes. This implementation is similar to JAX while_loop but
        adapted for multi-party computation.
    """
    cur_tracer = _tracer()

    # Flatten init into loop-carried MPObject leaves, disallow non-MPObject leaves for now
    is_mpobj = lambda x: isinstance(x, MPObject)
    init_vars, init_imms, _init_struct = var_morph(init, is_mpobj)

    if len(init_vars) == 0:
        raise ValueError("while_loop requires at least one MPObject in init state")
    if len(init_imms) != 0:
        raise TypeError(
            "while_loop init must be a PyTree of MPObjects (no Python/immediate leaves)"
        )

    cond_tracer = cur_tracer.fork()
    cond_tfn = trace(cond_tracer, cond_fn, init)

    body_tracer = cur_tracer.fork()
    body_tfn = trace(body_tracer, body_fn, init)

    # Validate cond returns single value
    if len(cond_tfn.out_vars) != 1:
        raise ValueError(
            f"Condition function must return a single boolean variable: got {len(cond_tfn.out_vars)} outputs"
        )
    cond_out_var = cond_tfn.out_vars[0]
    if len(cond_out_var.mptype.shape) != 0:
        raise TypeError(
            f"Condition function must return a scalar, but got shape {cond_out_var.mptype.shape}"
        )
    # Enforce boolean dtype for condition
    if cond_out_var.mptype.dtype != BOOL:
        raise TypeError(
            f"Condition function must return a boolean scalar, got dtype {cond_out_var.mptype.dtype}"
        )

    # Static pmask rule:
    # If the predicate's pmask is statically known it must match the trace context
    # mask. Otherwise some parties in this context would lack a boolean to drive
    # control flow (previously could lead to hang via None). To restrict to a subset:
    #   1. Trace the entire while_loop under a subset context (ctx.fork(mask=submask)), or
    #   2. Broadcast predicate to full mask (e.g. pshfl_s) before while_loop.
    # Dynamic predicates (pmask=None) are allowed; runtime guard (evaluator) raises
    # if any participating party observes None.
    pred_pmask = cond_out_var.mptype.pmask
    if pred_pmask is not None and pred_pmask != cur_tracer.mask:
        raise ValueError(
            "while_loop predicate static pmask mismatch: predicate pmask="
            f"{pred_pmask} trace mask={cur_tracer.mask}. Trace under subset context "
            "or broadcast predicate to all parties."
        )

    # Validate body returns same number of leaves and same dtype/shape per leaf
    if len(body_tfn.out_vars) != len(cond_tfn.in_vars):
        raise ValueError(
            "Body function must return the same number of MPObject leaves as the init state"
        )
    for i, (out_v, in_v) in enumerate(
        zip(body_tfn.out_vars, cond_tfn.in_vars, strict=True)
    ):
        if out_v.mptype != in_v.mptype:
            raise TypeError(
                f"Body output leaf {i} type mismatch: {out_v.mptype} vs {in_v.mptype}"
            )

    # Handle variable captures from outer scopes (union of both functions)
    all_captures = list((cond_tfn.capture_map | body_tfn.capture_map).keys())
    capture_vars = [
        var if var.ctx is cur_tracer else cur_tracer.capture(var)
        for var in all_captures
    ]
    assert all(isinstance(var, TraceVar) for var in capture_vars), capture_vars

    # Build WhileExpr with all state leaves followed by captures
    state_exprs = [cast(TraceVar, v).expr for v in init_vars]
    capture_exprs = [cast(TraceVar, var).expr for var in capture_vars]

    cond_fn_expr = cond_tfn.make_expr(
        cond_tfn.in_names() + cond_tfn.capture_names(all_captures)
    )
    body_fn_expr = body_tfn.make_expr(
        body_tfn.in_names() + body_tfn.capture_names(all_captures)
    )

    assert cond_fn_expr is not None and body_fn_expr is not None
    all_args = [*state_exprs, *capture_exprs]
    out_expr = WhileExpr(cond_fn_expr, body_fn_expr, all_args)

    # Materialize outputs and reconstruct the original PyTree of init (args part)
    rets_expr = [AccessExpr(out_expr, idx) for idx in range(out_expr.num_outputs)]
    out_vars = [TraceVar(cur_tracer, res) for res in rets_expr]

    # Reconstruct the Python return using the body function's output structure
    # This preserves the exact PyTree the body returns (matching JAX semantics).
    return var_demorph(out_vars, body_tfn.out_imms, body_tfn.out_struct)


@function
def pshfl(src: MPObject, index: MPObject) -> MPObject:
    """Shuffle the input tensor to the specified index (dynamic version).

    This operation redistributes data from the source tensor to the target index
    based on the provided index tensor. Each output party receives data from the
    corresponding index in the source tensor.

    Semantics:
    - If index[i] is None (runtime pmask of the i'th party is None), then the
      i'th party will receive None as the result.
    - If src[index[i]] is None (cannot source the variable from the index[i]'th
      party because that party doesn't hold the data), the runtime will raise
      an exception.
    - The operation requires that for each valid index[i], the corresponding
      party index[i] must actually hold the source data in src.

    Args:
        src: The input tensor to be shuffled. Must be held by the parties
             that will be referenced by the index values.
        index: The index tensor indicating which source parties to fetch
               data from. Must be a scalar tensor. Each party uses its
               local index value to determine which source party to
               fetch data from.

    Returns:
        MPObject: The shuffled tensor with data redistributed according to the index.
                 Parties with None index will receive None. The output pmask is
                 inherited from index.pmask.

    Raises:
        ValueError: If the index tensor is not a scalar.
        RuntimeError: If src[index[i]] is None for any valid index[i] (i.e.,
                     trying to fetch from a party that doesn't hold the data).

    Examples:
        `index` is a distributed tensor where each party holds the rank of the
        party it wants to pull data from.

        **Example 1: Basic dynamic shuffle**
                     P0   P1   P2
                     --   --   --
            Input:   x0   -    x2
            Index:   -    0    -   (P1's index is 0, fetches from P0)
        -----------------------------------------------------------
            Output:  -    x0   -

        **Example 2: Cross shuffle**
                     P0   P1   P2
                     --   --   --
            Input:   x0   x1   x2
            Index:   2    0    1   (P0←P2, P1←P0, P2←P1)
        -----------------------------------------------------------
            Output:  x2   x0   x1

        **Example 3: Error case - invalid source**
                     P0   P1   P2
                     --   --   --
            Input:   x0   -    -   (only P0 has data)
            Index:   -    1    -   (P1 tries to fetch from P1, which has no data)
        -----------------------------------------------------------
            Result:  RuntimeError
    """
    src_expr = cast(TraceVar, src).expr
    index_expr = cast(TraceVar, index).expr

    shfl_expr = ShflExpr(src_expr, index_expr)
    return TraceVar(_tracer(), shfl_expr)


@function
def pshfl_s(src_val: MPObject, pmask: Mask, src_ranks: list[Rank]) -> MPObject:
    """Shuffle the input tensor to the specified rank, static version.

    This operation redistributes data from source ranks to target ranks based on
    the specified mapping. Each output party receives data from its corresponding
    source rank.

    Args:
        src_val: The input tensor to be shuffled.
        pmask: The mask indicating which parties will hold the output.
               Only parties with non-zero bits in pmask will receive output.
        src_ranks: List of source ranks. The i-th output party
                   (i-th non-zero bit in pmask) receives data from
                   src_ranks[i].

    Returns:
        MPObject: The shuffled tensor with data redistributed according to the
                 src_ranks mapping.

    Raises:
        ValueError: If any rank in src_ranks is not present in src_val.pmask,
                   or if src_ranks length doesn't match the number of bits in pmask.

    Examples:
        `pmask` and `src_ranks` define the shuffle. `pmask` selects the parties
        that will produce an output. `src_ranks` provides the source rank for
        each of these active parties. The "Logical Index" below illustrates
        the source for each party.

        **Example 1: Basic shuffle from P1 to P0**
                     P0   P1   P2
                     --   --   --
            Input:   -    x1   -
    Logical Index:   1    -    -    ; pmask=[0], src_ranks=[1]
        -----------------------------------------------------------
            Output:  x1   -    -

        **Example 2: Multiple party shuffle**
                     P0   P1   P2   P3
                     --   --   --   --
            Input:   x0   x1   -    x3
    Logical Index:   -    0    -    3    ; pmask=[1,3], src_ranks=[0,3]
        -----------------------------------------------------------
            Output:  -    x0   -    x3

        **Example 3: Cross shuffle**
                     P0   P1   P2
                     --   --   --
            Input:   x0   x1   x2
    Logical Index:   2    0    1    ; pmask=[0,1,2], src_ranks=[2,0,1]
        -----------------------------------------------------------
            Output:  x2   x0   x1
    """
    src_expr = cast(TraceVar, src_val).expr
    shfl_s_expr = ShflSExpr(src_expr, pmask, src_ranks)
    return TraceVar(_tracer(), shfl_s_expr)


@function
def pconv(vars: list[MPObject]) -> MPObject:
    """Combine multiple variables that share the same dtype and shape into one.

    This function combines multiple variables that share the same dtype and shape
    into one. The input variables are assumed to have non-intersecting pmasks (actual holders),
    meaning each variable is held by different parties.

    If the pmasks intersect, the compiler or runtime will raise an error.

    Args:
        vars: A list of MPObject variables with identical dtype
              and shape but disjoint pmasks.

    Returns:
        MPObject: A single variable that represents the convergence of all input
                 variables.

    Raises:
        ValueError: If vars is empty or if the pmasks of input variables intersect,
                   indicating conflicting ownership of the same data partitions.
        TypeError: If the input variables don't have identical types (dtype and shape).

    Examples:
        **Example 1 – merge two disjoint variables**

                     P0   P1   P2
                     --   --   --
            x0:      x0   -    -
            x1:      -    x1   -
         ---------------------------------
            Output:  x0   x1   -

        **Example 2 – merge three parties**

                     P0   P1   P2
                     --   --   --
            a:       a0   -    -
            b:       -    b1   -
            c:       -    -    c2
         ---------------------------------
            Output:  a0   b1   c2

        **Example 3 – error (overlapping pmask)**

                     P0   P1
                     --   --
            u:       u0   -
            v:       v0   -      ← overlap on P0
         ---------------------------------
            pconv([u, v])   # raises ValueError

    Note:
        This operation is used to combine multiple variables into a single object,
        typically for unifying data held by different parties. The resulting variable
        has a pmask that is the union of all input pmasks.
    """
    var_exprs = [cast(TraceVar, var).expr for var in vars]
    conv_expr = ConvExpr(var_exprs)
    return TraceVar(_tracer(), conv_expr)
