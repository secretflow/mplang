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

from mplang.core.base import (
    Mask,
    MPContext,
    MPObject,
    Rank,
    ScalarType,
    Shape,
    TensorInfo,
    TensorLike,
)
from mplang.core.context_mgr import cur_ctx
from mplang.core.dtype import UINT64
from mplang.core.interp import InterpContext, InterpVar, apply
from mplang.core.pfunc import PFunction
from mplang.core.trace import TraceContext, TraceVar, trace
from mplang.expr.ast import (
    AccessExpr,
    CondExpr,
    ConstExpr,
    ConvExpr,
    EvalExpr,
    RandExpr,
    RankExpr,
    ShflExpr,
    ShflSExpr,
    WhileExpr,
)
from mplang.plib import basic
from mplang.utils.func_utils import var_demorph


def _switch_ctx(ctx: MPContext, obj: MPObject | Any) -> MPObject | Any:
    assert isinstance(ctx, MPContext), f"Expect MPContext, got {ctx}"

    if not isinstance(obj, MPObject):
        # If obj is not an MPObject, return it as is
        return obj

    if ctx is obj.ctx:
        # If the object is already in the correct context, return it directly
        return obj

    if obj.ctx.psize() != ctx.psize():
        # TODO(jint): strict check if source and target context are compatible.
        raise ValueError(f"{obj} world_size mismatch, expect {ctx.psize()}.")

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


def primitive(fn: Callable[P, R]) -> Callable[P, R]:
    """A decorator to make all primitive call in trace context."""

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        current_ctx = cur_ctx()
        if isinstance(current_ctx, TraceContext):
            # If we are in a tracer context, just call the function.
            # Note: switch_ctx will do the capture if needed.
            args, kwargs = tree_map(partial(_switch_ctx, current_ctx), (args, kwargs))
            return fn(*args, **kwargs)
        elif isinstance(current_ctx, InterpContext):
            trace_ctx = TraceContext(current_ctx.psize(), attrs=current_ctx.attrs())
            # TODO(jint): should we add trace_and_apply to improve the performance?
            traced_fn = trace(trace_ctx, fn, *args, **kwargs)
            # Return back to the original context.
            return cast(R, apply(current_ctx, traced_fn, *args, **kwargs))
        else:
            raise ValueError(f"Unsupported context type: {type(current_ctx)}")

    return wrapped


function = primitive


# ============================================================================
# Basic Primitive Operations
# ============================================================================


def _tracer() -> TraceContext:
    """Get the current context and ensure it's a Tracer."""
    ctx = cur_ctx()
    if not isinstance(ctx, TraceContext):
        raise ValueError(f"Expect tracer, got {ctx}")
    return ctx


@primitive
def psize() -> int:
    """Get the size of the current party world.

    Returns:
        int: The total number of parties in the current multi-party computation context.
    """
    ctx = _tracer()
    return ctx.psize()


@primitive
def pmask() -> Mask:
    """Get the current party mask in this computation context.

    Returns:
        Mask: The current party mask indicating which parties are active
              in the current computation context.
    """
    ctx = _tracer()
    return ctx.mask


@primitive
def prank() -> MPObject:
    """Multi-party get the rank (party identifier) of each party.

    This function returns a scalar tensor containing the rank (party identifier)
    for each party in the current party mask. Each party independently produces
    its own rank value, which serves as a unique identifier within the multi-party
    computation context.

    The rank values range from 0 to world_size-1, where world_size is the total
    number of parties in the computation. Each party's rank is private to that
    party and represents its position in the multi-party protocol.

    Returns:
        MPObject: A variable representing a scalar tensor with:
                  - dtype: UINT64
                  - shape: () (scalar)

    Note:
        Each party in the current party mask independently produces its own rank value.
    """
    ctx = _tracer()
    return TraceVar(ctx, RankExpr(ctx.mask))


@primitive
def prand(shape: Shape = ()) -> MPObject:
    """Multi-party generate a private random (uint64) tensor with the given shape.

    This function creates a private random tensor where each party independently
    generates its own local random values. Each party's random values are private
    and unknown to other parties. The output tensor contains 64-bit unsigned
    integers, with each party holding its own privately generated values.

    Args:
        shape: The shape of the random tensor to generate.
               Must be a tuple of positive integers. Defaults to () for scalar.

    Returns:
        MPObject: A variable representing the generated private random tensor with:
                  - dtype: UINT64
                  - shape: As specified by the shape parameter

    Note:
        Each party in the current party mask independently generates its own
        private random values. The randomness is local to each party and is
        not shared or revealed to other parties.
    """
    ctx = _tracer()
    typ = TensorInfo(UINT64, shape)
    return TraceVar(ctx, RandExpr(typ, ctx.mask))


@primitive
def constant(data: TensorLike | ScalarType) -> MPObject:
    """Create a constant tensor from tensor data or scalar value.

    This function creates a constant tensor that can be used in multi-party
    computations. The constant value is embedded directly into the computation
    graph and is available to all parties in the current party mask.

    Args:
        data: The constant data to embed. Can be:
              - A scalar value (int, float, bool)
              - A numpy array or other tensor-like object
              - Any object that can be converted to tensor

    Returns:
        MPObject: A variable representing the constant tensor with:
                  - dtype: Inferred from the input data
                  - shape: Inferred from the input data
                  - data: The embedded constant values

    Note:
        The constant data is embedded at graph construction time and is available
        to all parties during execution. Large constants may impact graph size.
    """
    import numpy as np

    # Convert data to TensorInfo + bytes for cacheable pconst
    if isinstance(data, ScalarType):
        tensor_info = TensorInfo.from_obj(data)
        # For scalars, convert to numpy array then to bytes
        np_data = np.array(data)
        data_bytes = np_data.tobytes()
    elif hasattr(data, "tobytes"):
        # For numpy arrays and other TensorLike objects with tobytes method
        tensor_info = TensorInfo.from_obj(data)
        data_bytes = data.tobytes()  # type: ignore
    else:
        # For other TensorLike objects, convert to numpy first
        np_data = np.array(data)
        tensor_info = TensorInfo.from_obj(np_data)
        data_bytes = np_data.tobytes()

    ctx = _tracer()
    return TraceVar(ctx, ConstExpr(tensor_info, data_bytes, ctx.mask))


@primitive
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
        # If no rmask is provided and no args, use full mask
        rmask = Mask.all(ctx.psize())
    if rmask is not None and not Mask(rmask).is_subset(ctx.mask):
        raise ValueError(
            f"Specified rmask {rmask} is not a subset of deduced pmask {ctx.mask}"
        )

    arg_exprs = [arg.expr for arg in cast(list[TraceVar], args)]
    fn_expr = EvalExpr(pfunc, arg_exprs, rmask)
    ret_exprs = [AccessExpr(fn_expr, idx) for idx in range(fn_expr.num_outputs)]

    return [TraceVar(ctx, res) for res in ret_exprs]


def set_mask(arg: MPObject, mask: Mask) -> MPObject:
    """Set the mask of an MPObject to a new value.

    This function allows changing the party mask of an existing MPObject variable.
    The behavior depends on whether the input MPObject has a dynamic or static pmask:

    **Case 1: Dynamic pmask (arg.pmask is None)**
    - The input MPObject has a runtime-determined pmask
    - The return value's pmask will be exactly the specified mask
    - No validation is performed at compile time

    **Case 2: Static pmask (arg.pmask is not None)**
    - If mask is a subset of arg.pmask: return_var.pmask == arg.pmask (unchanged)
    - If mask is NOT a subset of arg.pmask: raises ValueError at compile time

    Args:
        arg: The MPObject whose mask needs to be changed.
        mask: The target mask to apply. Must be a valid party mask.

    Returns:
        MPObject: A new variable with the specified mask behavior:
                 - For dynamic inputs: pmask = mask
                 - For static inputs (valid subset): pmask = arg.pmask

    Raises:
        ValueError: When arg has a static pmask and mask is not a subset of arg.pmask.
                   This validation occurs at compile time during graph construction.

    Examples:
        **Example 1: Dynamic pmask - mask assignment**
                     P0   P1   P2
                     --   --   --
            Input:   ?    ?    ?     (pmask=None, runtime-determined)
            mask:    [0,2]            (target mask)
        -----------------------------------------------------------
            Output:  x0   -    x2    (pmask=[0,2])

        **Example 2: Static pmask - valid subset**
                     P0   P1   P2
                     --   --   --
            Input:   x0   x1   x2    (pmask=[0,1,2])
            mask:    [0,2]            (subset of input pmask)
        -----------------------------------------------------------
            Output:  x0   -    x2    (pmask=[0,2])

        **Example 3: Static pmask - invalid subset (compile error)**
                     P0   P1   P2
                     --   --   --
            Input:   x0   -    x2     (pmask=[0,2])
            mask:    [1,2]            (NOT subset of [0,2])
        -----------------------------------------------------------
            Result:  ValueError at compile time

    Note:
        This function is typically used for constraining the execution scope
        of variables or for type casting between different pmask contexts.
        The underlying implementation uses JAX identity function with the
        specified execution mask.
    """
    pfunc, eval_args, out_tree = basic.identity(arg)
    results = peval(pfunc, eval_args, mask)
    return out_tree.unflatten(results)  # type: ignore[no-any-return]


@primitive
def cond(
    pred: MPObject,
    then_fn: Callable[..., MPObject],
    else_fn: Callable[..., MPObject],
    *args: Any,
) -> MPObject:
    """Multi-party conditional execution based on a predicate.

    This function implements conditional branching in multi-party computation,
    where the execution path (then_fn or else_fn) is determined by a predicate
    value. All parties evaluate the same predicate and execute the same branch,
    maintaining the SPMD (Single Program, Multiple Data) execution model.

    The predicate must be a scalar value, and both branches must have compatible
    input and output signatures. The function ensures type safety by validating
    that both branches accept the same input types and produce compatible outputs.

    Args:
        pred: A scalar variable representing the boolean predicate that
              determines which branch to execute. Must be a scalar tensor.
        then_fn: The multi-party function to execute when
                 the predicate is true. Must have compatible
                 input/output signatures with else_fn.
        else_fn: The multi-party function to execute when
                 the predicate is false. Must have compatible
                 input/output signatures with then_fn.
        *args: Input arguments to pass to the selected branch function.
               Must be compatible with both then_fn and else_fn input signatures.

    Returns:
        MPObject: The result of the executed branch. The output type is inferred
                 from both branches, with pmask set conservatively if the branches
                 differ in these properties.

    Raises:
        ValueError: If then_fn or else_fn don't have compatible input/output signatures,
                   or if function signatures do not match between branches.

    Examples:

        def add_one(x):
            return x + constant(1)

        def sub_one(x):
            return x - constant(1)

        pred = ...  # predicate from computation
        result = cond(pred, add_one, sub_one, x)

        **Example 1: Constant predicate - all parties execute same branch**
                     P0   P1   P2
                     --   --   --
            Input:   x0   -    x2
            Pred:    True True True   (constant shared by all parties)
        then_fn:     x+1  x+1  x+1    (add_one)
        else_fn:     -    -    -      (sub_one)
        -----------------------------------------------------------
            Output:  x0+1 -    x2+1   (all parties execute then_fn)

        **Example 2: Different predicate values - SPMD constraint violation**
                     P0   P1   P2
                     --   --   --
            Input:   x0   x1   x2
            Pred:    True False False  (different results per party)
        then_fn:     x+1  -    -       (add_one)
        else_fn:     -    x-1  x-1     (sub_one)
        -----------------------------------------------------------
            Output:  x0+1 x1-1 x2-1

        Note: This scenario violates SPMD consistency. In practice, the predicate
        must evaluate to the same value across all participating parties.

        **Example 3: Predicate with different pmask from input**
                     P0   P1   P2
                     --   --   --
            Input:   x0   -    x2     (pmask=[0,2])
            Pred:    True True False  (pmask=[0,1,2], superset of input pmask)
        then_fn:     x+1  -    -      (add_one)
        else_fn:     -    -    x-1    (sub_one)
        -----------------------------------------------------------
            Output:  x0+1 -    x2-1   (only parties with input data produce output)

    Note:
        Both branches must have identical input and output type signatures.
        The output pmask is set conservatively if branches have different pmasks.
        Both functions can capture variables from outer scopes.
    """
    assert all(isinstance(x, MPObject) for x in args), args

    cur_tracer = _tracer()

    # Step 1: Trace both branches in separate contexts
    then_tracer = cur_tracer.fork(pred.pmask)
    then_tfn = trace(then_tracer, then_fn, *args)

    else_tracer = cur_tracer.fork(pred.pmask)
    else_tfn = trace(else_tracer, else_fn, *args)

    if not then_tfn.is_signature_match(else_tfn, check_captures=False):
        raise ValueError(f"Function signatures do not match: {then_tfn} vs {else_tfn}")

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
    fn_expr = CondExpr(pred_expr, then_fn_expr, else_fn_expr, in_exprs)

    rets_expr = [AccessExpr(fn_expr, idx) for idx in range(fn_expr.num_outputs)]
    out_vars = [TraceVar(cur_tracer, res) for res in rets_expr]

    return var_demorph(out_vars, then_tfn.out_imms, then_tfn.out_struct)  # type: ignore[no-any-return]


@primitive
def while_loop(
    cond_fn: Callable[[MPObject], MPObject],
    body_fn: Callable[[MPObject], MPObject],
    init: MPObject,
) -> MPObject:
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
            sealed_sum = smpc.reveal(smpc.srun(lambda x: jnp.sum(x))(smpc.seal(x)))
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
        The output pmask is set conservatively if the body function changes the pmask
        during iteration. Both functions can capture variables from outer scopes.
        This implementation is similar to JAX while_loop but adapted for multi-party computation.
    """
    # TODO(jint): support multiple initial states
    cur_tracer = _tracer()

    # Step 1: Trace both condition and body functions in separate contexts
    cond_tracer = cur_tracer.fork(init.pmask)
    cond_tfn = trace(cond_tracer, cond_fn, init)

    body_tracer = cur_tracer.fork(init.pmask)
    body_tfn = trace(body_tracer, body_fn, init)

    # Step 2: Validate function signatures
    if len(body_tfn.out_vars) != 1:
        raise ValueError(
            f"Body function must return a single variable: got {len(body_tfn.out_vars)} outputs"
        )
    if body_tfn.out_vars[0].mptype != init.mptype:
        raise ValueError(
            f"Body function output type {body_tfn.out_vars[0].mptype} "
            f"does not match initial state type {init.mptype}"
        )

    if len(cond_tfn.out_vars) != 1:
        raise ValueError(
            f"Condition function must return a single boolean variable: "
            f"got {len(cond_tfn.out_vars)} outputs"
        )

    # Step 3: Handle variable captures from outer scopes
    # Collect all variables captured by either function
    # Similar to cond: union of captures from both functions
    all_captures = list((cond_tfn.capture_map | body_tfn.capture_map).keys())

    # Re-capture all outer variables into current scope for expression building
    capture_vars = [
        var if var.ctx is cur_tracer else cur_tracer.capture(var)
        for var in all_captures
    ]

    assert all(isinstance(var, TraceVar) for var in capture_vars), capture_vars

    # Step 4: Build the while loop expression
    init_expr = cast(TraceVar, init).expr
    capture_exprs = [cast(TraceVar, var).expr for var in capture_vars]

    # Generate function expressions with correct parameter mapping:
    # Parameter order: [state_param, capture_params...]
    cond_fn_expr = cond_tfn.make_expr(
        cond_tfn.in_names() + cond_tfn.capture_names(all_captures)
    )
    body_fn_expr = body_tfn.make_expr(
        body_tfn.in_names() + body_tfn.capture_names(all_captures)
    )

    # Create WhileExpr with init value and captured variables as arguments
    assert cond_fn_expr is not None and body_fn_expr is not None
    all_args = [init_expr, *capture_exprs]
    out_expr = WhileExpr(cond_fn_expr, body_fn_expr, all_args)
    assert out_expr.mptype == init.mptype

    return TraceVar(cur_tracer, out_expr)


@primitive
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
        RuntimeError: If src[index[i]] is None for any valid index[i] (i.e/,
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
    ctx = _tracer()

    src_expr = cast(TraceVar, src).expr
    index_expr = cast(TraceVar, index).expr

    shfl_expr = ShflExpr(src_expr, index_expr)
    return TraceVar(ctx, shfl_expr)


@primitive
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
    ctx = _tracer()

    src_expr = cast(TraceVar, src_val).expr

    shfl_s_expr = ShflSExpr(src_expr, pmask, src_ranks)
    return TraceVar(ctx, shfl_s_expr)


@primitive
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
    ctx = _tracer()

    var_exprs = [cast(TraceVar, var).expr for var in vars]

    conv_expr = ConvExpr(var_exprs)
    return TraceVar(ctx, conv_expr)
