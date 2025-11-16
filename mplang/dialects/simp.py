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

import mplang.edsl as el
import mplang.edsl.typing as elt

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Whether to verify predicate uniformity at runtime in uniform_cond
# Set to False to disable runtime checks (useful for testing or when uniformity is guaranteed)
VERIFY_UNIFORM_DEFAULT = True

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_trace_object(ctx: el.Tracer, obj: el.TraceObject) -> el.TraceObject:
    """Ensure a TraceObject belongs to the current tracer context."""
    if obj._tracer is ctx:
        return obj
    lifted = ctx.lift(obj)
    # Internal invariant: lift() must return TraceObject for Object inputs
    assert isinstance(lifted, el.TraceObject), (
        "TraceContext.lift must return TraceObject for Objects"
    )
    return lifted


def _recapture_object(ctx: el.Tracer, obj: el.Object) -> el.TraceObject:
    """Capture any Object into the current tracer as a TraceObject.

    This is a simple wrapper around lift() with an assertion.
    The lift() method already handles both TraceObject (same/different context)
    and InterpObject cases correctly.
    """
    lifted = ctx.lift(obj)
    # Internal invariant: lift() must return TraceObject for Object inputs
    assert isinstance(lifted, el.TraceObject), (
        "Tracer.lift must return TraceObject for Objects"
    )
    return lifted


def _merge_captures(*capture_lists: list[el.Object]) -> list[el.Object]:
    """Merge capture lists while preserving first-seen order."""
    merged: list[el.Object] = []
    seen_ids: set[int] = set()
    for capture_list in capture_lists:
        for obj in capture_list:
            obj_id = id(obj)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            merged.append(obj)
    return merged


def _align_region_inputs(
    traced_fn: el.TracedFunction, leading_count: int, capture_order: list[el.Object]
) -> None:
    """Align region graph inputs as [leading_values..., captures...] sequence."""

    if len(traced_fn.graph.inputs) < leading_count:
        raise TypeError(
            "Region inputs shorter than required leading values; expected "
            f"{leading_count}, got {len(traced_fn.graph.inputs)}"
        )

    leading_inputs = traced_fn.graph.inputs[:leading_count]
    capture_inputs = traced_fn.graph.inputs[leading_count:]
    if traced_fn.captured:
        capture_map = dict(zip(traced_fn.captured, capture_inputs, strict=True))
    else:
        capture_map = {}

    new_capture_inputs = []
    for capture_obj in capture_order:
        value = capture_map.get(capture_obj)
        if value is None:
            value = traced_fn.graph.add_input(
                name=f"%capture{len(traced_fn.graph.inputs)}",
                type=capture_obj.type,
            )
        new_capture_inputs.append(value)

    traced_fn.graph.inputs = leading_inputs + new_capture_inputs
    traced_fn.captured = list(capture_order)


def _normalize_parties(value: tuple[int, ...] | None) -> tuple[int, ...] | None:
    """Validate and normalize party tuple (sorted, deduplicated).

    Args:
        value: Party tuple or None. Should be consistent with MPType.parties format.

    Returns:
        Normalized sorted tuple or None.

    Raises:
        TypeError: If value contains non-integer elements.
        ValueError: If any rank is negative.
    """
    if value is None:
        return None

    # Validate all elements are non-negative integers
    for rank in value:
        if not isinstance(rank, int):
            raise TypeError(
                f"parties tuple must contain integers, got {rank!r} of type {type(rank)}"
            )
        if rank < 0:
            raise ValueError(f"party rank must be non-negative, got {rank}")

    # Return sorted deduplicated tuple (consistent with MPType.parties)
    return tuple(sorted(set(value)))


def _parties_from_type(base_type: elt.BaseType) -> tuple[int, ...] | None:
    """Extract party tuple from MP typed values."""
    if isinstance(base_type, elt.MPType):
        return tuple(base_type.parties)
    return None


def _deduce_parties(types: list[elt.BaseType]) -> tuple[int, ...] | None:
    """Deduce common parties by intersecting all known party sets."""
    masks: list[tuple[int, ...] | None] = [_parties_from_type(tp) for tp in types]
    if not masks or any(m is None for m in masks):
        return None
    current = set(masks[0])  # at least one element
    for parties in masks[1:]:
        assert parties is not None  # Type narrowing: guarded by check above
        current &= set(parties)
    return tuple(sorted(current))


def _wrap_with_mp(
    base_type: elt.BaseType, parties: tuple[int, ...] | None
) -> elt.BaseType:
    """Wrap base_type with MP typing when a party mask is available."""
    if parties is None or isinstance(base_type, elt.MPType):
        return base_type
    # MPType.__class_getitem__ expects (value_type, parties)
    return elt.MP[base_type, parties]


class _LocalMPTracer(el.Tracer):
    """Tracer for single-party regions executed under MP context."""

    def _lift(self, obj: el.Object) -> el.TraceObject:
        """Override _lift to unwrap MP-typed Objects to their value types.

        This enables single-party regions to work with the underlying value types
        while enforcing that all inputs are MP-typed in the outer context.

        Args:
            obj: Object to lift (must be MP-typed)

        Returns:
            TraceObject with value_type (unwrapped from MPType)

        Raises:
            TypeError: If obj is not MP-typed
        """
        obj_type = obj.type
        if not isinstance(obj_type, elt.MPType):
            raise TypeError(
                f"simp.peval local regions expect MP-typed values, got type {obj_type}"
            )

        # Call base _lift to create graph input
        lifted = super()._lift(obj)

        # Unwrap MPType → value_type
        lifted._graph_value.type = obj_type.value_type
        return lifted


# ---------------------------------------------------------------------------
# Control flow (scaffold)
# ---------------------------------------------------------------------------


uniform_cond_p = el.Primitive("simp.uniform_cond")


@uniform_cond_p.def_trace
def _uniform_cond_trace(
    pred: el.Object,
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
    cur_ctx = el.get_current_context()
    if not isinstance(cur_ctx, el.Tracer):
        raise TypeError(
            f"uniform_cond must be called within a Tracer context, got {type(cur_ctx).__name__}"
        )

    # ------------------------------------------------------------------
    # Validate predicate / branch signatures
    # ------------------------------------------------------------------
    if not isinstance(pred, el.TraceObject):
        raise TypeError(f"predicate must be TraceObject, got {type(pred)}")
    pred_shape = getattr(pred.type, "shape", None)
    if pred_shape is not None and pred_shape != ():
        raise TypeError(f"uniform_cond predicate must be scalar, got type {pred.type}")
    if not callable(then_fn) or not callable(else_fn):
        raise TypeError("In trace mode, both branches must be callable functions")

    pred = _ensure_trace_object(cur_ctx, pred)

    # Trace both branches (trace() handles lifting/capture inside each branch)
    then_traced = el.trace(then_fn, *args, **kwargs)
    else_traced = el.trace(else_fn, *args, **kwargs)

    # Validate branch output signatures match exactly
    if not then_traced.is_output_signature_match(else_traced):
        then_types = [v.type for v in then_traced.graph.outputs]
        else_types = [v.type for v in else_traced.graph.outputs]
        raise TypeError(
            "uniform_cond branch output signature mismatch: "
            f"then={then_types}, else={else_types}"
        )

    # ------------------------------------------------------------------
    # Collect argument TraceObjects (positional + keyword) for cond inputs
    # ------------------------------------------------------------------
    flat_inputs, _ = tree_flatten((args, kwargs))
    arg_trace_objs: list[el.TraceObject] = [
        _ensure_trace_object(cur_ctx, val)
        for val in flat_inputs
        if isinstance(val, el.TraceObject)
    ]
    arg_values = [obj._graph_value for obj in arg_trace_objs]
    num_arg_vars = len(arg_trace_objs)

    # ------------------------------------------------------------------
    # Align captures from both branches and recapture into current context
    # ------------------------------------------------------------------
    all_captures = _merge_captures(then_traced.captured, else_traced.captured)

    _align_region_inputs(then_traced, num_arg_vars, all_captures)
    _align_region_inputs(else_traced, num_arg_vars, all_captures)

    # Recapture each capture object into the current cond tracer context
    capture_trace_objs = [_recapture_object(cur_ctx, obj) for obj in all_captures]
    capture_values = [obj._graph_value for obj in capture_trace_objs]

    # ------------------------------------------------------------------
    # Build cond operation with aligned inputs / captures
    # ------------------------------------------------------------------
    output_types = [v.type for v in then_traced.graph.outputs]
    cond_inputs = [pred._graph_value, *arg_values, *capture_values]

    result_values = cur_ctx.graph.add_op(
        opcode="simp.uniform_cond",
        inputs=cond_inputs,
        output_types=output_types,
        attrs={"verify_uniform": VERIFY_UNIFORM_DEFAULT},
        regions=[then_traced.graph, else_traced.graph],
    )

    return cur_ctx.reconstruct_outputs(
        then_traced.out_var_pos,
        then_traced.out_imms,
        then_traced.out_tree,
        result_values,
    )


def uniform_cond(
    pred: el.Object,
    then_fn: Callable[..., Any],
    else_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Uniform conditional that executes only the selected branch at runtime.

    Args:
        pred: Boolean scalar TraceObject that is uniform across parties.
        then_fn: Callable evaluated when `pred` is True.
        else_fn: Callable evaluated when `pred` is False.
        *args: Additional positional arguments forwarded to both branches.
        **kwargs: Additional keyword arguments forwarded to both branches.

    Returns:
        The PyTree produced by the selected branch.

    Raises:
        TypeError: If predicate/branches are invalid or branch outputs mismatch.
    """

    return uniform_cond_p.bind(pred, then_fn, else_fn, *args, **kwargs)


# ---------------------------------------------------------------------------
# While loop (scaffold)
# ---------------------------------------------------------------------------

while_loop_p = el.Primitive("simp.while_loop")


@while_loop_p.def_trace
def _while_loop_trace(
    cond_fn: Callable[[el.Object], Any],
    body_fn: Callable[[el.Object], Any],
    init: el.Object,
) -> Any:
    """Trace-mode implementation for SIMP while_loop."""

    cur_ctx = el.get_current_context()
    if not isinstance(cur_ctx, el.Tracer):
        raise TypeError(
            f"while_loop must be called within a Tracer context, got {type(cur_ctx).__name__}"
        )

    if not callable(cond_fn) or not callable(body_fn):
        raise TypeError("while_loop requires callable cond_fn and body_fn")

    state_flat, state_treedef = tree_flatten(init)
    if not state_flat:
        raise TypeError("while_loop init must contain at least one Object")

    state_trace_objs: list[el.TraceObject] = []
    for leaf in state_flat:
        if not isinstance(leaf, el.TraceObject):
            raise TypeError(
                f"while_loop init leaves must be TraceObject, got {type(leaf)}"
            )
        state_trace_objs.append(_ensure_trace_object(cur_ctx, leaf))

    state_values = [obj._graph_value for obj in state_trace_objs]
    state_types = [obj.type for obj in state_trace_objs]
    state_count = len(state_trace_objs)

    # Trace cond/body with the current state structure
    cond_traced = el.trace(cond_fn, init)
    body_traced = el.trace(body_fn, init)

    cond_outputs = cond_traced.graph.outputs
    if len(cond_outputs) != 1:
        raise TypeError(
            "while_loop cond_fn must return exactly one output, "
            f"got {len(cond_outputs)}"
        )
    cond_shape = getattr(cond_outputs[0].type, "shape", None)
    if cond_shape is not None and cond_shape != ():
        raise TypeError(
            f"while_loop cond_fn output must be scalar, got shape {cond_shape}"
        )

    body_outputs = body_traced.graph.outputs
    if len(body_outputs) != state_count:
        raise TypeError(
            "while_loop body_fn must return same number of values as init state: "
            f"{state_count} expected, got {len(body_outputs)}"
        )
    for idx, (out_val, state_obj) in enumerate(
        zip(body_outputs, state_trace_objs, strict=True)
    ):
        if out_val.type != state_obj.type:
            raise TypeError(
                "while_loop body_fn output type mismatch at index "
                f"{idx}: {out_val.type} vs {state_obj.type}"
            )

    all_captures = _merge_captures(cond_traced.captured, body_traced.captured)

    _align_region_inputs(cond_traced, state_count, all_captures)
    _align_region_inputs(body_traced, state_count, all_captures)

    capture_trace_objs = [_recapture_object(cur_ctx, obj) for obj in all_captures]
    capture_values = [obj._graph_value for obj in capture_trace_objs]

    loop_inputs = [*state_values, *capture_values]
    result_values = cur_ctx.graph.add_op(
        opcode="simp.while_loop",
        inputs=loop_inputs,
        output_types=state_types,
        regions=[cond_traced.graph, body_traced.graph],
    )

    result_trace_objs = [el.TraceObject(val, cur_ctx) for val in result_values]
    return state_treedef.unflatten(result_trace_objs)


def while_loop(
    cond_fn: Callable[[el.Object], Any],
    body_fn: Callable[[el.Object], Any],
    init: el.Object,
) -> Any:
    """Execute a SIMP while loop that synchronizes across parties.

    Args:
        cond_fn: Receives the current loop state and returns a boolean scalar.
        body_fn: Receives the current loop state and returns the next state
            with the same PyTree structure and per-leaf types as `init`.
        init: Initial loop state (PyTree of Objects) shared by all parties.

    Returns:
        Final state after `cond_fn` evaluates to False.

    Raises:
        TypeError: If `cond_fn`/`body_fn` outputs violate the required shape or
            type constraints.
    """

    return while_loop_p.bind(cond_fn, body_fn, init)


# ---------------------------------------------------------------------------
# Communication primitives (scaffold)
# ---------------------------------------------------------------------------

peval_p = el.Primitive("simp.peval")
pshfl_p = el.Primitive("simp.pshfl")
pshfl_s_p = el.Primitive("simp.pshfl_s")
pconv_p = el.Primitive("simp.pconv")


@peval_p.def_trace
def _peval_trace(
    parties: tuple[int, ...] | None,
    local_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Trace a local single-party region executed under a specific party mask.

    Args:
        parties: Optional tuple of participating party ranks (consistent with
            MPType.parties format). ``None`` for auto-deduction from arguments.
        local_fn: Callable representing the single-party function body.
        *args: Positional arguments forming a PyTree of MPObjects /
            TraceObjects / immediates passed to the region.
        **kwargs: Keyword arguments forwarded to ``local_fn``.

    Returns:
        PyTree of TraceObjects matching the structure of ``local_fn`` outputs.

    Raises:
        TypeError: If ``local_fn`` is not callable or arguments contain invalid
            types for a SIMP region.
        ValueError: When the explicitly provided parties are not covered by the
            deduced parties from arguments.
    """

    cur_ctx = el.get_current_context()
    if not isinstance(cur_ctx, el.Tracer):
        raise TypeError(f"simp.peval must run inside Tracer, got {type(cur_ctx)}")
    if not callable(local_fn):
        raise TypeError(f"local_fn must be callable, got {type(local_fn)}")

    requested_parties = _normalize_parties(parties)

    # Trace local region with _LocalMPTracer to discover all inputs (args + captures)
    local_tracer = _LocalMPTracer()
    local_traced = local_tracer.run(local_fn, *args, **kwargs)

    # Extract all freevars (args + captures) from the tracer
    # _freevars contains: {obj_id: (original_obj, graph_value)}
    all_input_objs = [obj for obj, _ in local_tracer._freevars.values()]
    all_input_types = [obj.type for obj in all_input_objs]

    # Deduce parties from ALL inputs that flow into the region
    deduced_parties = _deduce_parties(all_input_types)

    # Determine effective parties: requested takes precedence, then deduced
    if requested_parties is not None:
        effective_parties = requested_parties
        # Validate: requested parties should be covered by input parties
        if deduced_parties is not None:
            if not set(requested_parties).issubset(set(deduced_parties)):
                raise ValueError(
                    f"Requested parties {requested_parties} not covered by "
                    f"input argument parties {deduced_parties}"
                )
    else:
        effective_parties = deduced_parties
        # If neither requested nor deduced, we cannot determine execution context
        # This is acceptable if all args are non-MP (e.g., constants), but may be an error
        # For now, allow None to propagate (runtime/interpreter should handle)

    # Separate explicit args from captures for proper region input ordering
    # The tracer already separated them: in_vars = args, captured = captures
    flat_inputs, _ = tree_flatten((args, kwargs))
    num_arg_vars = sum(1 for val in flat_inputs if isinstance(val, el.Object))

    # Align region inputs as [args..., captures...]
    _align_region_inputs(local_traced, num_arg_vars, local_traced.captured)

    # Recapture all freevars into outer context
    # _freevars already contains all inputs in insertion order (args first, then captures)
    all_input_objs = [obj for obj, _ in local_tracer._freevars.values()]
    recaptured_objs = [_recapture_object(cur_ctx, obj) for obj in all_input_objs]
    region_inputs = [obj._graph_value for obj in recaptured_objs]
    result_types = [
        _wrap_with_mp(value.type, effective_parties)
        for value in local_traced.graph.outputs
    ]

    attrs = {
        "fn_name": local_traced.name,
        "parties": list(effective_parties) if effective_parties is not None else None,
        "requested_parties": (
            list(requested_parties) if requested_parties is not None else None
        ),
    }

    result_values = cur_ctx.graph.add_op(
        opcode="simp.peval",
        inputs=region_inputs,
        output_types=result_types,
        attrs=attrs,
        regions=[local_traced.graph],
    )

    return cur_ctx.reconstruct_outputs(
        local_traced.out_var_pos,
        local_traced.out_imms,
        local_traced.out_tree,
        result_values,
    )


def peval(
    parties: tuple[int, ...] | None,
    local_fn: Callable[..., Any],
    *call_args: Any,
    **call_kwargs: Any,
) -> Any:
    """Convenience wrapper for the SIMP peval primitive.

    Args:
        parties: Optional tuple of party ranks (consistent with MPType.parties).
            ``None`` lets the primitive infer parties from its arguments.
        local_fn: Callable representing the single-party computation.
        *call_args: Positional arguments forwarded to ``local_fn``.
        **call_kwargs: Keyword arguments forwarded to ``local_fn``.
    """
    return peval_p.bind(parties, local_fn, *call_args, **call_kwargs)


@pshfl_p.def_abstract_eval
def _pshfl_ae(src_t: elt.BaseType, index_t: elt.BaseType) -> elt.BaseType:
    # TODO: validate index_t is scalar; output type matches src_t (shape/dtype)
    return src_t


@pshfl_s_p.def_abstract_eval
def _pshfl_s_ae(src_t: elt.BaseType, pmask: Any, src_ranks: Any) -> elt.BaseType:
    # pmask/src_ranks are attributes until edsl.typing models them; passthrough type
    return src_t


@pconv_p.def_abstract_eval
def _pconv_ae(*vars_t: elt.BaseType) -> elt.BaseType:
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
    "pconv_p",
    # Communication
    "peval",
    "peval_p",
    "pshfl_p",
    "pshfl_s_p",
    # Control flow
    "uniform_cond",
    "uniform_cond_p",
    "while_loop",
    "while_loop_p",
]
