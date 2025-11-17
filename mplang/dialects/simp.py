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


def _validate_scalar_predicate(value: el.graph.Value, context: str) -> None:
    """Validate that a graph value represents a scalar predicate."""
    shape = getattr(value.type, "shape", None)
    if shape is not None and shape != ():
        raise TypeError(
            f"{context} must be scalar, got shape {shape} with type {value.type}"
        )


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
    assert len(traced_fn.graph.inputs) >= leading_count

    leading_inputs = traced_fn.graph.inputs[:leading_count]
    capture_inputs = traced_fn.graph.inputs[leading_count:]
    capture_map = (
        dict(zip(traced_fn.captured, capture_inputs, strict=True))
        if traced_fn.captured
        else {}
    )

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

    for rank in value:
        if not isinstance(rank, int):
            raise TypeError(
                f"parties tuple must contain integers, got {rank!r} of type {type(rank)}"
            )
        if rank < 0:
            raise ValueError(f"party rank must be non-negative, got {rank}")

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
    current = set(masks[0])
    for parties in masks[1:]:
        assert parties is not None
        current &= set(parties)
    return tuple(sorted(current))


def _wrap_with_mp(
    base_type: elt.BaseType, parties: tuple[int, ...] | None
) -> elt.BaseType:
    """Wrap base_type with MP typing when a party mask is available."""
    if parties is None or isinstance(base_type, elt.MPType):
        return base_type
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

        lifted = super()._lift(obj)
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
    assert isinstance(cur_ctx, el.Tracer)

    if not isinstance(pred, el.TraceObject):
        raise TypeError(f"predicate must be TraceObject, got {type(pred)}")
    _validate_scalar_predicate(pred._graph_value, "uniform_cond predicate")
    if not callable(then_fn) or not callable(else_fn):
        raise TypeError("In trace mode, both branches must be callable functions")

    then_traced = el.trace(then_fn, *args, **kwargs)
    else_traced = el.trace(else_fn, *args, **kwargs)
    if not then_traced.is_output_signature_match(else_traced):
        then_types = [v.type for v in then_traced.graph.outputs]
        else_types = [v.type for v in else_traced.graph.outputs]
        raise TypeError(
            "uniform_cond branch output signature mismatch: "
            f"then={then_types}, else={else_types}"
        )

    num_arg_vars = len(then_traced.in_var_pos)
    arg_values = then_traced.graph.inputs[:num_arg_vars]

    all_captures = _merge_captures(then_traced.captured, else_traced.captured)

    _align_region_inputs(then_traced, num_arg_vars, all_captures)
    _align_region_inputs(else_traced, num_arg_vars, all_captures)

    capture_trace_objs = [cur_ctx.lift(obj) for obj in all_captures]
    capture_values = [obj._graph_value for obj in capture_trace_objs]

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
    cond_fn: Callable[[Any], Any],
    body_fn: Callable[[Any], Any],
    init: Any,
) -> Any:
    """Trace-mode implementation for SIMP while_loop."""

    cur_ctx = el.get_current_context()
    assert isinstance(cur_ctx, el.Tracer)
    assert callable(cond_fn) and callable(body_fn)

    state_flat, state_treedef = tree_flatten(init)
    if not state_flat:
        raise TypeError("while_loop init must contain at least one Object")

    state_trace_objs: list[el.TraceObject] = []
    for leaf in state_flat:
        if not isinstance(leaf, el.TraceObject):
            raise TypeError(
                f"while_loop init leaves must be TraceObject, got {type(leaf)}"
            )
        state_trace_objs.append(leaf)

    state_values = [obj._graph_value for obj in state_trace_objs]
    state_types = [obj.type for obj in state_trace_objs]
    state_count = len(state_trace_objs)

    cond_traced = el.trace(cond_fn, init)
    body_traced = el.trace(body_fn, init)

    cond_output_count = len(cond_traced.out_var_pos) + len(cond_traced.out_imms)
    if cond_output_count != 1:
        raise TypeError(
            "while_loop cond_fn must return exactly one output, "
            f"got {cond_output_count}"
        )
    if cond_traced.out_var_pos:
        _validate_scalar_predicate(
            cond_traced.graph.outputs[0], "while_loop cond_fn output"
        )

    body_output_count = len(body_traced.out_var_pos) + len(body_traced.out_imms)
    if body_output_count != state_count:
        raise TypeError(
            "while_loop body_fn must return same number of values as init state: "
            f"{state_count} expected, got {body_output_count}"
        )
    body_outputs = body_traced.graph.outputs
    if len(body_outputs) != state_count:
        raise TypeError(
            "while_loop body_fn must return all Variables (no immediates allowed in loop state), "
            f"expected {state_count} Variables, got {len(body_outputs)}"
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

    capture_trace_objs = [cur_ctx.lift(obj) for obj in all_captures]
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
    cond_fn: Callable[[Any], Any],
    body_fn: Callable[[Any], Any],
    init: Any,
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
    assert isinstance(cur_ctx, el.Tracer)
    assert callable(local_fn)

    requested_parties = _normalize_parties(parties)

    local_tracer = _LocalMPTracer()
    local_traced = local_tracer.run(local_fn, *args, **kwargs)

    all_input_objs = [obj for obj, _ in local_tracer._freevars.values()]
    all_input_types = [obj.type for obj in all_input_objs]
    deduced_parties = _deduce_parties(all_input_types)

    if requested_parties is not None:
        effective_parties = requested_parties
        if deduced_parties is not None:
            if not set(requested_parties).issubset(set(deduced_parties)):
                raise ValueError(
                    f"Requested parties {requested_parties} not covered by "
                    f"input argument parties {deduced_parties}"
                )
    else:
        effective_parties = deduced_parties

    num_arg_vars = len(local_traced.in_var_pos)

    _align_region_inputs(local_traced, num_arg_vars, local_traced.captured)

    all_input_objs = [obj for obj, _ in local_tracer._freevars.values()]
    recaptured_objs = [cur_ctx.lift(obj) for obj in all_input_objs]
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
    """Type inference for dynamic shuffle.

    Args:
        src_t: Source value type (must be MPType)
        index_t: Index value type (must be MPType with scalar shape)

    Returns:
        Output type with dynamic mask (parties=None)

    Raises:
        TypeError: If src or index are not MP-typed, or index is not scalar
    """
    if not isinstance(src_t, elt.MPType):
        raise TypeError(f"pshfl requires MP-typed src, got {src_t}")
    if not isinstance(index_t, elt.MPType):
        raise TypeError(f"pshfl requires MP-typed index, got {index_t}")

    # Validate index is scalar
    index_shape = getattr(index_t.value_type, "shape", None)
    if index_shape is not None and index_shape != ():
        raise TypeError(
            f"pshfl index must be scalar, got shape {index_shape} with type {index_t.value_type}"
        )

    # Output: dynamic mask (None parties)
    return elt.MP[src_t.value_type, None]


@pshfl_s_p.def_abstract_eval
def _pshfl_s_ae(
    src_t: elt.BaseType, parties: tuple[int, ...], src_ranks: list[int]
) -> elt.BaseType:
    """Type inference for static shuffle.

    Args:
        src_t: Source value type (must be MPType)
        parties: Target party tuple (static, compile-time known)
        src_ranks: Source rank for each target party

    Returns:
        Output type with static mask (parties=parties)

    Raises:
        TypeError: If src is not MP-typed or parties is None
        ValueError: If src_ranks length doesn't match parties, or
                    src_ranks reference parties not in src.parties
    """
    if not isinstance(src_t, elt.MPType):
        raise TypeError(f"pshfl_s requires MP-typed src, got {src_t}")

    parties_normalized = _normalize_parties(parties)
    if parties_normalized is None:
        raise TypeError("pshfl_s requires explicit parties tuple")

    if len(src_ranks) != len(parties_normalized):
        raise ValueError(
            f"pshfl_s: src_ranks length {len(src_ranks)} != parties count {len(parties_normalized)}"
        )

    # Validate src_ranks are in src.parties (if src.parties is known)
    if src_t.parties is not None:
        for rank in src_ranks:
            if rank not in src_t.parties:
                raise ValueError(
                    f"pshfl_s: src_rank {rank} not in src.parties {src_t.parties}"
                )

    # Output: static mask
    return elt.MP[src_t.value_type, parties_normalized]


@pconv_p.def_abstract_eval
def _pconv_ae(in_types: list[elt.BaseType], attrs: dict) -> elt.BaseType:
    """Type inference for converge operation.

    Args:
        in_types: List of input types (all must be MPType with same value_type)
        attrs: Attributes dict (unused)

    Returns:
        Output type with union of input parties (or None if any input is dynamic)

    Raises:
        TypeError: If inputs are not all MP-typed or have inconsistent value_types
        ValueError: If static parties are not disjoint
    """
    if not in_types:
        raise TypeError("pconv requires at least one input")

    # Validate all are MPType
    for i, t in enumerate(in_types):
        if not isinstance(t, elt.MPType):
            raise TypeError(f"pconv input {i} must be MP-typed, got {t}")

    mp_types = [t for t in in_types if isinstance(t, elt.MPType)]

    # Check value_type consistency
    first_vtype = mp_types[0].value_type
    for i, mt in enumerate(mp_types[1:], 1):
        if mt.value_type != first_vtype:
            raise TypeError(
                f"pconv value type mismatch at input {i}: "
                f"{mt.value_type} vs {first_vtype}"
            )

    # Deduce output parties
    parties_list = [mt.parties for mt in mp_types]

    if any(p is None for p in parties_list):
        # Dynamic case: propagate None
        output_parties = None
    else:
        # Static case: check disjoint and union
        for i, p1 in enumerate(parties_list):
            for j, p2 in enumerate(parties_list[i + 1 :], i + 1):
                if p1 is not None and p2 is not None:
                    if set(p1) & set(p2):
                        raise ValueError(
                            f"pconv requires disjoint parties, inputs {i} and {j} "
                            f"overlap: {set(p1) & set(p2)}"
                        )

        # Union all parties
        all_parties = set()
        for p in parties_list:
            if p is not None:
                all_parties.update(p)
        output_parties = tuple(sorted(all_parties)) if all_parties else None

    return elt.MP[first_vtype, output_parties]


def pshfl(src: el.Object, index: el.Object) -> el.Object:
    """Dynamic shuffle: redistribute src data based on runtime index values.

    Each party uses its local index value to fetch data from the corresponding
    source party. The output has dynamic mask (parties=None) since the data
    distribution depends on runtime index values.

    Args:
        src: Source data (MP-typed)
        index: Index indicating which source party to fetch from (MP-typed scalar)

    Returns:
        Shuffled data with dynamic mask

    Example:
        >>> # P0, P1, P2 each hold different index values at runtime
        >>> result = pshfl(src, index)
        >>> # result.type.parties == None (dynamic)
    """
    return pshfl_p.bind(src, index)


def pshfl_s(
    src: el.Object, parties: tuple[int, ...], src_ranks: list[int]
) -> el.Object:
    """Static shuffle: redistribute src data to specified parties from src_ranks.

    Unlike pshfl, the shuffle pattern is known at compile time. The i-th party
    in `parties` receives data from `src_ranks[i]`.

    Args:
        src: Source data (MP-typed)
        parties: Target party tuple (compile-time known)
        src_ranks: Source rank for each target party (must match parties length)

    Returns:
        Shuffled data with static mask (parties=parties)

    Example:
        >>> # Shuffle from P1 to P0
        >>> result = pshfl_s(src, parties=(0,), src_ranks=[1])
        >>> # result.type.parties == (0,)
    """
    return pshfl_s_p.bind(src, parties=parties, src_ranks=src_ranks)


def pconv(*vars: el.Object) -> el.Object:
    """Converge multiple disjoint-masked variables into a single variable.

    Combines data from multiple parties into one logical variable. In static case,
    validates that input parties are disjoint and unions them. In dynamic case,
    propagates the dynamic mask.

    Args:
        *vars: Variable number of MP-typed inputs with disjoint parties

    Returns:
        Converged variable with union of input parties (or None if any input is dynamic)

    Raises:
        ValueError: If static parties are not disjoint

    Example:
        >>> # P0 has x, P1 has y (disjoint)
        >>> result = pconv(x, y)
        >>> # result.type.parties == (0, 1)
    """
    return pconv_p.bind(*vars)


__all__ = [
    # Communication primitives
    "pconv",
    "pconv_p",
    "peval",
    "peval_p",
    "pshfl",
    "pshfl_p",
    "pshfl_s",
    "pshfl_s_p",
    # Control flow
    "uniform_cond",
    "uniform_cond_p",
    "while_loop",
    "while_loop_p",
]
