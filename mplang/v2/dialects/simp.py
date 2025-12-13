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

"""SIMP dialect: SPMD multi-party primitives for EDSL.

Provides control flow and communication primitives:
- pcall_static: Party call with explicit static parties
- pcall_dynamic: Party call where all parties attempt execution (output always dynamic)
- shuffle_dynamic, shuffle: Data redistribution
- converge: Merge disjoint partitions
- uniform_cond: Uniform conditional (eager mode)
- while_loop: While loop (eager mode)

Primitive definition guideline:
- Simple ops (add, mul) → use def_abstract_eval
- Complex ops (control flow, fork tracer) → use def_trace

See individual primitive docstrings for detailed documentation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from jax.tree_util import tree_flatten, tree_unflatten

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Whether to verify predicate uniformity at runtime in uniform_cond
# Set to False to disable runtime checks (useful for testing or when
# uniformity is guaranteed)
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
    """Merge capture lists while preserving first-seen order and deduplicating by id."""
    seen: dict[int, el.Object] = {}
    for obj in (o for lst in capture_lists for o in lst):
        seen.setdefault(id(obj), obj)
    return list(seen.values())


def _deduce_parties(types: Sequence[elt.BaseType]) -> tuple[int, ...] | None:
    """Deduce common parties by intersecting all known party sets."""
    if not types:
        return None

    # Extract parties from MPType objects
    parties_list = []
    for tp in types:
        if isinstance(tp, elt.MPType):
            parties_list.append(tp.parties)

    if not parties_list:
        return None

    if any(p is None for p in parties_list):
        return None

    # Intersect all party sets (we know all parties are not None here)
    first_parties = parties_list[0]
    assert first_parties is not None
    current = set(first_parties)
    for parties in parties_list[1:]:
        assert parties is not None
        current &= set(parties)
    return tuple(sorted(current))


class _LocalMPTracer(el.Tracer):
    """Tracer for single-party regions executed under MP context."""

    def _lift_type(self, obj: el.Object) -> elt.BaseType:
        """Override to unwrap MP-typed Objects to their value types.

        This enables single-party regions to work with the underlying value types.
        MP-typed objects are unwrapped to their value types.
        Other types (e.g. TensorType) are passed through as-is (treated as public/replicated).

        Args:
            obj: Object to lift

        Returns:
            value_type (unwrapped from MPType) or original type
        """
        obj_type = obj.type
        if isinstance(obj_type, elt.MPType):
            return cast(elt.BaseType, obj_type.value_type)
        return cast(elt.BaseType, obj_type)


# ---------------------------------------------------------------------------
# Control flow (scaffold)
# ---------------------------------------------------------------------------


uniform_cond_p: el.Primitive[Any] = el.Primitive("simp.uniform_cond")


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
        The verify_uniform flag is controlled by the global
        VERIFY_UNIFORM_DEFAULT config. To change it, set
        mplang.dialects.simp.VERIFY_UNIFORM_DEFAULT = False

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

    # Get outer graph values for parameters
    # then_traced.params contains the original TraceObjects from args/kwargs
    outer_arg_values = [
        param._graph_value
        for param in then_traced.params
        if isinstance(param, el.TraceObject)
    ]

    if len(outer_arg_values) != num_arg_vars:
        raise RuntimeError(
            f"uniform_cond: argument count mismatch. Expected {num_arg_vars} variables, "
            f"got {len(outer_arg_values)} from params."
        )

    all_captures = _merge_captures(then_traced.captured, else_traced.captured)

    then_traced.align_region_inputs(num_arg_vars, all_captures)
    else_traced.align_region_inputs(num_arg_vars, all_captures)

    capture_trace_objs = [cur_ctx.lift(obj) for obj in all_captures]
    capture_values = [obj._graph_value for obj in capture_trace_objs]

    output_types = [v.type for v in then_traced.graph.outputs]
    cond_inputs = [pred._graph_value, *outer_arg_values, *capture_values]

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

while_loop_p: el.Primitive[Any] = el.Primitive("simp.while_loop")


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
    assert state_treedef is not None
    if not state_flat:
        raise TypeError("while_loop init must contain at least one Object")

    # Validate all leaves are TraceObjects
    for leaf in state_flat:
        if not isinstance(leaf, el.TraceObject):
            raise TypeError(
                f"while_loop init leaves must be TraceObject, got {type(leaf)}"
            )

    cond_traced = el.trace(cond_fn, init)
    body_traced = el.trace(body_fn, init)

    # Use params from traced function (same as state_flat filtered to Objects)
    # These are TraceObjects since we're in trace mode
    state_trace_objs = cast(list[el.TraceObject], cond_traced.params)
    state_values = [obj._graph_value for obj in state_trace_objs]
    state_types = [obj.type for obj in state_trace_objs]
    state_count = len(state_trace_objs)

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
            "while_loop body_fn must return all Variables "
            "(no immediates allowed in loop state), "
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

    cond_traced.align_region_inputs(state_count, all_captures)
    body_traced.align_region_inputs(state_count, all_captures)

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
    return tree_unflatten(state_treedef, result_trace_objs)


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


# Core primitives with clear semantic names
pcall_static_p = el.Primitive[Any]("simp.pcall_static")
pcall_dynamic_p = el.Primitive[Any]("simp.pcall_dynamic")
shuffle_dynamic_p = el.Primitive[el.Object]("simp.shuffle_dynamic")
shuffle_static_p = el.Primitive[el.Object]("simp.shuffle")
converge_p = el.Primitive[el.Object]("simp.converge")


@pcall_static_p.def_trace
def _pcall_static_trace(
    parties: tuple[int, ...],
    local_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Trace a local single-party region with explicit static parties.

    Args:
        parties: Required tuple of participating party ranks.
        local_fn: Callable representing the single-party function body.
        *args: Positional arguments forming a PyTree of MPObjects /
            TraceObjects / immediates passed to the region.
        **kwargs: Keyword arguments forwarded to ``local_fn``.

    Returns:
        PyTree of TraceObjects with static parties mask.

    Raises:
        TypeError: If ``local_fn`` is not callable or arguments contain invalid types.
        ValueError: When explicitly provided parties are not covered by input parties.
    """
    cur_ctx = el.get_current_context()
    assert isinstance(cur_ctx, el.Tracer)
    assert callable(local_fn)

    if parties is None:
        raise ValueError("pcall_static requires explicit parties, got None")

    requested_parties = tuple(sorted(set(parties)))

    local_tracer = _LocalMPTracer()
    local_traced = local_tracer.run(local_fn, *args, **kwargs)

    # Get all input objects: params (function arguments) + captured (closures)
    # TracedFunction guarantees: graph.inputs = [*params_inputs, *captured_inputs]
    all_input_objs = local_traced.params + local_traced.captured

    # All types are guaranteed to be MPType by _LocalMPTracer._lift
    all_input_types: list[elt.MPType] = [obj.type for obj in all_input_objs]  # type: ignore[misc]
    deduced_parties = _deduce_parties(all_input_types)

    if deduced_parties is not None:
        if not set(requested_parties).issubset(set(deduced_parties)):
            raise ValueError(
                f"Requested parties {requested_parties} not covered by "
                f"input argument parties {deduced_parties}"
            )

    # Re-capture all input objects in outer context
    recaptured_objs = [cur_ctx.lift(obj) for obj in all_input_objs]
    region_inputs = [obj._graph_value for obj in recaptured_objs]
    result_types: list[elt.BaseType] = [
        elt.MPType(value.type, requested_parties)
        for value in local_traced.graph.outputs
    ]

    result_values = cur_ctx.graph.add_op(
        opcode="simp.pcall_static",
        inputs=region_inputs,
        output_types=result_types,
        attrs={
            "fn_name": local_traced.name,
            "parties": list(requested_parties),
        },
        regions=[local_traced.graph],
    )

    return cur_ctx.reconstruct_outputs(
        local_traced.out_var_pos,
        local_traced.out_imms,
        local_traced.out_tree,
        result_values,
    )


@pcall_dynamic_p.def_trace
def _pcall_dynamic_trace(
    local_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Trace a party call with dynamic execution.

    All parties attempt to execute. Runtime behavior: each party executes
    if all inputs are present, otherwise outputs None. Output always has
    dynamic parties (None).

    Args:
        local_fn: Callable representing the single-party function body.
        *args: Positional arguments forming a PyTree of MPObjects /
            TraceObjects / immediates passed to the region.
        **kwargs: Keyword arguments forwarded to ``local_fn``.

    Returns:
        PyTree of TraceObjects with dynamic parties (None).

    Raises:
        TypeError: If ``local_fn`` is not callable or arguments contain invalid types.
    """
    cur_ctx = el.get_current_context()
    assert isinstance(cur_ctx, el.Tracer)
    assert callable(local_fn)

    local_tracer = _LocalMPTracer()
    local_traced = local_tracer.run(local_fn, *args, **kwargs)

    # Get all input objects: params (function arguments) + captured (closures)
    # TracedFunction guarantees: graph.inputs = [*params_inputs, *captured_inputs]
    all_input_objs = local_traced.params + local_traced.captured

    recaptured_objs = [cur_ctx.lift(obj) for obj in all_input_objs]
    region_inputs = [obj._graph_value for obj in recaptured_objs]

    # Output always has dynamic parties (None)
    result_types: list[elt.BaseType] = [
        elt.MPType(value.type, None) for value in local_traced.graph.outputs
    ]

    result_values = cur_ctx.graph.add_op(
        opcode="simp.pcall_dynamic",
        inputs=region_inputs,
        output_types=result_types,
        attrs={
            "fn_name": local_traced.name,
        },
        regions=[local_traced.graph],
    )

    return cur_ctx.reconstruct_outputs(
        local_traced.out_var_pos,
        local_traced.out_imms,
        local_traced.out_tree,
        result_values,
    )


def pcall_static(
    parties: tuple[int, ...],
    local_fn: Callable[..., Any],
    *call_args: Any,
    **call_kwargs: Any,
) -> Any:
    """Execute a function on explicitly specified parties (static).

    This primitive requires explicit party specification and always produces
    static party masks in the output. Use this when the execution parties
    are known at compile time.

    Args:
        parties: Required tuple of party ranks (must be explicit, not None).
        local_fn: Callable representing the single-party computation.
        *call_args: Positional arguments forwarded to ``local_fn``.
        **call_kwargs: Keyword arguments forwarded to ``local_fn``.

    Returns:
        Result with static parties mask matching the parties argument.

    Example:
        >>> # Compute on parties 0 and 1 (static)
        >>> result = pcall_static(parties=(0, 1), local_fn=lambda x: x + 1, x)
    """
    return pcall_static_p.bind(parties, local_fn, *call_args, **call_kwargs)


def pcall_dynamic(
    local_fn: Callable[..., Any],
    *call_args: Any,
    **call_kwargs: Any,
) -> Any:
    """Execute a function on all parties with runtime-determined execution.

    All parties attempt to execute the function. At runtime, each party executes
    if all inputs are present, otherwise outputs None. Output always has dynamic
    party mask (None).

    Args:
        local_fn: Callable representing the single-party computation.
        *call_args: Positional arguments forwarded to ``local_fn``.
        **call_kwargs: Keyword arguments forwarded to ``local_fn``.

    Returns:
        Result with dynamic parties (None). At runtime, parties with all inputs
        execute, others output None.

    Example:
        >>> # All parties attempt execution based on input availability
        >>> result = pcall_dynamic(local_fn=lambda x: x + 1, x)
    """
    return pcall_dynamic_p.bind(local_fn, *call_args, **call_kwargs)


@shuffle_dynamic_p.def_abstract_eval
def _shuffle_dynamic_ae(src_t: elt.BaseType, index_t: elt.BaseType) -> elt.BaseType:
    """Type inference for dynamic shuffle (runtime-determined data redistribution).

    Args:
        src_t: Source value type (must be MPType)
        index_t: Index value type (must be MPType with scalar shape)

    Returns:
        Output type with dynamic mask (parties=None)

    Raises:
        TypeError: If src or index are not MP-typed, or index is not scalar
    """
    if not isinstance(src_t, elt.MPType):
        raise TypeError(f"shuffle_dynamic requires MP-typed src, got {src_t}")
    if not isinstance(index_t, elt.MPType):
        raise TypeError(f"shuffle_dynamic requires MP-typed index, got {index_t}")

    # Validate index is scalar
    index_shape = getattr(index_t.value_type, "shape", None)
    if index_shape is not None and index_shape != ():
        raise TypeError(
            f"shuffle_dynamic index must be scalar, got shape {index_shape} "
            f"with type {index_t.value_type}"
        )

    # Output: dynamic mask (None parties)
    return elt.MPType(src_t.value_type, None)


@shuffle_static_p.def_abstract_eval
def _shuffle_ae(src_t: elt.BaseType, routing: dict[int, int]) -> elt.BaseType:
    """Type inference for static shuffle (compile-time known data routing).

    Args:
        src_t: Source value type (must be MPType)
        routing: Dict mapping target_party -> source_rank

    Returns:
        Output type with static mask (parties=tuple(sorted(routing.keys())))

    Raises:
        TypeError: If src is not MP-typed or routing is not a dict
        ValueError: If routing references parties not in src.parties
    """
    if not isinstance(src_t, elt.MPType):
        raise TypeError(f"shuffle_static requires MP-typed src, got {src_t}")

    if not isinstance(routing, dict):
        raise TypeError(f"shuffle_static requires routing dict, got {type(routing)}")

    if not routing:
        raise ValueError("shuffle_static requires non-empty routing dict")

    # Target parties are the keys of routing dict
    target_parties = tuple(sorted(routing.keys()))

    # Validate source ranks are in src.parties (if src.parties is known)
    if src_t.parties is not None:
        for target, source in routing.items():
            if source not in src_t.parties:
                raise ValueError(
                    f"shuffle_static: routing[{target}]={source} not in "
                    f"src.parties {src_t.parties}"
                )

    # Output: static mask with target parties
    return elt.MPType(src_t.value_type, target_parties)


@converge_p.def_abstract_eval
def _converge_ae(in_types: list[elt.BaseType], *, mask: int = -1) -> elt.BaseType:
    """Type inference for converge operation (merge disjoint partitions).

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
        raise TypeError("converge requires at least one input")

    # Validate all are MPType
    for i, t in enumerate(in_types):
        if not isinstance(t, elt.MPType):
            raise TypeError(f"converge input {i} must be MP-typed, got {t}")

    mp_types = [t for t in in_types if isinstance(t, elt.MPType)]

    # Check value_type consistency
    first_vtype = mp_types[0].value_type
    for i, mt in enumerate(mp_types[1:], 1):
        if mt.value_type != first_vtype:
            raise TypeError(
                f"converge value type mismatch at input {i}: "
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
                            f"converge requires disjoint parties, inputs {i} and {j} "
                            f"overlap: {set(p1) & set(p2)}"
                        )

        # Union all parties
        all_parties: set[int] = set()
        for p in parties_list:
            if p is not None:
                all_parties.update(p)
        output_parties = tuple(sorted(all_parties)) if all_parties else None

    return elt.MPType(first_vtype, output_parties)


def shuffle_dynamic(src: el.Object, index: el.Object) -> el.Object:
    """Dynamic shuffle: redistribute data based on runtime index values.

    Each party uses its local index value to fetch data from the corresponding
    source party. The output has dynamic mask (parties=None) since the data
    distribution depends on runtime index values.

    This is the most flexible shuffle primitive but requires runtime communication
    pattern determination.

    Args:
        src: Source data (MP-typed)
        index: Index indicating which source party to fetch from (MP-typed scalar)

    Returns:
        Shuffled data with dynamic mask (parties=None)

    Example:
        >>> # P0, P1, P2 each hold different index values at runtime
        >>> result = shuffle_dynamic(src, index)
        >>> # result.type.parties == None (dynamic)
    """
    return shuffle_dynamic_p.bind(src, index)


def shuffle_static(src: el.Object, routing: dict[int, int]) -> el.Object:
    """Static shuffle: redistribute data with compile-time known routing pattern.

    Unlike shuffle_dynamic, the routing pattern is known at compile time.
    Each entry in routing specifies: target_party -> source_rank.

    This enables compile-time optimization and produces a static output mask.

    Design rationale:
        Uses receiver-oriented routing {target: source} to naturally express:
        - Permutation: {0: 1, 1: 0} (swap parties)
        - Broadcast: {0: 1, 2: 1} (multiple targets from same source)
        Maintains SIMP single-input-single-output semantics at MP value level.

    Args:
        src: Source data (MP-typed)
        routing: Dict mapping target_party -> source_rank
                 e.g., {0: 1, 2: 0} means:
                 - party 0 receives from rank 1
                 - party 2 receives from rank 0

    Returns:
        Shuffled data with static mask (parties=sorted keys of routing)

    Example:
        >>> # Party 0 gets data from rank 1
        >>> result = shuffle_static(src, routing={0: 1})
        >>> # result.type.parties == (0,)
        >>>
        >>> # Multiple parties
        >>> result = shuffle_static(src, routing={0: 1, 2: 0})
        >>> # result.type.parties == (0, 2)
    """
    return shuffle_static_p.bind(src, routing=routing)


def converge(*vars: el.Object) -> el.Object:
    """Converge multiple disjoint-partitioned variables into one.

    Merges data from multiple parties into one logical variable. In static case,
    validates that input parties are disjoint and produces their union. In dynamic
    case, propagates the dynamic mask.

    This is the fundamental operation for combining results from different parties.

    Args:
        *vars: Variable number of MP-typed inputs with disjoint parties

    Returns:
        Converged variable with union of input parties (or None if any input is dynamic)

    Raises:
        ValueError: If static parties are not disjoint

    Example:
        >>> # P0 has x, P1 has y (disjoint)
        >>> result = converge(x, y)
        >>> # result.type.parties == (0, 1)
    """
    return converge_p.bind(*vars)


def constant(parties: tuple[int, ...], data: Any) -> el.Object:
    """Create a constant value distributed to specific parties.

    This is a helper function that creates a constant value on the specified
    parties. It is equivalent to calling `pcall_static` with a function that
    returns the constant data.

    Args:
        parties: Tuple of party ranks where the constant should be placed.
        data: The constant data (scalar, array, etc.).

    Returns:
        MP[Tensor, parties] object representing the distributed constant.
    """
    import jax.numpy as jnp
    import numpy as np

    from mplang.v2.dialects import table, tensor

    # 1. Scalars (int, float, bool, numpy scalars)
    if isinstance(data, (int, float, bool, np.number, np.bool_)):
        return cast(el.Object, pcall_static(parties, tensor.constant, data))

    # 2. Tensor-like (numpy array or JAX array)
    if isinstance(data, (np.ndarray, jnp.ndarray)):
        return cast(el.Object, pcall_static(parties, tensor.constant, data))

    # 3. Table-like (dict, DataFrame)
    is_dataframe = False
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            is_dataframe = True
    except ImportError:
        pass

    if is_dataframe or isinstance(data, dict):
        return cast(el.Object, pcall_static(parties, table.constant, data))

    # 4. Lists/Tuples (Ambiguous, default to tensor)
    if isinstance(data, (list, tuple)):
        return cast(el.Object, pcall_static(parties, tensor.constant, data))

    raise TypeError(f"Unsupported data type for simp.constant: {type(data)}")


# Backward compatibility aliases
def peval(
    parties: tuple[int, ...] | None,
    local_fn: Callable[..., Any],
    *call_args: Any,
    **call_kwargs: Any,
) -> Any:
    """Backward compatible peval function.

    Routes to pcall_static if parties is explicit, pcall_dynamic if None.
    """
    if parties is None:
        return pcall_dynamic(local_fn, *call_args, **call_kwargs)
    else:
        return pcall_static(parties, local_fn, *call_args, **call_kwargs)


# =============================================================================
# Factory functions for creating configured Interpreters
# =============================================================================


def make_simulator(
    world_size: int,
    *,
    cluster_spec: Any = None,
    enable_tracing: bool = False,
    enable_profiling: bool = False,
) -> Any:
    """Create an Interpreter configured for local SIMP simulation.

    This factory creates a LocalCluster with workers and returns an
    Interpreter with the simp dialect state attached.

    Args:
        world_size: Number of simulated parties.
        cluster_spec: Optional ClusterSpec for metadata.
        enable_tracing: If True, enable execution tracing.
        enable_profiling: If True, enable primitive profiling for benchmarking.

    Returns:
        Configured Interpreter with simp state attached.

    Example:
        >>> interp = simp.make_simulator(2)
        >>> with interp:
        ...     result = my_func()
    """
    if enable_profiling:
        from mplang.v2.edsl import registry

        registry.enable_profiling()

    from mplang.v2.backends.simp_driver.mem import make_simulator as _make_sim

    return _make_sim(
        world_size, cluster_spec=cluster_spec, enable_tracing=enable_tracing
    )


def make_driver(endpoints: list[str], *, cluster_spec: Any = None) -> Any:
    """Create an Interpreter configured for remote SIMP execution.

    This factory creates a RemoteSimpState and returns an Interpreter
    with the simp dialect state attached.

    Args:
        endpoints: List of HTTP endpoints for workers.
        cluster_spec: Optional ClusterSpec for metadata.

    Returns:
        Configured Interpreter with simp state attached.

    Example:
        >>> interp = simp.make_driver(["http://worker1:8000", "http://worker2:8000"])
        >>> with interp:
        ...     result = my_func()
    """
    from mplang.v2.backends.simp_driver.http import make_driver as _make_drv

    return _make_drv(endpoints, cluster_spec=cluster_spec)


__all__ = [
    "constant",
    "converge",
    "converge_p",
    "make_driver",
    "make_simulator",
    "pcall_dynamic",
    "pcall_dynamic_p",
    "pcall_static",
    "pcall_static_p",
    "peval",
    "shuffle_dynamic",
    "shuffle_dynamic_p",
    "shuffle_static",
    "shuffle_static_p",
    "uniform_cond",
    "uniform_cond_p",
    "while_loop",
    "while_loop_p",
]
