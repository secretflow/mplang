"""SIMP dialect (experimental, EDSL-based)

Goal
-----
Migrate a subset of `mplang.core.primitive` into an EDSL-first dialect named
"simp" to support SPMD multi-party programming in the new architecture.

Design principles
-----------------
- Keep primitives as thin EDSL `Primitive`s with clear namespaced opcodes, e.g.:
  - "simp.add", "simp.uniform_cond", "simp.while", "simp.pshfl", ...
- For eager mode, implementations operate on `InterpObject`.
- For trace mode, we record Graph ops via Tracer.bind_primitive; initially single-output.
- Reuse existing generic EDSL arithmetic where possible; add SIMP-specific ops separately.

Initial scope
-------------
- Re-export basic arithmetic (add/mul/sub/div) under the SIMP namespace.
- Scaffold control-flow and communication primitives with clear contracts:
  - uniform_cond, while_loop (placeholders for region-based ops)
  - pshfl, pshfl_s, pconv (placeholders with signatures and TODOs)

Limitations (today)
-------------------
- EDSL Tracer.bind_primitive supports single-output, no regions; control-flow will
  be stubbed for tracing and only work in eager mode initially.
- Mask/party types are not modeled in edsl.typing; SIMP primitives will annotate
  expected semantics in docstrings and raise NotImplementedError where needed.

Usage
-----
>>> from mplang.dialects import simp
>>> # arithmetic works immediately (re-exported)
>>> z = simp.add.bind(x, y)

Roadmap
-------
This file is the starting point for incremental migration. See the docstring of
each primitive for planned semantics and TODOs.
"""

from __future__ import annotations

from typing import Any

from mplang.edsl.primitive import (
    Primitive,
)
from mplang.edsl.primitive import (
    add_p as _add,
)
from mplang.edsl.primitive import (
    div_p as _div,
)
from mplang.edsl.primitive import (
    mul_p as _mul,
)
from mplang.edsl.primitive import (
    sub_p as _sub,
)
from mplang.edsl.typing import BaseType

# Namespace: expose arithmetic under SIMP names for consistency with dialects
add = _add  # opcode currently "add"; may be renamed to "simp.add" later
mul = _mul
sub = _sub
div = _div


# ---------------------------------------------------------------------------
# Control flow (scaffold)
# ---------------------------------------------------------------------------

uniform_cond_p = Primitive("simp.uniform_cond")


@uniform_cond_p.def_abstract_eval
def _uniform_cond_ae(
    pred_t: BaseType,
    then_t: BaseType,
    else_t: BaseType,
    *args_t: BaseType,
    verify_uniform: bool = True,
) -> BaseType:
    """Abstract evaluation for uniform_cond.

    Contract (target):
    - pred_t: boolean scalar (to be enforced when edsl.typing has BOOL)
    - then/else result types must match exactly and are returned
    - verify_uniform is a static attribute that may guide lowering/runtime checks

    Current: placeholder returns then_t (caller responsible for ensuring match).
    """
    # TODO: validate pred_t is bool scalar when edsl.typing exposes BOOL/shape
    # TODO: ensure then_t == else_t; for now return then_t
    return then_t


@uniform_cond_p.def_impl
def _uniform_cond_impl(
    pred: Any, then_res: Any, else_res: Any, *args: Any, verify_uniform: bool = True
) -> Any:
    """Eager implementation for uniform_cond (scaffold).

    Current behavior:
    - If pred is a Python/NumPy boolean, returns then_res or else_res.
    - This is not side-effect safe for branch functions; this scaffold assumes
      the caller has pre-computed branch results (mirrors jax.where-like usage).

    Planned behavior:
    - Accept callables for then_fn/else_fn and execute only one branch.
    - When tracing, record a region-based op with two regions and proper inputs.
    """
    import numpy as np

    if isinstance(pred, (bool, np.bool_)):
        return then_res if bool(pred) else else_res
    raise NotImplementedError("uniform_cond eager impl expects boolean predicate value")


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
# Define stubs that make that explicit.
@pshfl_p.def_impl
def _pshfl_impl(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("simp.pshfl eager execution is not implemented yet")


@pshfl_s_p.def_impl
def _pshfl_s_impl(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("simp.pshfl_s eager execution is not implemented yet")


@pconv_p.def_impl
def _pconv_impl(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("simp.pconv eager execution is not implemented yet")


__all__ = [
    "add",
    "div",
    "mul",
    "pconv_p",
    "pshfl_p",
    "pshfl_s_p",
    "sub",
    "uniform_cond_p",
]
