"""Func dialect: Function definition and call primitives for EDSL.

Provides function-related operations:
- call: Call a graph with arguments (Primitive for use in EDSL programs)

Note: TracedFunction and trace (formerly make_graph) have been moved to mplang.edsl.tracer.

See individual function docstrings for detailed documentation.
"""

from __future__ import annotations

from typing import Any

from mplang.edsl.primitive import Primitive
from mplang.edsl.typing import BaseType

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
    "call_p",
]
