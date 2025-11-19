"""Registry for primitive implementations.

This module decouples the Primitive definition from the Interpreter execution.
Primitives register their implementations here, and the Interpreter looks them up here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Global registry for primitive implementations
# Key: opcode (str), Value: implementation function
_IMPL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_impl(opcode: str, fn: Callable[..., Any]) -> None:
    """Register an implementation for an opcode.

    Args:
        opcode: The unique name of the primitive (e.g. "add", "mul").
        fn: The function implementing the logic.
            Signature: (interpreter, op, *args) -> result
    """
    _IMPL_REGISTRY[opcode] = fn


def get_impl(opcode: str) -> Callable[..., Any] | None:
    """Get the registered implementation for an opcode."""
    return _IMPL_REGISTRY.get(opcode)
