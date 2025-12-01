"""JIT Decorator: Compile and cache Graph IR."""

from collections.abc import Callable
from typing import Any

from jax.tree_util import tree_map

from mplang.v2.edsl.context import get_current_context, get_default_context
from mplang.v2.edsl.interpreter import Interpreter
from mplang.v2.edsl.tracer import Tracer


def jit(fn: Callable) -> Callable:
    """JIT compilation decorator.

    Traces the function to Graph IR on first call, then executes the cached
    Graph on subsequent calls.

    Example:
        >>> @jit
        ... def compute(x, y):
        ...     return x + y
        >>> result = compute(x_interp, y_interp)  # First call: trace
        >>> result = compute(x_interp, y_interp)  # Subsequent: execute cached graph
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Tracer():
            result = fn(*args, **kwargs)

        # Use current context if available (e.g., SimpSimulator), otherwise use default
        cur_ctx = get_current_context() or get_default_context()
        assert isinstance(cur_ctx, Interpreter), (
            "JIT execution requires Interpreter context"
        )
        return tree_map(cur_ctx.lift, result)

    return wrapper
