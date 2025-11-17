"""JIT Decorator: Compile and cache Graph IR."""

from collections.abc import Callable

from mplang2.edsl.graph import Graph
from mplang2.edsl.interpreter import interpret
from mplang2.edsl.tracer import Tracer


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

    cached_graph: Graph | None = None

    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal cached_graph

        # TODO: Argument validation

        if cached_graph is None:
            # First call: trace
            tracer = Tracer()
            cached_graph = tracer.trace(fn, *args, **kwargs)
            print(f"[JIT] Traced function '{fn.__name__}' to graph:")
            print(cached_graph)

        result = interpret(cached_graph, args)
        return result

    return wrapper
