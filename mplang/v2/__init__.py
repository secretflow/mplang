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

"""MPLang2: Next generation EDSL for multi-party computation.

This is the temporary home for the new EDSL implementation during migration.
Once migration is complete, this will replace the original mplang package.

Public API is designed to be compatible with mplang v1 where possible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__version__ = "0.1.0"

# =============================================================================
# Core EDSL components
# =============================================================================
# =============================================================================
# Dialects
# =============================================================================
# =============================================================================
# Backend / Runtime
# =============================================================================
import mplang.v2.backends.func_impl  # Register func handlers
from mplang.v2 import dialects
from mplang.v2.backends.simp_driver.ops import DRIVER_HANDLERS
from mplang.v2.backends.simp_worker import SimpWorker
from mplang.v2.backends.simp_worker.mem import LocalMesh
from mplang.v2.backends.simp_worker.ops import WORKER_HANDLERS
from mplang.v2.dialects.simp import make_driver, make_simulator
from mplang.v2.edsl import (
    Graph,
    GraphPrinter,
    Object,
    Operation,
    Primitive,
    TracedFunction,
    Tracer,
    Value,
    format_graph,
    get_current_context,
    get_default_context,
    get_root_context,
    jit,
    pop_context,
    primitive,
    push_context,
    register_default_context_factory,
    trace,
)
from mplang.v2.edsl.registry import get_profiler

# Type system
from mplang.v2.edsl.typing import (
    MPType,
    ScalarType,
    SSType,
    TableType,
    TensorType,
    VectorType,
)

# =============================================================================
# Device API (compatible with mplang v1)
# =============================================================================
from mplang.v2.libs.device import (
    ClusterSpec,
    Device,
    Node,
    device,
    get_dev_attr,
    is_device_obj,
    jax_fn,
    put,
    set_dev_attr,
)
from mplang.v2.runtime.interpreter import Interpreter

# =============================================================================
# Context Management API (JAX-like pattern)
# =============================================================================


def set_root_context(context: Interpreter, force: bool = False) -> None:
    """Set the global/root execution context.

    This explicitly sets the provided interpreter as the Root Context.
    All subsequent operations (compile, evaluate, device resolution) will
    use this context as the default environment.

    Args:
        context: Interpreter to use as the root context.
        force: If True, clears the existing context stack before setting.
               If False (default), pushes onto the stack.
    """
    from mplang.v2.edsl.context import _context_stack, get_current_context

    if force:
        _context_stack.clear()
        _context_stack.append(context)
        return

    if get_current_context() is not None:
        raise RuntimeError(
            "Cannot set root context: Context stack is not empty. "
            "Use force=True to overwrite the existing root context."
        )

    push_context(context)


def _get_context(context: Interpreter | None) -> Interpreter:
    """Get context from parameter or context stack."""
    if context is not None:
        return context
    ctx = get_current_context()
    if ctx is None:
        raise RuntimeError(
            "No context available. Either pass context explicitly or use "
            "set_context()/push_context() to set a default context."
        )
    if not isinstance(ctx, Interpreter):
        raise RuntimeError(
            f"Current context is not an Interpreter: {type(ctx).__name__}. "
            "Use mp.set_context(interpreter) to set the execution context."
        )
    return ctx


# =============================================================================
# Meta-APIs (compile, evaluate, fetch)
# =============================================================================


def evaluate(
    fn: Callable[..., Any] | TracedFunction,
    *args: Any,
    context: Interpreter | None = None,
    **kwargs: Any,
) -> Any:
    """Evaluate a function or traced function.

    Args:
        fn: The function or TracedFunction to evaluate.
        *args: Positional arguments to pass to the function.
        context: Optional interpreter context. If None, uses current context.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function evaluation.

    Example:
        >>> with mp.make_simulator(3) as sim:
        ...     result = mp.evaluate(traced)  # uses sim from context
        >>> # Or explicitly:
        >>> result = mp.evaluate(traced, context=sim)
    """
    from mplang.v2.edsl.tracer import TracedFunction
    from mplang.v2.runtime.interpreter import InterpObject

    interp = _get_context(context)

    def unwrap_if_interp(val: Any) -> Any:
        """Unwrap InterpObject to runtime value at execution boundary."""
        if isinstance(val, InterpObject):
            return val.runtime_obj
        return val

    with interp:
        if isinstance(fn, TracedFunction):
            inputs = fn.prepare_inputs(*args, **kwargs)
            inputs = [unwrap_if_interp(v) for v in inputs]
            raw_result = interp.evaluate_graph(fn.graph, inputs)
            return fn.reconstruct_outputs(raw_result)

        return fn(*args, **kwargs)


def fetch(
    result: Any,
    party: int | str | None = None,
    *,
    context: Interpreter | None = None,
) -> Any:
    """Fetch the result, optionally for a specific party.

    Args:
        result: The result to fetch (DriverVar or other value).
        party: Optional party index or device name (e.g., "P0").
        context: Optional interpreter context. If None, uses current context.

    Returns:
        The fetched data (unwrapped from Value wrappers).

    Example:
        >>> with mp.make_simulator(3) as sim:
        ...     data = mp.fetch(result, "P0")  # uses sim from context
    """
    from typing import cast

    from mplang.v2.backends.simp_driver.state import SimpDriver
    from mplang.v2.backends.simp_driver.values import DriverVar
    from mplang.v2.backends.table_impl import TableValue
    from mplang.v2.backends.tensor_impl import TensorValue
    from mplang.v2.runtime.interpreter import InterpObject

    interp = _get_context(context)

    def _unwrap_value(val: Any) -> Any:
        """Unwrap Value types to get the underlying data."""
        if isinstance(val, TensorValue):
            return val.data
        elif isinstance(val, TableValue):
            return val.data
        return val

    # Unwrap InterpObject to get the runtime value
    if isinstance(result, InterpObject):
        result = result.runtime_obj

    # Get simp state for fetching
    simp_state = cast(SimpDriver | None, interp.get_dialect_state("simp"))
    cluster_spec = getattr(interp, "_cluster_spec", None)

    # Fetch from DriverVar
    if isinstance(result, DriverVar):
        resolved_values = []
        for rank, val in enumerate(result.values):
            if isinstance(val, str) and "://" in val:
                if simp_state is not None:
                    fut = simp_state.fetch(rank, val)
                    resolved_values.append(fut.result())
                else:
                    resolved_values.append(val)
            else:
                resolved_values.append(val)

        # Select party if needed
        if party is not None:
            if isinstance(party, str) and cluster_spec is not None:
                device_info = cluster_spec.devices.get(party)
                if device_info and device_info.members:
                    party = device_info.members[0].rank
                else:
                    raise ValueError(f"Unknown party: {party}")
            else:
                # Default logic for int
                pass

            p_idx = cast(int, party)
            return _unwrap_value(resolved_values[p_idx])  # type: ignore[no-any-return]
        return [_unwrap_value(v) for v in resolved_values]

    # Unwrap Value types to get the underlying data
    return _unwrap_value(result)


# Alias for compatibility
def function(fn: Callable[..., Any] | None = None) -> Callable[..., Any]:
    """Decorator defining a Multi-Party Function (MP Program).

    This decorator "lifts" a local function into a distributed program by
    automatically wrapping it in a `simp.pcall_static` that targets ALL available
    parties in the current context.

    Semantics: f(args) -> pcall(ALL, f, args)

    Args:
        fn: The function to decorate.

    Returns:
        A wrapper function that, when called, executes the original function
        on all workers.
    """
    import functools

    from mplang.v2.dialects import simp

    if fn is None:
        return function

    def has_simp_state(ctx: Any) -> bool:
        if hasattr(ctx, "get_dialect_state"):
            state = ctx.get_dialect_state("simp")
            return state is not None and hasattr(state, "world_size")
        return False

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from mplang.v2.edsl.context import find_context

        # Find context with simp dialect state
        ctx = find_context(has_simp_state)
        if ctx is None:
            raise RuntimeError(
                "mp.function requires a context with world_size information "
                "(e.g. SimpSimulator or Driver initialized)."
            )

        # ctx found by predicate so we know it has get_dialect_state
        simp_state = ctx.get_dialect_state("simp")  # type: ignore[attr-defined]
        world_size = simp_state.world_size  # type: ignore

        all_parties = tuple(range(world_size))
        return simp.pcall_static(all_parties, fn, *args, **kwargs)

    return wrapper


def compile(
    fn: Callable[..., Any],
    *args: Any,
    context: Interpreter | None = None,
    **kwargs: Any,
) -> TracedFunction:
    """Compile a function to get its IR without executing it.

    Args:
        fn: The function to compile.
        *args: Example arguments for tracing.
        context: Optional interpreter context. If None, uses current context.
        **kwargs: Example keyword arguments for tracing.

    Returns:
        TracedFunction with the compiled graph.

    Example:
        >>> with mp.make_simulator(3) as sim:
        ...     traced = mp.compile(job)  # uses sim from context
    """
    # If a context is explicitly provided, push it before tracing
    # so that _resolve_cluster() can find it.
    if context is not None:
        with context:
            return trace(fn, *args, **kwargs)

    # Otherwise, rely on the caller having pushed an interpreter context.
    # _resolve_cluster() will traverse the stack to find the interpreter.
    return trace(fn, *args, **kwargs)


# =============================================================================
# Public API
# =============================================================================
__all__ = [  # noqa: RUF022
    # Version
    "__version__",
    # Device API
    "ClusterSpec",
    "Device",
    "Node",
    "device",
    "get_dev_attr",
    "is_device_obj",
    "jax_fn",
    "put",
    "set_dev_attr",
    # Core EDSL
    "Graph",
    "GraphPrinter",
    "Object",
    "Operation",
    "Primitive",
    "TracedFunction",
    "Tracer",
    "Value",
    "compile",
    "evaluate",
    "fetch",
    "format_graph",
    "function",
    "get_current_context",
    "get_default_context",
    "jit",
    "mplang",
    "pop_context",
    "primitive",
    "push_context",
    "set_root_context",
    "trace",
    # Type system
    "MPType",
    "ScalarType",
    "SSType",
    "TableType",
    "TensorType",
    "VectorType",
    # Backend / Runtime
    "DRIVER_HANDLERS",
    "Interpreter",
    "LocalMesh",
    "SimpWorker",
    "WORKER_HANDLERS",
    "make_driver",
    "make_simulator",
    # Dialects
    "dialects",
    "register_default_context_factory",
    "get_root_context",
    "get_profiler",
]

# Register Interpreter as default context factory
register_default_context_factory(Interpreter)
