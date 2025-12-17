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
    find_context,
    find_context_with_state,
    find_interpreter,
    format_graph,
    get_current_context,
    get_default_context,
    is_tracing,
    jit,
    pop_context,
    primitive,
    push_context,
    register_default_context_factory,
    set_root_context,
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
from mplang.v2.libs.device import fetch as device_fetch
from mplang.v2.runtime.interpreter import Interpreter

# =============================================================================
# Context Management API (JAX-like pattern)
# =============================================================================


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
            wrapped = [
                InterpObject(v, fn.graph.outputs[i].type, interp)
                for i, v in enumerate(raw_result)
            ]
            return fn.reconstruct_outputs(wrapped)

        return fn(*args, **kwargs)


def fetch(
    result: Any,
    *,
    follow_device: bool = True,
    context: Interpreter | None = None,
) -> Any:
    """Fetch results from interpreter context to Python.

    This is a meta-function that operates at execution boundaries, not a traced
    dialect operation. It brings data from the distributed/MPC runtime back to
    the Python host.

    Behavior in different contexts:
    - **Tracing (compile)**: Returns the input unchanged (identity). The graph
      outputs are determined by the function's return statement, not fetch calls.
    - **Execution (evaluate)**: Actually fetches data from workers/parties.

    Design Note (A vs B tradeoff):
        Two designs were considered for fetch behavior during tracing:

        - **Design A (chosen)**: fetch = identity during tracing. Graph outputs
          are determined solely by the return statement. This is simpler and
          avoids ambiguity when fetch and return reference different values.

        - **Design B (alternative)**: fetch marks output points in the graph.
          This would allow fetch(a), fetch(b), return b to output both a and b.
          However, it complicates the semantics and requires tracking fetch
          points separately from return values.

        Design A was chosen for simplicity. If a value needs to be an output,
        it should be returned. fetch's role is purely for execution-time I/O.

    Args:
        result: Object(s) to fetch. Can be a single InterpObject, DriverVar,
            or nested structure containing them.
        follow_device: If True and object has device attribute, dispatch to
            device.fetch which fetches from the correct rank based on device.
            If False, fetch from all parties.
        context: Interpreter context. If None, uses current context.

    Returns:
        Fetched Python values. For device objects with follow_device=True,
        returns single value from the device's rank(s). Otherwise returns
        list of values (one per party) or single value for world_size=1.
        During tracing, returns the input unchanged.
    """
    from jax.tree_util import tree_map

    from mplang.v2.backends.simp_driver.values import DriverVar
    from mplang.v2.edsl.context import is_tracing
    from mplang.v2.runtime.interpreter import InterpObject
    from mplang.v2.runtime.value import WrapValue

    # Check if we are in tracing context - if so, return identity
    if is_tracing():
        # Design A: fetch = identity during tracing
        # Graph outputs are determined by return statement, not fetch calls
        return result

    # Execution context - actually fetch data
    interp = _get_context(context)

    def _fetch_single(var: Any) -> Any:
        """Fetch a single value from InterpObject."""
        # InterpObject (from mp.evaluate) - extract runtime_obj
        if isinstance(var, InterpObject):
            if follow_device and is_device_obj(var):
                return device_fetch(var)
            var = var.runtime_obj  # extract and continue processing

        # DriverVar (simp dialect) - remote fetch from workers
        if isinstance(var, DriverVar):
            from mplang.v2.backends.simp_driver.state import SimpDriver

            simp_state = interp.get_dialect_state("simp")
            assert isinstance(simp_state, SimpDriver), "DriverVar requires simp state"

            resolved: list[Any] = []
            for rank, uri in enumerate(var.values):
                if uri is None:
                    resolved.append(None)
                else:
                    fetched = simp_state.fetch(rank, uri).result()
                    if isinstance(fetched, WrapValue):
                        fetched = fetched.data
                    resolved.append(fetched)

            return resolved[0] if len(resolved) == 1 else resolved

        # WrapValue (TensorValue, TableValue, etc.) - unwrap
        if isinstance(var, WrapValue):
            return var.data

        # Plain values pass through
        return var

    with interp:
        return tree_map(_fetch_single, result)


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
    "find_context",
    "find_context_with_state",
    "find_interpreter",
    "format_graph",
    "function",
    "get_current_context",
    "get_default_context",
    "is_tracing",
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
    "get_profiler",
]

# Register Interpreter as default context factory
register_default_context_factory(Interpreter)
