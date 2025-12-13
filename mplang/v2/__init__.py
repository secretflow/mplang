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
from mplang.v2.backends.simp_driver.ops import HOST_HANDLERS
from mplang.v2.backends.simp_worker import SimpWorker
from mplang.v2.backends.simp_worker.mem import LocalMesh
from mplang.v2.backends.simp_worker.ops import WORKER_HANDLERS
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
    jit,
    pop_context,
    primitive,
    push_context,
    register_default_context_factory,
    trace,
)

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
    get_global_cluster,
    is_device_obj,
    jax_fn,
    put,
    set_dev_attr,
    set_global_cluster,
)
from mplang.v2.runtime.interpreter import Interpreter, interpret


def evaluate(
    interp: Interpreter,
    fn: Callable[..., Any] | TracedFunction,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Evaluate a function using the interpreter.

    Args:
        interp: The Interpreter instance.
        fn: The function to evaluate.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function evaluation.
    """
    from mplang.v2.edsl.tracer import TracedFunction
    from mplang.v2.runtime.interpreter import InterpObject

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


def fetch(interp: Interpreter, result: Any, party: int | str | None = None) -> Any:
    """Fetch the result from the interpreter.

    This version handles fetching specific parties from HostVars.
    """
    from typing import cast

    from mplang.v2.backends.simp_driver.base import SimpDriver
    from mplang.v2.backends.simp_driver.values import HostVar
    from mplang.v2.backends.table_impl import TableValue
    from mplang.v2.backends.tensor_impl import TensorValue
    from mplang.v2.runtime.interpreter import InterpObject

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

    # Fetch from HostVar
    if isinstance(result, HostVar):
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
function = jit  # @mp.function -> @mp2.function (JIT compilation)


def compile(
    interp: Interpreter, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> TracedFunction:
    """Compile a function to get its IR without executing it."""
    cluster_spec = getattr(interp, "_cluster_spec", None)
    if cluster_spec is not None:
        set_global_cluster(cluster_spec)
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
    "get_global_cluster",
    "is_device_obj",
    "jax_fn",
    "put",
    "set_dev_attr",
    "set_global_cluster",
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
    "interpret",
    "jit",
    "mplang",
    "pop_context",
    "primitive",
    "push_context",
    "trace",
    # Type system
    "MPType",
    "ScalarType",
    "SSType",
    "TableType",
    "TensorType",
    "VectorType",
    # Backend / Runtime
    "HOST_HANDLERS",
    "Interpreter",
    "LocalMesh",
    "SimpWorker",
    "WORKER_HANDLERS",
    # Dialects
    "dialects", "register_default_context_factory",
]

# Register Interpreter as default context factory
register_default_context_factory(Interpreter)
