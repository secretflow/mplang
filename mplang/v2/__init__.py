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

import os
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Self

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
from mplang.v2.backends.simp_client import HttpClient, ThreadingClient
from mplang.v2.backends.simp_impl import HOST_HANDLERS, WORKER_HANDLERS
from mplang.v2.backends.simp_local_comm import LocalMesh
from mplang.v2.backends.simp_worker import SimpWorkerContext
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
from mplang.v2.runtime.interpreter import Interpreter, interpret

# Register Interpreter as default context factory
register_default_context_factory(Interpreter)

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


# =============================================================================
# Compatibility layer: Simulator class
# =============================================================================
class Simulator:
    """Simulator compatible with mplang v1 API.

    Usage:
        sim = Simulator.simple(2)  # 2-party simulation
        result = evaluate(sim, my_function)
        value = fetch(sim, result)
    """

    def __init__(self, cluster_spec: ClusterSpec, enable_tracing: bool = False):
        """Create a Simulator from a ClusterSpec."""
        from mplang.v2.runtime.interpreter import ExecutionTracer
        from mplang.v2.runtime.object_store import ObjectStore

        self._cluster = cluster_spec
        world_size = len(cluster_spec.nodes)

        # Construct root_dir from cluster_id
        data_root = pathlib.Path(os.environ.get("MPLANG_DATA_ROOT", ".mpl"))
        cluster_root = data_root / cluster_spec.cluster_id
        host_root = cluster_root / "__host__"

        # 1. Create Local Mesh (Orchestrator)
        self.mesh = LocalMesh(world_size)

        # 2. Create Execution Tracer
        self.tracer = ExecutionTracer(
            enabled=enable_tracing, trace_dir=host_root / "trace"
        )
        self.tracer.start()

        # 3. Create Workers
        self.workers: list[Interpreter] = []
        for rank in range(world_size):
            worker_root = cluster_root / f"node{rank}"
            store = ObjectStore(fs_root=str(worker_root / "store"))

            # Pure Context object
            context = SimpWorkerContext(
                rank=rank,
                world_size=world_size,
                communicator=self.mesh.comms[rank],
                store=store,
            )

            # Pure Interpreter instance
            w_handlers = {**WORKER_HANDLERS}
            w_interp = Interpreter(
                name=f"Worker-{rank}",
                tracer=self.tracer,
                trace_pid=rank,
                store=store,
                root_dir=worker_root,
                context=context,
                handlers=w_handlers,
            )
            # Default async ops config for workers
            w_interp.async_ops = {
                "bfv.add", "bfv.mul", "bfv.rotate", "bfv.batch_encode",
                "bfv.relinearize", "bfv.encrypt", "bfv.decrypt",
                "field.solve_okvs", "field.decode_okvs", "field.aes_expand", "field.mul",
                "simp.shuffle",
            }
            # Do NOT share executor with mesh. Workers need their own thread pool
            # for async operations to avoid deadlock when mesh executor is saturated.
            # w_interp.executor = self.mesh.executor

            self.workers.append(w_interp)

        # 4. Create Client Context for Host
        self._client_ctx = ThreadingClient(
            world_size=world_size,
            context=self.mesh,
            root_dir=host_root,
        )
        self._client_ctx.workers = self.workers  # Link workers to client for local exec

        # 5. Create Host Interpreter
        h_handlers = {**HOST_HANDLERS}
        self.interpreter = Interpreter(
            name="HostInterpreter",
            root_dir=host_root,
            context=self._client_ctx,
            handlers=h_handlers,
            tracer=self.tracer,
        )

        set_global_cluster(cluster_spec)

    @classmethod
    def simple(cls, world_size: int, **kwargs: Any) -> Simulator:
        """Create a simple simulator with the given number of parties."""
        enable_tracing = kwargs.pop("enable_tracing", False)
        cluster = ClusterSpec.simple(
            world_size,
            enable_ppu_device=kwargs.pop("enable_ppu_device", True),
            enable_spu_device=kwargs.pop("enable_spu_device", True),
            **kwargs,
        )
        return cls(cluster, enable_tracing=enable_tracing)

    @property
    def cluster(self) -> ClusterSpec:
        """Get the cluster specification."""
        return self._cluster

    @property
    def backend(self) -> Interpreter:
        """Get the underlying Interpreter (Host)."""
        return self.interpreter

    @property
    def client_ctx(self) -> ThreadingClient:
        """Get the simulation client context."""
        return self._client_ctx

    def fetch(self, obj: Any) -> Any:
        """Fetch data from the simulator."""
        return self.client_ctx.fetch(obj.rank, obj.uri).result()

    def __enter__(self) -> Self:
        """Enter context: push interpreter."""
        push_context(self.interpreter)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context: pop interpreter."""
        pop_context()
        # Do not shutdown mesh here, as 'evaluate' uses 'with sim:' which would kill
        # the simulator for subsequent calls (like fetch).
        # User should call sim.shutdown() explicitly if needed, or rely on GC/atexit.

    def shutdown(self) -> None:
        """Shutdown the simulator and release resources."""
        self.mesh.shutdown(wait=False)


class Driver:
    """Driver for distributed execution compatible with mplang v1 API."""

    def __init__(self, cluster_spec: ClusterSpec):
        """Create a Driver from a ClusterSpec."""
        self._cluster = cluster_spec
        world_size = len(cluster_spec.nodes)

        # Construct root_dir from cluster_id
        data_root = pathlib.Path(os.environ.get("MPLANG_DATA_ROOT", ".mpl"))
        host_root = data_root / cluster_spec.cluster_id / "__host__"

        # Ensure endpoints have http:// prefix
        endpoints = []
        for node in cluster_spec.nodes.values():
            ep = node.endpoint
            if not ep.startswith("http://") and not ep.startswith("https://"):
                ep = f"http://{ep}"
            endpoints.append(ep)

        # 1. Create Client Context
        self.client_ctx = HttpClient(
            world_size=world_size,
            endpoints=endpoints,
        )

        # 2. Create Host Interpreter
        self.interpreter = Interpreter(
            name="HostInterpreter",
            root_dir=host_root,
            context=self.client_ctx,
            handlers=HOST_HANDLERS,
        )
        set_global_cluster(cluster_spec)

    @classmethod
    def simple(cls, cluster_spec: ClusterSpec | list[str] | dict | None = None, **kwargs: Any) -> Driver:
        """Create a simple driver with default settings.

        Args:
            cluster_spec: The cluster specification, or list of endpoints, or dict.
            **kwargs: Additional keyword arguments to pass to the Driver.

        Returns:
            A Driver instance.
        """
        if cluster_spec is None:
            cluster_spec = ClusterSpec.simple()
        elif isinstance(cluster_spec, list):
            # Assume list of endpoints
            cluster_spec = ClusterSpec.simple(world_size=len(cluster_spec), endpoints=cluster_spec)
        elif isinstance(cluster_spec, dict):
            cluster_spec = ClusterSpec.from_dict(cluster_spec)

        return cls(cluster_spec, **kwargs)

    @property
    def cluster(self) -> ClusterSpec:
        """Get the cluster specification."""
        return self._cluster

    @property
    def backend(self) -> Interpreter:
        """Get the underlying Host Interpreter."""
        return self.interpreter

    def fetch(self, obj: Any) -> Any:
        """Fetch data from the driver."""
        return self.client_ctx.fetch(obj.rank, obj.uri).result()

    def __enter__(self) -> Self:
        """Enter context: push driver as the default interpreter."""
        push_context(self.interpreter)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context: pop the driver."""
        pop_context()

    def shutdown(self) -> None:
        """Shutdown the driver and release resources."""
        if hasattr(self.client_ctx, "shutdown"):
            self.client_ctx.shutdown()


def evaluate(
    sim: Simulator | Driver,
    fn: Callable[..., Any] | TracedFunction,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Evaluate a function using the simulator or driver.

    Compatible with mplang v1 API: mp.evaluate(sim, fn)

    Args:
        sim: The Simulator or Driver instance.
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

    with sim:
        if isinstance(fn, TracedFunction):
            inputs = fn.prepare_inputs(*args, **kwargs)
            # Unwrap InterpObject at execution boundary
            inputs = [unwrap_if_interp(v) for v in inputs]
            interpreter = sim.backend
            raw_result = interpreter.evaluate_graph(fn.graph, inputs)
            return fn.reconstruct_outputs(raw_result)

        return fn(*args, **kwargs)


def fetch(sim: Simulator | Driver, result: Any, party: int | str | None = None) -> Any:
    """Fetch the result from the simulator or driver.

    This version handles fetching specific parties from HostVars.
    """
    from mplang.v2.backends.simp_structs import HostVar
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

    if hasattr(sim, "fetch"):
        # For HostVar, we need to iterate and fetch each
        if isinstance(result, HostVar):
            resolved_values = []
            for rank, val in enumerate(result.values):
                if isinstance(val, str) and "://" in val:
                    # Fetch from remote using sim.client_ctx
                    # sim.fetch(obj) assumes InterpObject? No, 'sim.fetch' in our new class
                    # doesn't handle rank/uri pair easily.
                    # Let's use sim.client_ctx direct access for efficiency here.
                    fut = sim.client_ctx.fetch(rank, val)
                    resolved_values.append(fut.result())
                else:
                    resolved_values.append(val)

            # Select party if needed
            if party is not None:
                if isinstance(party, str):
                    device_info = sim.cluster.devices.get(party)
                    if device_info and device_info.members:
                        party = device_info.members[0].rank
                    else:
                        raise ValueError(f"Unknown party: {party}")
                return _unwrap_value(resolved_values[party])
            return [_unwrap_value(v) for v in resolved_values]

    # Simulator didn't support fetch? Or fallback.
    if isinstance(result, HostVar):
        values = result.values
        if party is not None:
            # ... party resolution ...
            if isinstance(party, str):
                device_info = sim.cluster.devices.get(party)
                if device_info and device_info.members:
                    party = device_info.members[0].rank
            return _unwrap_value(values[party])  # type: ignore[index]
        return [_unwrap_value(v) for v in values]

    # Unwrap Value types to get the underlying data
    return _unwrap_value(result)


# Alias for compatibility
function = jit  # @mp.function -> @mp2.function (JIT compilation)


def compile(
    sim: Simulator | Driver, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> TracedFunction:
    """Compile a function to get its IR without executing it."""
    # Set up cluster context, then trace
    set_global_cluster(sim.cluster)
    return trace(fn, *args, **kwargs)


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Device API
    "ClusterSpec",
    "Device",
    "Driver",
    # Core EDSL
    "Graph",
    "GraphPrinter",
    "Interpreter",
    # Type system
    "MPType",
    "Node",
    "Object",
    "Operation",
    "Primitive",
    "SSType",
    "ScalarType",
    # Runtime
    "Simulator",
    "TableType",
    "TensorType",
    "TracedFunction",
    "Tracer",
    "Value",
    "VectorType",
    # Version
    "__version__",
    "compile",
    "device",
    # Dialects
    "dialects",
    "evaluate",
    "fetch",
    "format_graph",
    "function",
    "get_current_context",
    "get_default_context",
    "get_dev_attr",
    "get_global_cluster",
    "interpret",
    "is_device_obj",
    "jax_fn",
    "jit",
    "mplang",
    "pop_context",
    "primitive",
    "push_context",
    "put",
    "set_dev_attr",
    "set_global_cluster",
    "trace",
]
