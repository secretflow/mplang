"""MPLang2: Next generation EDSL for multi-party computation.

This is the temporary home for the new EDSL implementation during migration.
Once migration is complete, this will replace the original mplang package.

Public API is designed to be compatible with mplang v1 where possible.
"""

__version__ = "0.1.0"

# =============================================================================
# Core EDSL components
# =============================================================================
# =============================================================================
# Dialects
# =============================================================================
from mplang.v2 import dialects

# =============================================================================
# Backend / Runtime
# =============================================================================
from mplang.v2.backends.simp_http_driver import SimpHttpDriver
from mplang.v2.backends.simp_simulator import SimpSimulator
from mplang.v2.edsl import (
    Graph,
    GraphPrinter,
    Interpreter,
    Object,
    Operation,
    Primitive,
    TracedFunction,
    Tracer,
    Value,
    format_graph,
    get_current_context,
    get_default_context,
    interpret,
    jit,
    pop_context,
    primitive,
    push_context,
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


# =============================================================================
# Compatibility layer: Simulator class (wraps SimpSimulator with mplang v1 API)
# =============================================================================
class Simulator:
    """Simulator compatible with mplang v1 API.

    Usage:
        sim = Simulator.simple(2)  # 2-party simulation
        result = evaluate(sim, my_function)
        value = fetch(sim, result)
    """

    def __init__(self, cluster_spec: ClusterSpec):
        """Create a Simulator from a ClusterSpec."""
        self._cluster = cluster_spec
        self._sim = SimpSimulator(world_size=len(cluster_spec.nodes))
        set_global_cluster(cluster_spec)

    @classmethod
    def simple(cls, world_size: int, **kwargs) -> "Simulator":
        """Create a simple simulator with the given number of parties.

        Args:
            world_size: Number of parties (physical nodes).
            **kwargs: Additional arguments passed to ClusterSpec.simple().

        Returns:
            A Simulator instance.
        """
        cluster = ClusterSpec.simple(
            world_size,
            enable_ppu_device=kwargs.pop("enable_ppu_device", True),
            enable_spu_device=kwargs.pop("enable_spu_device", True),
            **kwargs,
        )
        return cls(cluster)

    @property
    def cluster(self) -> ClusterSpec:
        """Get the cluster specification."""
        return self._cluster

    @property
    def backend(self) -> SimpSimulator:
        """Get the underlying SimpSimulator backend."""
        return self._sim

    def __enter__(self) -> "Simulator":
        """Enter context: push simulator as the default interpreter."""
        push_context(self._sim)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: pop the simulator."""
        pop_context()


class Driver:
    """Driver for distributed execution compatible with mplang v1 API.

    Connects to a running cluster of workers via HTTP and executes
    multi-party computations in a distributed manner.

    Usage:
        driver = Driver(cluster_spec)
        result = evaluate(driver, my_function)
        value = fetch(driver, result)

    Note:
        Before using Driver, you must start the worker servers.
        See mplang.v2.backends.simp_http for worker implementation.
    """

    def __init__(self, cluster_spec: ClusterSpec):
        """Create a Driver from a ClusterSpec.

        Args:
            cluster_spec: The cluster specification with node endpoints.
        """
        self._cluster = cluster_spec
        # Ensure endpoints have http:// prefix
        endpoints = []
        for node in cluster_spec.nodes.values():
            ep = node.endpoint
            if not ep.startswith("http://") and not ep.startswith("https://"):
                ep = f"http://{ep}"
            endpoints.append(ep)
        self._driver = SimpHttpDriver(
            world_size=len(cluster_spec.nodes),
            endpoints=endpoints,
        )
        set_global_cluster(cluster_spec)

    @property
    def cluster(self) -> ClusterSpec:
        """Get the cluster specification."""
        return self._cluster

    @property
    def backend(self) -> SimpHttpDriver:
        """Get the underlying SimpHttpDriver backend."""
        return self._driver

    def __enter__(self) -> "Driver":
        """Enter context: push driver as the default interpreter."""
        push_context(self._driver)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: pop the driver."""
        pop_context()

    def shutdown(self) -> None:
        """Shutdown the driver and release resources."""
        self._driver.shutdown()


def evaluate(sim: "Simulator | Driver", fn, *args, **kwargs):
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
    with sim:
        return fn(*args, **kwargs)


def fetch(sim: "Simulator | Driver", result, party: int | str | None = None):
    """Fetch the result from the simulator or driver.

    Compatible with mplang v1 API: mp.fetch(sim, result)

    For mplang.v2, since we use eager execution in simulation mode,
    results are typically already available as concrete values.

    Args:
        sim: The Simulator instance.
        result: The result object to fetch.
        party: Optional party index or name to fetch from (for HostVar).
               If None, returns values from all parties as a list.

    Returns:
        The concrete Python value, or list of values from all parties if party is None.
    """
    from mplang.v2.backends.simp_host import HostVar
    from mplang.v2.edsl.interpreter import InterpObject

    # Unwrap InterpObject to get the runtime value
    if isinstance(result, InterpObject):
        result = result.runtime_obj

    if isinstance(result, HostVar):
        if party is not None:
            if isinstance(party, str):
                # Look up party by name (e.g., "P0" -> rank 0)
                device_info = sim.cluster.devices.get(party)
                if device_info and device_info.members:
                    party = device_info.members[0].rank
                else:
                    raise ValueError(f"Unknown party: {party}")
            return result[party]
        # Return all parties' values as a list
        return result.values
    # Already a concrete value
    return result


# Alias for compatibility
function = jit  # @mp.function -> @mp2.function (JIT compilation)


def compile(sim: "Simulator | Driver", fn, *args, **kwargs) -> TracedFunction:
    """Compile a function to get its IR without executing it.

    Compatible with mplang v1 API: mp.compile(sim, fn)

    Args:
        sim: The Simulator or Driver instance (provides cluster context).
        fn: The function to compile.
        *args: Arguments to pass during tracing.
        **kwargs: Keyword arguments to pass during tracing.

    Returns:
        TracedFunction: The traced function with inspectable IR.

    Example:
        traced = compile(sim, my_fn)
        print(traced.compiler_ir())
    """
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
    "SimpHttpDriver",
    "SimpSimulator",
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
    "pop_context",
    "primitive",
    "push_context",
    "put",
    "set_dev_attr",
    "set_global_cluster",
    "trace",
]
