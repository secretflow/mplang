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

"""Interpreter: Execute Graph IR and Eager Operations.

Interpreter is a Context that executes operations immediately.
It can execute both:
1. Graph IR (via GraphInterpreter)
2. Eager operations on InterpObject (via backend executors)
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import os
import pathlib
import queue
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from mplang.v2.edsl.context import AbstractInterpreter
from mplang.v2.edsl.graph import Graph
from mplang.v2.edsl.object import Object
from mplang.v2.edsl.registry import get_impl
from mplang.v2.edsl.typing import BaseType
from mplang.v2.runtime.dialect_state import DialectState
from mplang.v2.runtime.object_store import ObjectStore

if TYPE_CHECKING:
    from mplang.v2.edsl.primitive import Primitive


class ExecutionTracer:
    """Tracer for DAG execution events (Chrome Tracing format)."""

    def __init__(self, enabled: bool = False, *, trace_dir: str | pathlib.Path):
        self.enabled = enabled
        self.start_time = 0.0
        self.end_time = 0.0
        self.active_tasks_samples: list[tuple[float, int]] = []
        self.queue_size_samples: list[tuple[float, int]] = []
        self.completed_ops = 0
        self.total_ops = 0
        self.trace_dir = pathlib.Path(trace_dir)

        # Tracing
        self.trace_events: list[dict[str, Any]] = []
        self.op_schedule_times: dict[
            tuple[int, Any], float
        ] = {}  # (id(op), namespace) -> ts (us)
        self.pid = os.getpid()

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self, filename_prefix: str = "dag_trace") -> None:
        self.end_time = time.time()
        self.save_trace(filename_prefix)

    def sample(self, active_tasks: int, queue_size: int) -> None:
        now = time.time() - self.start_time
        self.active_tasks_samples.append((now, active_tasks))
        self.queue_size_samples.append((now, queue_size))

    def log_schedule(self, op: Any, namespace: Any = None) -> None:
        if not self.enabled:
            return
        key = (id(op), namespace)
        self.op_schedule_times[key] = time.time() * 1e6

    def log_start(
        self, op: Any, pid: int | None = None, namespace: Any = None
    ) -> float:
        if not self.enabled:
            return 0.0
        start_ts = time.time() * 1e6
        if pid is None:
            pid = self.pid

        # Record scheduling latency (Queue Time)
        key = (id(op), namespace)
        if key in self.op_schedule_times:
            sched_ts = self.op_schedule_times.pop(key)
            self.trace_events.append({
                "name": f"Queue: {op.opcode}",
                "cat": "scheduler",
                "ph": "X",
                "ts": sched_ts,
                "dur": start_ts - sched_ts,
                "pid": pid,
                "tid": "SchedulerQueue",
            })
        return start_ts

    def log_end(self, op: Any, start_ts: float, pid: int | None = None) -> None:
        if not self.enabled:
            return
        end_ts = time.time() * 1e6
        tid = threading.get_ident()
        if pid is None:
            pid = self.pid

        self.trace_events.append({
            "name": op.opcode,
            "cat": "op",
            "ph": "X",
            "ts": start_ts,
            "dur": end_ts - start_ts,
            "pid": pid,
            "tid": tid,
            "args": {
                "opcode": op.opcode,
            },
        })

    def log_custom_event(
        self,
        name: str,
        start_ts: float,
        end_ts: float,
        cat: str = "custom",
        args: dict[str, Any] | None = None,
    ) -> None:
        """Log a custom event with explicit start/end timestamps (in seconds)."""
        if not self.enabled:
            return
        tid = threading.get_ident()

        # Convert to microseconds
        ts_us = start_ts * 1e6
        dur_us = (end_ts - start_ts) * 1e6

        self.trace_events.append({
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": ts_us,
            "dur": dur_us,
            "pid": self.pid,
            "tid": tid,
            "args": args or {},
        })

    def save_trace(
        self,
        filename_prefix: str = "dag_trace",
        job_id: str | None = None,
        rank: int | None = None,
    ) -> None:
        if not self.enabled or not self.trace_events:
            return
        try:
            if len(self.trace_events) < 100:
                return  # Skip small graphs

            # Use unique filename to avoid overwriting
            if job_id:
                # Format: trace_<job_id>_rank_<rank>.json
                rank_str = f"_rank_{rank}" if rank is not None else ""
                filename = f"trace_{job_id}{rank_str}.json"
            else:
                timestamp = int(time.time() * 1000)
                tid = threading.get_ident()
                filename = f"{filename_prefix}_{timestamp}_{tid}.json"

            # Save trace to trace_dir
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.trace_dir / filename

            with open(filepath, "w") as f:
                json.dump({"traceEvents": self.trace_events}, f)
            print(f"\n[Tracer] Trace saved to {filepath.absolute()}")
        except Exception as e:
            print(f"[Tracer] Failed to save trace: {e}")

    def print_summary(self) -> None:
        duration = self.end_time - self.start_time
        if duration <= 0:
            return

        avg_active = (
            sum(c for _, c in self.active_tasks_samples)
            / len(self.active_tasks_samples)
            if self.active_tasks_samples
            else 0
        )
        max_active = (
            max(c for _, c in self.active_tasks_samples)
            if self.active_tasks_samples
            else 0
        )
        avg_queue = (
            sum(c for _, c in self.queue_size_samples) / len(self.queue_size_samples)
            if self.queue_size_samples
            else 0
        )

        print("\n" + "=" * 80)
        print("DAG EXECUTION PROFILER")
        print("=" * 80)
        print(f"Total Duration: {duration:.3f}s")
        print(f"Total Ops:      {self.total_ops}")
        print(f"Throughput:     {self.total_ops / duration:.1f} ops/s")
        print("-" * 80)
        print(f"Active Tasks:   Avg={avg_active:.1f}, Max={max_active}")
        print(f"Ready Queue:    Avg={avg_queue:.1f}")
        print("=" * 80 + "\n")


class _NullTracer:
    """No-op tracer stub for when tracing is disabled."""

    enabled = False
    total_ops = 0

    def log_schedule(self, op: Any, namespace: Any = None) -> None:
        pass

    def log_start(
        self, op: Any, pid: int | None = None, namespace: Any = None
    ) -> float:
        return 0.0

    def log_end(self, op: Any, start_ts: float, pid: int | None = None) -> None:
        pass

    def stop(self) -> None:
        pass

    def save_trace(self, **kwargs: Any) -> None:
        pass


class InterpObject(Object):
    """Interp-time object (during eager execution).

    Holds a runtime object (the actual data/handle owned by the backend executor)
    and a reference to the Interpreter (Context).
    Operations delegate to primitives which execute immediately.

    The runtime object can be:
    - FHE backend: Local TenSEAL/SEAL ciphertext
    - JAX backend: Local jax.Array
    - MP backend: Backend handle (pointer to party-side data)
    - SQL backend: DatabaseHandle
    - etc.

    Example:
        >>> # FHE backend (local execution)
        >>> x = fhe.encrypt([1, 2, 3])  # InterpObject with local ciphertext
        >>> y = fhe.encrypt([4, 5, 6])
        >>> z = x + y  # InterpObject.__add__ → add_p.bind(x, y)

        >>> # MP backend (distributed execution)
        >>> x = mp.random.uniform(shape=(10,))  # InterpObject with backend handle
        >>> y = mp.random.uniform(shape=(10,))
        >>> z = x + y  # InterpObject.__add__ → add_p.bind(x, y)
    """

    def __init__(
        self,
        runtime_obj: Any,
        obj_type: BaseType,
        interpreter: Interpreter | None = None,
    ):
        """Initialize InterpObject.

        Args:
            runtime_obj: Backend-specific runtime object (ciphertext, array, handle, etc.)
            obj_type: Type of the object (BaseType from edsl.typing)
            interpreter: Interpreter context (if None, uses default interpreter)
        """
        self._runtime_obj = runtime_obj
        self._type = obj_type
        self._context = interpreter  # InterpObject holds its Interpreter (Context)

    @property
    def type(self) -> BaseType:
        return self._type

    @property
    def runtime_obj(self) -> Any:
        """Get the underlying runtime object (backend-specific)."""
        return self._runtime_obj

    def __repr__(self) -> str:
        runtime_repr = repr(self._runtime_obj)
        # Truncate long representations
        if len(runtime_repr) > 50:
            runtime_repr = runtime_repr[:47] + "..."
        return f"InterpObject({runtime_repr}, type={self.type})"


class Interpreter(AbstractInterpreter):
    """Execution context for eager execution.

    Inherits from Context and implements bind_primitive() by executing immediately.

    Responsibilities:
    1. Execute primitives on InterpObject immediately
    2. Delegate to backend-specific executors
    3. Execute Graph IR (via GraphInterpreter)

    Example:
        >>> interp = Interpreter()
        >>> x = InterpObject(np.array([1, 2, 3]), Tensor[f32, (3,)])
        >>> y = InterpObject(np.array([4, 5, 6]), Tensor[f32, (3,)])
        >>> z = x + y  # InterpObject.__add__ → add_p.bind(x, y)
    """

    def __init__(
        self,
        executor: concurrent.futures.Executor | None = None,
        name: str = "Interpreter",
        tracer: ExecutionTracer | None = None,
        trace_pid: int | None = None,
        store: ObjectStore | None = None,
        root_dir: str | pathlib.Path | None = None,
        handlers: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        # Persistence Root
        self.root_dir = (
            pathlib.Path(root_dir)
            if root_dir
            else pathlib.Path(os.environ.get("MPLANG_DATA_ROOT", ".mpl"))
        )

        # Initialize Context base class (for state management)
        super().__init__()

        # Instance-level handler registry (overrides global registry)
        self.handlers: dict[str, Callable] = handlers or {}
        self.tracer = tracer

        # GraphValue -> InterpObject cache
        # Maps a GraphValue (IR node) to its computed InterpObject (Runtime result).
        # This serves two purposes:
        # 1. Caching: Avoid re-evaluating the same graph node multiple times.
        # 2. MIMO Optimization: When one output of a multi-output op is computed,
        #    all sibling outputs are cached here to avoid re-execution.
        self._execution_cache: dict[Any, InterpObject] = {}
        self.executor = executor
        self.async_ops: set[str] = set()
        self.name = name
        self.trace_pid = trace_pid
        self.store: ObjectStore | None = store

    def shutdown(self) -> None:
        """Shutdown the interpreter and release resources.

        This method is idempotent and safe to call multiple times.
        It performs the following cleanup:
        1. Shuts down the internal executor (if any).
        2. Stops the execution tracer (if any).
        3. Shuts down any attached dialect states (e.g., stopping drivers).
        """
        # 1. Shutdown Executor
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        # 2. Stop Tracer
        if self.tracer:
            self.tracer.stop()
            # Don't clear self.tracer, as we might want to read stats later

        # 3. Shutdown Dialect States
        # Iterate over all attached states (e.g., drivers, cluster managers)
        # and shut them down if they support it.
        for state in self._states.values():
            if hasattr(state, "shutdown") and callable(state.shutdown):
                state.shutdown()

    # =========================================================================
    # Dialect State Management
    # =========================================================================
    def get_dialect_state(self, dialect: str) -> DialectState | None:
        """Get the state object for a specific dialect.

        This is a convenience wrapper around get_state("dialect.{dialect}").

        Args:
            dialect: Name of the dialect (e.g., "simp", "bfv", "spu")

        Returns:
            The dialect state object, or None if not set.

        Example:
            simp_state = interpreter.get_dialect_state("simp")
            if simp_state is not None:
                simp_state.submit(rank, graph, inputs)
        """
        state = self.get_state(f"dialect.{dialect}")
        # Type assertion: dialect states are always DialectState or None
        return cast(DialectState | None, state)

    def set_dialect_state(self, dialect: str, state: DialectState) -> None:
        """Set the state object for a specific dialect.

        This is a convenience wrapper around set_state("dialect.{dialect}", state).

        Args:
            dialect: Name of the dialect (e.g., "simp", "bfv", "spu")
            state: The dialect state object (should implement DialectState protocol)

        Example:
            interpreter.set_dialect_state("simp", cluster.connect())
        """
        self.set_state(f"dialect.{dialect}", state)

    def bind_primitive(
        self, primitive: Primitive, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> InterpObject | list[InterpObject] | Any:
        """Execute primitive by tracing and interpreting.

        Implements the unified trace → interpret flow:
        1. All InterpObject arguments already registered via lift()
        2. Create a Tracer and push it as context
        3. Call primitive.bind() to build Graph IR (uses obj id in value names)
        4. Execute the graph via evaluate_graph() (resolves inputs via registry)

        Args:
            primitive: The primitive to execute
            args: Positional arguments (already lifted by Primitive.bind)
            kwargs: Keyword arguments (already lifted by Primitive.bind)

        Returns:
            Execution result (InterpObject or list of InterpObject or mixed with immediates)
        """
        from mplang.v1.utils.func_utils import var_demorph, var_morph
        from mplang.v2.edsl.tracer import Tracer

        # Create tracer and build graph
        # Note: primitive.bind() internally calls Tracer.lift() with is_param=False,
        # so all args become captures (not params). This is correct because we're
        # tracing a primitive execution, not a user function with explicit parameters.
        with Tracer() as ctx:
            # Finalize graph by setting outputs
            result_traced = primitive.bind(*args, **kwargs)

            # Separate outputs into variables (Objects) and immediates (constants)
            out_vars, out_imms, morph_struct = var_morph(
                result_traced, lambda x: isinstance(x, Object)
            )

            if out_vars:
                graph = ctx.finalize(out_vars)
            else:
                # All outputs are immediates, no graph outputs
                graph = ctx.graph
                graph.outputs = []

        # Build inputs list for interpret
        # _captured_vars contains all inputs (no params in this context)
        inputs_list = [
            obj.runtime_obj if isinstance(obj, InterpObject) else obj
            for obj, _ in ctx._captured_vars.values()
        ]

        # Execute graph (may have 0 outputs if all were immediates)
        if graph.outputs:
            result_runtime_list = self.evaluate_graph(graph, inputs_list)
        else:
            result_runtime_list = []

        # Wrap runtime results as InterpObjects
        interp_results = [
            InterpObject(rt_val, tr_obj.type, self)
            for rt_val, tr_obj in zip(result_runtime_list, out_vars, strict=True)
        ]

        # Reconstruct the output tree: merge InterpObjects and immediates
        return var_demorph(interp_results, out_imms, morph_struct)

    def lift(self, obj: Any) -> InterpObject | Any:
        """Lift an object to the Interpreter's native representation.

        This is THE central method that manages the boundary between
        InterpObject and TraceObject:

        1. **InterpObject → TraceObject** (during nested tracing):
           - Register the InterpObject in self._objects for later resolution
           - The InterpObject must belong to this Interpreter
           - When the object flows into Tracer.lift() during bind_primitive,
             it will be captured as input with a clean SSA name like "%arg0"

        2. **TraceObject → InterpObject** (evaluate traced computation):
           - Extract the graph from the TraceObject's context (Tracer)
           - Execute the graph via evaluate_graph() to get runtime result
           - Wrap result as InterpObject and register it

        3. **Constants**: Pass through unchanged

        Args:
            obj: Object to lift (InterpObject, TraceObject, or constant)

        Returns:
            InterpObject (if Object input) or constant (pass-through)

        Example:
            >>> # InterpObject case
            >>> x = InterpObject(np.array([1, 2]), Tensor[f32, (2,)])
            >>> x_lifted = interp.lift(x)  # registers in _objects, returns x
            >>>
            >>> # TraceObject case
            >>> tracer = Tracer()
            >>> push_context(tracer)
            >>> z_trace = some_primitive.bind(x, y)  # TraceObject
            >>> pop_context()
            >>> interp = Interpreter()
            >>> z_interp = interp.lift(z_trace)  # evaluate graph → InterpObject
        """
        from mplang.v2.edsl.tracer import TraceObject

        if isinstance(obj, InterpObject):
            # InterpObject must belong to this interpreter
            if obj._context is not None and obj._context is not self:
                raise ValueError(
                    f"InterpObject belongs to a different Interpreter. "
                    f"Object context: {obj._context}, Current interpreter: {self}"
                )
            return obj

        elif isinstance(obj, TraceObject):
            # Check execution cache
            # If this value was computed as part of a previous execution (e.g. sibling output)
            # we can return it immediately without re-execution.
            graph_value = obj._graph_value
            if graph_value in self._execution_cache:
                return self._execution_cache[graph_value]

            # First time seeing this Value.
            # We need to execute the graph to compute it.
            # MIMO Optimization:
            # Instead of just asking for this single value, we ask for ALL outputs
            # of the operation that produced this value. This ensures that if we
            # later ask for a sibling output, it will be in the cache.

            tracer = obj._context
            graph = tracer.graph
            defining_op = graph_value.defining_op

            if defining_op is None:
                # Value is likely a constant or input (no defining op in graph)
                # Just execute graph for this single value
                target_outputs = [graph_value]
            else:
                # Fetch all outputs of the defining op
                target_outputs = defining_op.outputs

            # Temporarily set graph outputs to the target outputs
            # We must save/restore original outputs to avoid side effects
            original_outputs = graph.outputs
            graph.outputs = target_outputs

            try:
                # Resolve inputs from Tracer's captured vars
                # _captured_vars preserves insertion order which matches graph.inputs order
                inputs_list = []
                for captured_obj, _ in tracer._captured_vars.values():
                    # Recursively lift captured objects to ensure they are ready
                    lifted = self.lift(captured_obj)
                    if isinstance(lifted, InterpObject):
                        inputs_list.append(lifted.runtime_obj)
                    else:
                        inputs_list.append(lifted)

                # Execute graph
                results_runtime = self.evaluate_graph(graph, inputs_list)

                # Cache all results
                for val, res in zip(target_outputs, results_runtime, strict=True):
                    # Wrap as InterpObject and cache
                    # Note: We use obj.type for the requested value, but for siblings
                    # we should ideally use their types. However, we don't have TraceObjects
                    # for siblings here, only GraphValues.
                    # InterpObject needs a type. GraphValue has a type.
                    self._execution_cache[val] = InterpObject(res, val.type, self)

            finally:
                # Restore original outputs
                graph.outputs = original_outputs

            # Now the result for our requested object should be in the cache
            if graph_value not in self._execution_cache:
                raise RuntimeError(
                    f"Failed to compute value for {obj} even after graph execution"
                )

            return self._execution_cache[graph_value]

        else:
            # Constants: pass through unchanged
            return obj

    def evaluate_graph(
        self, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> list[Any]:
        """Execute a Graph IR with runtime data.

        Can be overridden by subclasses to implement remote execution or compilation.

        Args:
            graph: Finalized Graph IR to execute
            inputs: Runtime objects corresponding to graph.inputs (positional)
            job_id: Optional unique ID for this execution job (for profiling/tracing).

        Returns:
            List of runtime execution results corresponding to graph.outputs.
        """
        if self.executor:
            return self._evaluate_graph_async(graph, inputs, job_id)
        else:
            return self._evaluate_graph_sync(graph, inputs, job_id)

    def _evaluate_graph_sync(
        self, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> list[Any]:
        """Synchronous execution (Baseline)."""
        # Local environment: Value -> Runtime Object
        env = dict(zip(graph.inputs, inputs, strict=True))

        for op in graph.operations:
            # Resolve inputs
            try:
                args = [env[val] for val in op.inputs]
            except KeyError as e:
                missing_keys = [str(k) for k in op.inputs if k not in env]
                # Limit available keys output to avoid flooding logs if env is huge
                available_keys = [str(k) for k in list(env.keys())[:20]]
                if len(env) > 20:
                    available_keys.append("...")

                raise RuntimeError(
                    f"Failed to resolve inputs for op '{op.opcode}'.\n"
                    f"Missing values: {missing_keys}\n"
                    f"Available values (partial): {available_keys}"
                ) from e

            # Dispatch
            # 1. Check instance-level handlers
            handler = self.handlers.get(op.opcode)
            # 2. Check global registry
            if not handler:
                handler = get_impl(op.opcode)

            if handler:
                # Pass interpreter to support recursive execution (HOFs)
                # Pass op to access attributes and regions
                # Pass args as runtime values
                results = handler(self, op, *args)
            else:
                raise NotImplementedError(
                    f"No implementation registered for opcode: {op.opcode}"
                )

            # Update environment with outputs
            # Handler should return a single value or a tuple/list of values
            if len(op.outputs) == 0:
                pass  # Void operation
            elif len(op.outputs) == 1:
                env[op.outputs[0]] = results
            else:
                if len(results) != len(op.outputs):
                    raise RuntimeError(
                        f"Op {op.opcode} returned {len(results)} values, expected {len(op.outputs)}"
                    )
                for out_val, res in zip(op.outputs, results, strict=True):
                    env[out_val] = res

        # Return outputs
        if self.tracer and job_id:
            self.tracer.save_trace(job_id=job_id, rank=self.trace_pid)

        return [env[out] for out in graph.outputs]

    def _evaluate_graph_async(
        self, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> list[Any]:
        """Asynchronous execution with non-blocking DAG scheduling."""
        # Tracer setup (if not provided, use a disabled stub)
        tracer: ExecutionTracer | _NullTracer
        if self.tracer:
            tracer = self.tracer
            tracer.total_ops += len(graph.operations)
        else:
            # No tracer provided - use minimal stub (no trace_dir needed)
            tracer = _NullTracer()

        active_tasks = 0

        # 1. Setup State
        # Value -> Runtime Object (initially inputs)
        env = dict(zip(graph.inputs, inputs, strict=True))

        # Op -> Pending Input Count
        pending_counts = {}
        # Value -> list[Op] (Consumers)
        value_to_consumers: dict[Any, list[Any]] = collections.defaultdict(list)
        # Value -> Remaining Consumers Count (for GC)
        remaining_consumers: dict[Any, int] = collections.defaultdict(int)

        # 2. Build Dependency Graph
        for op in graph.operations:
            count = 0
            for val in op.inputs:
                if val not in env:  # If not already resolved (input or constant)
                    value_to_consumers[val].append(op)
                    remaining_consumers[val] += 1
                    count += 1
            pending_counts[op] = count

        # Mark graph outputs as having an extra consumer (the user)
        # so they are not GC'd before return
        for out in graph.outputs:
            remaining_consumers[out] += 1

        # 3. Synchronization
        lock = threading.Lock()
        ready_queue: queue.Queue[Any] = queue.Queue()
        remaining_ops = len(graph.operations)

        # Error propagation
        error_occurred = False

        # 4. Execution Helper
        def on_op_done(op: Any, result: Any, error: Exception | None = None) -> None:
            nonlocal remaining_ops, error_occurred, active_tasks

            if error:
                with lock:
                    if not error_occurred:
                        error_occurred = True
                        ready_queue.put(error)
                return

            with lock:
                if op.opcode in self.async_ops and self.executor:
                    active_tasks -= 1
                    # profiler.sample(active_tasks, ready_queue.qsize())

                if error_occurred:
                    return

                # Store results
                if len(op.outputs) == 1:
                    env[op.outputs[0]] = result
                else:
                    for out_val, res in zip(op.outputs, result, strict=True):
                        env[out_val] = res

                # Trigger consumers
                for out_val in op.outputs:
                    if out_val in value_to_consumers:
                        for consumer_op in value_to_consumers[out_val]:
                            pending_counts[consumer_op] -= 1
                            if pending_counts[consumer_op] == 0:
                                tracer.log_schedule(
                                    consumer_op, namespace=self.trace_pid
                                )
                                ready_queue.put(consumer_op)

                # GC Inputs
                for val in op.inputs:
                    if val in remaining_consumers:
                        remaining_consumers[val] -= 1
                        if remaining_consumers[val] == 0:
                            env.pop(val, None)

                remaining_ops -= 1
                if remaining_ops == 0:
                    ready_queue.put(None)  # Sentinel

        def execute_op(op: Any) -> None:
            nonlocal active_tasks
            # Extract args from env (must be ready)
            args = [env[val] for val in op.inputs]

            handler = self.handlers.get(op.opcode)
            if not handler:
                handler = get_impl(op.opcode)

            if not handler:
                raise NotImplementedError(
                    f"No implementation registered for opcode: {op.opcode}"
                )

            if op.opcode in self.async_ops and self.executor:
                with lock:
                    active_tasks += 1
                    # profiler.sample(active_tasks, ready_queue.qsize())

                # Submit to executor
                def task() -> Any:
                    start_ts = tracer.log_start(
                        op, pid=self.trace_pid, namespace=self.trace_pid
                    )
                    res = handler(self, op, *args)
                    tracer.log_end(op, start_ts, pid=self.trace_pid)
                    return res

                def callback(fut: Any) -> None:
                    try:
                        res = fut.result()
                        on_op_done(op, res)
                    except Exception as e:
                        on_op_done(op, None, error=e)

                fut = self.executor.submit(task)
                fut.add_done_callback(callback)
            else:
                # Sync execution (run immediately)
                try:
                    start_ts = tracer.log_start(
                        op, pid=self.trace_pid, namespace=self.trace_pid
                    )
                    res = handler(self, op, *args)
                    tracer.log_end(op, start_ts, pid=self.trace_pid)
                    on_op_done(op, res)
                except Exception as e:
                    on_op_done(op, None, error=e)

        # 5. Initial Submission
        # Submit all ops with 0 pending inputs
        initial_ops = [op for op, count in pending_counts.items() if count == 0]
        if not initial_ops and remaining_ops > 0:
            # Cycle detected or empty graph?
            pass

        for op in initial_ops:
            tracer.log_schedule(op, namespace=self.trace_pid)
            ready_queue.put(op)

        # Handle empty graph case
        if remaining_ops == 0:
            ready_queue.put(None)

        # 6. Main Loop
        while True:
            item = ready_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            # It's an op
            execute_op(item)

        # 7. Return outputs
        if not self.tracer:
            tracer.stop()

        if self.tracer and job_id:
            self.tracer.save_trace(job_id=job_id, rank=self.trace_pid)

        return [env[out] for out in graph.outputs]
