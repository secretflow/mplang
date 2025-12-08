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

"""SIMP operation implementations for WorkerInterpreter.

This module registers all simp.* operation implementations. It should be
imported by worker entry points (simp_simulator, simp_http_worker) to
trigger registration.

Note: These implementations assume they run inside WorkerInterpreter.
They should NOT be used directly from Host-side code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mplang.v2.dialects import simp
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import Interpreter

if TYPE_CHECKING:
    from mplang.v2.backends.simp_worker import WorkerInterpreter


def _ensure_worker_interpreter(
    interpreter: Interpreter, op_name: str
) -> WorkerInterpreter:
    """Validate that interpreter is a WorkerInterpreter.

    Args:
        interpreter: The interpreter instance
        op_name: Operation name for error message

    Returns:
        The interpreter cast to WorkerInterpreter

    Raises:
        RuntimeError: If interpreter is not a WorkerInterpreter
    """
    from mplang.v2.backends.simp_worker import WorkerInterpreter

    if not isinstance(interpreter, WorkerInterpreter):
        raise RuntimeError(
            f"{op_name} must be run in WorkerInterpreter, got {type(interpreter)}"
        )
    return interpreter


@simp.pcall_static_p.def_impl
def pcall_static_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.pcall_static (Worker side).

    Executes the region if this worker's rank is in the parties list.
    Otherwise returns None (or list of Nones for multi-output).
    """
    worker = _ensure_worker_interpreter(interpreter, "pcall_static_impl")

    parties = op.attrs.get("parties")
    if parties is None:
        raise ValueError("pcall_static requires 'parties' attribute")

    if worker.rank in parties:
        # Execute region
        fn_graph = op.regions[0]
        # Inject parties info into interpreter for downstream ops (e.g. spu.exec)
        prev_parties = getattr(worker, "current_parties", None)
        worker.current_parties = parties  # type: ignore[attr-defined]

        try:
            return worker.evaluate_graph(fn_graph, list(args))
        finally:
            if prev_parties is None:
                del worker.current_parties  # type: ignore[attr-defined]
            else:
                worker.current_parties = prev_parties  # type: ignore[attr-defined]
    else:
        # Return dummy values (None) for each output
        if len(op.outputs) == 1:
            return None
        else:
            return [None] * len(op.outputs)


@simp.pcall_dynamic_p.def_impl
def pcall_dynamic_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.pcall_dynamic (Worker side).

    Dynamic pcall runs on all parties. Each party executes the region.
    """
    fn_graph = op.regions[0]
    return interpreter.evaluate_graph(fn_graph, list(args))


@simp.shuffle_static_p.def_impl
def shuffle_static_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.shuffle_static (Worker side).

    Performs point-to-point communication based on the routing dict.
    routing: {target_rank: source_rank}
    """
    worker = _ensure_worker_interpreter(interpreter, "shuffle_static_impl")

    routing = op.attrs.get("routing")
    if routing is None:
        return args[0]  # Identity if no routing

    comm = worker.communicator
    my_rank = worker.rank
    data = args[0]

    # Send phase: if I am a source, send to targets
    for tgt, src in routing.items():
        if src == my_rank and tgt != my_rank:
            key = f"shuffle_{op.name}_{tgt}"
            comm.send(tgt, key, data)

    # Recv phase: if I am a target, receive from source
    if my_rank in routing:
        src = routing[my_rank]
        if src == my_rank:
            return data  # Local copy
        key = f"shuffle_{op.name}_{my_rank}"
        return comm.recv(src, key)
    else:
        return None  # Not a target


@simp.converge_p.def_impl
def converge_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.converge (Worker side).

    Merges disjoint partitions by returning the first non-None input.
    In SPMD execution, each worker has exactly one non-None input.
    """
    for arg in args:
        if arg is not None:
            return arg
    return None


@simp.uniform_cond_p.def_impl
def uniform_cond_impl(
    interpreter: Interpreter, op: Operation, pred: Any, *args: Any
) -> Any:
    """Implementation of simp.uniform_cond (Worker side).

    Executes the selected branch based on the predicate value.
    Assumes the predicate is uniform across all parties.
    """
    from mplang.v2.backends.tensor_impl import TensorValue

    # TODO: Implement AllReduce verification if verify_uniform is True
    if op.attrs.get("verify_uniform", True):
        pass

    # Unwrap TensorValue if needed to get Python bool
    if isinstance(pred, TensorValue):
        pred = bool(pred.unwrap())

    if pred:
        return interpreter.evaluate_graph(op.regions[0], list(args))
    else:
        return interpreter.evaluate_graph(op.regions[1], list(args))


@simp.while_loop_p.def_impl
def while_loop_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.while_loop (Worker side).

    Executes the loop body while the condition is true.
    """
    from mplang.v2.backends.tensor_impl import TensorValue

    cond_graph = op.regions[0]
    body_graph = op.regions[1]

    num_state = len(op.outputs)
    current_state = list(args[:num_state])
    captures = list(args[num_state:])

    while True:
        region_inputs = current_state + captures

        # Execute condition
        cond_res = interpreter.evaluate_graph(cond_graph, region_inputs)

        # Unwrap TensorValue if needed to get Python bool
        if isinstance(cond_res, TensorValue):
            cond_res = bool(cond_res.unwrap())

        if not cond_res:
            break

        # Execute body
        body_res = interpreter.evaluate_graph(body_graph, region_inputs)
        if isinstance(body_res, list):
            current_state = body_res
        else:
            current_state = [body_res]

    if len(current_state) == 1:
        return current_state[0]
    return current_state
