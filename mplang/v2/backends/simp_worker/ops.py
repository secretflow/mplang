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

"""Simp Worker ops (WORKER_HANDLERS).

This module contains all simp operation implementations for the Worker Interpreter.
These implementations execute locally on a single party.
"""

from __future__ import annotations

from typing import Any

from mplang.v2.dialects import simp
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import Interpreter


def _ensure_worker_context(interpreter: Any, op_name: str) -> Any:
    """Validate that interpreter has a Worker context."""
    state = interpreter.get_dialect_state("simp")
    if state is None or not hasattr(state, "communicator"):
        raise RuntimeError(f"{op_name} requires simp Worker state (with communicator)")
    return state


def _pcall_static_worker_impl(
    interpreter: Interpreter, op: Operation, *args: Any
) -> Any:
    """Worker implementation of pcall_static."""
    worker = _ensure_worker_context(interpreter, "pcall_static_impl")

    parties = op.attrs.get("parties")
    if parties is None:
        raise ValueError("pcall_static requires 'parties' attribute")

    if worker.rank in parties:
        fn_graph = op.regions[0]
        prev_parties = getattr(interpreter, "current_parties", None)
        interpreter.current_parties = parties  # type: ignore[attr-defined]

        try:
            return interpreter.evaluate_graph(fn_graph, list(args))
        finally:
            if prev_parties is None:
                del interpreter.current_parties  # type: ignore[attr-defined]
            else:
                interpreter.current_parties = prev_parties  # type: ignore[attr-defined]
    else:
        if len(op.outputs) == 1:
            return None
        else:
            return [None] * len(op.outputs)


def _pcall_dynamic_worker_impl(
    interpreter: Interpreter, op: Operation, *args: Any
) -> Any:
    """Worker implementation of pcall_dynamic."""
    fn_graph = op.regions[0]
    return interpreter.evaluate_graph(fn_graph, list(args))


def _shuffle_static_worker_impl(
    interpreter: Interpreter, op: Operation, *args: Any
) -> Any:
    """Worker implementation of shuffle_static."""
    worker = _ensure_worker_context(interpreter, "shuffle_static_impl")

    routing = op.attrs.get("routing")
    if routing is None:
        return args[0]

    comm = worker.communicator
    my_rank = worker.rank
    data = args[0]

    for tgt, src in routing.items():
        if src == my_rank and tgt != my_rank:
            key = f"shuffle_{op.name}_{tgt}"
            comm.send(tgt, key, data)

    if my_rank in routing:
        src = routing[my_rank]
        if src == my_rank:
            return data
        key = f"shuffle_{op.name}_{my_rank}"
        return comm.recv(src, key)
    else:
        return None


def _converge_worker_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Worker implementation of simp.converge."""
    for arg in args:
        if arg is not None:
            return arg
    return None


def _uniform_cond_worker_impl(
    interpreter: Interpreter, op: Operation, pred: Any, *args: Any
) -> Any:
    """Worker implementation of simp.uniform_cond."""
    from mplang.v2.backends.tensor_impl import TensorValue

    if op.attrs.get("verify_uniform", True):
        pass  # TODO: Implement AllReduce verification

    if isinstance(pred, TensorValue):
        pred = bool(pred.unwrap())

    if pred:
        return interpreter.evaluate_graph(op.regions[0], list(args))
    else:
        return interpreter.evaluate_graph(op.regions[1], list(args))


def _while_loop_worker_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Worker implementation of simp.while_loop."""
    from mplang.v2.backends.tensor_impl import TensorValue

    cond_graph = op.regions[0]
    body_graph = op.regions[1]

    num_state = len(op.outputs)
    current_state = list(args[:num_state])
    captures = list(args[num_state:])

    while True:
        region_inputs = current_state + captures

        cond_res = interpreter.evaluate_graph(cond_graph, region_inputs)

        if isinstance(cond_res, TensorValue):
            cond_res = bool(cond_res.unwrap())

        if not cond_res:
            break

        body_res = interpreter.evaluate_graph(body_graph, region_inputs)
        if isinstance(body_res, list):
            current_state = body_res
        else:
            current_state = [body_res]

    if len(current_state) == 1:
        return current_state[0]
    return current_state


WORKER_HANDLERS = {
    simp.pcall_static_p.name: _pcall_static_worker_impl,
    simp.pcall_dynamic_p.name: _pcall_dynamic_worker_impl,
    simp.shuffle_static_p.name: _shuffle_static_worker_impl,
    simp.converge_p.name: _converge_worker_impl,
    simp.uniform_cond_p.name: _uniform_cond_worker_impl,
    simp.while_loop_p.name: _while_loop_worker_impl,
}
