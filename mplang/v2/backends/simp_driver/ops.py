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

"""Simp Driver ops (HOST_HANDLERS).

This module contains all simp operation implementations for the Driver Interpreter.
These implementations dispatch work to workers and collect results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from mplang.v2.backends.simp_driver.values import HostVar
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import simp
from mplang.v2.edsl.graph import Graph, Operation, Value
from mplang.v2.runtime.interpreter import InterpObject

if TYPE_CHECKING:
    pass


def _get_client_context(interpreter: Any) -> Any:
    """Get the simp dialect state from interpreter."""
    state = interpreter.get_dialect_state("simp")
    if state is None:
        raise RuntimeError("Interpreter must have simp dialect state attached")
    return state


def _pcall_static_host_impl(
    interpreter: Any, op: Operation, *args: Any
) -> Any:
    """Driver implementation of pcall_static (SPMD dispatch)."""

    parties = op.attrs["parties"]
    fn_graph = op.regions[0]
    futures = []

    host = _get_client_context(interpreter)

    for rank in parties:
        party_inputs = []
        for arg in args:
            if isinstance(arg, HostVar):
                party_inputs.append(arg[rank])
            else:
                party_inputs.append(arg)

        wrapper_graph = Graph()

        wrapper_inputs = []
        for original_in in fn_graph.inputs:
            new_in = Value(name=original_in.name, type=original_in.type)
            wrapper_graph.inputs.append(new_in)
            wrapper_inputs.append(new_in)
            wrapper_graph.values[new_in.name] = new_in

        out_types = [val.type for val in fn_graph.outputs]
        op_outputs = [Value(name=f"wrapper_out_{i}", type=t) for i, t in enumerate(out_types)]
        for v in op_outputs:
            wrapper_graph.values[v.name] = v

        pcall_op = Operation(
            opcode=op.opcode,
            inputs=wrapper_inputs,
            outputs=op_outputs,
            attrs=op.attrs.copy(),
            regions=[fn_graph]
        )
        for v in op_outputs:
            v.defining_op = pcall_op

        wrapper_graph.operations.append(pcall_op)
        wrapper_graph.outputs = op_outputs

        futures.append(host.submit(rank, wrapper_graph, party_inputs))

    results_flat = host.collect(futures)

    num_outputs = len(op.regions[0].outputs)
    if num_outputs == 0:
        return None

    world_size = host.world_size
    output_vars = []
    for _ in range(num_outputs):
        output_vars.append([None] * world_size)

    for idx, rank in enumerate(parties):
        party_res = results_flat[idx]
        if num_outputs == 1:
            output_vars[0][rank] = party_res
        else:
            for out_idx in range(num_outputs):
                output_vars[out_idx][rank] = party_res[out_idx]

    final_results = [HostVar(values) for values in output_vars]
    if len(final_results) == 1:
        return final_results[0]
    return final_results


def _pcall_dynamic_host_impl(
    interpreter: Any, op: Operation, *args: Any
) -> Any:
    """Driver implementation of pcall_dynamic (SPMD dispatch)."""
    host = _get_client_context(interpreter)
    world_size = host.world_size
    parties = tuple(range(world_size))
    fn_graph = op.regions[0]
    futures = []

    for rank in parties:
        party_inputs = []
        for arg in args:
            if isinstance(arg, HostVar):
                party_inputs.append(arg[rank])
            else:
                party_inputs.append(arg)
        futures.append(host.submit(rank, fn_graph, party_inputs))

    results_flat = host.collect(futures)

    num_outputs = len(op.regions[0].outputs)
    if num_outputs == 0:
        return None

    if num_outputs == 1:
        return HostVar(results_flat)
    else:
        transposed = []
        for i in range(num_outputs):
            transposed.append(HostVar([res[i] for res in results_flat]))
        return transposed


def _shuffle_static_host_impl(
    interpreter: Any, op: Operation, src: Any
) -> Any:
    """Driver implementation of shuffle (Dispatch to workers)."""
    from mplang.v2.edsl.typing import CustomType

    if not isinstance(src, HostVar):
        raise TypeError(f"shuffle input must be HostVar on Host, got {type(src)}")

    g = Graph()
    any_type = CustomType("Any")
    in_val = Value(name="shuffle_in", type=any_type)
    g.inputs.append(in_val)

    types = [any_type]
    op_out_values = [Value(name="shuffle_out", type=t) for t in types]

    shuffle_op = Operation(
        opcode=op.opcode,
        inputs=[in_val],
        outputs=op_out_values,
        attrs=op.attrs,
        regions=[],
    )
    for v in op_out_values:
        v.defining_op = shuffle_op

    g.operations.append(shuffle_op)
    g.outputs.append(op_out_values[0])
    g.values[in_val.name] = in_val
    for v in op_out_values:
        g.values[v.name] = v

    host = _get_client_context(interpreter)
    world_size = host.world_size
    futures = []

    for rank in range(world_size):
        if rank < len(src.values):
            party_inp = [src.values[rank]]
        else:
            party_inp = [None]

        futures.append(host.submit(rank, g, party_inp))

    results_flat = host.collect(futures)

    return HostVar(results_flat)


def _converge_host_impl(
    interpreter: Any, op: Operation, *args: Any
) -> Any:
    """Driver implementation of converge (Merge disjoint HostVars)."""
    if not args:
        return None

    world_size = args[0].world_size
    merged = [None] * world_size

    for arg in args:
        if not isinstance(arg, HostVar):
            raise TypeError(f"converge host impl expects HostVar, got {type(arg)}")

        if arg.world_size != world_size:
            raise ValueError("World size mismatch in converge inputs")

        for i, val in enumerate(arg.values):
            if val is not None:
                if merged[i] is not None:
                    raise ValueError(f"converge collision at rank {i}")
                merged[i] = val

    return HostVar(merged)


def _uniform_cond_host_impl(
    ctx: Any,
    op: Any,
    pred: HostVar,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Driver implementation for uniform_cond (Distributed IF)."""

    if not isinstance(pred, HostVar):
        if isinstance(pred, bool):
            val = pred
        else:
            raise TypeError(f"uniform_cond predicate must be HostVar or bool, got {type(pred)}")
    else:
        val_idx = next((i for i, v in enumerate(pred.values) if v is not None), -1)
        if val_idx == -1:
            raise ValueError("uniform_cond predicate has no values")

        val = pred.values[val_idx]

        if isinstance(val, str) and val.startswith("mem://"):
            state = ctx.get_dialect_state("simp")
            # Use abstract SimpDriver interface - all drivers support fetch
            if hasattr(state, "fetch"):
                future = state.fetch(val_idx, val)
                val = future.result()

        if isinstance(val, TensorValue):
            val = val.unwrap()
        elif isinstance(val, InterpObject):
            val = val.runtime_obj

        if isinstance(val, (jnp.ndarray, np.ndarray, np.generic)):
            val = val.item()

    branch_inputs = list(args)

    if val:
        if not hasattr(op, "regions") or len(op.regions) != 2:
            raise ValueError("uniform_cond op expects 2 regions")
        region = op.regions[0]
        return ctx.evaluate_graph(region, branch_inputs)
    else:
        if not hasattr(op, "regions") or len(op.regions) != 2:
            raise ValueError("uniform_cond op expects 2 regions")
        region = op.regions[1]
        return ctx.evaluate_graph(region, branch_inputs)


def _constant_host_impl(
    interpreter: Any, op: Operation, parties: tuple[int, ...], value: Any
) -> Any:
    """Driver implementation of simp.constant."""
    host = _get_client_context(interpreter)
    world_size = host.world_size

    data = [None] * world_size
    for rank in parties:
        data[rank] = value

    return HostVar(data)


HOST_HANDLERS = {
    simp.pcall_static_p.name: _pcall_static_host_impl,
    simp.pcall_dynamic_p.name: _pcall_dynamic_host_impl,
    simp.shuffle_static_p.name: _shuffle_static_host_impl,
    simp.converge_p.name: _converge_host_impl,
    simp.uniform_cond_p.name: _uniform_cond_host_impl,
}
