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

import jax.numpy as jnp
import numpy as np

from mplang.v2.backends.simp_client import ThreadingClient
from mplang.v2.backends.simp_structs import HostVar
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import simp
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import InterpObject, Interpreter

if TYPE_CHECKING:
    pass


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _ensure_worker_context(interpreter: Any, op_name: str) -> Any:
    """Validate that interpreter has a Worker context."""
    context = getattr(interpreter, "context", None)
    if context and hasattr(context, "communicator"):
        return context

    raise RuntimeError(
        f"{op_name} requires Worker context (communicator), got {type(context)}"
    )


def _get_client_context(interpreter: Any) -> Any:
    """Get the Client context from interpreter."""
    context = getattr(interpreter, "context", None)
    if context and hasattr(context, "submit"):
        return context

    # Fallback for legacy tests or if context IS the client (duck typing)
    if hasattr(interpreter, "submit"):
        return interpreter
    raise RuntimeError(f"Interpreter context must support 'submit' for Host ops. Got: {context}")


# ------------------------------------------------------------------------------
# Implementations
# ------------------------------------------------------------------------------

def _pcall_static_host_impl(
    interpreter: Any, op: Operation, *args: Any
) -> Any:
    """Host implementation of pcall_static (SPMD dispatch)."""
    # Import locally to avoid circular dep
    from mplang.v2.backends.simp_structs import HostVar
    from mplang.v2.edsl.graph import Graph, Operation, Value

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

        # Wrap fn_graph in a pcall_static op to ensure context (current_parties) is set on worker
        # This is critical for ops like spu.exec that need to know the parties

        wrapper_graph = Graph()

        # Create inputs for wrapper graph matching fn_graph inputs
        wrapper_inputs = []
        for original_in in fn_graph.inputs:
            new_in = Value(name=original_in.name, type=original_in.type)
            new_in.provenance = None
            wrapper_graph.inputs.append(new_in)
            wrapper_inputs.append(new_in)
            wrapper_graph.values[new_in.name] = new_in

        # Create output values based on fn_graph output types
        out_types = [val.type for val in fn_graph.outputs]
        op_outputs = [Value(name=f"wrapper_out_{i}", type=t) for i, t in enumerate(out_types)]
        for v in op_outputs:
            wrapper_graph.values[v.name] = v

        # Create the pcall_static op
        # We reuse fn_graph as the region. Note: This might modify parent pointers of fn_graph
        # but since we are serializing immediately/soon for dispatch, it should be fine.
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
    """Host implementation of pcall_dynamic (SPMD dispatch)."""
    from mplang.v2.backends.simp_structs import HostVar

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
    """Host implementation of shuffle (Dispatch to workers)."""
    from mplang.v2.backends.simp_structs import HostVar
    from mplang.v2.edsl.graph import Graph, Operation, Value
    from mplang.v2.edsl.typing import CustomType

    if not isinstance(src, HostVar):
        raise TypeError(f"shuffle input must be HostVar on Host, got {type(src)}")

    # Construct a 1-op graph: [input] -> shuffle -> [output]
    g = Graph()
    any_type = CustomType("Any")
    in_val = Value(name="shuffle_in", type=any_type)
    in_val.provenance = None
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
    # Dispatch execution
    host = _get_client_context(interpreter)
    world_size = host.world_size
    futures = []

    for rank in range(world_size):
        # Input for this rank
        if rank < len(src.values):
            party_inp = [src.values[rank]]
        else:
            party_inp = [None]

        futures.append(host.submit(rank, g, party_inp))

    results_flat = host.collect(futures)

    # Results are the new handles (URIs) on the receiving parties
    return HostVar(results_flat)


# ------------------------------------------------------------------------------
# Worker Implementations
# ------------------------------------------------------------------------------

def _pcall_static_worker_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Worker implementation of pcall_static."""
    worker = _ensure_worker_context(interpreter, "pcall_static_impl")

    parties = op.attrs.get("parties")
    if parties is None:
        raise ValueError("pcall_static requires 'parties' attribute")

    if worker.rank in parties:
        # Execute region
        fn_graph = op.regions[0]
        # Inject parties info into interpreter for downstream ops (e.g. spu.exec)
        prev_parties = getattr(interpreter, "current_parties", None)
        interpreter.current_parties = parties  # type: ignore[attr-defined]
        print(f"DEBUG: pcall_static_worker_impl set current_parties={parties} on interp {id(interpreter)}")

        try:
            return interpreter.evaluate_graph(fn_graph, list(args))
        finally:
            if prev_parties is None:
                del interpreter.current_parties  # type: ignore[attr-defined]
            else:
                interpreter.current_parties = prev_parties
    else:
        # Return dummy values (None) for each output
        if len(op.outputs) == 1:
            return None
        else:
            return [None] * len(op.outputs)


def _pcall_dynamic_worker_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Worker implementation of pcall_dynamic."""
    fn_graph = op.regions[0]
    return interpreter.evaluate_graph(fn_graph, list(args))


def _shuffle_static_worker_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Worker implementation of shuffle_static."""
    worker = _ensure_worker_context(interpreter, "shuffle_static_impl")

    routing = op.attrs.get("routing")
    if routing is None:
        return args[0]  # Identity if no routing

    comm = worker.communicator
    my_rank = worker.rank
    data = args[0]

    # Send phase
    for tgt, src in routing.items():
        if src == my_rank and tgt != my_rank:
            key = f"shuffle_{op.name}_{tgt}"
            comm.send(tgt, key, data)

    # Recv phase
    if my_rank in routing:
        src = routing[my_rank]
        if src == my_rank:
            return data  # Local copy
        key = f"shuffle_{op.name}_{my_rank}"
        return comm.recv(src, key)
    else:
        return None  # Not a target


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

    # TODO: Implement AllReduce verification if verify_uniform is True
    if op.attrs.get("verify_uniform", True):
        pass

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

        # Execute condition
        cond_res = interpreter.evaluate_graph(cond_graph, region_inputs)

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


# ------------------------------------------------------------------------------
# Handler Collections
# ------------------------------------------------------------------------------

def _converge_host_impl(
    interpreter: Any, op: Operation, *args: Any
) -> Any:
    """Host implementation of converge (Merge disjoint HostVars)."""
    from mplang.v2.backends.simp_structs import HostVar

    if not args:
        return None

    world_size = args[0].world_size
    merged = [None] * world_size

    for arg in args:
        if not isinstance(arg, HostVar):
            # Try to handle non-HostVar (unlikely in SIMP Host context for MP types)
            # But converge might be called with concrete values?
            # For now assume HostVar.
            raise TypeError(f"converge host impl expects HostVar, got {type(arg)}")

        if arg.world_size != world_size:
            raise ValueError("World size mismatch in converge inputs")

        for i, val in enumerate(arg.values):
            if val is not None:
                if merged[i] is not None:
                    # Overlap detected!
                    # For converge, we expect disjoint inputs.
                    # However, if we are just merging partial views, maybe it is fine?
                    # But primitive spec says "disjoint".
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
    """Host implementation for uniform_cond (Distributed IF)."""
    # Predicate must be a HostVar with boolean values.
    # We assume "uniform" means all parties agree (replicated).
    # We check the first party's value.
    if not isinstance(pred, HostVar):
        # Fallback if passed a raw bool?
        if isinstance(pred, bool):
            val = pred
        else:
            raise TypeError(f"uniform_cond predicate must be HostVar or bool, got {type(pred)}")
    else:
        # Check all values match? Or just take first non-None?
        # Replicated invariant implies they match.
        val_idx = next((i for i, v in enumerate(pred.values) if v is not None), -1)
        if val_idx == -1:
            raise ValueError("uniform_cond predicate has no values")

        val = pred.values[val_idx]

        # If val is URI, fetch it
        if isinstance(val, str) and val.startswith("mem://"):
            if isinstance(ctx.context, ThreadingClient):
                future = ctx.context.fetch(val_idx, val)
                val = future.result()
            else:
                # If we can't fetch, maybe we are simulating locally and store allows access?
                # But standard path requires context.fetch.
                # Fallback: Check if store has explicit backdoor?
                # Default to assumption it's a value if context missing.
                pass

        # Unwrap TensorValue if needed (e.g. from fetch)
        if isinstance(val, (TensorValue, InterpObject)):
            val = val.data  # type: ignore[union-attr]

        # JAX/Numpy scalar unwrap
        if isinstance(val, (jnp.ndarray, np.ndarray, np.generic)):
            val = val.item()

    # args[0] is pred (passed as named arg 'pred', also in *args? No, interpreter passes positional)
    # Wait, interpreter passes *args corresponds to op.inputs.
    # op.inputs has [pred, arg1, arg2...]
    # So *args = (pred, arg1, arg2...)
    # But python matches 'pred' to first arg?
    # Signature: _uniform_cond_host_impl(ctx, op, pred, *args)
    # So 'pred' eats first arg. '*args' has the rest.
    # So branch_inputs = args.

    branch_inputs = list(args)

    if val:
        # True branch -> Region 0
        if not hasattr(op, "regions") or len(op.regions) != 2:
            raise ValueError(f"uniform_cond op expects 2 regions, got {len(op.regions) if hasattr(op, 'regions') else 'None'}")

        region = op.regions[0]
        # Execute region graph
        # We must use ctx.evaluate_graph
        # Region inputs match branch_inputs.
        # region is likely the Graph object itself
        return ctx.evaluate_graph(region, branch_inputs)
    else:
        # False branch -> Region 1
        if not hasattr(op, "regions") or len(op.regions) != 2:
            raise ValueError("uniform_cond op expects 2 regions")

        region = op.regions[1]
        return ctx.evaluate_graph(region, branch_inputs)


def _constant_host_impl(
    interpreter: Any, op: Operation, parties: tuple[int, ...], value: Any
) -> Any:
    """Host implementation of simp.constant."""
    from mplang.v2.backends.simp_structs import HostVar

    # We assume usage in Host context implies creating a HostVar (Replicated or Distributed).
    # 'value' is the constant value.
    # 'parties' indicates where this constant is valid.

    host = _get_client_context(interpreter)
    world_size = host.world_size

    data = [None] * world_size
    for rank in parties:
        # We store the value directly.
        # Note: If value is a large array, this duplicates it on Host.
        # But HostVar is just a container.
        data[rank] = value

    return HostVar(data)


HOST_HANDLERS = {
    simp.pcall_static_p.name: _pcall_static_host_impl,
    simp.pcall_dynamic_p.name: _pcall_dynamic_host_impl,
    simp.shuffle_static_p.name: _shuffle_static_host_impl,
    simp.converge_p.name: _converge_host_impl,
    simp.uniform_cond_p.name: _uniform_cond_host_impl,
}

WORKER_HANDLERS = {
    simp.pcall_static_p.name: _pcall_static_worker_impl,
    simp.pcall_dynamic_p.name: _pcall_dynamic_worker_impl,
    simp.shuffle_static_p.name: _shuffle_static_worker_impl,
    simp.converge_p.name: _converge_worker_impl,
    simp.uniform_cond_p.name: _uniform_cond_worker_impl,
    simp.while_loop_p.name: _while_loop_worker_impl,
}

# Keep global registration for backward compatibility or default fallback
# Using Host Impl as default for Dispatch ops, but really it should be based on handlers.
# Since we are moving to Explicit handlers, we can remove these decorators.
# If users rely on global registry, they will break unless they provide handlers.
# For now, let's leave them UNREGISTERED by default in this file, forcing usage via handlers.
