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

"""Simp Driver ops (DRIVER_HANDLERS).

Unified SPMD dispatch pattern for all SIMP operations.
All ops: wrap → dispatch to ALL workers → collect DriverVar(s).
Op-specific logic lives in Worker handlers (simp_worker/ops.py).
"""

from __future__ import annotations

from typing import Any

from mplang.v2.backends.simp_driver.values import DriverVar
from mplang.v2.dialects import simp
from mplang.v2.edsl.graph import Graph, Operation
from mplang.v2.edsl.typing import CustomType


def _get_driver_context(interpreter: Any) -> Any:
    """Get the simp driver state from interpreter."""
    state = interpreter.get_dialect_state("simp")
    if state is None:
        raise RuntimeError("Interpreter must have simp dialect state attached")
    return state


def _wrap_op_as_graph(op: Operation) -> Graph:
    """Wrap an Operation into a single-op Graph for worker submission."""
    g = Graph()
    any_type = CustomType("Any")

    # Create graph inputs
    graph_inputs = [g.add_input(f"in_{i}", any_type) for i in range(len(op.inputs))]

    # Determine output types
    output_types = [out.type for out in op.outputs] if op.outputs else [any_type]

    # Add the operation (this handles outputs and value registration)
    g.add_op(
        opcode=op.opcode,
        inputs=graph_inputs,
        output_types=output_types,
        attrs=op.attrs.copy(),
        regions=op.regions,
    )

    # Mark outputs
    for v in g.operations[-1].outputs:
        g.add_output(v)

    return g


def _collect_to_hostvars(results: list[Any], num_outputs: int, world_size: int) -> Any:
    """Collect worker results into DriverVar(s).

    Args:
        results: List of results from each worker (length = world_size).
                 Each result is a list of URIs (one per output).
        num_outputs: Number of outputs per worker
        world_size: Total number of workers

    Returns:
        Single DriverVar if num_outputs == 1, else list of DriverVars
    """
    if num_outputs == 0:
        return None

    # Transpose [worker][output] -> [output][worker]
    # results[worker_idx] is a list of URIs for that worker's outputs
    transposed = []
    for out_idx in range(num_outputs):
        transposed.append(
            DriverVar([res[out_idx] if res is not None else None for res in results])
        )

    if num_outputs == 1:
        return transposed[0]
    return transposed


def _generic_simp_dispatch(interpreter: Any, op: Operation, *args: Any) -> Any:
    """Unified SIMP dispatch: wrap op, SPMD submit, collect DriverVar(s).

    This is the ONLY driver handler needed for all SIMP ops.
    Worker handlers implement the actual op-specific logic.
    """
    driver = _get_driver_context(interpreter)
    world_size = driver.world_size

    # 1. Wrap operation into a Graph
    wrapper_graph = _wrap_op_as_graph(op)

    # 2. SPMD dispatch to ALL workers
    futures = []
    for rank in range(world_size):
        # Extract per-party inputs from DriverVars
        party_inputs = [
            arg[rank] if isinstance(arg, DriverVar) else arg for arg in args
        ]
        futures.append(driver.submit(rank, wrapper_graph, party_inputs))

    # 3. Collect results
    results = driver.collect(futures)

    # 4. Assemble into DriverVar(s)
    num_outputs = len(op.outputs) if op.outputs else 1
    return _collect_to_hostvars(results, num_outputs, world_size)


# =============================================================================
# All SIMP ops use unified dispatch
# =============================================================================

DRIVER_HANDLERS = {
    simp.pcall_static_p.name: _generic_simp_dispatch,
    simp.pcall_dynamic_p.name: _generic_simp_dispatch,
    simp.shuffle_static_p.name: _generic_simp_dispatch,
    simp.converge_p.name: _generic_simp_dispatch,
    simp.uniform_cond_p.name: _generic_simp_dispatch,
    simp.while_loop_p.name: _generic_simp_dispatch,
}
