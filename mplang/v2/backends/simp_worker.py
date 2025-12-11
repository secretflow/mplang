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

"""SIMP worker module.

Provides WorkerInterpreter for executing Graph IR on a single party.
This module only defines the interpreter class - operation implementations
are registered in simp_impl.py.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import mplang.v2.backends.field_impl  # noqa: F401
import mplang.v2.backends.tensor_impl  # noqa: F401
from mplang.v2.edsl.graph import Graph
from mplang.v2.runtime.interpreter import ExecutionTracer, Interpreter
from mplang.v2.runtime.object_store import ObjectStore


class WorkerInterpreter(Interpreter):
    """Interpreter running on a single party (worker).

    Attributes:
        rank: The global rank of this worker.
        world_size: Total number of workers.
        communicator: Communication backend for shuffle operations.
        spu_endpoints: Optional dict mapping global_rank -> BRPC endpoint for SPU.
                       If None, SPU uses in-memory link (simulation mode).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        communicator: Any,
        spu_endpoints: dict[int, str] | None = None,
        tracer: ExecutionTracer | None = None,
        *,
        root_dir: str | pathlib.Path,
    ):
        store = ObjectStore(fs_root=str(pathlib.Path(root_dir) / "store"))
        super().__init__(
            name=f"Worker-{rank}",
            tracer=tracer,
            trace_pid=rank,
            store=store,
            root_dir=root_dir,
        )
        self.rank = rank
        self.world_size = world_size
        self.communicator: Any = communicator
        self.spu_endpoints = spu_endpoints

        # Enable multi-threaded execution for BFV ops by default
        import concurrent.futures

        max_workers = os.cpu_count() or 4
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.async_ops = {
            "bfv.add",
            "bfv.mul",
            "bfv.rotate",
            "bfv.batch_encode",
            "bfv.relinearize",
            "bfv.encrypt",
            "bfv.decrypt",
            # Field Ops (Heavy Computation)
            "field.solve_okvs",
            "field.decode_okvs",
            "field.aes_expand",
            "field.mul",
            # Communication Ops (I/O Overlap)
            "simp.shuffle",
        }

    def execute_job(
        self, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Execute graph and return URIs to results."""
        # 1. Resolve inputs (URI -> Value)
        resolved_inputs = []
        for inp in inputs:
            if isinstance(inp, str) and "://" in inp:
                resolved_inputs.append(self.store.get(inp))
            else:
                resolved_inputs.append(inp)

        # 2. Execute
        results = self.evaluate_graph(graph, resolved_inputs, job_id)

        # 3. Store results (Value -> URI)
        if not graph.outputs:
            return None

        if len(graph.outputs) == 1:
            # Single result
            return self.store.put(results)
        else:
            # List of results
            return [self.store.put(res) for res in results]
