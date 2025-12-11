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

"""SIMP host base module."""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
import uuid
from typing import Any

from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Graph
from mplang.v2.runtime.interpreter import Interpreter
from mplang.v2.runtime.value import Value


@serde.register_class
class HostVar(Value):
    """Runtime value for SIMP dialect holding values for all parties."""

    _serde_kind = "simp.HostVar"

    def __init__(self, values: list[Any]):
        self.values = values

    def __repr__(self) -> str:
        return f"HostVar({self.values})"

    def __getitem__(self, rank: int) -> Any:
        return self.values[rank]

    def to_json(self) -> dict[str, Any]:
        return {"values": serde.to_json(self.values)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HostVar:
        return cls(values=serde.from_json(data["values"]))


class SimpHost(Interpreter):
    """Base class for SIMP host interpreters.

    This interpreter runs on the coordinator (Host) and dispatches
    SPMD operations to workers.
    """

    def __init__(self, world_size: int, root_dir: pathlib.Path | None = None):
        super().__init__(root_dir=root_dir)
        self.world_size = world_size

    def evaluate_graph(
        self, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Execute graph by distributing it to all parties."""
        if job_id is None:
            job_id = str(uuid.uuid4())

        # Optionally save graph to file for debugging
        if os.environ.get("MPLANG_PRINT_GRAPH", "").lower() in ("1", "true", "yes"):
            if self.root_dir is not None:
                from mplang.v2.edsl.printer import format_graph

                graphs_dir = self.root_dir / "graphs"
                graphs_dir.mkdir(parents=True, exist_ok=True)
                graph_file = graphs_dir / f"graph_{job_id}.txt"
                graph_file.write_text(format_graph(graph))
                print(f"[Graph] Saved to {graph_file}")

        # inputs: list of runtime objects (HostVar or constant), matching graph.inputs order

        futures = []
        for rank in range(self.world_size):
            # Prepare inputs for this rank
            party_inputs = []
            for runtime_obj in inputs:
                if isinstance(runtime_obj, HostVar):
                    party_inputs.append(runtime_obj[rank])
                else:
                    party_inputs.append(runtime_obj)

            futures.append(self._submit(rank, graph, party_inputs, job_id=job_id))

        results = self._collect(futures)

        # Reassemble outputs
        if not graph.outputs:
            return None

        if len(graph.outputs) == 1:
            # results is list of single values
            return HostVar(results)
        else:
            # results is list of tuples/lists
            # We need to transpose: list of [out1, out2] -> [HostVar(out1s), HostVar(out2s)]
            num_outputs = len(graph.outputs)
            transposed = []
            for i in range(num_outputs):
                transposed.append(HostVar([res[i] for res in results]))
            return transposed

    def _submit(
        self, rank: int, graph: Graph, inputs: list[Any], job_id: str | None = None
    ) -> Any:
        """Submit a graph execution to a specific rank.

        Args:
            rank: The target party rank.
            graph: The graph to execute.
            inputs: The inputs for the graph.
            job_id: Optional unique ID for this execution job.

        Returns:
            A future or handle representing the asynchronous execution.
        """
        raise NotImplementedError("SimpHost subclasses must implement _submit")

    def _collect(self, futures: list[Any]) -> list[Any]:
        """Collect results from futures.

        Args:
            futures: List of futures returned by _submit.

        Returns:
            List of results (one per rank).
        """
        raise NotImplementedError("SimpHost subclasses must implement _collect")

    def fetch(self, obj: Any) -> Any:
        """Fetch data from workers if obj contains URIs."""
        if isinstance(obj, HostVar):
            # obj.values are URIs (or data)
            futures = []
            for rank, val in enumerate(obj.values):
                if isinstance(val, str) and "://" in val:
                    futures.append(self._fetch(rank, val))
                else:
                    # Already data
                    f: concurrent.futures.Future[Any] = concurrent.futures.Future()
                    f.set_result(val)
                    futures.append(f)

            return self._collect(futures)
        return obj

    def _fetch(self, rank: int, uri: str) -> Any:
        """Fetch data from a specific worker."""
        raise NotImplementedError("SimpHost subclasses must implement _fetch")
