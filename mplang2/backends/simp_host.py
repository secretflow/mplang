"""SIMP host base module."""

from __future__ import annotations

from typing import Any

from mplang2.edsl.graph import Graph
from mplang2.edsl.interpreter import Interpreter


class HostVar:
    """Runtime value for SIMP dialect holding values for all parties."""

    def __init__(self, values: list[Any]):
        self.values = values

    def __repr__(self) -> str:
        return f"HostVar({self.values})"

    def __getitem__(self, rank: int) -> Any:
        return self.values[rank]


class SimpHost(Interpreter):
    """Base class for SIMP host interpreters."""

    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = world_size

    def evaluate_graph(self, graph: Graph, inputs: list[Any]) -> Any:
        """Execute graph by distributing it to all parties."""
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

            futures.append(self._submit(rank, graph, party_inputs))

        results = self._collect(futures)

        # Reassemble outputs
        if not graph.outputs:
            return None

        if len(graph.outputs) == 1:
            return HostVar(results)

        # Multiple outputs
        # results is list of tuples/lists (one per party)
        # We want list of HostVars (one per output)
        num_outs = len(graph.outputs)
        outs = []
        for i in range(num_outs):
            outs.append(HostVar([res[i] for res in results]))
        return outs

    def _submit(self, rank: int, graph: Graph, inputs: list[Any]) -> Any:
        raise NotImplementedError

    def _collect(self, futures: list[Any]) -> list[Any]:
        raise NotImplementedError
