"""SIMP host base module."""

from __future__ import annotations

from typing import Any

from mplang.v2.edsl.graph import Graph
from mplang.v2.edsl.interpreter import Interpreter


class HostVar:
    """Runtime value for SIMP dialect holding values for all parties."""

    def __init__(self, values: list[Any]):
        self.values = values

    def __repr__(self) -> str:
        return f"HostVar({self.values})"

    def __getitem__(self, rank: int) -> Any:
        return self.values[rank]


class SimpHost(Interpreter):
    """Base class for SIMP host interpreters.

    This interpreter runs on the coordinator (Host) and dispatches
    SPMD operations to workers.
    """

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

    def _submit(self, rank: int, graph: Graph, inputs: list[Any]) -> Any:
        """Submit a graph execution to a specific rank.

        Args:
            rank: The target party rank.
            graph: The graph to execute.
            inputs: The inputs for the graph.

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
