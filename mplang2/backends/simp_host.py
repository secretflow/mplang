"""SIMP host base module."""

from __future__ import annotations

from typing import Any

from mplang2.edsl.graph import Operation
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

    def evaluate_graph(self, graph: Operation, inputs: dict[Any, Any]) -> Any:
        """Execute graph by distributing it to all parties."""
        # inputs: Value -> HostVar (or constant)

        futures = []
        for rank in range(self.world_size):
            # Prepare inputs for this rank
            party_inputs = {}
            for val, runtime_obj in inputs.items():
                if isinstance(runtime_obj, HostVar):
                    party_inputs[val] = runtime_obj[rank]
                else:
                    party_inputs[val] = runtime_obj

            futures.append(self._submit(rank, graph, party_inputs))

        results = self._collect(futures)

        # Reassemble outputs
        # results is list of (out1, out2, ...) for each party
        # We want (HostVar(out1s), HostVar(out2s), ...)

        num_outputs = len(graph.outputs)
        if num_outputs == 0:
            return None

        if num_outputs == 1:
            # results is list of single values
            return HostVar(results)
        else:
            # results is list of tuples
            transposed = list(zip(*results, strict=True))
            return [HostVar(list(vals)) for vals in transposed]

    def _submit(self, rank: int, graph: Operation, inputs: dict[Any, Any]) -> Any:
        raise NotImplementedError

    def _collect(self, futures: list[Any]) -> list[Any]:
        raise NotImplementedError
