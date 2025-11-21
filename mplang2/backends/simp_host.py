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

    def evaluate_graph(self, graph: Graph, inputs: dict[Any, Any]) -> Any:
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

        # If graph returns single value, results is list of values.
        # If graph returns tuple, results is list of tuples.

        # We want to return HostVar(s).

        # Assume graph returns single value for now or handle tuple.
        # But we don't know the structure easily without inspecting graph outputs.

        # For now, just return list of results (HostVar-like structure manually constructed if needed)
        # But HostInterpreter usually returns HostVar.

        # Let's just return the raw results list for now, or construct HostVar.
        if not results:
            return None

        first_res = results[0]
        if isinstance(first_res, (list, tuple)):
            # Multiple outputs
            num_outs = len(first_res)
            outs = []
            for i in range(num_outs):
                outs.append(HostVar([res[i] for res in results]))
            return tuple(outs)
        else:
            return HostVar(results)

    def _submit(self, rank: int, graph: Graph, inputs: dict[Any, Any]) -> Any:
        raise NotImplementedError

    def _collect(self, futures: list[Any]) -> list[Any]:
        raise NotImplementedError
