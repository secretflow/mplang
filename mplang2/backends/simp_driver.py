"""SIMP driver module."""

from __future__ import annotations

from typing import Any

from mplang2.backends.simp_host import SimpHost
from mplang2.edsl.graph import Operation


class SimpDriver(SimpHost):
    """SIMP driver for remote execution (RPC)."""

    def __init__(self, world_size: int = 3):
        super().__init__(world_size)
        # TODO: Initialize RPC connections

    def _submit(self, rank: int, graph: Operation, inputs: dict[Any, Any]) -> Any:
        # TODO: Submit via RPC
        raise NotImplementedError("RPC execution not implemented yet")

    def _collect(self, futures: list[Any]) -> list[Any]:
        # TODO: Wait for RPC results
        raise NotImplementedError("RPC execution not implemented yet")
