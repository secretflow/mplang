"""SIMP worker module.

Provides WorkerInterpreter for executing Graph IR on a single party.
This module only defines the interpreter class - operation implementations
are registered in simp_impl.py.
"""

from __future__ import annotations

from typing import Any

from mplang2.edsl.interpreter import Interpreter


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
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.communicator: Any = communicator
        self.spu_endpoints = spu_endpoints
