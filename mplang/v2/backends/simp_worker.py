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

from typing import Any

from mplang.v2.edsl.interpreter import DagProfiler, Interpreter


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
        profiler: DagProfiler | None = None,
    ):
        super().__init__(name=f"Worker-{rank}", profiler=profiler, trace_pid=rank)
        self.rank = rank
        self.world_size = world_size
        self.communicator: Any = communicator
        self.spu_endpoints = spu_endpoints

        # Enable multi-threaded execution for BFV ops by default
        import concurrent.futures
        import os

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
        }
