# Copyright 2026 Ant Group Co., Ltd.
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

"""WorkerInfra: shared infrastructure container for per-request Interpreter creation.

Created once at Worker startup. Passed to ``create_request_interpreter()`` for
each incoming request. All fields are either immutable or thread-safe.
"""

from __future__ import annotations

import concurrent.futures
import pathlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mplang.backends.simp_worker.base import CommunicatorProtocol
from mplang.runtime.interpreter import ExecutionTracer
from mplang.runtime.object_store import ObjectStore

# Opcodes eligible for async DAG scheduling when an Executor is present.
# Kept as a module-level constant so that both MemCluster and HTTP Worker
# share the same set without duplication.
DEFAULT_ASYNC_OPS: frozenset[str] = frozenset({
    "bfv.add",
    "bfv.mul",
    "bfv.rotate",
    "bfv.batch_encode",
    "bfv.relinearize",
    "bfv.encrypt",
    "bfv.decrypt",
    "field.solve_okvs",
    "field.decode_okvs",
    "field.aes_expand",
    "field.mul",
    "simp.shuffle",
})


@dataclass
class WorkerInfra:
    """Shared infrastructure for a Worker process.

    Created once at Worker startup. Passed to each per-request Interpreter
    factory. All fields are either immutable or thread-safe.
    """

    rank: int
    world_size: int
    communicator: CommunicatorProtocol
    store: ObjectStore
    handlers: dict[str, Callable[..., Any]]
    spu_endpoints: dict[int, str] | None = None
    tracer: ExecutionTracer | None = None
    trace_pid: int | None = None
    root_dir: pathlib.Path | None = None
    executor: concurrent.futures.Executor | None = None
    async_ops: frozenset[str] = field(default_factory=frozenset)

    # SPU template links (lazily populated, protected by lock).
    # Cache keys are (local_rank, spu_world_size, protocol, field, link_mode).
    # The number of distinct keys is bounded by the Cartesian product of SPU
    # configurations actually used at runtime -- typically 1-3 entries per
    # worker process (one per distinct SPU device declaration).
    _spu_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _spu_template_links: dict[tuple, Any] = field(default_factory=dict, repr=False)

    def get_or_create_spu_link(
        self,
        cache_key: tuple,
        create_fn: Callable[[], Any],
    ) -> Any:
        """Thread-safe lazy creation of template SPU links.

        Args:
            cache_key: Tuple identifying the SPU configuration.
            create_fn: Factory function to create a new link context.

        Returns:
            A libspu.link.Context (template link) for the given configuration.
        """
        with self._spu_lock:
            if cache_key not in self._spu_template_links:
                self._spu_template_links[cache_key] = create_fn()
            return self._spu_template_links[cache_key]

    def shutdown(self) -> None:
        """Release cached SPU template links.

        Safe to call multiple times.  Should be called during process
        shutdown to eagerly close BRPC connections rather than waiting
        for GC / process exit.
        """
        with self._spu_lock:
            self._spu_template_links.clear()
