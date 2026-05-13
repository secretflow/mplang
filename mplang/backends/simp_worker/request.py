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

"""Per-request Interpreter factory.

Creates lightweight Interpreter instances for each incoming request,
providing full isolation of mutable state (CommContext, SimpWorker,
SPUState) while sharing immutable infrastructure (Communicator, Store,
handlers).
"""

from __future__ import annotations

from mplang.backends.simp_worker.comm_context import CommContext
from mplang.backends.simp_worker.infra import WorkerInfra
from mplang.backends.simp_worker.state import SimpWorker
from mplang.backends.spu_state import SPUState
from mplang.runtime.interpreter import Interpreter


def create_request_interpreter(
    infra: WorkerInfra,
    job_id: str,
) -> Interpreter:
    """Create a lightweight Interpreter for a single request.

    Cost: ~2μs (dict/TLS allocation only).
    SPU Runtime created on-demand via link.spawn() (~120μs).

    Args:
        infra: Shared WorkerInfra (process-lifetime).
        job_id: Unique request identifier (used as CommContext context_id).

    Returns:
        Per-request Interpreter with isolated state.
    """
    # Per-request CommContext with unique context_id
    comm_ctx = CommContext(infra.communicator, context_id=job_id, my_rank=infra.rank)

    # Per-request Interpreter (does not own shared executor/tracer)
    interp = Interpreter(
        name=f"Worker-{infra.rank}-{job_id}",
        tracer=infra.tracer,
        trace_pid=infra.trace_pid,
        store=infra.store,
        root_dir=infra.root_dir,
        handlers=infra.handlers,
        executor=infra.executor,
        comm_ctx=comm_ctx,
        owns_executor=False,
        owns_tracer=False,
    )
    interp.async_ops = set(infra.async_ops)

    # Per-request SimpWorker: isolates current_parties
    worker_state = SimpWorker(
        rank=infra.rank,
        world_size=infra.world_size,
        communicator=infra.communicator,  # raw comm kept for SPU BaseChannel
        store=infra.store,
        spu_endpoints=infra.spu_endpoints,
    )
    interp.set_dialect_state("simp", worker_state)

    # Per-request SPUState: will use link.spawn() for Runtime isolation
    spu_state = SPUState(infra=infra)
    interp.set_dialect_state("spu", spu_state)

    return interp
