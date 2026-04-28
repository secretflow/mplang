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

"""E2E test: concurrent requests on the same 2-party shuffle graph.

Uses MemCluster (in-memory workers with ThreadCommunicator) to verify that
two concurrent job_id-isolated requests produce correct results with no
communication key collision.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, cast

import numpy as np

from mplang.backends.simp_driver.mem import MemCluster
from mplang.backends.simp_worker import SimpWorker
from mplang.dialects import simp
from mplang.edsl.graph import Graph
from mplang.edsl.typing import IntegerType, MPType, TensorType
from mplang.runtime.interpreter import Interpreter


def _build_shuffle_graph() -> Graph:
    """Build a simple graph: Party 0 sends input to Party 1 via shuffle_static.

    %arg0: MP[Tensor, (0,)]  — Party 0's input
    %0 = simp.shuffle_static(%arg0, routing={1: 0})
    return %0
    """
    g = Graph()
    t = TensorType(IntegerType(bitwidth=32), ())
    arg0 = g.add_input("arg0", MPType(t, parties=(0,)))
    (out,) = g.add_op(
        simp.shuffle_static_p.name,
        [arg0],
        output_types=[MPType(t, parties=(1,))],
        attrs={"routing": {1: 0}},
    )
    g.add_output(out)
    return g


def _run_on_workers(
    cluster: MemCluster,
    graph: Graph,
    per_rank_inputs: dict[int, list[Any]],
    job_id: str,
) -> list[Any]:
    """Submit graph to all workers and collect results."""
    futures: list[concurrent.futures.Future[Any]] = []
    for rank in range(cluster.world_size):
        worker = cluster.workers[rank]
        ctx = cast(SimpWorker, worker.get_dialect_state("simp"))
        inputs = per_rank_inputs.get(rank, [None])
        # Store input data and get URIs
        uri_inputs = [ctx.store.put(v) if v is not None else None for v in inputs]
        futures.append(
            concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
                _worker_execute, worker, ctx, graph, uri_inputs, job_id
            )
        )
    return [f.result() for f in futures]


def _worker_execute(
    worker: Interpreter,
    ctx: SimpWorker,
    graph: Graph,
    uri_inputs: list[Any],
    job_id: str,
) -> Any:
    """Execute graph on a single worker (runs in thread)."""
    resolved = [ctx.store.get(u) if u is not None else None for u in uri_inputs]
    results = worker.evaluate_graph(graph, resolved, job_id=job_id)
    if not graph.outputs:
        return None
    return [ctx.store.put(r) if r is not None else None for r in results]


def test_concurrent_shuffle_with_job_ids() -> None:
    """Two concurrent requests executing the same shuffle graph.

    Party 0 sends data to Party 1.  Two requests run in parallel with
    different job_ids.  Both should complete correctly without mailbox
    overflow or key mismatch.
    """
    cluster = MemCluster(world_size=2)
    graph = _build_shuffle_graph()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        # Request 1: send value 100
        f1_workers: list[concurrent.futures.Future[Any]] = []
        for rank in range(2):
            worker = cluster.workers[rank]
            ctx = cast(SimpWorker, worker.get_dialect_state("simp"))
            inp = np.array(100, dtype=np.int32) if rank == 0 else None
            uri_inp = [ctx.store.put(inp) if inp is not None else None]
            f1_workers.append(
                pool.submit(_worker_execute, worker, ctx, graph, uri_inp, "req-001")
            )

        # Request 2: send value 200
        f2_workers: list[concurrent.futures.Future[Any]] = []
        for rank in range(2):
            worker = cluster.workers[rank]
            ctx = cast(SimpWorker, worker.get_dialect_state("simp"))
            inp = np.array(200, dtype=np.int32) if rank == 0 else None
            uri_inp = [ctx.store.put(inp) if inp is not None else None]
            f2_workers.append(
                pool.submit(_worker_execute, worker, ctx, graph, uri_inp, "req-002")
            )

        r1 = [f.result() for f in f1_workers]
        r2 = [f.result() for f in f2_workers]

    # Party 1 (rank=1) should have received the value via shuffle
    ctx1 = cast(SimpWorker, cluster.workers[1].get_dialect_state("simp"))
    val1 = ctx1.store.get(r1[1][0])
    val2 = ctx1.store.get(r2[1][0])
    assert int(val1) == 100, f"Expected 100, got {val1}"
    assert int(val2) == 200, f"Expected 200, got {val2}"

    cluster.shutdown()


def test_concurrent_same_graph_without_job_id_blocked() -> None:
    """Without job_id, concurrent execution of the same graph should be blocked."""
    import threading

    from mplang.edsl.registry import register_impl

    hold_event = threading.Event()

    def _blocking_impl(interpreter: Any, op: Any, x: Any) -> Any:
        """Block until signaled — ensures thread 1 holds the graph lock."""
        hold_event.wait(timeout=5)
        return x

    register_impl("test._blocking_op", _blocking_impl)

    # Build a graph with a blocking op
    g = Graph()
    t = TensorType(IntegerType(bitwidth=32), ())
    arg0 = g.add_input("arg0", MPType(t, parties=(0,)))
    (out,) = g.add_op("test._blocking_op", [arg0], output_types=[None])
    g.add_output(out)

    cluster = MemCluster(world_size=2)
    worker = cluster.workers[0]

    errors: list[Exception | None] = [None, None]
    started = threading.Event()

    def run_first() -> None:
        try:
            started.set()  # Signal that thread 1 has started
            worker.evaluate_graph(g, [np.int32(1)])  # No job_id, will block
        except Exception as e:
            errors[0] = e

    def run_second() -> None:
        try:
            started.wait(timeout=5)  # Wait for thread 1 to start
            import time

            time.sleep(0.05)  # Brief delay to ensure thread 1 holds the lock
            worker.evaluate_graph(g, [np.int32(2)])  # No job_id, should fail
        except Exception as e:
            errors[1] = e
        finally:
            hold_event.set()  # Release thread 1 regardless

    t1 = threading.Thread(target=run_first)
    t2 = threading.Thread(target=run_second)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    # Thread 2 should have gotten a RuntimeError
    assert isinstance(errors[1], RuntimeError), (
        f"Expected RuntimeError from thread 2, got: {errors[1]}"
    )
    assert errors[0] is None, f"Thread 1 should succeed, got: {errors[0]}"
    cluster.shutdown()
