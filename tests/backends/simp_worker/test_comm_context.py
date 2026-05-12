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

"""Tests for CommContext automatic key generation."""

from __future__ import annotations

import concurrent.futures
from typing import Any

from mplang.backends.simp_worker.comm_context import CommContext
from mplang.backends.simp_worker.mem import LocalMesh


def test_send_recv_key_symmetry() -> None:
    """send on rank 0 and recv on rank 1 produce matching keys."""
    mesh = LocalMesh(world_size=2)

    def run(rank: int) -> Any:
        ctx = CommContext(mesh.comms[rank], context_id="ctx0", my_rank=rank)
        if rank == 0:
            ctx.send(1, "hello")
            return None
        else:
            return ctx.recv(0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(run, range(2)))

    assert results[1] == "hello"
    mesh.shutdown()


def test_multiple_exchanges() -> None:
    """Multiple send/recv pairs produce distinct keys."""
    mesh = LocalMesh(world_size=2)

    def run(rank: int) -> Any:
        ctx = CommContext(mesh.comms[rank], context_id="ctx0", my_rank=rank)
        if rank == 0:
            ctx.send(1, "msg0")
            ctx.send(1, "msg1")
            ctx.send(1, "msg2")
            return None
        else:
            return [ctx.recv(0), ctx.recv(0), ctx.recv(0)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(run, range(2)))

    assert results[1] == ["msg0", "msg1", "msg2"]
    mesh.shutdown()


def test_bidirectional() -> None:
    """Interleaved send/recv in both directions."""
    mesh = LocalMesh(world_size=2)

    def run(rank: int) -> Any:
        ctx = CommContext(mesh.comms[rank], context_id="bidir", my_rank=rank)
        if rank == 0:
            ctx.send(1, "from0")
            got = ctx.recv(1)
            return got
        else:
            got = ctx.recv(0)
            ctx.send(0, "from1")
            return got

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(run, range(2)))

    assert results[0] == "from1"
    assert results[1] == "from0"
    mesh.shutdown()


def test_spawn_independent_counters() -> None:
    """Spawned child contexts have independent counter namespaces."""
    mesh = LocalMesh(world_size=2)

    def run(rank: int) -> Any:
        root = CommContext(mesh.comms[rank], context_id="root", my_rank=rank)
        child0 = root.spawn()
        child1 = root.spawn()

        if rank == 0:
            child0.send(1, "c0_msg")
            child1.send(1, "c1_msg")
            return None
        else:
            # Receive in same spawn order
            return [child0.recv(0), child1.recv(0)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(run, range(2)))

    assert results[1] == ["c0_msg", "c1_msg"]
    mesh.shutdown()


def test_spawn_hierarchy() -> None:
    """Nested spawn produces correct hierarchical IDs."""
    mesh = LocalMesh(world_size=2)
    ctx = CommContext(mesh.comms[0], context_id="root", my_rank=0)
    child = ctx.spawn()
    grandchild = child.spawn()
    assert child._id == "root.0"
    assert grandchild._id == "root.0.0"

    child2 = ctx.spawn()
    assert child2._id == "root.1"
    mesh.shutdown()


def test_barrier_via_comm_context() -> None:
    """CommContext works correctly with collective barrier pattern."""
    from mplang.backends.simp_worker import collective_algorithms as algo

    mesh = LocalMesh(world_size=3)

    def run(rank: int) -> int:
        ctx = CommContext(mesh.comms[rank], context_id="barrier", my_rank=rank)
        algo.barrier(ctx, participants=(0, 1, 2))
        return rank

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run, range(3)))

    assert sorted(results) == [0, 1, 2]
    mesh.shutdown()


def test_three_party_allgather() -> None:
    """CommContext allgather across 3 parties."""
    from mplang.backends.simp_worker import collective_algorithms as algo

    mesh = LocalMesh(world_size=3)

    def run(rank: int) -> list[int]:
        ctx = CommContext(mesh.comms[rank], context_id="ag", my_rank=rank)
        return algo.allgather(ctx, rank * 10, participants=(0, 1, 2))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run, range(3)))

    assert results == [[0, 10, 20], [0, 10, 20], [0, 10, 20]]
    mesh.shutdown()
