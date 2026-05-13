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

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any

from mplang.backends.simp_worker.collectives import (
    allgather_obj,
    allreduce_bool_and,
    allreduce_bool_or,
    barrier,
    broadcast,
)
from mplang.backends.simp_worker.comm_context import CommContext
from mplang.backends.simp_worker.mem import LocalMesh


@dataclass
class _Worker:
    rank: int
    world_size: int
    communicator: Any
    current_parties: tuple[int, ...] | None = None


def test_broadcast_roundtrip() -> None:
    mesh = LocalMesh(world_size=3)

    def run_rank(rank: int) -> Any:
        ctx = CommContext(mesh.comms[rank], context_id="test", my_rank=rank)
        worker = _Worker(rank, 3, mesh.comms[rank])
        return broadcast(
            ctx,
            worker,
            {"x": 123},
            root=0,
            participants=(0, 1, 2),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run_rank, range(3)))

    assert results == [{"x": 123}, {"x": 123}, {"x": 123}]
    mesh.shutdown()


def test_allgather_obj_order() -> None:
    mesh = LocalMesh(world_size=3)

    def run_rank(rank: int) -> Any:
        ctx = CommContext(mesh.comms[rank], context_id="test", my_rank=rank)
        worker = _Worker(rank, 3, mesh.comms[rank])
        gathered = allgather_obj(
            ctx,
            worker,
            rank,
            participants=(0, 1, 2),
        )
        return tuple(gathered)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run_rank, range(3)))

    assert results == [(0, 1, 2), (0, 1, 2), (0, 1, 2)]
    mesh.shutdown()


def test_allreduce_bool_and_or() -> None:
    mesh = LocalMesh(world_size=3)

    inputs = {0: True, 1: True, 2: False}

    def run_rank(rank: int) -> tuple[bool, bool]:
        ctx = CommContext(mesh.comms[rank], context_id="test", my_rank=rank)
        worker = _Worker(rank, 3, mesh.comms[rank])
        v = inputs[rank]
        r_and = allreduce_bool_and(
            ctx,
            worker,
            v,
            participants=(0, 1, 2),
        )
        r_or = allreduce_bool_or(
            ctx,
            worker,
            v,
            participants=(0, 1, 2),
        )
        return r_and, r_or

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run_rank, range(3)))

    assert results == [(False, True), (False, True), (False, True)]
    mesh.shutdown()


def test_barrier_completes() -> None:
    mesh = LocalMesh(world_size=3)

    def run_rank(rank: int) -> int:
        ctx = CommContext(mesh.comms[rank], context_id="test", my_rank=rank)
        worker = _Worker(rank, 3, mesh.comms[rank])
        barrier(
            ctx,
            worker,
            participants=(0, 1, 2),
        )
        return rank

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run_rank, range(3)))

    assert sorted(results) == [0, 1, 2]
    mesh.shutdown()
