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

"""Tests for WorkerInfra and create_request_interpreter isolation."""

from __future__ import annotations

import concurrent.futures
from typing import Any

from mplang.backends.simp_worker.infra import DEFAULT_ASYNC_OPS, WorkerInfra
from mplang.backends.simp_worker.mem import LocalMesh
from mplang.backends.simp_worker.request import create_request_interpreter
from mplang.runtime.object_store import ObjectStore


def _make_infra(mesh: LocalMesh, rank: int) -> WorkerInfra:
    """Helper to build a minimal WorkerInfra for testing."""
    return WorkerInfra(
        rank=rank,
        world_size=mesh.world_size,
        communicator=mesh.comms[rank],
        store=ObjectStore(),
        handlers={},
        async_ops=DEFAULT_ASYNC_OPS,
    )


# -- WorkerInfra tests --


def test_default_async_ops_is_frozenset() -> None:
    """DEFAULT_ASYNC_OPS should be a frozenset (immutable)."""
    assert isinstance(DEFAULT_ASYNC_OPS, frozenset)
    assert len(DEFAULT_ASYNC_OPS) > 0


def test_infra_async_ops_immutable() -> None:
    """WorkerInfra.async_ops should be a frozenset."""
    mesh = LocalMesh(world_size=2)
    infra = _make_infra(mesh, 0)
    assert isinstance(infra.async_ops, frozenset)
    mesh.shutdown()


def test_infra_spu_link_cache() -> None:
    """get_or_create_spu_link caches and returns the same object."""
    mesh = LocalMesh(world_size=2)
    infra = _make_infra(mesh, 0)

    call_count = 0

    def factory() -> str:
        nonlocal call_count
        call_count += 1
        return f"link_{call_count}"

    key = ("rank0", 2, "semi2k", "fm64", "http")
    link1 = infra.get_or_create_spu_link(key, factory)
    link2 = infra.get_or_create_spu_link(key, factory)

    assert link1 is link2
    assert call_count == 1  # factory called only once
    mesh.shutdown()


def test_infra_shutdown_clears_spu_cache() -> None:
    """WorkerInfra.shutdown() clears SPU template link cache."""
    mesh = LocalMesh(world_size=2)
    infra = _make_infra(mesh, 0)

    key = ("rank0", 2, "semi2k", "fm64", "http")
    infra.get_or_create_spu_link(key, lambda: "link")
    assert len(infra._spu_template_links) == 1

    infra.shutdown()
    assert len(infra._spu_template_links) == 0

    # Idempotent
    infra.shutdown()
    mesh.shutdown()


# -- create_request_interpreter tests --


def test_request_interpreter_has_isolated_comm_ctx() -> None:
    """Each request interpreter gets a unique CommContext (context_id = job_id)."""
    mesh = LocalMesh(world_size=2)
    infra = _make_infra(mesh, 0)

    interp_a = create_request_interpreter(infra, "job-aaa")
    interp_b = create_request_interpreter(infra, "job-bbb")

    assert interp_a.comm_ctx is not interp_b.comm_ctx
    assert interp_a.comm_ctx._id == "job-aaa"
    assert interp_b.comm_ctx._id == "job-bbb"

    interp_a.shutdown()
    interp_b.shutdown()
    mesh.shutdown()


def test_request_interpreter_async_ops_copied() -> None:
    """async_ops on the interpreter is a copy, not an alias of infra's set."""
    mesh = LocalMesh(world_size=2)
    infra = _make_infra(mesh, 0)

    interp = create_request_interpreter(infra, "job-copy")

    # Mutating the interpreter's set should not affect infra
    original_len = len(infra.async_ops)
    interp.async_ops.add("test.fake_op")

    assert len(infra.async_ops) == original_len
    assert "test.fake_op" not in infra.async_ops
    assert "test.fake_op" in interp.async_ops

    interp.shutdown()
    mesh.shutdown()


def test_request_interpreter_shares_store() -> None:
    """Per-request interpreters share the same ObjectStore as infra."""
    mesh = LocalMesh(world_size=2)
    infra = _make_infra(mesh, 0)

    interp = create_request_interpreter(infra, "job-store")
    assert interp.store is infra.store

    interp.shutdown()
    mesh.shutdown()


def test_request_interpreter_does_not_own_executor() -> None:
    """Per-request interpreter should not own executor or tracer."""
    mesh = LocalMesh(world_size=2)
    infra = _make_infra(mesh, 0)

    interp = create_request_interpreter(infra, "job-own")
    assert interp._owns_executor is False
    assert interp._owns_tracer is False

    interp.shutdown()
    mesh.shutdown()


def test_request_interpreters_isolated_comm_cross_rank() -> None:
    """Two ranks can exchange messages via per-request interpreters."""
    mesh = LocalMesh(world_size=2)
    infra0 = _make_infra(mesh, 0)
    infra1 = _make_infra(mesh, 1)

    def run(rank: int) -> Any:
        infra = infra0 if rank == 0 else infra1
        interp = create_request_interpreter(infra, "job-cross")
        ctx = interp.comm_ctx
        try:
            if rank == 0:
                ctx.send(1, "hello")
                return None
            else:
                return ctx.recv(0)
        finally:
            interp.shutdown()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(run, range(2)))

    assert results[1] == "hello"
    mesh.shutdown()
