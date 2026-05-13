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

"""Simp worker-side collectives (wrapper layer).

This module is the *context-aware wrapper* on top of
``mplang.backends.simp_worker.collective_algorithms``.

Responsibilities here:
- Resolve "participants" from (explicit arg / op.attrs["parties"] /
  worker.current_parties / world).
- Pass a ``CommContext`` to the underlying algorithms (which generate
  collision-free keys automatically via per-peer counters).

The underlying algorithms only depend on the CommContext interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from mplang.backends.simp_worker import collective_algorithms as algo
from mplang.backends.simp_worker.base import CommunicatorProtocol
from mplang.backends.simp_worker.comm_context import CommContext
from mplang.edsl.graph import Operation


class _Worker(Protocol):
    rank: int
    world_size: int
    communicator: CommunicatorProtocol
    current_parties: tuple[int, ...] | None


def resolve_participants(
    worker: _Worker,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
) -> Sequence[int]:
    """Resolve participant ranks.

    Priority:
    1) explicit participants argument
    2) op.attrs["parties"] if present
    3) worker.current_parties if set (pcall_static dynamic scope)
    4) all ranks [0, world_size)

    Note:
        Normalization/validation (sorting, emptiness, range checks, rank
        inclusion) is intentionally delegated to the lower-level algorithms in
        `collective_algorithms`.
    """

    if participants is not None:
        return participants

    if op is not None:
        parties = op.attrs.get("parties")
        if parties is not None:
            if not isinstance(parties, Sequence):
                raise TypeError(
                    "op.attrs['parties'] must be a sequence of rank integers"
                )
            return tuple(int(r) for r in parties)

    if worker.current_parties is not None:
        return worker.current_parties

    return tuple(range(worker.world_size))


def barrier(
    ctx: CommContext,
    worker: _Worker,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
) -> None:
    ps = resolve_participants(worker, op=op, participants=participants)
    algo.barrier(ctx, participants=ps)


def broadcast(
    ctx: CommContext,
    worker: _Worker,
    value: Any,
    *,
    root: int,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
) -> Any:
    ps = resolve_participants(worker, op=op, participants=participants)
    return algo.broadcast(ctx, value, root=int(root), participants=ps)


def allgather_obj(
    ctx: CommContext,
    worker: _Worker,
    value: Any,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
) -> list[Any]:
    ps = resolve_participants(worker, op=op, participants=participants)
    return algo.allgather(ctx, value, participants=ps)


def allgather_bool(
    ctx: CommContext,
    worker: _Worker,
    value: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
) -> list[bool]:
    gathered = allgather_obj(ctx, worker, bool(value), op=op, participants=participants)
    return [bool(v) for v in gathered]


def allreduce_bool_and(
    ctx: CommContext,
    worker: _Worker,
    value: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
) -> bool:
    ps = resolve_participants(worker, op=op, participants=participants)
    return algo.allreduce_bool_and(ctx, bool(value), participants=ps)


def allreduce_bool_or(
    ctx: CommContext,
    worker: _Worker,
    value: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
) -> bool:
    ps = resolve_participants(worker, op=op, participants=participants)
    return algo.allreduce_bool_or(ctx, bool(value), participants=ps)


def allreduce_bool_xor(
    ctx: CommContext,
    worker: _Worker,
    value: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
) -> bool:
    ps = resolve_participants(worker, op=op, participants=participants)
    return algo.allreduce_bool_xor(ctx, bool(value), participants=ps)


def verify_uniform_predicate(
    ctx: CommContext,
    worker: _Worker,
    pred: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
) -> bool:
    """Verify that ``pred`` is uniform across participants.

    Uses AND/OR all-reduce to detect mismatch. If mismatch is detected, runs an
    allgather to provide a helpful error message. All participants execute the
    same comm steps to avoid deadlocks.
    """

    ps = resolve_participants(worker, op=op, participants=participants)

    all_and = allreduce_bool_and(ctx, worker, bool(pred), op=op, participants=ps)
    all_or = allreduce_bool_or(ctx, worker, bool(pred), op=op, participants=ps)

    if all_and != all_or:
        gathered = allgather_bool(ctx, worker, bool(pred), op=op, participants=ps)
        ps_norm = algo.normalize_participants(ctx, ps)
        dist = dict(zip(ps_norm, gathered, strict=True))
        raise RuntimeError(
            "simp.uniform_cond predicate is not uniform across participants: "
            f"participants={ps_norm}, values={dist}"
        )

    return bool(pred)
