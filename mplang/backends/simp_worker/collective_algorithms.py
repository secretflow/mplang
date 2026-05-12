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

"""Collective communication algorithms.

This module contains *pure* collective algorithms implemented only in terms of
a :class:`CommContext` and an explicit participant set.

It intentionally does NOT depend on:
- Interpreter execution IDs / graph keys
- SimpWorker current_parties
- Operation objects

``CommContext`` generates collision-free message keys automatically via
per-peer counters, so callers no longer need to supply a ``key_prefix``.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

from mplang.backends.simp_worker.comm_context import CommContext


def normalize_participants(
    ctx: CommContext, participants: Sequence[int]
) -> tuple[int, ...]:
    ps = tuple(sorted({int(r) for r in participants}))
    if not ps:
        raise ValueError("participants must be non-empty")
    if any(r < 0 or r >= ctx.world_size for r in ps):
        raise ValueError(
            f"participants out of range: {ps}, world_size={ctx.world_size}"
        )
    if ctx.rank not in ps:
        raise ValueError(f"rank {ctx.rank} is not in participants {ps}")
    return ps


def barrier(ctx: CommContext, *, participants: Sequence[int]) -> None:
    """Barrier using root gather + root release."""

    ps = normalize_participants(ctx, participants)
    root = ps[0]

    if ctx.rank != root:
        ctx.send(root, True)
        ctx.recv(root)
        return

    for r in ps:
        if r == root:
            continue
        ctx.recv(r)

    for r in ps:
        if r == root:
            continue
        ctx.send(r, True)


def broadcast(
    ctx: CommContext,
    value: Any,
    *,
    root: int,
    participants: Sequence[int],
) -> Any:
    """Broadcast a value from root to all participants."""

    ps = normalize_participants(ctx, participants)
    if root not in ps:
        raise ValueError(f"root {root} must be in participants {ps}")

    if ctx.rank == root:
        for r in ps:
            if r == root:
                continue
            ctx.send(r, value)
        return value

    return ctx.recv(root)


def allgather(
    ctx: CommContext,
    value: Any,
    *,
    participants: Sequence[int],
) -> list[Any]:
    """Allgather implemented as gather-to-root then root broadcast."""

    ps = normalize_participants(ctx, participants)
    root = ps[0]

    if ctx.rank != root:
        ctx.send(root, value)
        gathered = ctx.recv(root)
        if not isinstance(gathered, list):
            raise TypeError(f"expected list from root broadcast, got {type(gathered)}")
        return gathered

    values_by_rank: dict[int, Any] = {root: value}
    for r in ps:
        if r == root:
            continue
        values_by_rank[r] = ctx.recv(r)

    gathered = [values_by_rank[r] for r in ps]

    for r in ps:
        if r == root:
            continue
        ctx.send(r, gathered)

    return gathered


def allreduce_bool_and(
    ctx: CommContext,
    value: bool,
    *,
    participants: Sequence[int],
) -> bool:
    return _allreduce_bool(ctx, value, participants=participants, combine=operator.and_)


def allreduce_bool_or(
    ctx: CommContext,
    value: bool,
    *,
    participants: Sequence[int],
) -> bool:
    return _allreduce_bool(ctx, value, participants=participants, combine=operator.or_)


def allreduce_bool_xor(
    ctx: CommContext,
    value: bool,
    *,
    participants: Sequence[int],
) -> bool:
    return _allreduce_bool(ctx, value, participants=participants, combine=operator.xor)


def _allreduce_bool(
    ctx: CommContext,
    value: bool,
    *,
    participants: Sequence[int],
    combine: Callable[[bool, bool], bool],
) -> bool:
    ps = normalize_participants(ctx, participants)
    root = ps[0]

    if ctx.rank != root:
        ctx.send(root, bool(value))
        return bool(ctx.recv(root))

    acc = bool(value)
    for r in ps:
        if r == root:
            continue
        acc = combine(acc, bool(ctx.recv(r)))

    for r in ps:
        if r == root:
            continue
        ctx.send(r, acc)

    return acc
