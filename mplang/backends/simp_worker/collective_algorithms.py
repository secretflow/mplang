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

"""Collective communication algorithms (communicator-only).

This module contains *pure* collective algorithms implemented only in terms of
(a) a communicator and (b) an explicit participant set.

It intentionally does NOT depend on:
- Interpreter execution IDs / graph keys
- SimpWorker current_parties
- Operation objects

Callers are expected to provide a collision-free key prefix for each collective
instance.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

from mplang.backends.simp_worker.base import CommunicatorProtocol


def normalize_participants(
    comm: CommunicatorProtocol, participants: Sequence[int]
) -> tuple[int, ...]:
    ps = tuple(sorted({int(r) for r in participants}))
    if not ps:
        raise ValueError("participants must be non-empty")
    if any(r < 0 or r >= comm.world_size for r in ps):
        raise ValueError(
            f"participants out of range: {ps}, world_size={comm.world_size}"
        )
    if comm.rank not in ps:
        raise ValueError(f"rank {comm.rank} is not in participants {ps}")
    return ps


def barrier(
    comm: CommunicatorProtocol, *, participants: Sequence[int], key_prefix: str
) -> None:
    """Barrier using root gather + root release."""

    ps = normalize_participants(comm, participants)
    root = ps[0]

    arrive_key = f"{key_prefix}_arrive"
    release_key = f"{key_prefix}_release"

    if comm.rank != root:
        comm.send(root, arrive_key, True)
        comm.recv(root, release_key)
        return

    for r in ps:
        if r == root:
            continue
        _ = comm.recv(r, arrive_key)

    for r in ps:
        if r == root:
            continue
        comm.send(r, release_key, True)


def broadcast(
    comm: CommunicatorProtocol,
    value: Any,
    *,
    root: int,
    participants: Sequence[int],
    key_prefix: str,
) -> Any:
    """Broadcast a value from root to all participants."""

    ps = normalize_participants(comm, participants)
    if root not in ps:
        raise ValueError(f"root {root} must be in participants {ps}")

    bcast_key = f"{key_prefix}_bcast"

    if comm.rank == root:
        for r in ps:
            if r == root:
                continue
            comm.send(r, bcast_key, value)
        return value

    return comm.recv(root, bcast_key)


def allgather(
    comm: CommunicatorProtocol,
    value: Any,
    *,
    participants: Sequence[int],
    key_prefix: str,
) -> list[Any]:
    """Allgather implemented as gather-to-root then root broadcast."""

    ps = normalize_participants(comm, participants)
    root = ps[0]

    gather_key = f"{key_prefix}_gather"
    bcast_key = f"{key_prefix}_bcast"

    if comm.rank != root:
        comm.send(root, gather_key, value)
        gathered = comm.recv(root, bcast_key)
        if not isinstance(gathered, list):
            raise TypeError(f"expected list from root broadcast, got {type(gathered)}")
        return gathered

    values_by_rank: dict[int, Any] = {root: value}
    for r in ps:
        if r == root:
            continue
        values_by_rank[r] = comm.recv(r, gather_key)

    gathered = [values_by_rank[r] for r in ps]

    for r in ps:
        if r == root:
            continue
        comm.send(r, bcast_key, gathered)

    return gathered


def allreduce_bool_and(
    comm: CommunicatorProtocol,
    value: bool,
    *,
    participants: Sequence[int],
    key_prefix: str,
) -> bool:
    return _allreduce_bool(
        comm,
        value,
        participants=participants,
        key_prefix=key_prefix,
        combine=operator.and_,
    )


def allreduce_bool_or(
    comm: CommunicatorProtocol,
    value: bool,
    *,
    participants: Sequence[int],
    key_prefix: str,
) -> bool:
    return _allreduce_bool(
        comm,
        value,
        participants=participants,
        key_prefix=key_prefix,
        combine=operator.or_,
    )


def allreduce_bool_xor(
    comm: CommunicatorProtocol,
    value: bool,
    *,
    participants: Sequence[int],
    key_prefix: str,
) -> bool:
    return _allreduce_bool(
        comm,
        value,
        participants=participants,
        key_prefix=key_prefix,
        combine=operator.xor,
    )


def _allreduce_bool(
    comm: CommunicatorProtocol,
    value: bool,
    *,
    participants: Sequence[int],
    key_prefix: str,
    combine: Callable[[bool, bool], bool],
) -> bool:
    ps = normalize_participants(comm, participants)
    root = ps[0]

    gather_key = f"{key_prefix}_gather"
    bcast_key = f"{key_prefix}_bcast"

    if comm.rank != root:
        comm.send(root, gather_key, bool(value))
        return bool(comm.recv(root, bcast_key))

    acc = bool(value)
    for r in ps:
        if r == root:
            continue
        acc = combine(acc, bool(comm.recv(r, gather_key)))

    for r in ps:
        if r == root:
            continue
        comm.send(r, bcast_key, acc)

    return acc
