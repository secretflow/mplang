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
`mplang.backends.simp_worker.collective_algorithms`.

Responsibilities here:
- Resolve "participants" from (explicit arg / op.attrs["parties"] /
  worker.current_parties / world).
- Build collision-free `key_prefix` using interpreter execution IDs.

The underlying algorithms only depend on the communicator interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from mplang.backends.simp_worker import collective_algorithms as algo
from mplang.edsl.graph import Operation


class _ExecContext(Protocol):
    def current_op_exec_id(self) -> int: ...

    def current_graph_exec_key(self) -> str: ...


class _Worker(Protocol):
    rank: int
    world_size: int
    communicator: algo.Communicator
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


def _collective_prefix(
    interpreter: _ExecContext, *, op: Operation | None, name: str
) -> str:
    exec_id = interpreter.current_op_exec_id()
    graph_key = interpreter.current_graph_exec_key()
    op_name = op.name if op is not None else "_"
    return f"coll_{graph_key}_{op_name}_{exec_id}_{name}"


def barrier(
    interpreter: _ExecContext,
    worker: _Worker,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
    name: str = "barrier",
) -> None:
    ps = resolve_participants(worker, op=op, participants=participants)
    prefix = _collective_prefix(interpreter, op=op, name=name)
    algo.barrier(worker.communicator, participants=ps, key_prefix=prefix)


def broadcast(
    interpreter: _ExecContext,
    worker: _Worker,
    value: Any,
    *,
    root: int,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
    name: str = "broadcast",
) -> Any:
    ps = resolve_participants(worker, op=op, participants=participants)
    prefix = _collective_prefix(interpreter, op=op, name=name)
    return algo.broadcast(
        worker.communicator,
        value,
        root=int(root),
        participants=ps,
        key_prefix=prefix,
    )


def allgather_obj(
    interpreter: _ExecContext,
    worker: _Worker,
    value: Any,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
    name: str = "allgather_obj",
) -> list[Any]:
    ps = resolve_participants(worker, op=op, participants=participants)
    prefix = _collective_prefix(interpreter, op=op, name=name)
    return algo.allgather(
        worker.communicator, value, participants=ps, key_prefix=prefix
    )


def allgather_bool(
    interpreter: _ExecContext,
    worker: _Worker,
    value: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
    name: str = "allgather_bool",
) -> list[bool]:
    gathered = allgather_obj(
        interpreter,
        worker,
        bool(value),
        op=op,
        participants=participants,
        name=name,
    )
    return [bool(v) for v in gathered]


def allreduce_bool_and(
    interpreter: _ExecContext,
    worker: _Worker,
    value: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
    name: str = "allreduce_bool_and",
) -> bool:
    ps = resolve_participants(worker, op=op, participants=participants)
    prefix = _collective_prefix(interpreter, op=op, name=name)
    return algo.allreduce_bool_and(
        worker.communicator,
        bool(value),
        participants=ps,
        key_prefix=prefix,
    )


def allreduce_bool_or(
    interpreter: _ExecContext,
    worker: _Worker,
    value: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
    name: str = "allreduce_bool_or",
) -> bool:
    ps = resolve_participants(worker, op=op, participants=participants)
    prefix = _collective_prefix(interpreter, op=op, name=name)
    return algo.allreduce_bool_or(
        worker.communicator,
        bool(value),
        participants=ps,
        key_prefix=prefix,
    )


def allreduce_bool_xor(
    interpreter: _ExecContext,
    worker: _Worker,
    value: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
    name: str = "allreduce_bool_xor",
) -> bool:
    ps = resolve_participants(worker, op=op, participants=participants)
    prefix = _collective_prefix(interpreter, op=op, name=name)
    return algo.allreduce_bool_xor(
        worker.communicator,
        bool(value),
        participants=ps,
        key_prefix=prefix,
    )


def verify_uniform_predicate(
    interpreter: _ExecContext,
    worker: _Worker,
    pred: bool,
    *,
    op: Operation | None = None,
    participants: Sequence[int] | None = None,
    name: str = "uniform_predicate",
) -> bool:
    """Verify that `pred` is uniform across participants.

    Uses AND/OR all-reduce to detect mismatch. If mismatch is detected, runs an
    allgather to provide a helpful error message. All participants execute the
    same comm steps to avoid deadlocks.
    """

    ps = resolve_participants(worker, op=op, participants=participants)

    all_and = allreduce_bool_and(
        interpreter,
        worker,
        bool(pred),
        op=op,
        participants=ps,
        name=f"{name}_and",
    )
    all_or = allreduce_bool_or(
        interpreter,
        worker,
        bool(pred),
        op=op,
        participants=ps,
        name=f"{name}_or",
    )

    if all_and != all_or:
        gathered = allgather_bool(
            interpreter,
            worker,
            bool(pred),
            op=op,
            participants=ps,
            name=f"{name}_gather",
        )
        ps_norm = algo.normalize_participants(worker.communicator, ps)
        dist = dict(zip(ps_norm, gathered, strict=True))
        raise RuntimeError(
            "simp.uniform_cond predicate is not uniform across participants: "
            f"participants={ps_norm}, values={dist}"
        )

    return bool(pred)
