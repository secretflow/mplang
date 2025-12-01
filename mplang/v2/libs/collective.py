# Copyright 2025 Ant Group Co., Ltd.
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

"""Collective communication library for multi-party data redistribution.

This module provides high-level collective operations built on top of
SIMP dialect primitives (shuffle_static, shuffle_dynamic, converge).

Design Philosophy:
- Single-controller perspective: all operations describe data flow from
  the orchestrator's view, not individual party's view
- MPObject represents distributed values across parties
- Operations transform the distribution pattern

Naming Convention:
- transfer: point-to-point (1 party → 1 party)
- replicate: broadcast (1 party → N parties, same value)
- distribute: scatter (1 party with N values → N parties, one each)
- collect: gather (N parties → 1 party, stacked)

Example:
    >>> from mplang.v2.libs.collective import transfer, replicate, distribute, collect
    >>> from mplang.v2.dialects.simp import constant, converge
    >>>
    >>> # Create data on party 0
    >>> x = constant((0,), 42)
    >>>
    >>> # Transfer to party 1
    >>> y = transfer(x, to=1)
    >>>
    >>> # Replicate to all parties
    >>> z = replicate(x, to=(0, 1, 2))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mplang.v2.dialects.simp import converge, shuffle_static
from mplang.v2.edsl import Object
from mplang.v2.edsl.typing import MPType

if TYPE_CHECKING:
    pass


# =============================================================================
# Helpers
# =============================================================================


def _get_parties(obj: Object) -> tuple[int, ...] | None:
    """Extract static parties from object type."""
    if isinstance(obj.type, MPType):
        return obj.type.parties
    return None


def _get_single_party(obj: Object) -> int:
    """Extract the single party from an object (must have exactly one).

    Args:
        obj: Object with static parties containing exactly one party

    Returns:
        The single party rank

    Raises:
        ValueError: If parties is None (dynamic) or has != 1 party
    """
    parties = _get_parties(obj)
    if parties is None:
        raise ValueError(
            "Operation requires static parties, got dynamic (parties=None)"
        )
    if len(parties) != 1:
        raise ValueError(
            f"Operation requires single-party source, got parties={parties}"
        )
    return parties[0]


def _require_static_parties(obj: Object, op_name: str) -> tuple[int, ...]:
    """Require and return static parties from object.

    Args:
        obj: Object to check
        op_name: Operation name for error message

    Returns:
        Static parties tuple

    Raises:
        ValueError: If parties is None (dynamic)
    """
    parties = _get_parties(obj)
    if parties is None:
        raise ValueError(
            f"{op_name} requires static parties, got dynamic (parties=None)"
        )
    return parties


# =============================================================================
# Point-to-Point Communication
# =============================================================================


def transfer(data: Object, *, to: int) -> Object:
    """Transfer data from one party to another.

    Single-controller perspective:
    - Input: MPObject held by exactly one party
    - Output: MPObject held by party `to`

    The source party is automatically inferred from data.type.parties.

    Args:
        data: Data to transfer (must have static parties with exactly one party)
        to: Target party rank

    Returns:
        Data held by party `to` (parties=(to,))

    Raises:
        ValueError: If data has dynamic parties or more than one party

    Example:
        >>> x = constant((0,), 42)  # x held by party 0
        >>> y = transfer(x, to=1)  # y held by party 1
        >>> y.type.parties  # (1,)
    """
    frm = _get_single_party(data)
    if frm == to:
        return data
    return shuffle_static(data, routing={to: frm})


# =============================================================================
# One-to-Many Operations
# =============================================================================


def replicate(data: Object, *, to: tuple[int, ...]) -> Object:
    """Replicate data from one party to multiple parties.

    Single-controller perspective:
    - Input: MPObject held by exactly one party
    - Output: MPObject replicated across all parties in `to`

    Each target party receives an identical copy of the data.

    Args:
        data: Data to replicate (must have static parties with exactly one party)
        to: Target party ranks (tuple)

    Returns:
        Data replicated across all target parties (parties=to)

    Raises:
        ValueError: If data has dynamic parties or more than one party

    Example:
        >>> x = constant((0,), 42)
        >>> y = replicate(x, to=(0, 1, 2))
        >>> y.type.parties  # (0, 1, 2)
        >>> # All three parties now hold the value 42
    """
    frm = _get_single_party(data)
    routing = dict.fromkeys(to, frm)
    return shuffle_static(data, routing=routing)


def distribute(values: list[Object], *, frm: int) -> Object:
    """Distribute a list of values from one party to multiple parties.

    Single-controller perspective:
    - Input: N MPObjects, all held by party `frm`
    - Output: 1 MPObject distributed across N parties (party i holds values[i])

    This is the inverse of collect().

    Args:
        values: List of N objects, all must be held by party `frm`
        frm: Source party rank

    Returns:
        Single MPObject with parties=(0, 1, ..., N-1)
        Party i holds the value from values[i]

    Raises:
        ValueError: If values is empty or any value is not held by `frm`

    Example:
        >>> xs = [constant((0,), i) for i in range(3)]  # all held by party 0
        >>> y = distribute(xs, frm=0)
        >>> y.type.parties  # (0, 1, 2)
        >>> # Party 0 has 0, party 1 has 1, party 2 has 2
    """
    if not values:
        raise ValueError("distribute requires at least one value")

    # Validate all values are held by frm
    for i, v in enumerate(values):
        parties = _get_parties(v)
        if parties is None:
            raise ValueError(
                f"distribute requires static parties, value[{i}] has dynamic parties"
            )
        if parties != (frm,):
            raise ValueError(
                f"distribute requires all values from party {frm}, "
                f"value[{i}] has parties={parties}"
            )

    pieces = [shuffle_static(v, routing={i: frm}) for i, v in enumerate(values)]
    return converge(*pieces)


# =============================================================================
# Many-to-One Operations
# =============================================================================


def collect(data: Object, *, to: int) -> list[Object]:
    """Collect distributed data to one party.

    Single-controller perspective:
    - Input: 1 MPObject distributed across N parties
    - Output: N MPObjects, each held by party `to`, preserving source order

    Note: Returns a list because we preserve the logical separation of values
    from different source parties. Use pcall_static to stack/concat if needed.

    Args:
        data: Distributed data (must have static parties)
        to: Target party rank

    Returns:
        List of N objects, all held by party `to`
        result[i] contains the value from source party i

    Raises:
        ValueError: If data has dynamic parties

    Example:
        >>> x = converge(x0, x1, x2)  # x.parties = (0, 1, 2)
        >>> ys = collect(x, to=0)  # List of 3 objects
        >>> ys[0].type.parties  # (0,)
        >>> ys[1].type.parties  # (0,)
        >>> # ys[0] has x0's value, ys[1] has x1's value, etc.
    """
    src_parties = _require_static_parties(data, "collect")
    return [shuffle_static(data, routing={to: src}) for src in src_parties]


# =============================================================================
# Many-to-Many Operations
# =============================================================================


def allreplicate(data: Object) -> list[Object]:
    """Replicate each party's data to all parties.

    Single-controller perspective:
    - Input: 1 MPObject distributed across N parties
    - Output: N MPObjects, each replicated across all N parties

    result[i] contains party i's original value, replicated to all parties.

    Args:
        data: Distributed data (must have static parties)

    Returns:
        List of N objects, each with parties equal to the original parties
        result[i] is the value from source party i, replicated to all parties

    Raises:
        ValueError: If data has dynamic parties

    Example:
        >>> x = converge(x0, x1, x2)  # x.parties = (0, 1, 2)
        >>> ys = allreplicate(x)  # List of 3 objects
        >>> ys[0].type.parties  # (0, 1, 2) - contains x0's value
        >>> ys[1].type.parties  # (0, 1, 2) - contains x1's value
    """
    src_parties = _require_static_parties(data, "allreplicate")

    result = []
    for src in src_parties:
        # Replicate from src to all parties
        routing = dict.fromkeys(src_parties, src)
        result.append(shuffle_static(data, routing=routing))
    return result


def permute(data: Object, *, mapping: dict[int, int]) -> Object:
    """Permute data according to a party mapping.

    Single-controller perspective:
    - Input: 1 MPObject distributed across parties
    - Output: 1 MPObject with data permuted according to mapping

    The mapping specifies: target_party -> source_party.
    This is a thin wrapper around shuffle_static for clarity.

    Args:
        data: Distributed data
        mapping: Dict mapping target_party -> source_party

    Returns:
        Permuted data with parties = tuple(sorted(mapping.keys()))

    Example:
        >>> x = converge(x0, x1)  # x.parties = (0, 1)
        >>> y = permute(x, mapping={0: 1, 1: 0})  # swap
        >>> # Party 0 now has x1's value, party 1 has x0's value
    """
    return shuffle_static(data, routing=mapping)
