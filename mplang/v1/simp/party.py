# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import pathlib
import pkgutil
from collections.abc import Callable
from functools import wraps
from types import ModuleType
from typing import Any

from mplang.v1.ops.base import FeOperation
from mplang.v1.simp.api import run_at, run_jax_at
from mplang.v1.simp.mpi import p2p


def P2P(src: Party, dst: Party, value: Any) -> Any:
    """Point-to-point transfer using Party objects instead of raw ranks.

    Equivalent to ``p2p(src.rank, dst.rank, value)`` but improves readability
    and reduces magic numbers in user code / tutorials.

    Parameters
    ----------
    src : Party
        Source party object.
    dst : Party
        Destination party object.
    value : Any
        Value to transfer.

    Returns
    -------
    Any
        The same value representation at destination context (as defined by
        underlying ``p2p`` primitive semantics).
    """
    if not isinstance(src, Party) or not isinstance(dst, Party):  # defensive
        raise TypeError("P2P expects Party objects, e.g. P2P(P0, P2, value)")
    return p2p(src.rank, dst.rank, value)


"""Party-scoped module registration & dispatch.

This module provides a light-weight mechanism to expose *module-like* groups
of callable operations bound to a specific party (rank) via attribute access:

    load_module("mplang.ops.crypto", alias="crypto")
    P0.crypto.encrypt(x)  # executes encrypt() with pmask = {rank 0}

Core concepts:
* Registry (``_NAMESPACE_REGISTRY``): maps alias -> importable module path.
* Lazy import: underlying module is imported on first attribute access.
* Wrapping: fetched callables are wrapped so that invocation automatically
    routes through ``run_impl`` with that party's mask.

Only *callable* attributes are exposed; non-callable attributes raise
``AttributeError`` to avoid surprising divergent local vs. distributed
semantics.

The public API surface intentionally stays small (`Party`, `P`, `run`,
`runAt`, and `load_module`). Internal details (proxy class / registry) are
not part of the stability guarantee.
"""

_NAMESPACE_REGISTRY: dict[str, str] = {}


class _PartyModuleProxy:
    """Lazy module proxy bound to a specific party.

    Attribute access resolves a callable inside the registered module and
    returns a wrapped function that executes with the party's mask.
    Non-callable attributes are rejected explicitly to keep semantics clear.
    """

    def __init__(self, party: Party, name: str):
        self._party: Party = party
        self._name: str = name
        self._module: ModuleType | None = None  # loaded lazily

    def _ensure(self) -> None:
        if self._module is None:
            self._module = importlib.import_module(_NAMESPACE_REGISTRY[self._name])

    def __getattr__(self, item: str) -> Callable[..., Any]:
        self._ensure()
        op = getattr(self._module, item)
        if not callable(op):
            raise AttributeError(
                f"Attribute '{item}' of party module '{self._name}' is not callable (got {type(op).__name__})"
            )

        @wraps(op)
        def _wrapped(*args: Any, **kw: Any) -> Any:
            # Inline runAt to reduce an extra partial layer while preserving semantics.
            return run_at(self._party.rank, op, *args, **kw)

        # Provide a party-qualified name for debugging / logs without losing original metadata.
        base_name = getattr(op, "__name__", None)
        if base_name is None:
            # Frontend FeOperation or object without __name__; try .name attribute (FeOperation contract) or fallback to repr
            base_name = getattr(op, "name", None) or type(op).__name__
        try:
            _wrapped.__name__ = f"{base_name}@P{self._party.rank}"
        except Exception:  # pragma: no cover - assignment may fail for exotic wrappers
            pass
        return _wrapped


class Party:
    def __init__(self, rank: int) -> None:
        self.rank: int = int(rank)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Party(rank={self.rank})"

    def __call__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not callable(fn):
            raise TypeError(
                f"First argument to Party({self.rank}) must be callable, got {fn!r}"
            )
        # Use run_op_at for FeOperation, run_jax_at for plain callables
        if isinstance(fn, FeOperation):
            return run_at(self.rank, fn, *args, **kwargs)
        else:
            # TODO(jint): implicitly assume non-FeOperation as JAX function is a bit too magical?
            return run_jax_at(self.rank, fn, *args, **kwargs)

    def __getattr__(self, name: str) -> _PartyModuleProxy:
        if name in _NAMESPACE_REGISTRY:
            return _PartyModuleProxy(self, name)
        raise AttributeError(
            f"Party has no attribute '{name}'. Registered: {list(_NAMESPACE_REGISTRY)}"
        )


class _PartyIndex:
    def __getitem__(self, rank: int) -> Party:
        return Party(rank)


def _load_prelude_modules() -> None:
    """Auto-register public frontend submodules for party namespace access.

    Implementation detail: we treat every non-underscore immediate child of
    ``mplang.ops`` as public and make it available as ``P0.<name>``.
    This keeps user ergonomics high (no manual load_module calls for core
    frontends) but slightly increases implicit surface area. If this grows
    unwieldy we can switch to an allowlist.
    """
    try:
        import mplang.v1.ops as _fe  # type: ignore
    except (ImportError, ModuleNotFoundError):  # pragma: no cover
        # Frontend package not present (minimal install); safe to skip.
        return

    pkg_path = pathlib.Path(_fe.__file__).parent
    for m in pkgutil.iter_modules([str(pkg_path)]):
        if m.name.startswith("_"):
            continue
        if m.name not in _NAMESPACE_REGISTRY:
            _NAMESPACE_REGISTRY[m.name] = f"mplang.v1.ops.{m.name}"


def load_module(module: str, alias: str | None = None) -> None:
    """Register a module for party-scoped (per-rank) callable access.

    After registration, each party object (e.g. ``P0``) can access callable
    attributes of the target module through the chosen alias and have them
    executed under that party's mask automatically. Non-callable attributes
    are intentionally not exposed to avoid ambiguity between local data and
    distributed execution semantics.

    Parameters
    ----------
    module : str
        The fully-qualified import path of the module to expose. It must be
        importable via ``importlib.import_module``.
    alias : str | None, optional
        The short name used as an attribute on ``Party``/``P0``/``P1``/... .
        If omitted, the last path segment of ``module`` is used.

    Raises
    ------
    ValueError
        If the alias is already registered to a *different* module path.

    Notes
    -----
    Registration is idempotent when the alias maps to the same module. The
    actual module object is imported lazily on first attribute access, so
    calling ``load_module`` has negligible upfront cost.

    Examples
    --------
    >>> load_module("mplang.ops.crypto", alias="crypto")
    >>> # Now call an op on party 0
    >>> P0.crypto.encrypt(data)
    """
    if alias is None:
        alias = module.rsplit(".", 1)[-1]
    prev = _NAMESPACE_REGISTRY.get(alias)
    if prev and prev != module:
        raise ValueError(f"Alias '{alias}' already registered for '{prev}'")
    _NAMESPACE_REGISTRY[alias] = module


P = _PartyIndex()
P0, P1, P2 = Party(0), Party(1), Party(2)

_load_prelude_modules()
