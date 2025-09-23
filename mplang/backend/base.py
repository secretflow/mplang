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

"""Flat backend kernel registry & per-participant runtime.

Design revision:
- Global, stateless kernel function catalog (fn_type -> callable).
- BackendRuntime: per-rank state & cache; executes kernels.
- Legacy global helpers removed after full migration to explicit runtimes.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from collections.abc import Callable as TypingCallable
from dataclasses import dataclass
from typing import Any

from mplang.core.pfunc import PFunction

__all__ = [
    "BackendRuntime",
    "KernelContext",
    "backend_kernel",
    "create_runtime",
    "cur_kctx",
    "list_registered_kernels",
]

# ---------------- Context ----------------


@dataclass
class KernelContext:
    """Ephemeral call context set via contextvar while a kernel runs."""

    rank: int
    world_size: int
    state: dict[str, dict[str, Any]]  # backend namespace -> pocket
    cache: dict[str, Any]  # runtime-level shared cache (per BackendRuntime)


_CTX_VAR: contextvars.ContextVar[KernelContext | None] = contextvars.ContextVar(
    "_flat_backend_ctx", default=None
)


def cur_kctx() -> KernelContext:
    ctx = _CTX_VAR.get()
    if ctx is None:
        raise RuntimeError("cur_kctx() called outside backend kernel execution")
    return ctx


# ---------------- Registry ----------------

# Canonical kernel callable signature: (pfunc, args_tuple) -> tuple(outputs)
KernelFn = TypingCallable[[PFunction, tuple[Any, ...]], tuple[Any, ...]]

_KERNELS: dict[str, KernelFn] = {}


def backend_kernel(fn_type: str) -> TypingCallable[[KernelFn], KernelFn]:
    """Decorator to register a flat backend kernel.

    Kernel signature:  fn(pfunc: PFunction, args: tuple) -> tuple
    Return value length must equal len(pfunc.outs_info).
    """

    def _decorator(fn: KernelFn) -> KernelFn:
        if fn_type in _KERNELS:
            raise ValueError(f"duplicate backend kernel fn_type={fn_type}")
        _KERNELS[fn_type] = fn
        return fn

    return _decorator


def list_registered_kernels() -> list[str]:  # public API unchanged
    return sorted(_KERNELS.keys())


class BackendRuntime:
    """Per-rank backend execution environment.

    Holds mutable backend state (namespaced pockets) and a cache. Stateless
    kernel implementations look up their state through cur_kctx().
    """

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.state: dict[str, dict[str, Any]] = {}
        self.cache: dict[str, Any] = {}

    # Main entry
    def run_kernel(self, pfunc: PFunction, arg_list: list[Any]) -> list[Any]:
        fn_type = pfunc.fn_type
        fn = _KERNELS.get(fn_type)
        if fn is None:
            raise NotImplementedError(f"no backend kernel registered for {fn_type}")
        kctx = KernelContext(
            rank=self.rank,
            world_size=self.world_size,
            state=self.state,
            cache=self.cache,
        )
        token = _CTX_VAR.set(kctx)
        try:
            result = fn(pfunc, tuple(arg_list))
            if not isinstance(result, tuple):
                raise TypeError(
                    f"backend kernel {fn_type} must return tuple, got {type(result).__name__}"
                )
            if len(result) != len(pfunc.outs_info):
                raise ValueError(
                    f"backend kernel {fn_type} produced {len(result)} outputs, expected {len(pfunc.outs_info)}"
                )
            return list(result)
        finally:
            _CTX_VAR.reset(token)

    # Optional helper
    def reset(self) -> None:  # pragma: no cover - simple
        self.state.clear()
        self.cache.clear()


def create_runtime(rank: int, world_size: int) -> BackendRuntime:
    """Factory for BackendRuntime (allows future policy injection)."""
    return BackendRuntime(rank, world_size)


# Convenience alias for future import stability (keep old name references harmless)
Kernel = Callable  # type: ignore
BackendModule = None  # type: ignore
backend_module = None  # type: ignore
