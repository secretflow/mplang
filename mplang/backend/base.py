"""Flat backend kernel registry.

This supersedes the earlier BackendModule abstraction. Kernels are registered
globally via @backend_kernel(fn_type="<namespace>.<op>") and executed by the
evaluator through fn_type lookup.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mplang.core.pfunc import PFunction

__all__ = [
    "KernelContext",
    "backend_kernel",
    "cur_kctx",
    "initialize_backend",
    "list_registered_kernels",
]

# ---------------- Context ----------------


@dataclass
class KernelContext:
    rank: int
    world_size: int
    global_state: dict[str, Any]
    kernel_state: dict[str, dict[str, Any]]  # per fn_type mutable pocket
    cache: dict[str, Any]


_CTX_VAR: contextvars.ContextVar[KernelContext | None] = contextvars.ContextVar(
    "_flat_backend_ctx", default=None
)


def cur_kctx() -> KernelContext:
    ctx = _CTX_VAR.get()
    if ctx is None:
        raise RuntimeError("cur_kctx() called outside backend kernel execution")
    return ctx


# ---------------- Registry ----------------

_KERNELS: dict[str, Callable[[PFunction, tuple], tuple]] = {}
_KERNEL_STATE: dict[str, dict[str, Any]] = {}
_GLOBAL_STATE: dict[str, Any] = {}
_HANDLERS: list[object] = []  # deprecated; kept to avoid import errors


def backend_kernel(
    fn_type: str,
) -> Callable[
    [Callable[[PFunction, tuple], tuple]], Callable[[PFunction, tuple], tuple]
]:
    """Decorator to register a flat backend kernel.

    Kernel signature:  fn(pfunc: PFunction, args: tuple) -> tuple
    Return value length must equal len(pfunc.outs_info).
    """

    def _decorator(fn: Callable[[PFunction, tuple], tuple]):
        if fn_type in _KERNELS:
            raise ValueError(f"duplicate backend kernel fn_type={fn_type}")
        _KERNELS[fn_type] = fn
        return fn

    return _decorator


def register_handler_as_kernels(_handler):  # pragma: no cover - deprecated
    raise RuntimeError(
        "register_handler_as_kernels deprecated; rewrite backend with @backend_kernel"
    )


def list_registered_kernels() -> list[str]:
    return sorted(_KERNELS.keys())


def initialize_backend(rank: int, world_size: int) -> None:
    """Initialize global backend context (idempotent per rank).

    (Legacy handler.setup removed.)
    """

    _GLOBAL_STATE["rank"] = rank
    _GLOBAL_STATE["world_size"] = world_size
    # legacy handlers no longer supported


def run_kernel(pfunc: PFunction, arg_list: list[Any]) -> list[Any]:
    fn_type = pfunc.fn_type
    if fn_type not in _KERNELS:
        raise NotImplementedError(f"no backend kernel registered for {fn_type}")
    fn = _KERNELS[fn_type]
    kctx = KernelContext(
        rank=_GLOBAL_STATE.get("rank", -1),
        world_size=_GLOBAL_STATE.get("world_size", -1),
        global_state=_GLOBAL_STATE,
        kernel_state=_KERNEL_STATE,
        cache={},  # placeholder for future shared cache
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


# Convenience alias for future import stability (keep old name references harmless)
Kernel = Callable  # type: ignore
BackendModule = None  # type: ignore
backend_module = None  # type: ignore
