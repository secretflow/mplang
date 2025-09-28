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

"""Backend kernel registry: mapping kernel_id -> implementation.

This module provides a lightweight registry for backend kernel implementations.
It does not track or decide which kernel handles a given semantic operation;
that policy (op -> kernel_id) is managed externally by each ``RuntimeContext``.

Exposed primitives:
* ``@kernel_def(kernel_id)``: decorator to register a kernel implementation.
* ``get_kernel_spec(kernel_id)``: look up a previously registered kernel.
* ``cur_kctx()`` / ``KernelContext``: execution context available only
    inside a kernel body (rank, world_size, per-backend state pockets, and a
    runtime-wide cache shared by kernels of the same runtime instance).

No global op binding table exists here; callers resolve an op to a kernel_id
before invoking the kernel function.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "KernelContext",
    "KernelSpec",
    "cur_kctx",
    "get_kernel_spec",
    "kernel_exists",
    "list_kernels",
]


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
    """Return the current kernel execution context (only valid inside a kernel).

    Two storages:
      - state: namespaced pockets (dict[str, dict]) for backend-local mutable helpers
      - cache: global (per runtime) shared dict; prefer state unless truly cross-backend

    Examples:
      1) Compile cache::
            @kernel_def("mlir.stablehlo")
            def _exec(pfunc, args):
                ctx = cur_kctx()
                pocket = ctx.state.setdefault("stablehlo", {})
                cache = pocket.setdefault("compile_cache", {})
                text = pfunc.fn_text
                mod = cache.get(text)
                if mod is None:
                    mod = compile_mlir(text)
                    cache[text] = mod
                return run(mod, args)

      2) Deterministic RNG::
            @kernel_def("crypto.keygen")
            def _keygen(pfunc, args):
                ctx = cur_kctx()
                pocket = ctx.state.setdefault("crypto", {})
                rng = pocket.get("rng")
                if rng is None:
                    rng = np.random.default_rng(1234 + ctx.rank * 7919)
                    pocket["rng"] = rng
                return (rng.integers(0, 256, size=(32,), dtype=np.uint8),)
    """
    ctx = _CTX_VAR.get()
    if ctx is None:
        raise RuntimeError("cur_kctx() called outside backend kernel execution")
    return ctx


# ---------------- Registry ----------------

# Kernel callable signature: (pfunc, *args) -> Any | sequence (no **kwargs)
KernelFn = Callable[..., Any]


@dataclass
class KernelSpec:
    kernel_id: str
    fn: KernelFn
    meta: dict[str, Any]


# All registered kernel implementations: kernel_id -> spec
_KERNELS: dict[str, KernelSpec] = {}


def kernel_def(kernel_id: str, /, **meta: Any) -> Callable[[KernelFn], KernelFn]:
    """Decorator to register a concrete kernel implementation.

    This ONLY registers the implementation (kernel_id -> fn). It does NOT bind
    any op. Higher layer must call ``bind_op(op_type, kernel_id)`` explicitly.
    """

    def _decorator(fn: KernelFn) -> KernelFn:
        if kernel_id in _KERNELS:
            raise ValueError(f"duplicate kernel_id={kernel_id}")
        _KERNELS[kernel_id] = KernelSpec(kernel_id=kernel_id, fn=fn, meta=dict(meta))
        return fn

    return _decorator


def get_kernel_spec(kernel_id: str) -> KernelSpec:
    """Return KernelSpec for a registered kernel_id (no op binding lookup)."""
    spec = _KERNELS.get(kernel_id)
    if spec is None:
        raise KeyError(f"kernel_id {kernel_id} not registered")
    return spec


def list_kernels() -> list[str]:
    return sorted(_KERNELS.keys())


def kernel_exists(kernel_id: str) -> bool:
    """Return True if a kernel_id has been registered."""
    return kernel_id in _KERNELS
