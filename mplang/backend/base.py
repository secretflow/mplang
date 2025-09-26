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

"""Backend kernel registry & per-participant runtime (explicit op->kernel binding).

This version decouples *kernel implementation registration* from *operation binding*.

Concepts:
    * kernel_id: unique identifier of a concrete backend implementation.
    * op_type: semantic operation name carried by ``PFunction.fn_type``.
    * bind_op(op_type, kernel_id): performed by higher layer (see ``backend.context``)
        to select which implementation handles an op. Runtime dispatch is now a 2-step:
        pfunc.fn_type -> active kernel_id -> KernelSpec.fn

The previous implicit "import == register+bind" coupling is removed. Kernel
modules only call ``@kernel_def(kernel_id)``. Default bindings are established
centrally (lazy) the first time a runtime executes a kernel.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "KernelContext",
    "KernelSpec",
    "bind_op",
    "cur_kctx",
    "get_kernel_for_op",
    "list_kernels",
    "list_ops",
    "unbind_op",
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

# Active op bindings: op_type -> kernel_id
_BINDINGS: dict[str, str] = {}


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


def bind_op(op_type: str, kernel_id: str, *, force: bool = True) -> None:
    """Bind an op_type to a registered kernel implementation.

    Args:
        op_type: Semantic operation name.
        kernel_id: Previously registered kernel identifier.
        force: If False and op_type already bound, keep existing binding.
               If True (default), overwrite.
    """
    if kernel_id not in _KERNELS:
        raise KeyError(f"kernel_id {kernel_id} not registered")
    if not force and op_type in _BINDINGS:
        return
    _BINDINGS[op_type] = kernel_id


def unbind_op(op_type: str) -> None:
    _BINDINGS.pop(op_type, None)


def get_kernel_for_op(op_type: str) -> KernelSpec:
    kid = _BINDINGS.get(op_type)
    if kid is None:
        # Tests expect NotImplementedError for unsupported operations
        raise NotImplementedError(f"no backend kernel registered for op {op_type}")
    spec = _KERNELS.get(kid)
    if spec is None:  # inconsistent state
        raise RuntimeError(f"active kernel_id {kid} missing spec")
    return spec


def list_kernels() -> list[str]:
    return sorted(_KERNELS.keys())


def list_ops() -> list[str]:
    return sorted(_BINDINGS.keys())
