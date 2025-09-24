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
from dataclasses import dataclass
from typing import Any

from mplang.core.dtype import UINT8, DType
from mplang.core.pfunc import PFunction
from mplang.core.table import TableLike, TableType
from mplang.core.tensor import TensorLike, TensorType

__all__ = [
    "BackendRuntime",
    "KernelContext",
    "create_runtime",
    "cur_kctx",
    "kernel_def",
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

# Canonical kernel callable signature (new style): (pfunc, *args) -> Any | sequence
# - No **kwargs (explicitly disallowed)
# - Return normalization handled by BackendRuntime.run_kernel
KernelFn = Callable[..., Any]

_KERNELS: dict[str, KernelFn] = {}


def _validate_table_arg(
    fn_type: str, arg_index: int, spec: TableType, value: Any
) -> None:
    if not isinstance(value, TableLike):
        raise TypeError(
            f"kernel {fn_type} input[{arg_index}] expects TableLike, got {type(value).__name__}"
        )
    if len(value.columns) != len(spec.columns):
        raise ValueError(
            f"kernel {fn_type} input[{arg_index}] column count mismatch: got {len(value.columns)}, expected {len(spec.columns)}"
        )


def _validate_tensor_arg(
    fn_type: str, arg_index: int, spec: TensorType, value: Any
) -> None:
    # Backend-only handle sentinel (e.g., PHE keys) bypasses all structural checks
    if tuple(spec.shape) == (-1, 0) and spec.dtype == UINT8:
        return

    if isinstance(value, (int, float, bool, complex)):
        val_shape: tuple[Any, ...] = ()
        duck_dtype: Any = type(value)
    else:
        if not isinstance(value, TensorLike):
            raise TypeError(
                f"kernel {fn_type} input[{arg_index}] expects TensorLike, got {type(value).__name__}"
            )
        val_shape = getattr(value, "shape", ())
        duck_dtype = getattr(value, "dtype", None)

    if len(spec.shape) != len(val_shape):
        raise ValueError(
            f"kernel {fn_type} input[{arg_index}] rank mismatch: got {val_shape}, expected {spec.shape}"
        )

    for dim_idx, (spec_dim, val_dim) in enumerate(
        zip(spec.shape, val_shape, strict=True)
    ):
        if spec_dim >= 0 and spec_dim != val_dim:
            raise ValueError(
                f"kernel {fn_type} input[{arg_index}] shape mismatch at dim {dim_idx}: got {val_dim}, expected {spec_dim}"
            )

    try:
        val_dtype = DType.from_any(duck_dtype)
    except (ValueError, TypeError):  # pragma: no cover
        raise TypeError(
            f"kernel {fn_type} input[{arg_index}] has unsupported dtype object {duck_dtype!r}"
        ) from None
    if val_dtype != spec.dtype:
        raise ValueError(
            f"kernel {fn_type} input[{arg_index}] dtype mismatch: got {val_dtype}, expected {spec.dtype}"
        )


def kernel_def(fn_type: str) -> Callable[[KernelFn], KernelFn]:
    """Decorator to register a backend kernel (new signature).

    Expected Python signature form:

        @kernel_def("namespace.op")
        def _op(pfunc: PFunction, *args): ...

    Rules:
      * First parameter MUST be the PFunction object.
      * Positional arguments correspond 1:1 to pfunc.ins_info order.
      * **kwargs are NOT supported (will raise at call site if used).
      * Return value forms accepted (n = len(pfunc.outs_info)):
          - n == 0: return None / () / []
          - n == 1: return scalar/object OR (value,) / [value]
          - n > 1 : return tuple/list of length n
        Anything else raises a ValueError.
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

        # Strict positional arg count validation (no kernel-managed arity bypass)
        if len(arg_list) != len(pfunc.ins_info):
            raise ValueError(
                f"kernel {fn_type} arg count mismatch: got {len(arg_list)}, expect {len(pfunc.ins_info)}"
            )

        for idx, (spec, val) in enumerate(zip(pfunc.ins_info, arg_list, strict=True)):
            if isinstance(spec, TableType):
                _validate_table_arg(fn_type, idx, spec, val)
                continue

            if isinstance(spec, TensorType):
                _validate_tensor_arg(fn_type, idx, spec, val)
                continue

            # Unknown spec type: silently skip validation (legacy behavior)
            continue

        kctx = KernelContext(
            rank=self.rank,
            world_size=self.world_size,
            state=self.state,
            cache=self.cache,
        )
        token = _CTX_VAR.set(kctx)
        try:
            raw = fn(pfunc, *arg_list)
        finally:
            _CTX_VAR.reset(token)

        # Normalize return values
        expected = len(pfunc.outs_info)
        if expected == 0:
            if raw in (None, (), []):
                return []
            raise ValueError(
                f"kernel {fn_type} should return no values; got {type(raw).__name__}"
            )

        # If multi-output expected, raw must be sequence of right length
        if expected == 1:
            if isinstance(raw, (tuple, list)):
                if len(raw) != 1:
                    raise ValueError(
                        f"kernel {fn_type} produced {len(raw)} outputs, expected 1"
                    )
                return [raw[0]]
            # Single object
            return [raw]

        # expected > 1
        if not isinstance(raw, (tuple, list)):
            raise TypeError(
                f"kernel {fn_type} must return sequence (len={expected}), got {type(raw).__name__}"
            )
        if len(raw) != expected:
            raise ValueError(
                f"kernel {fn_type} produced {len(raw)} outputs, expected {expected}"
            )
        return list(raw)

    # Optional helper
    def reset(self) -> None:  # pragma: no cover - simple
        self.state.clear()
        self.cache.clear()


def create_runtime(rank: int, world_size: int) -> BackendRuntime:
    """Factory for BackendRuntime (allows future policy injection)."""
    return BackendRuntime(rank, world_size)
