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

from collections.abc import Mapping
from typing import Any

from mplang.backend import base
from mplang.backend.base import KernelContext, get_kernel_spec, kernel_exists
from mplang.core.dtype import UINT8, DType
from mplang.core.pfunc import PFunction
from mplang.core.table import TableLike, TableType
from mplang.core.tensor import TensorLike, TensorType

# Default bindings
# Import kernel implementation modules explicitly so their @kernel_def entries
# register at import time. Keep imports grouped; alias with leading underscore
# to silence unused variable warnings without F401 pragmas.
_IMPL_IMPORTED = False


def _ensure_impl_imported() -> None:
    global _IMPL_IMPORTED
    if _IMPL_IMPORTED:
        return
    from mplang.backend import builtin as _impl_builtin  # noqa: F401
    from mplang.backend import crypto as _impl_crypto  # noqa: F401
    from mplang.backend import phe as _impl_phe  # noqa: F401
    from mplang.backend import spu as _impl_spu  # noqa: F401
    from mplang.backend import sql_duckdb as _impl_sql_duckdb  # noqa: F401
    from mplang.backend import stablehlo as _impl_stablehlo  # noqa: F401
    from mplang.backend import tee as _impl_tee  # noqa: F401

    _IMPL_IMPORTED = True


# imports consolidated above

_DEFAULT_BINDINGS: dict[str, str] = {
    # builtin
    "builtin.identity": "builtin.identity",
    "builtin.read": "builtin.read",
    "builtin.write": "builtin.write",
    "builtin.constant": "builtin.constant",
    "builtin.rank": "builtin.rank",
    "builtin.prand": "builtin.prand",
    "builtin.table_to_tensor": "builtin.table_to_tensor",
    "builtin.tensor_to_table": "builtin.tensor_to_table",
    "builtin.debug_print": "builtin.debug_print",
    "builtin.pack": "builtin.pack",
    "builtin.unpack": "builtin.unpack",
    # crypto
    "crypto.keygen": "crypto.keygen",
    "crypto.enc": "crypto.enc",
    "crypto.dec": "crypto.dec",
    "crypto.kem_keygen": "crypto.kem_keygen",
    "crypto.kem_derive": "crypto.kem_derive",
    "crypto.hkdf": "crypto.hkdf",
    # phe
    "phe.keygen": "phe.keygen",
    "phe.encrypt": "phe.encrypt",
    "phe.mul": "phe.mul",
    "phe.add": "phe.add",
    "phe.decrypt": "phe.decrypt",
    "phe.dot": "phe.dot",
    "phe.gather": "phe.gather",
    "phe.scatter": "phe.scatter",
    "phe.concat": "phe.concat",
    "phe.reshape": "phe.reshape",
    "phe.transpose": "phe.transpose",
    # spu
    "spu.seed_env": "spu.seed_env",
    "spu.makeshares": "spu.makeshares",
    "spu.reconstruct": "spu.reconstruct",
    "spu.run_pphlo": "spu.run_pphlo",
    # stablehlo
    "mlir.stablehlo": "mlir.stablehlo",
    # sql
    # generic SQL op; backend-specific kernel id for duckdb
    "sql.run": "duckdb.run_sql",
    # tee
    "tee.quote": "tee.quote",
    "tee.attest": "tee.attest",
}


# --- RuntimeContext ---


class RuntimeContext:
    """Per-runtime execution context with isolated op->kernel bindings.

    Parameters
    ----------
    rank : int
        Local rank of this participant.
    world_size : int
        Total number of participants.
    initial_bindings : Mapping[str, str] | None, optional
        Optional partial overrides applied on top of the default binding table
        during construction (override semantics, not replace). After
        initialization, all (re)binding must go through ``bind_op`` /
        ``rebind_op``.
    state / cache / stats : dict, optional
        Mutable pockets reused across kernel invocations. If omitted, new
        dictionaries are created.
    """

    __slots__ = ("_ibindings", "cache", "rank", "state", "stats", "world_size")

    def __init__(
        self,
        rank: int,
        world_size: int,
        initial_bindings: Mapping[str, str] | None = None,
        *,
        state: dict[str, dict[str, Any]] | None = None,
        cache: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        _ensure_impl_imported()
        self.rank = rank
        self.world_size = world_size
        # Merge defaults with user overrides (override semantics)
        self._ibindings: dict[str, str] = {
            **_DEFAULT_BINDINGS,
            **(initial_bindings or {}),
        }
        self.state = state if state is not None else {}
        self.cache = cache if cache is not None else {}
        self.stats = stats if stats is not None else {}
        self.stats.setdefault("op_calls", {})

    def run_kernel(self, pfunc: PFunction, arg_list: list[Any]) -> list[Any]:
        fn_type = pfunc.fn_type
        kid = self._ibindings.get(fn_type)
        if kid is None:
            raise NotImplementedError(f"no backend kernel registered for op {fn_type}")
        spec = get_kernel_spec(kid)
        fn = spec.fn  # kernel implementation
        if len(arg_list) != len(pfunc.ins_info):
            raise ValueError(
                f"kernel {fn_type} arg count mismatch: got {len(arg_list)}, expect {len(pfunc.ins_info)}"
            )
        for idx, (ins_spec, val) in enumerate(
            zip(pfunc.ins_info, arg_list, strict=True)
        ):
            if isinstance(ins_spec, TableType):
                _validate_table_arg(fn_type, idx, ins_spec, val)
                continue
            if isinstance(ins_spec, TensorType):
                _validate_tensor_arg(fn_type, idx, ins_spec, val)
                continue
        # install kernel context
        kctx = KernelContext(
            rank=self.rank,
            world_size=self.world_size,
            state=self.state,
            cache=self.cache,
        )
        token = base._CTX_VAR.set(kctx)  # type: ignore[attr-defined]
        try:
            raw = fn(pfunc, *arg_list)
        finally:
            base._CTX_VAR.reset(token)  # type: ignore[attr-defined]
        # Stats (best effort)
        try:
            op_calls = self.stats.setdefault("op_calls", {})
            op_calls[fn_type] = op_calls.get(fn_type, 0) + 1
        except Exception:  # pragma: no cover - never raise due to stats
            pass
        expected = len(pfunc.outs_info)
        if expected == 0:
            if raw in (None, (), []):
                return []
            raise ValueError(
                f"kernel {fn_type} should return no values; got {type(raw).__name__}"
            )
        if expected == 1:
            if isinstance(raw, (tuple, list)):
                if len(raw) != 1:
                    raise ValueError(
                        f"kernel {fn_type} produced {len(raw)} outputs, expected 1"
                    )
                return [raw[0]]
            return [raw]
        if not isinstance(raw, (tuple, list)):
            raise TypeError(
                f"kernel {fn_type} must return sequence (len={expected}), got {type(raw).__name__}"
            )
        if len(raw) != expected:
            raise ValueError(
                f"kernel {fn_type} produced {len(raw)} outputs, expected {expected}"
            )
        return list(raw)

    def reset(self) -> None:
        self.state.clear()
        self.cache.clear()

    # ---- explicit (re)binding API ----
    def bind_op(self, op_type: str, kernel_id: str, *, force: bool = False) -> None:
        """Bind an operation to a kernel for THIS context only.

        force=False (default) keeps existing binding (no silent override).
        """
        if not kernel_exists(kernel_id):
            raise KeyError(f"kernel_id {kernel_id} not registered")
        if not force and op_type in self._ibindings:
            return
        self._ibindings[op_type] = kernel_id

    def rebind_op(self, op_type: str, kernel_id: str) -> None:
        """Force rebind an operation to a different kernel (shorthand)."""
        self.bind_op(op_type, kernel_id, force=True)

    # Introspection helpers
    def list_bound_ops(self) -> list[str]:  # pragma: no cover - convenience
        return sorted(self._ibindings.keys())

    def get_binding(self, op_type: str) -> str | None:  # pragma: no cover
        return self._ibindings.get(op_type)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"RuntimeContext(rank={self.rank}, world_size={self.world_size}, "
            f"bound_ops={len(self._ibindings)})"
        )


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
