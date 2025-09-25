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

"""Runtime execution context & explicit op->kernel binding setup.

`RuntimeContext` executes previously bound operations. Binding is performed
by `bind_all_ops()` which imports builtin kernel implementation modules and
manually maps each semantic op_type to a kernel_id (currently identical).

This keeps initialization explicit & deterministic (no import-order magic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mplang.backend import base
from mplang.backend.base import KernelContext, bind_op, get_kernel_for_op
from mplang.core.dtype import UINT8, DType
from mplang.core.pfunc import PFunction
from mplang.core.table import TableLike, TableType
from mplang.core.tensor import TensorLike, TensorType

# --- Binding bootstrap ---

_BOUND = False

# Enumerate builtin ops (semantic names). kernel_id is same string.
_BUILTIN_OPS = [
    # builtin
    "builtin.identity",
    "builtin.read",
    "builtin.write",
    "builtin.constant",
    "builtin.rank",
    "builtin.prand",
    "builtin.table_to_tensor",
    "builtin.tensor_to_table",
    "builtin.debug_print",
    "builtin.pack",
    "builtin.unpack",
    # crypto
    "crypto.keygen",
    "crypto.enc",
    "crypto.dec",
    "crypto.kem_keygen",
    "crypto.kem_derive",
    "crypto.hkdf",
    # phe
    "phe.keygen",
    "phe.encrypt",
    "phe.mul",
    "phe.add",
    "phe.decrypt",
    # spu
    "spu.seed_env",
    "spu.makeshares",
    "spu.reconstruct",
    "mlir.pphlo",
    # stablehlo
    "mlir.stablehlo",
    # sql
    "sql[duckdb]",
    # tee
    "tee.quote",
    "tee.attest",
]

_IMPLEMENTATION_MODULES = [
    ".builtin",
    ".crypto",
    ".phe",
    ".spu",
    ".stablehlo",
    ".sql_duckdb",
    ".tee",
]


def bind_all_ops(force: bool = False) -> None:
    """Import builtin implementation modules then bind op->kernel.

    Idempotent unless force=True.
    """
    global _BOUND
    if _BOUND and not force:
        return
    # import implementations to register kernels
    pkg = __name__.rsplit(".", 1)[0]
    for rel in _IMPLEMENTATION_MODULES:
        __import__(pkg + rel, fromlist=["*"])
    # perform 1:1 bindings
    for op in _BUILTIN_OPS:
        bind_op(op, op)
    _BOUND = True


# --- RuntimeContext ---


@dataclass
class RuntimeContext:
    rank: int
    world_size: int
    state: dict[str, dict[str, Any]]
    cache: dict[str, Any]

    @classmethod
    def create(cls, rank: int, world_size: int) -> RuntimeContext:
        bind_all_ops()  # ensure bindings loaded
        return cls(rank=rank, world_size=world_size, state={}, cache={})

    def run_kernel(self, pfunc: PFunction, arg_list: list[Any]) -> list[Any]:
        fn_type = pfunc.fn_type
        spec = get_kernel_for_op(fn_type)
        fn = spec.fn
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
