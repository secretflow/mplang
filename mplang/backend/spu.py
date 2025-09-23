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

from dataclasses import dataclass
from typing import Any

import numpy as np
import spu.api as spu_api
import spu.libspu as libspu

from mplang.backend.base import backend_kernel, cur_kctx
from mplang.core.mptype import TensorLike
from mplang.core.pfunc import PFunction
from mplang.runtime.link_comm import LinkCommunicator


def shape_spu_to_np(spu_shape: Any) -> tuple[int, ...]:
    """Convert SPU shape to numpy tuple."""
    return tuple(spu_shape.dims)


def dtype_spu_to_np(spu_dtype: Any) -> np.dtype:
    """Convert SPU dtype to numpy dtype."""
    MAP = {
        libspu.DataType.DT_F32: np.float32,
        libspu.DataType.DT_F64: np.float64,
        libspu.DataType.DT_I1: np.bool_,
        libspu.DataType.DT_I8: np.int8,
        libspu.DataType.DT_U8: np.uint8,
        libspu.DataType.DT_I16: np.int16,
        libspu.DataType.DT_U16: np.uint16,
        libspu.DataType.DT_I32: np.int32,
        libspu.DataType.DT_U32: np.uint32,
        libspu.DataType.DT_I64: np.int64,
        libspu.DataType.DT_U64: np.uint64,
    }
    return MAP[spu_dtype]  # type: ignore[return-value]


@dataclass
class SpuValue:
    """SPU value container for secure computation."""

    shape: tuple[int, ...]
    dtype: Any
    vtype: libspu.Visibility
    share: libspu.Share

    def __repr__(self) -> str:
        return f"SpuValue({self.shape},{self.dtype},{self.vtype})"


# SpuHandler removed (legacy handler API deprecated)


def _get_spu_config_and_world() -> tuple[libspu.RuntimeConfig, int]:
    kctx = cur_kctx()
    pocket = kctx.kernel_state.setdefault("spu", {})
    cfg = pocket.get("config")
    world = pocket.get("world")
    if cfg is None or world is None:
        raise RuntimeError("SPU kernel state not initialized (config/world)")
    return cfg, int(world)


def initialize_spu_runtime(
    config: libspu.RuntimeConfig,
    world_size: int,
    link_ctxs: list[LinkCommunicator] | None,
) -> None:
    """Seed SPU kernel state.

    If called inside a backend kernel execution, we use the active KernelContext.
    If called outside (e.g., test setup), we fall back to global kernel_state so
    that later kernel invocations see the config/world/link info.
    """
    try:
        kctx = cur_kctx()
        pocket = kctx.kernel_state.setdefault("spu", {})
    except RuntimeError:  # outside kernel execution
        from mplang.backend import base as _base  # local import to avoid cycle

        pocket = _base._KERNEL_STATE.setdefault("spu", {})  # type: ignore[attr-defined]
    # Always override to ensure clean test isolation
    pocket["config"] = config
    pocket["world"] = world_size
    if link_ctxs is not None:
        pocket["links"] = link_ctxs


@backend_kernel("spu.makeshares")
def _spu_makeshares(pfunc: PFunction, args: tuple) -> tuple:
    """Create SPU shares from input data.

    Args:
        pfunc: PFunction containing makeshares metadata
        args: Input data to be shared (single tensor)

    Returns:
        Tuple of SPU shares (SpuValue), one for each party.
    """
    assert len(args) == 1

    visibility_value = pfunc.attrs.get("visibility", libspu.Visibility.VIS_SECRET.value)
    if isinstance(visibility_value, int):
        visibility = libspu.Visibility(visibility_value)
    else:
        visibility = visibility_value

    arg = np.array(args[0], copy=False)
    cfg, world = _get_spu_config_and_world()
    spu_io = spu_api.Io(world, cfg)
    shares = spu_io.make_shares(arg, visibility)
    assert len(shares) == world, f"Expected {world} shares, got {len(shares)}"
    return tuple(
        SpuValue(
            shape=arg.shape,
            dtype=arg.dtype,
            vtype=visibility,
            share=share,
        )
        for share in shares
    )


@backend_kernel("spu.reconstruct")
def _spu_reconstruct(pfunc: PFunction, args: tuple) -> tuple:
    """Reconstruct plaintext data from SPU shares."""
    cfg, world = _get_spu_config_and_world()
    assert len(args) == world, f"Expected {world} shares, got {len(args)}"
    for i, arg in enumerate(args):
        if not isinstance(arg, SpuValue):
            raise ValueError(
                f"Input {i} must be SpuValue, got {type(arg)}. Reconstruction requires SPU shares as input."
            )
    spu_args: list[SpuValue] = list(args)  # type: ignore
    shares = [spu_arg.share for spu_arg in spu_args]
    spu_io = spu_api.Io(world, cfg)
    reconstructed = spu_io.reconstruct(shares)
    return (reconstructed,)


@backend_kernel("mlir.pphlo")
def _spu_run_mlir(pfunc: PFunction, args: tuple) -> tuple:
    """Execute compiled SPU function (mlir.pphlo) and return SpuValue outputs."""
    if pfunc.fn_type != "mlir.pphlo":
        raise ValueError(f"Unsupported format: {pfunc.fn_type}. Expected 'mlir.pphlo'")

    cfg, _ = _get_spu_config_and_world()
    pocket = cur_kctx().kernel_state.setdefault("spu", {})
    link_ctxs: list[LinkCommunicator] | None = pocket.get("links")
    rank = cur_kctx().rank
    link_ctx = None if link_ctxs is None or rank >= len(link_ctxs) else link_ctxs[rank]
    if link_ctx is None:
        raise RuntimeError(
            "Link context not set for this rank; cannot execute mlir.pphlo"
        )

        # Create the real SPU runtime
    spu_rt = spu_api.Runtime(link_ctx.get_lctx(), cfg)
    if spu_rt is None:  # pragma: no cover - defensive
        raise RuntimeError("SPU runtime not set up. Call setup() first.")

        # Validate that all inputs are SpuValue objects
    for i, arg in enumerate(args):
        if not isinstance(arg, SpuValue):
            raise ValueError(
                f"Input {i} must be SpuValue, got {type(arg)}. In real SPU environments, all inputs must be SpuValue objects."
            )

        # Cast for type checking (we've validated above)
    spu_args: list[SpuValue] = list(args)  # type: ignore

    # Reconstruct SPU executable from MLIR code and metadata
    if pfunc.fn_text is None:
        raise ValueError("PFunction does not contain executable data")
    if not isinstance(pfunc.fn_text, str):
        raise ValueError(f"Expected str, got {type(pfunc.fn_text)}")

        # Extract metadata for executable reconstruction
    attrs: dict[str, Any] = dict(pfunc.attrs or {})
    input_names = attrs.get("input_names", [])
    output_names = attrs.get("output_names", [])
    executable_name = attrs.get("executable_name", pfunc.fn_name)

    # Create executable from MLIR code and metadata
    executable = libspu.Executable(
        name=executable_name,
        input_names=input_names,
        output_names=output_names,
        code=pfunc.fn_text,
    )

    # Set input variables in SPU runtime
    for idx, spu_arg in enumerate(spu_args):
        spu_rt.set_var(input_names[idx], spu_arg.share)
    spu_rt.run(executable)
    shares = [spu_rt.get_var(out_name) for out_name in output_names]
    metas = [spu_rt.get_var_meta(out_name) for out_name in output_names]
    results: list[TensorLike] = [
        SpuValue(
            shape=shape_spu_to_np(meta.shape),
            dtype=dtype_spu_to_np(meta.data_type),
            vtype=meta.visibility,
            share=shares[idx],
        )
        for idx, meta in enumerate(metas)
    ]
    return tuple(results)
