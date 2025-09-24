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

from mplang.backend.base import cur_kctx, kernel_def
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


def _get_spu_pocket() -> dict[str, Any]:
    return cur_kctx().state.setdefault("spu", {})


def _get_spu_config_and_world() -> tuple[libspu.RuntimeConfig, int]:
    pocket = _get_spu_pocket()
    cfg = pocket.get("config")
    world = pocket.get("world")
    if cfg is None or world is None:
        raise RuntimeError("SPU kernel state not initialized (config/world)")
    return cfg, int(world)


def _register_spu_env(
    config: libspu.RuntimeConfig, world_size: int, link_ctx: LinkCommunicator | None
) -> None:
    """Register SPU config/world/link inside current kernel context.

    Idempotent: if config/world already set, they must match; link is recorded per rank.
    This replaces previous global fallback seeding logic.
    """
    pocket = _get_spu_pocket()
    prev_cfg = pocket.get("config")
    prev_world = pocket.get("world")
    if prev_cfg is None:
        pocket["config"] = config
        pocket["world"] = world_size
    else:
        # libspu RuntimeConfig may not implement __eq__; compare serialized repr
        same_cfg = (
            prev_cfg.SerializeToString() == config.SerializeToString()  # type: ignore[attr-defined]
            if hasattr(prev_cfg, "SerializeToString")
            and hasattr(config, "SerializeToString")
            else prev_cfg == config
        )
        if not (same_cfg and prev_world == world_size):
            raise RuntimeError("Conflicting SPU env registration")
    # Store single link per runtime (one runtime per rank)
    if link_ctx is not None:
        pocket["link"] = link_ctx


@kernel_def("spu.seed_env")
def _spu_seed_env(pfunc: PFunction, *args: Any) -> Any:
    """Backend kernel to seed SPU environment.

    NOTE: This is a control-plane style operation (side-effect: installs SPU
    config/link into the per-runtime state pocket) rather than a pure data
    transformation. It remains a kernel temporarily for minimal surface
    changes during the backend deglobalization refactor. Callers MUST invoke
    it explicitly via `runtime.run_kernel(seed_pfunc, [])`, never through
    `Evaluator.evaluate` (fast-path removed) to keep IR evaluation semantics
    clean. A future cleanup may promote this to a dedicated runtime helper
    (e.g. `seed_spu_env(runtime, config, world, link)`), at which point this
    kernel can be deprecated.

    Required attrs: config (RuntimeConfig), world (int)
    Optional attr: link (LinkCommunicator or None)
    """
    cfg = pfunc.attrs.get("config")
    world = pfunc.attrs.get("world")
    link_ctx = pfunc.attrs.get("link", None)
    if cfg is None or world is None:
        raise ValueError("spu.seed_env requires 'config' and 'world' attrs")
    _register_spu_env(cfg, int(world), link_ctx)
    return None


@kernel_def("spu.makeshares")
def _spu_makeshares(pfunc: PFunction, *args: Any) -> Any:
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


@kernel_def("spu.reconstruct")
def _spu_reconstruct(pfunc: PFunction, *args: Any) -> Any:
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
    return reconstructed


@kernel_def("mlir.pphlo")
def _spu_run_mlir(pfunc: PFunction, *args: Any) -> Any:
    """Execute compiled SPU function (mlir.pphlo) and return SpuValue outputs.

    Participation rule: a rank participates iff its entry in the stored
    link_ctx list is non-None. This allows us to allocate a world-sized list
    (indexed by global rank) and simply assign None for non-SPU parties.
    """
    if pfunc.fn_type != "mlir.pphlo":
        raise ValueError(f"Unsupported format: {pfunc.fn_type}. Expected 'mlir.pphlo'")

    cfg, _ = _get_spu_config_and_world()
    pocket = _get_spu_pocket()
    link_ctx: LinkCommunicator | None = pocket.get("link")
    if link_ctx is None:
        raise RuntimeError("Rank not participating in SPU; no link set via seed_env")

    # Lazy runtime cache
    spu_rt = pocket.get("runtime")
    if spu_rt is None:
        spu_rt = spu_api.Runtime(link_ctx.get_lctx(), cfg)
        pocket["runtime"] = spu_rt

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
