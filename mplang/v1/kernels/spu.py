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
from typing import Any, ClassVar

import numpy as np
import spu.api as spu_api
import spu.libspu as libspu

from mplang.v1.core import (
    BOOL,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    DType,
    PFunction,
)
from mplang.v1.kernels.base import cur_kctx, kernel_def
from mplang.v1.kernels.value import (
    TensorValue,
    Value,
    ValueDecodeError,
    ValueProtoBuilder,
    ValueProtoReader,
    register_value,
)
from mplang.v1.protos.v1alpha1 import value_pb2 as _value_pb2
from mplang.v1.runtime.link_comm import LinkCommunicator


def shape_spu_to_np(spu_shape: Any) -> tuple[int, ...]:
    """Convert SPU shape to numpy tuple."""
    return tuple(spu_shape.dims)


def dtype_spu_to_mpl(spu_dtype: libspu.DataType) -> DType:
    """Convert libspu.DataType to MPLang DType."""
    MAP = {
        libspu.DataType.DT_F32: FLOAT32,
        libspu.DataType.DT_F64: FLOAT64,
        libspu.DataType.DT_I1: BOOL,
        libspu.DataType.DT_I8: INT8,
        libspu.DataType.DT_U8: UINT8,
        libspu.DataType.DT_I16: INT16,
        libspu.DataType.DT_U16: UINT16,
        libspu.DataType.DT_I32: INT32,
        libspu.DataType.DT_U32: UINT32,
        libspu.DataType.DT_I64: INT64,
        libspu.DataType.DT_U64: UINT64,
    }
    return MAP[spu_dtype]


@register_value
@dataclass
class SpuValue(Value):
    """SPU value container for secure computation (Value type)."""

    KIND: ClassVar[str] = "mplang.spu.SpuValue"
    WIRE_VERSION: ClassVar[int] = 1

    shape: tuple[int, ...]
    dtype: DType  # Now uses MPLang's unified DType
    vtype: libspu.Visibility
    share: libspu.Share

    def __repr__(self) -> str:
        return f"SpuValue({self.shape},{self.dtype},{self.vtype})"

    def to_proto(self) -> _value_pb2.ValueProto:
        """Serialize SpuValue to wire format.

        libspu.Share has two attributes:
        - meta: bytes (protobuf serialized metadata)
        - share_chunks: list[bytes] (the actual secret share data)

        Strategy: Store shape/dtype/vtype in runtime_attrs, concatenate share.meta + all chunks in payload.
        """
        # Store metadata in runtime_attrs; keep chunk lengths for payload splitting
        chunk_lengths = [len(chunk) for chunk in self.share.share_chunks]

        # Payload contains only share chunks (meta stored in attrs)
        payload = b""
        for chunk in self.share.share_chunks:
            payload += chunk

        return (
            ValueProtoBuilder(self.KIND, self.WIRE_VERSION)
            .set_attr("shape", list(self.shape))
            .set_attr("dtype", self.dtype.name)  # Serialize DType name
            .set_attr("vtype", int(self.vtype))
            .set_attr("share_meta", self.share.meta)
            .set_attr("chunk_lengths", chunk_lengths)
            .set_payload(payload)
            .build()
        )

    @classmethod
    def from_proto(cls, proto: _value_pb2.ValueProto) -> SpuValue:
        """Deserialize SpuValue from wire format."""
        reader = ValueProtoReader(proto)
        if reader.version != cls.WIRE_VERSION:
            raise ValueDecodeError(f"Unsupported SpuValue version {reader.version}")

        # Read metadata from runtime_attrs
        shape = tuple(reader.get_attr("shape"))
        dtype_name = reader.get_attr("dtype")
        # Reconstruct DType from serialized name (numpy dtype string)
        dtype = DType.from_numpy(dtype_name)
        vtype = libspu.Visibility(reader.get_attr("vtype"))
        share_meta = reader.get_attr("share_meta")
        chunk_lengths = reader.get_attr("chunk_lengths")

        # Parse payload: [chunk_0][chunk_1]...
        payload = reader.payload
        offset = 0

        share_chunks: list[bytes] = []
        for chunk_len in chunk_lengths:
            chunk = payload[offset : offset + chunk_len]
            offset += chunk_len
            share_chunks.append(chunk)

        # Reconstruct libspu.Share
        share = libspu.Share()
        share.meta = share_meta
        share.share_chunks = share_chunks

        return cls(
            shape=shape,
            dtype=dtype,
            vtype=vtype,
            share=share,
        )


def _get_spu_config_and_world() -> tuple[libspu.RuntimeConfig, int]:
    kctx = cur_kctx()
    cfg = kctx.runtime.get_state("spu.config")
    world = kctx.runtime.get_state("spu.world")
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
    kctx = cur_kctx()
    prev_cfg = kctx.runtime.get_state("spu.config")
    prev_world = kctx.runtime.get_state("spu.world")
    if prev_cfg is None:
        kctx.runtime.set_state("spu.config", config)
        kctx.runtime.set_state("spu.world", world_size)
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
        kctx.runtime.set_state("spu.link", link_ctx)


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
def _spu_makeshares(pfunc: PFunction, tensor: TensorValue) -> tuple[SpuValue, ...]:
    """Create SPU shares from input TensorValue data."""
    visibility_value = pfunc.attrs.get("visibility", libspu.Visibility.VIS_SECRET.value)
    if isinstance(visibility_value, int):
        visibility = libspu.Visibility(visibility_value)
    else:
        visibility = visibility_value

    arg = tensor.to_numpy()
    cfg, world = _get_spu_config_and_world()
    spu_io = spu_api.Io(world, cfg)
    shares = spu_io.make_shares(arg, visibility)
    assert len(shares) == world, f"Expected {world} shares, got {len(shares)}"
    # Store MPLang DType instead of libspu.DataType
    dtype = DType.from_numpy(arg.dtype)
    return tuple(
        SpuValue(
            shape=arg.shape,
            dtype=dtype,
            vtype=visibility,
            share=share,
        )
        for share in shares
    )


@kernel_def("spu.reconstruct")
def _spu_reconstruct(pfunc: PFunction, *shares: SpuValue) -> TensorValue:
    """Reconstruct plaintext data from SPU shares."""
    cfg, world = _get_spu_config_and_world()
    assert len(shares) == world, f"Expected {world} shares, got {len(shares)}"
    for i, share in enumerate(shares):
        if not isinstance(share, SpuValue):
            raise ValueError(
                f"Input {i} must be SpuValue, got {type(share)}. Reconstruction requires SPU shares as input."
            )
    spu_args: list[SpuValue] = list(shares)  # type: ignore
    share_payloads = [spu_arg.share for spu_arg in spu_args]
    spu_io = spu_api.Io(world, cfg)
    reconstructed = spu_io.reconstruct(share_payloads)
    base = np.array(reconstructed, copy=False)
    # Respect semantic dtype/shape recorded on shares (all shares share same meta).
    semantic_dtype = shares[0].dtype.to_numpy()  # DType now has to_numpy() method
    semantic_shape = shares[0].shape
    restored = np.asarray(base, dtype=semantic_dtype).reshape(semantic_shape)
    return TensorValue(np.array(restored, copy=False))


@kernel_def("spu.run_pphlo")
def _spu_run_mlir(pfunc: PFunction, *args: SpuValue) -> tuple[SpuValue, ...]:
    """Execute compiled SPU function (spu.run_pphlo) and return SpuValue outputs.

    Participation rule: a rank participates iff its entry in the stored
    link_ctx list is non-None. This allows us to allocate a world-sized list
    (indexed by global rank) and simply assign None for non-SPU parties.
    """
    if pfunc.fn_type != "spu.run_pphlo":
        raise ValueError(
            f"Unsupported format: {pfunc.fn_type}. Expected 'spu.run_pphlo'"
        )

    cfg, _ = _get_spu_config_and_world()
    kctx = cur_kctx()
    link_ctx = kctx.runtime.get_state("spu.link")
    if link_ctx is None:
        raise RuntimeError("Rank not participating in SPU; no link set via seed_env")

    # Lazy runtime cache under key spu.runtime
    spu_rt = kctx.runtime.get_state("spu.runtime")
    if spu_rt is None:
        spu_rt = spu_api.Runtime(link_ctx.get_lctx(), cfg)
        kctx.runtime.set_state("spu.runtime", spu_rt)

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
    results: list[SpuValue] = [
        SpuValue(
            shape=shape_spu_to_np(meta.shape),
            dtype=dtype_spu_to_mpl(meta.data_type),
            vtype=meta.visibility,
            share=shares[idx],
        )
        for idx, meta in enumerate(metas)
    ]
    return tuple(results)
