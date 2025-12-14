# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SPU Runtime Implementation.

Implements execution logic for SPU primitives using libspu.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

import numpy as np
import spu.api as spu_api
import spu.libspu as libspu

from mplang.v2.backends.simp_worker import SimpWorker
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import spu
from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import Interpreter
from mplang.v2.runtime.value import WrapValue

# =============================================================================
# SPU Share Wrapper
# =============================================================================


@serde.register_class
class SPUShareValue(WrapValue[libspu.Share]):
    """Wrapper for libspu.Share representing an SPU secret share.

    This wraps the external libspu library's Share type to provide
    proper serialization support via the Value base class.

    In-memory, we hold the libspu.Share directly to avoid copying.
    Serialization extracts meta/share_chunks when needed.
    """

    _serde_kind: ClassVar[str] = "spu_impl.SPUShareValue"

    def _convert(self, data: Any) -> libspu.Share:
        if isinstance(data, SPUShareValue):
            return data.unwrap()
        if isinstance(data, libspu.Share):
            return data
        raise TypeError(f"Expected libspu.Share, got {type(data)}")

    @property
    def libspu_share(self) -> libspu.Share:
        """Get the underlying libspu.Share object."""
        return self._data

    def to_json(self) -> dict[str, Any]:
        return {
            "meta": base64.b64encode(self._data.meta).decode("ascii"),
            "share_chunks": [
                base64.b64encode(chunk).decode("ascii")
                for chunk in self._data.share_chunks
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SPUShareValue:
        share = libspu.Share()
        share.meta = base64.b64decode(data["meta"])
        share.share_chunks = [
            base64.b64decode(chunk_b64) for chunk_b64 in data["share_chunks"]
        ]
        return cls(share)

    @classmethod
    def from_libspu(cls, share: libspu.Share) -> SPUShareValue:
        """Create SPUShareValue from a libspu.Share (zero-copy)."""
        return cls(share)


# =============================================================================
# SPU Config Helpers
# =============================================================================


def to_runtime_config(config: spu.SPUConfig) -> libspu.RuntimeConfig:
    """Convert SPUConfig to libspu.RuntimeConfig.

    This is a runtime-only function that maps the string-based configuration
    to libspu enums. Should only be called in the backend implementation.
    """
    runtime_config = libspu.RuntimeConfig()
    # ProtocolKind uses "SEMI2K" not "PROT_SEMI2K"
    runtime_config.protocol = getattr(libspu.ProtocolKind, config.protocol)
    runtime_config.field = getattr(libspu.FieldType, config.field)
    runtime_config.fxp_fraction_bits = config.fxp_fraction_bits
    return runtime_config


# Global cache for SPU runtimes per (local_rank, world_size) pair
# Key: (local_rank, spu_world_size, protocol, field, link_mode), Value: (Runtime, Io)
_SPU_RUNTIMES: dict[
    tuple[int, int, str, str, str], tuple[spu_api.Runtime, spu_api.Io]
] = {}


def _create_mem_link(local_rank: int, spu_world_size: int) -> libspu.link.Context:
    """Create in-memory link for simulation."""
    desc = libspu.link.Desc()  # type: ignore
    desc.recv_timeout_ms = 30 * 1000
    for i in range(spu_world_size):
        desc.add_party(f"P{i}", f"mem:{i}")
    return libspu.link.create_mem(desc, local_rank)


def _create_brpc_link(local_rank: int, spu_endpoints: list[str]) -> libspu.link.Context:
    """Create BRPC link for distributed execution.

    Args:
        local_rank: The local rank within the SPU device (0-indexed).
        spu_endpoints: List of BRPC endpoints for all SPU parties.

    Returns:
        A libspu.link.Context for BRPC communication.
    """
    desc = libspu.link.Desc()  # type: ignore
    desc.recv_timeout_ms = 100 * 1000  # 100 seconds
    desc.http_max_payload_size = 32 * 1024 * 1024  # 32MB

    for i, endpoint in enumerate(spu_endpoints):
        desc.add_party(f"P{i}", endpoint)

    return libspu.link.create_brpc(desc, local_rank)


def _get_spu_ctx(
    local_rank: int,
    spu_world_size: int,
    config: spu.SPUConfig,
    spu_endpoints: list[str] | None = None,
) -> tuple[spu_api.Runtime, spu_api.Io]:
    """Get or create SPU runtime and IO for the given local rank within SPU.

    Args:
        local_rank: The local rank within the SPU device (0-indexed).
        spu_world_size: The number of parties in the SPU device.
        config: SPU configuration including protocol settings.
        spu_endpoints: Optional list of BRPC endpoints. If None, use mem link.

    Returns:
        A tuple of (Runtime, Io) for this party.
    """
    # Determine link mode
    link_mode = "brpc" if spu_endpoints else "mem"

    # Include protocol, field, and link_mode in cache key
    cache_key = (local_rank, spu_world_size, config.protocol, config.field, link_mode)
    if cache_key in _SPU_RUNTIMES:
        return _SPU_RUNTIMES[cache_key]

    # Create Link
    if spu_endpoints:
        link = _create_brpc_link(local_rank, spu_endpoints)
    else:
        link = _create_mem_link(local_rank, spu_world_size)

    # Use config from SPUConfig
    runtime_config = to_runtime_config(config)

    # Create Runtime and Io
    runtime = spu_api.Runtime(link, runtime_config)
    io = spu_api.Io(spu_world_size, runtime_config)

    _SPU_RUNTIMES[cache_key] = (runtime, io)
    return runtime, io


@spu.makeshares_p.def_impl
def makeshares_impl(
    interpreter: Interpreter, op: Operation, data: TensorValue
) -> tuple[SPUShareValue, ...]:
    """Generate secret shares for data using spu.Io."""
    count = op.attrs["count"]
    config: spu.SPUConfig = op.attrs["config"]

    # We create a standalone Io for share generation (no link needed for make_shares)
    runtime_config = to_runtime_config(config)
    io = spu_api.Io(count, runtime_config)

    # Unwrap TensorValue
    arr = data.unwrap()

    # data is expected to be numpy array
    arr = np.asarray(arr)

    # Generate shares (VIS_SECRET)
    libspu_shares = io.make_shares(arr, libspu.Visibility.VIS_SECRET)

    # Wrap libspu.Share objects in SPUShareValue
    return tuple(SPUShareValue.from_libspu(share) for share in libspu_shares)


@spu.reconstruct_p.def_impl
def reconstruct_impl(
    interpreter: Interpreter, op: Operation, *shares: SPUShareValue
) -> TensorValue:
    """Reconstruct data from secret shares using spu.Io."""
    count = len(shares)
    config: spu.SPUConfig = op.attrs["config"]

    runtime_config = to_runtime_config(config)
    io = spu_api.Io(count, runtime_config)

    # Unwrap SPUShareValue to libspu.Share
    libspu_shares = [share.libspu_share for share in shares]

    # Reconstruct
    result = io.reconstruct(libspu_shares)

    # Wrap result as TensorValue
    return TensorValue.wrap(result)


@spu.exec_p.def_impl
def exec_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Execute SPU kernel using spu.Runtime.

    The SPU config must contain parties info to correctly map global rank
    to local SPU rank and determine SPU world size.
    """
    # Get SPU config from attrs (passed through from run_jax)
    config: spu.SPUConfig = op.attrs["config"]

    # Get parties from interpreter context (injected by pcall_static_impl)
    parties = getattr(interpreter, "current_parties", None)
    if parties is None:
        raise RuntimeError(
            "spu.exec requires 'current_parties' in interpreter context. "
            "Ensure it is called within a pcall_static block."
        )

    # Get global rank from interpreter or its context
    # Use SimpWorker if available
    context = interpreter.get_dialect_state("simp")
    if isinstance(context, SimpWorker):
        global_rank = context.rank
    else:
        # Fallback for other contexts or direct interpreter usage?
        # User said: "directly ensure simp_context is there"
        # If not SimpWorker, we can't run spu.exec?
        # But maybe integration tests run differently?
        # Let's trust user: "ensure simp_context is there"
        raise RuntimeError(f"spu.exec requires SimpWorker, got {type(context)}")

    if global_rank not in parties:
        raise RuntimeError(
            f"Global rank {global_rank} is not in current parties {parties}"
        )

    # Convert global rank to local SPU rank
    local_rank = parties.index(global_rank)
    spu_world_size = len(parties)

    # Get SPU endpoints from interpreter (set by WorkerInterpreter for BRPC mode)
    # spu_endpoints is a dict mapping global_rank -> brpc_endpoint
    spu_endpoints_map: dict[int, str] | None = getattr(
        interpreter, "spu_endpoints", None
    )
    if spu_endpoints_map is None:
        context = interpreter.get_dialect_state("simp")
        if context is not None:
            spu_endpoints_map = getattr(context, "spu_endpoints", None)

    # Build ordered list of endpoints for SPU parties
    spu_endpoints: list[str] | None = None
    if spu_endpoints_map is not None:
        spu_endpoints = []
        for party_rank in parties:
            if party_rank not in spu_endpoints_map:
                raise RuntimeError(
                    f"SPU endpoint not found for party {party_rank}. "
                    f"Available: {list(spu_endpoints_map.keys())}"
                )
            spu_endpoints.append(spu_endpoints_map[party_rank])

    runtime, io = _get_spu_ctx(local_rank, spu_world_size, config, spu_endpoints)

    executable_code = op.attrs["executable"]
    input_names = op.attrs["input_names"]
    output_names = op.attrs["output_names"]

    # Create Executable
    executable = libspu.Executable(
        name="spu_kernel",
        input_names=input_names,
        output_names=output_names,
        code=executable_code,
    )

    # Set inputs
    for name, share in zip(input_names, args, strict=True):
        # Handle SPUShareValue wrapper - unwrap to libspu.Share
        if isinstance(share, SPUShareValue):
            libspu_share = share.libspu_share
        else:
            # Handle public input (numpy array)
            # Generate shares with VIS_PUBLIC
            # make_shares expects numpy array
            if not isinstance(share, (np.ndarray, np.generic, int, float)):
                share = np.array(share)

            shares = io.make_shares(share, libspu.Visibility.VIS_PUBLIC)
            libspu_share = shares[local_rank]

        runtime.set_var(name, libspu_share)

    # Run
    runtime.run(executable)

    # Get outputs and wrap in SPUShareValue
    results = []
    for name in output_names:
        libspu_share = runtime.get_var(name)
        results.append(SPUShareValue.from_libspu(libspu_share))

    if len(results) == 1:
        return results[0]
    return tuple(results)
