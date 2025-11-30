"""SPU Runtime Implementation.

Implements execution logic for SPU primitives using libspu.
"""

from typing import Any

import numpy as np
import spu.api as spu_api
import spu.libspu as libspu

from mplang2.dialects import spu
from mplang2.edsl.graph import Operation
from mplang2.edsl.interpreter import Interpreter

# Global cache for SPU runtimes per (local_rank, world_size) pair
# Key: (local_rank, spu_world_size, protocol, field, link_mode), Value: (Runtime, Io)
_SPU_RUNTIMES: dict[
    tuple[int, int, str, str, str], tuple[spu_api.Runtime, spu_api.Io]
] = {}


def _create_mem_link(local_rank: int, spu_world_size: int) -> "libspu.link.Context":
    """Create in-memory link for simulation."""
    desc = libspu.link.Desc()  # type: ignore
    desc.recv_timeout_ms = 30 * 1000
    for i in range(spu_world_size):
        desc.add_party(f"P{i}", f"mem:{i}")
    return libspu.link.create_mem(desc, local_rank)


def _create_brpc_link(
    local_rank: int, spu_endpoints: list[str]
) -> "libspu.link.Context":
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
    runtime_config = config.to_runtime_config()

    # Create Runtime and Io
    runtime = spu_api.Runtime(link, runtime_config)
    io = spu_api.Io(spu_world_size, runtime_config)

    _SPU_RUNTIMES[cache_key] = (runtime, io)
    return runtime, io


@spu.makeshares_p.def_impl
def makeshares_impl(interpreter: Interpreter, op: Operation, data: Any) -> Any:
    """Generate secret shares for data using spu.Io."""
    count = op.attrs["count"]
    config: spu.SPUConfig = op.attrs["config"]

    # We create a standalone Io for share generation (no link needed for make_shares)
    runtime_config = config.to_runtime_config()
    io = spu_api.Io(count, runtime_config)

    # data is expected to be numpy array or scalar
    data = np.array(data)

    # Generate shares (VIS_SECRET)
    shares = io.make_shares(data, libspu.Visibility.VIS_SECRET)

    # shares is a list of libspu.Share (C++ objects)
    # We return them as a tuple. They will be moved by simp.shuffle.
    return tuple(shares)


@spu.reconstruct_p.def_impl
def reconstruct_impl(interpreter: Interpreter, op: Operation, *shares: Any) -> Any:
    """Reconstruct data from secret shares using spu.Io."""
    count = len(shares)
    config: spu.SPUConfig = op.attrs["config"]

    runtime_config = config.to_runtime_config()
    io = spu_api.Io(count, runtime_config)

    # Reconstruct
    result = io.reconstruct(list(shares))

    # Result is numpy array
    return result


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

    # Get global rank from interpreter
    global_rank = getattr(interpreter, "rank", None)
    if global_rank is None:
        raise RuntimeError(
            "spu.exec requires an interpreter with 'rank' attribute (e.g. WorkerInterpreter)."
        )

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
        if not isinstance(share, libspu.Share):
            # Handle public input (numpy array)
            # Generate shares with VIS_PUBLIC
            # make_shares expects numpy array
            if not isinstance(share, (np.ndarray, np.generic, int, float)):
                share = np.array(share)

            shares = io.make_shares(share, libspu.Visibility.VIS_PUBLIC)
            share = shares[local_rank]

        runtime.set_var(name, share)

    # Run
    runtime.run(executable)

    # Get outputs
    results = []
    for name in output_names:
        results.append(runtime.get_var(name))

    if len(results) == 1:
        return results[0]
    return tuple(results)
