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
# Key: (local_rank, spu_world_size, protocol, field), Value: (Runtime, Io)
_SPU_RUNTIMES: dict[tuple[int, int, str, str], tuple[spu_api.Runtime, spu_api.Io]] = {}


def _get_spu_ctx(
    local_rank: int, spu_world_size: int, config: spu.SPUConfig
) -> tuple[spu_api.Runtime, spu_api.Io]:
    """Get or create SPU runtime and IO for the given local rank within SPU.

    Args:
        local_rank: The local rank within the SPU device (0-indexed).
        spu_world_size: The number of parties in the SPU device.
        config: SPU configuration including protocol settings.

    Returns:
        A tuple of (Runtime, Io) for this party.
    """
    # Include protocol and field in cache key to avoid mismatches between tests
    cache_key = (local_rank, spu_world_size, config.protocol, config.field)
    if cache_key in _SPU_RUNTIMES:
        return _SPU_RUNTIMES[cache_key]

    # Create Link using local rank and SPU world size
    desc = libspu.link.Desc()  # type: ignore
    desc.recv_timeout_ms = 30 * 1000
    for i in range(spu_world_size):
        desc.add_party(f"P{i}", f"mem:{i}")
    link = libspu.link.create_mem(desc, local_rank)

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

    runtime, io = _get_spu_ctx(local_rank, spu_world_size, config)

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
