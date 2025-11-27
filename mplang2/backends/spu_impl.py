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

# Global cache for SPU runtimes per rank
# Key: rank, Value: (Runtime, Io)
_SPU_RUNTIMES: dict[int, tuple[spu_api.Runtime, spu_api.Io]] = {}


def _get_spu_ctx(rank: int, world_size: int) -> tuple[spu_api.Runtime, spu_api.Io]:
    """Get or create SPU runtime and IO for the given rank."""
    if rank in _SPU_RUNTIMES:
        return _SPU_RUNTIMES[rank]

    # Create Link
    desc = libspu.link.Desc()  # type: ignore
    desc.recv_timeout_ms = 30 * 1000
    for i in range(world_size):
        desc.add_party(f"P{i}", f"mem:{i}")
    link = libspu.link.create_mem(desc, rank)

    # Create Config
    config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SEMI2K,
        field=libspu.FieldType.FM64,
        fxp_fraction_bits=18,
    )

    # Create Runtime and Io
    runtime = spu_api.Runtime(link, config)
    io = spu_api.Io(world_size, config)

    _SPU_RUNTIMES[rank] = (runtime, io)
    return runtime, io


@spu.makeshares_p.def_impl
def makeshares_impl(interpreter: Interpreter, op: Operation, data: Any) -> Any:
    """Generate secret shares for data using spu.Io."""
    count = op.attrs["count"]
    # We assume we are running on the source party, so we generate shares for everyone.
    # We need a config to create Io. We use a default config for now.
    # Note: In a real deployment, config should be consistent across parties.

    # We create a standalone Io for share generation (no link needed for make_shares)
    config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SEMI2K,
        field=libspu.FieldType.FM64,
        fxp_fraction_bits=18,
    )
    io = spu_api.Io(count, config)

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
    # We assume we are running on the target party and have received all shares.
    count = len(shares)

    config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SEMI2K,
        field=libspu.FieldType.FM64,
        fxp_fraction_bits=18,
    )
    io = spu_api.Io(count, config)

    # Reconstruct
    result = io.reconstruct(list(shares))

    # Result is numpy array
    return result


@spu.exec_p.def_impl
def exec_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Execute SPU kernel using spu.Runtime."""
    # We are running on an SPU party.
    # args are shares (libspu.Share) belonging to this party.

    # We need the rank of this party within the SPU device.
    # However, interpreter.rank is the global rank.
    # SPU device parties are defined in the op context?
    # Actually, spu.jit ensures all inputs are on SPU parties.
    # But we need to know the world_size of the SPU to initialize Runtime.
    # We can infer it from the number of parties involved in the pcall?
    # But exec_impl runs inside a pcall, so it doesn't know about other parties easily.

    # Assumption: The SPU world size is fixed for the simulation.
    # We expect the interpreter to provide world_size (e.g. WorkerInterpreter).

    rank = getattr(interpreter, "rank", 0)
    world_size = getattr(interpreter, "world_size", None)
    if world_size is None:
        raise RuntimeError(
            "spu.exec requires an interpreter with 'world_size' attribute (e.g. WorkerInterpreter)."
        )

    runtime, io = _get_spu_ctx(rank, world_size)

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
            share = shares[rank]

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
