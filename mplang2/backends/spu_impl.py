"""SPU Runtime Implementation.

Implements execution logic for SPU primitives using libspu.
"""

from typing import Any

import spu.libspu as libspu
from spu import Io, Runtime, Visibility

from mplang2.dialects import spu
from mplang2.edsl.graph import Operation
from mplang2.edsl.interpreter import Interpreter


def _get_link(rank: int, world_size: int) -> Any:
    # We construct the description deterministically based on world_size
    # This ensures all ranks use the same configuration.
    desc = libspu.link.Desc()  # type: ignore
    desc.recv_timeout_ms = 30 * 1000
    for i in range(world_size):
        desc.add_party(f"P{i}", f"mem:{i}")

    # Create the link for this rank specifically in the current thread.
    # This avoids potential issues with creating yacl::link::Context in one thread
    # and using it in another.
    return libspu.link.create_mem(desc, rank)


def _default_config() -> libspu.RuntimeConfig:
    """Default SPU config for simulation (SEMI2K, FM64)."""
    return libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SEMI2K,
        field=libspu.FieldType.FM64,
        fxp_fraction_bits=18,
    )


@spu.makeshares_p.def_impl
def makeshares_impl(interpreter: Interpreter, op: Operation, data: Any) -> Any:
    """Generate secret shares for data."""
    count = op.attrs["count"]
    # TODO: Get config from op or context. Using default for now.
    config = _default_config()

    # Io requires world size. 'count' is the number of shares we want.
    io = Io(count, config)

    # Generate shares
    # data is expected to be numpy array or compatible
    shares = io.make_shares(data, Visibility.VIS_SECRET)

    return tuple(shares)


@spu.reconstruct_p.def_impl
def reconstruct_impl(interpreter: Interpreter, op: Operation, *shares: Any) -> Any:
    """Reconstruct data from secret shares."""
    config = _default_config()
    io = Io(len(shares), config)
    return io.reconstruct(list(shares))


@spu.exec_p.def_impl
def exec_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Execute SPU kernel."""
    executable_code = op.attrs["executable"]
    output_shapes = op.attrs["output_shapes"]

    # Get runtime
    # We assume interpreter has rank/world_size (WorkerInterpreter)
    rank = getattr(interpreter, "rank", 0)
    world_size = getattr(interpreter, "world_size", 1)

    # Cache runtime on the interpreter instance
    rt_key = "_spu_runtime"
    if not hasattr(interpreter, rt_key):
        config = _default_config()
        link = _get_link(rank, world_size)
        rt = Runtime(link, config)
        setattr(interpreter, rt_key, rt)

    rt = getattr(interpreter, rt_key)

    # Prepare executable
    # Assume default naming convention used by jit
    input_names = [f"in{i}" for i in range(len(args))]
    output_names = [f"out{i}" for i in range(len(output_shapes))]

    executable = libspu.Executable(
        name="spu_kernel",
        input_names=input_names,
        output_names=output_names,
        code=executable_code,
    )

    # Set inputs
    for name, arg in zip(input_names, args, strict=True):
        rt.set_var(name, arg)

    # Run
    rt.run(executable)

    # Get outputs
    results = []
    for name in output_names:
        res = rt.get_var(name)
        results.append(res)

    if len(results) == 1:
        return results[0]
    return tuple(results)
