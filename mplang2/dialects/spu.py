"""SPU (Secure Processing Unit) dialect for the EDSL.

This dialect implements an "Encrypted Virtual Machine" model where the SPU
is treated as a logical device composed of multiple parties. It leverages
the `simp` dialect for data movement (encryption/decryption) and execution.

Concepts:
    - SPUDevice: Represents a set of parties forming the SPU.
    - encrypt: Moves data from a public party to the SPU (secret sharing).
    - decrypt: Moves data from the SPU to a public party (reconstruction).
    - run_jax: Executes JAX computations on the SPU.

Example:
```python
import jax.numpy as jnp
from mplang2.dialects import spu, tensor, simp
import mplang2.edsl.typing as elt

# 0. Setup
spu_device = spu.SPUDevice(parties=(0, 1, 2))


# 1. Define computation
@spu.jit
def secure_add(x, y):
    return x + y


# 2. Encrypt (Public -> SPU)
# Assume x, y are on party 0
x_enc = spu.encrypt(x, spu_device)  # MP[SS[Tensor], (0,1,2)]
y_enc = spu.encrypt(y, spu_device)

# 3. Execute (SPU -> SPU)
z_enc = secure_add(x_enc, y_enc)

# 4. Decrypt (SPU -> Public)
z = spu.decrypt(z_enc, target_party=0)
```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import spu.libspu as libspu
import spu.utils.frontend as spu_fe
from jax import ShapeDtypeStruct
from jax.tree_util import tree_flatten, tree_unflatten

import mplang2.edsl as el
import mplang2.edsl.typing as elt
from mplang2.dialects import simp

# ==============================================================================
# --- Configuration
# ==============================================================================


class SPUDevice(NamedTuple):
    """Configuration for an SPU device formed by a set of parties."""

    parties: tuple[int, ...]
    config: libspu.RuntimeConfig | None = None


# ==============================================================================
# --- Primitives (Local Operations)
# ==============================================================================

# These primitives operate locally on a single party.
# They are used inside simp.pcall to construct the distributed protocols.

makeshares_p = el.Primitive("spu.makeshares")
reconstruct_p = el.Primitive("spu.reconstruct")
exec_p = el.Primitive("spu.exec")


@makeshares_p.def_abstract_eval
def _makeshares_ae(data: elt.TensorType, *, count: int) -> tuple[elt.SSType, ...]:
    """Split a tensor into `count` secret shares."""
    if not isinstance(data, elt.TensorType):
        raise TypeError(f"makeshares expects TensorType, got {data}")
    # Shares have same shape/dtype as data (simplified additive sharing)
    # Return SS-typed shares directly
    return tuple(elt.SS(data) for _ in range(count))


@reconstruct_p.def_abstract_eval
def _reconstruct_ae(*shares: elt.SSType) -> elt.TensorType:
    """Reconstruct a tensor from shares."""
    if not shares:
        raise ValueError("reconstruct requires at least one share")
    first = shares[0]
    if not isinstance(first, elt.SSType):
        raise TypeError(f"reconstruct expects SSType shares, got {first}")
    if not isinstance(first.pt_type, elt.TensorType):
        raise TypeError(f"reconstruct expects SS[Tensor], got {first}")
    # Return the underlying plaintext type
    return first.pt_type


@exec_p.def_abstract_eval
def _exec_ae(
    *args: elt.SSType,
    executable: bytes,
    input_vis: list[libspu.Visibility],
    output_vis: list[libspu.Visibility],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[elt.ScalarType],
) -> tuple[elt.SSType, ...] | elt.SSType:
    """Execute SPU kernel on shares."""
    # Validate inputs are SS types
    for arg in args:
        if not isinstance(arg, elt.SSType):
            raise TypeError(f"spu.exec expects SSType inputs, got {arg}")

    # Outputs are SS[Tensor]
    outputs = []
    for shape, dtype in zip(output_shapes, output_dtypes, strict=True):
        outputs.append(elt.SS(elt.Tensor(dtype, shape)))

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


# ==============================================================================
# --- High-Level API (Distributed Protocols)
# ==============================================================================


def encrypt(data: el.Object, device: SPUDevice) -> el.Object:
    """Encrypt data by generating shares and distributing them to the SPU device.

    Protocol:
    1. Source party generates N shares locally.
    2. Shares are distributed to SPU parties (one share per party).
    3. Result is a logical SS value distributed across SPU parties.

    Args:
        data: Public tensor (MPType) on a single source party.
        device: Target SPU device configuration.

    Returns:
        MP[SS[Tensor], device.parties]

    Note:
        This implementation uses a generic "generate shares then distribute" flow
        via `simp` primitives. Backends may optimize this (e.g., using PRG seeds
        to generate shares locally on target parties) by intercepting the
        sequence of `makeshares` -> `shuffle` -> `converge`.
    """
    if not isinstance(data.type, elt.MPType):
        raise TypeError(f"encrypt expects MP-typed data, got {data.type}")

    source_parties = data.type.parties
    if source_parties is None:
        raise ValueError("encrypt requires static source party")
    if len(source_parties) != 1:
        raise ValueError(f"encrypt requires single source party, got {source_parties}")

    source_rank = source_parties[0]
    num_shares = len(device.parties)

    # 1. Generate shares on source party
    # Returns tuple of MP objects on source party
    shares_on_source = simp.pcall_static(
        (source_rank,), lambda x: makeshares_p.bind(x, count=num_shares), data
    )

    # shares_on_source is a PyTree (tuple) of MP objects.
    # We need to unpack it.
    # Since pcall returns the result structure, and makeshares returns a tuple,
    # shares_on_source should be a tuple of MP objects.
    # However, pcall returns a single Object if the function returns a single value,
    # or a PyTree of Objects.
    # Let's assume we can unpack it.

    # 2. Distribute shares
    distributed_shares = []
    for i, target_rank in enumerate(device.parties):
        # Extract i-th share (still on source)
        # Note: accessing element of tuple of MP objects
        share_i = shares_on_source[i]

        # Move to target party
        share_at_target = simp.shuffle_static(share_i, {target_rank: source_rank})
        distributed_shares.append(share_at_target)

    # 3. Converge to single logical variable
    # We have [MP[SS[T], (p0)], MP[SS[T], (p1)], ...]
    # Converge -> MP[SS[T], (p0, p1, ...)]
    converged_shares = simp.converge(*distributed_shares)

    return converged_shares


def decrypt(val: el.Object, target_party: int) -> el.Object:
    """Decrypt data by gathering shares and reconstructing.

    Protocol:
    1. SPU parties send their shares to target party.
    2. Target party reconstructs the secret.

    Args:
        val: Encrypted value (MP[SS[Tensor]]) on SPU parties.
        target_party: Rank of the party to receive the result.

    Returns:
        MP[Tensor, (target_party)]
    """
    if not isinstance(val.type, elt.MPType):
        raise TypeError(f"decrypt expects MP-typed value, got {val.type}")

    spu_parties = val.type.parties
    if spu_parties is None:
        raise ValueError("decrypt requires static SPU parties")

    # 1. Gather shares to target
    gathered_shares = []
    for source_rank in spu_parties:
        # Extract share from specific party
        # We use pcall to isolate the value on source_rank
        # This is effectively identity but narrows the parties
        share_on_source = simp.pcall_static((source_rank,), lambda x: x, val)

        # Move to target
        share_at_target = simp.shuffle_static(
            share_on_source, {target_party: source_rank}
        )
        gathered_shares.append(share_at_target)

    # 2. Reconstruct on target
    result = simp.pcall_static((target_party,), reconstruct_p.bind, *gathered_shares)

    return result


def jit(fn: Callable) -> Callable:
    """JIT compile a JAX function for SPU execution.

    The decorated function will:
    1. Compile the JAX function to SPU PPHLO.
    2. Execute the compiled kernel on the SPU parties using `simp.pcall`.
    """

    def wrapper(*args: el.Object, **kwargs: Any) -> el.Object:
        # 1. Inspect inputs to determine SPU parties
        flat_args, args_tree = tree_flatten((args, kwargs))
        spu_parties = None

        # Validate inputs and find common parties
        for arg in flat_args:
            if not isinstance(arg.type, elt.MPType):
                raise TypeError(f"spu.jit inputs must be MP-typed, got {arg.type}")
            if not isinstance(arg.type.value_type, elt.SSType):
                raise TypeError(f"spu.jit inputs must be SS-typed, got {arg.type}")

            if spu_parties is None:
                spu_parties = arg.type.parties
            elif spu_parties != arg.type.parties:
                raise ValueError("All inputs must be on the same SPU parties")

        if spu_parties is None:
            raise ValueError("No inputs provided or dynamic parties not supported")

        # 2. Prepare for compilation
        # We need to cast SS inputs to Tensors (shares) for the pcall
        # But for compilation, we need the logical plaintext types

        jax_args_flat = []
        input_vis = []

        for arg in flat_args:
            ss_type = arg.type.value_type
            pt_type = ss_type.pt_type
            if not isinstance(pt_type, elt.TensorType):
                raise TypeError(f"spu.jit inputs must be SS[Tensor], got {ss_type}")

            # Map to JAX
            dtype_map = {
                elt.f32: jnp.float32,
                elt.f64: jnp.float64,
                elt.i32: jnp.int32,
                elt.i64: jnp.int64,
            }
            jax_dtype = dtype_map.get(pt_type.element_type, jnp.float32)
            shape = tuple(d if d != -1 else 1 for d in pt_type.shape)

            jax_args_flat.append(ShapeDtypeStruct(shape, jax_dtype))
            input_vis.append(libspu.Visibility.VIS_SECRET)

        jax_args, jax_kwargs = tree_unflatten(args_tree, jax_args_flat)

        # 3. Compile
        def compiler_fn(*c_args, **c_kwargs):
            return fn(*c_args, **c_kwargs)

        executable, output_info = spu_fe.compile(
            spu_fe.Kind.JAX,
            compiler_fn,
            jax_args,
            jax_kwargs,
            input_names=[f"in{i}" for i in range(len(flat_args))],
            input_vis=input_vis,
            outputNameGen=lambda outs: [f"out{i}" for i in range(len(outs))],
        )

        # 4. Define fused execution function (Exec)
        # This function runs on each SPU party locally.
        # It executes the kernel directly on SS inputs.

        # Extract output metadata (needed for both exec and unflatten)
        flat_outputs_info, out_tree = tree_flatten(output_info)

        def fused_exec(*local_ss_inputs):
            # 4.1. Execute SPU Kernel
            output_shapes = [out.shape for out in flat_outputs_info]

            jax_to_elt = {
                jnp.dtype("float32"): elt.f32,
                jnp.dtype("float64"): elt.f64,
                jnp.dtype("int32"): elt.i32,
                jnp.dtype("int64"): elt.i64,
            }
            output_dtypes = [
                jax_to_elt.get(out.dtype, elt.f32) for out in flat_outputs_info
            ]
            output_vis_list = [libspu.Visibility.VIS_SECRET] * len(flat_outputs_info)

            res_shares = exec_p.bind(
                *local_ss_inputs,
                executable=executable.code,
                input_vis=input_vis,
                output_vis=output_vis_list,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
            )

            return res_shares

        # 5. Execute on SPU parties
        # We pass the MP[SS] objects directly. pcall unwraps them to SS objects
        # for the local function, and wraps the returned SS objects back into MP[SS].
        result_ss = simp.pcall_static(spu_parties, fused_exec, *flat_args)

        # 6. Unflatten results to match original structure
        # pcall returns a single object if the function returns a single value,
        # or a tuple if it returns multiple. tree_unflatten expects a list of leaves.
        leaves = result_ss if isinstance(result_ss, tuple) else [result_ss]
        final_result = tree_unflatten(out_tree, leaves)

        return final_result

    return wrapper
