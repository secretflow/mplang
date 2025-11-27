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
def secure_add(x, y):
    return x + y


# 2. Encrypt (Public -> SPU)
# Assume x, y are on party 0
x_enc = spu.encrypt(x, spu_device)  # MP[SS[Tensor], (0,1,2)]
y_enc = spu.encrypt(y, spu_device)

# 3. Execute (SPU -> SPU)
z_enc = spu.call(secure_add, spu_device.parties, x_enc, y_enc)

# 4. Decrypt (SPU -> Public)
z = spu.decrypt(z_enc, target_party=0)
```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import spu.libspu as libspu
import spu.utils.frontend as spu_fe
from jax import ShapeDtypeStruct
from jax.tree_util import tree_flatten, tree_unflatten

import mplang2.edsl as el
import mplang2.edsl.typing as elt
from mplang.utils.func_utils import normalize_fn
from mplang2.dialects import simp, type_utils

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

makeshares_p = el.Primitive[tuple[el.Object, ...]]("spu.makeshares")
reconstruct_p = el.Primitive[el.Object]("spu.reconstruct")
exec_p = el.Primitive[Any]("spu.exec")


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
    *args: elt.SSType | elt.TensorType,
    executable: bytes,
    input_vis: list[libspu.Visibility],
    output_vis: list[libspu.Visibility],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[elt.ScalarType],
    input_names: list[str],
    output_names: list[str],
) -> tuple[elt.SSType, ...] | elt.SSType:
    """Execute SPU kernel on shares."""
    # Validate inputs are SS types or Tensor types
    for arg in args:
        if not (isinstance(arg, elt.SSType) or isinstance(arg, elt.TensorType)):
            raise TypeError(f"spu.exec expects SSType or TensorType inputs, got {arg}")

    # Outputs are SS[Tensor]
    outputs: list[elt.SSType[Any]] = []
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

    # shares_on_source is a tuple of MP objects (one per share).

    # 2. Distribute shares
    distributed_shares = []
    for i, target_rank in enumerate(device.parties):
        # Extract i-th share (still on source)
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

    return cast(el.Object, result)


def call(
    fn: Callable, parties: tuple[int, ...] | None, *args: Any, **kwargs: Any
) -> el.Object:
    """Execute a function on SPU parties.

    Args:
        fn: The function to execute.
        parties: The SPU parties. If None, inferred from inputs.
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    """

    # 1. Inspect inputs to determine SPU parties
    # Use normalize_fn to separate EDSL objects (variables) from raw values (immediates)
    def is_variable(arg: Any) -> bool:
        return isinstance(arg, el.Object)

    normalized_fn, in_vars = normalize_fn(fn, args, kwargs, is_variable)

    spu_parties = parties

    # Validate inputs and find common parties
    for arg in in_vars:
        if isinstance(arg.type, elt.MPType):
            if not (
                isinstance(arg.type.value_type, elt.SSType)
                or isinstance(arg.type.value_type, elt.TensorType)
            ):
                raise TypeError(
                    f"spu.call inputs must be SS-typed or Tensor-typed, got {arg.type}"
                )

            if spu_parties is None:
                spu_parties = arg.type.parties
            elif spu_parties != arg.type.parties:
                # If parties were explicitly provided, check consistency
                if parties is not None and parties != arg.type.parties:
                    raise ValueError(
                        f"Input parties {arg.type.parties} mismatch with explicit parties {parties}"
                    )
                # If inferred from previous args, check consistency
                elif parties is None and spu_parties != arg.type.parties:
                    raise ValueError("All inputs must be on the same SPU parties")
        elif isinstance(arg.type, elt.TensorType):
            # Host object (Public)
            pass
        else:
            raise TypeError(
                f"spu.call inputs must be MPType or TensorType, got {arg.type}"
            )

    if spu_parties is None:
        raise ValueError(
            "No inputs provided or dynamic parties not supported, and no explicit parties given"
        )

    # 2. Prepare for compilation
    # We need to cast SS inputs to Tensors (shares) for the pcall
    # But for compilation, we need the logical plaintext types

    jax_args_flat = []
    input_vis = []

    for arg in in_vars:
        if isinstance(arg.type, elt.MPType):
            # 1.1 MP Type (SPU Resident)
            val_type = arg.type.value_type
            if isinstance(val_type, elt.SSType):
                pt_type = val_type.pt_type
                vis = libspu.Visibility.VIS_SECRET
            elif isinstance(val_type, elt.TensorType):
                pt_type = val_type
                vis = libspu.Visibility.VIS_PUBLIC
            else:
                raise TypeError(f"Unsupported input type: {val_type}")
        else:
            # 1.2 Host Object (Local EDSL Object)
            # Treat as Public input
            pt_type = arg.type
            vis = libspu.Visibility.VIS_PUBLIC

        if not isinstance(pt_type, elt.TensorType):
            raise TypeError(f"spu.jit inputs must be Tensor-based, got {pt_type}")

        # Map to JAX
        jax_dtype = type_utils.elt_to_jax_dtype(
            cast(elt.ScalarType, pt_type.element_type)
        )
        shape = tuple(d if d != -1 else 1 for d in pt_type.shape)

        jax_args_flat.append(ShapeDtypeStruct(shape, jax_dtype))
        input_vis.append(vis)

    # 3. Compile
    # Note: normalized_fn takes a list of variables as input
    executable, output_info = spu_fe.compile(
        spu_fe.Kind.JAX,
        normalized_fn,
        [jax_args_flat],
        {},
        input_names=[f"in{i}" for i in range(len(in_vars))],
        input_vis=input_vis,
        outputNameGen=lambda outs: [f"out{i}" for i in range(len(outs))],
    )

    # 4. Define fused execution function (Exec)
    # This function runs on each SPU party locally.
    # It executes the kernel directly on SS inputs.

    # Extract output metadata (needed for both exec and unflatten)
    flat_outputs_info, out_tree = tree_flatten(output_info)

    def fused_exec(*local_ss_inputs: Any) -> Any:
        # 4.1. Execute SPU Kernel

        output_shapes = [out.shape for out in flat_outputs_info]

        output_dtypes = [
            type_utils.jax_to_elt_dtype(out.dtype) for out in flat_outputs_info
        ]
        output_vis_list = [libspu.Visibility.VIS_SECRET] * len(flat_outputs_info)

        res_shares = exec_p.bind(
            *local_ss_inputs,
            executable=executable.code,
            input_vis=input_vis,
            output_vis=output_vis_list,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            input_names=executable.input_names,
            output_names=executable.output_names,
        )

        return res_shares

    # 5. Execute on SPU parties
    # We pass the MP[SS] objects directly. pcall unwraps them to SS objects
    # for the local function, and wraps the returned SS objects back into MP[SS].
    result_ss = simp.pcall_static(spu_parties, fused_exec, *in_vars)

    # 6. Unflatten results to match original structure
    # pcall returns a single object if the function returns a single value,
    # or a tuple if it returns multiple. tree_unflatten expects a list of leaves.
    leaves = result_ss if isinstance(result_ss, tuple) else [result_ss]
    final_result = tree_unflatten(out_tree, leaves)

    return cast(el.Object, final_result)
