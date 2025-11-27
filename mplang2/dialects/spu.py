"""SPU (Secure Processing Unit) dialect for the EDSL.

This dialect implements an "Encrypted Virtual Machine" model where the SPU
is treated as a logical device composed of multiple parties. It leverages
the `simp` dialect for data movement (encryption/decryption) and execution.

Concepts:
    - SPUDevice: Represents a set of parties forming the SPU.
    - make_shares: Generates secret shares on the source party.
    - reconstruct: Reconstructs secret from shares on the target party.
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
# Generate shares locally
x_shares = spu.make_shares(x, count=3)
y_shares = spu.make_shares(y, count=3)

# Distribute shares to SPU parties
x_dist = []
y_dist = []
for i, target in enumerate(spu_device.parties):
    x_dist.append(simp.shuffle_static(x_shares[i], {target: 0}))
    y_dist.append(simp.shuffle_static(y_shares[i], {target: 0}))

# Converge to logical SPU variables
x_enc = simp.converge(*x_dist)
y_enc = simp.converge(*y_dist)

# 3. Execute (SPU -> SPU)
z_enc = spu.run_jax(secure_add, spu_device.parties, x_enc, y_enc)

# 4. Decrypt (SPU -> Public)
# Gather shares to party 0
z_shares = []
for source in spu_device.parties:
    # Extract share from logical variable
    share = simp.pcall_static((source,), lambda x: x, z_enc)
    # Move to target
    z_shares.append(simp.shuffle_static(share, {0: source}))

# Reconstruct
z = spu.reconstruct(tuple(z_shares))
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
from mplang2.dialects import type_utils

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


def make_shares(data: el.Object, count: int) -> tuple[el.Object, ...]:
    """Generate shares locally (no transfer).

    This function should be called inside a `simp.pcall` region.

    Args:
        data: Local TensorType object.
        count: Number of shares to generate.

    Returns:
        Tuple of SSType objects (shares).
    """
    return makeshares_p.bind(data, count=count)


def reconstruct(shares: tuple[el.Object, ...]) -> el.Object:
    """Reconstruct data from shares locally (no transfer).

    This function should be called inside a `simp.pcall` region.

    Args:
        shares: Tuple of SSType objects (shares).

    Returns:
        TensorType object (reconstructed).
    """
    return reconstruct_p.bind(*shares)


def run_jax(fn: Callable, *args: Any, **kwargs: Any) -> el.Object:
    """Execute a function on SPU locally.

    This function should be called inside a `simp.pcall` region.
    It compiles the function and executes it using the SPU runtime.

    Args:
        fn: The function to execute.
        *args: Positional arguments (SSType or TensorType).
        **kwargs: Keyword arguments.
    """

    # 1. Inspect inputs
    # Use normalize_fn to separate EDSL objects (variables) from raw values (immediates)
    def is_variable(arg: Any) -> bool:
        return isinstance(arg, el.Object)

    normalized_fn, in_vars = normalize_fn(fn, args, kwargs, is_variable)

    # Validate inputs
    for arg in in_vars:
        if not (
            isinstance(arg.type, elt.SSType) or isinstance(arg.type, elt.TensorType)
        ):
            raise TypeError(
                f"spu.run_jax inputs must be SSType or TensorType, got {arg.type}"
            )

    # 2. Prepare for compilation
    jax_args_flat = []
    input_vis = []

    for arg in in_vars:
        if isinstance(arg.type, elt.SSType):
            pt_type = arg.type.pt_type
            vis = libspu.Visibility.VIS_SECRET
        elif isinstance(arg.type, elt.TensorType):
            pt_type = arg.type
            vis = libspu.Visibility.VIS_PUBLIC
        else:
            raise TypeError(f"Unsupported input type: {arg.type}")

        if not isinstance(pt_type, elt.TensorType):
            raise TypeError(f"spu.run_jax inputs must be Tensor-based, got {pt_type}")

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

    # 4. Execute SPU Kernel
    flat_outputs_info, out_tree = tree_flatten(output_info)
    output_shapes = [out.shape for out in flat_outputs_info]

    output_dtypes = [
        type_utils.jax_to_elt_dtype(out.dtype) for out in flat_outputs_info
    ]
    output_vis_list = [libspu.Visibility.VIS_SECRET] * len(flat_outputs_info)

    res_shares = exec_p.bind(
        *in_vars,
        executable=executable.code,
        input_vis=input_vis,
        output_vis=output_vis_list,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        input_names=executable.input_names,
        output_names=executable.output_names,
    )

    # 5. Unflatten results
    leaves = res_shares if isinstance(res_shares, tuple) else [res_shares]
    final_result = tree_unflatten(out_tree, leaves)

    return cast(el.Object, final_result)
