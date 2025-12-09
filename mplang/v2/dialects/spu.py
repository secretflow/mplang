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
from mplang.v2.dialects import spu, tensor, simp
import mplang.v2.edsl.typing as elt

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
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import spu.utils.frontend as spu_fe
from jax import ShapeDtypeStruct
from jax.tree_util import tree_flatten, tree_unflatten

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v1.utils.func_utils import normalize_fn
from mplang.v2.dialects import dtypes
from mplang.v2.edsl import serde

# ==============================================================================
# --- Configuration
# ==============================================================================


@serde.register_class
@dataclass(frozen=True)
class SPUConfig:
    """SPU configuration (subset of libspu.RuntimeConfig).

    Attributes:
        protocol: SPU protocol (e.g., "SEMI2K", "ABY3").
        field: SPU field type (e.g., "FM64", "FM128").
        fxp_fraction_bits: Fixed-point fraction bits.
    """

    protocol: str = "SEMI2K"
    field: str = "FM128"
    fxp_fraction_bits: int = 18

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SPUConfig:
        return cls(
            protocol=d.get("protocol", "SEMI2K"),
            field=d.get("field", "FM128"),
            fxp_fraction_bits=d.get("fxp_fraction_bits", 18),
        )

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "spu.SPUConfig"

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "field": self.field,
            "fxp_fraction_bits": self.fxp_fraction_bits,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SPUConfig:
        return cls(
            protocol=data["protocol"],
            field=data["field"],
            fxp_fraction_bits=data["fxp_fraction_bits"],
        )


# ==============================================================================
# --- Primitives (Local Operations)
# ==============================================================================

# These primitives operate locally on a single party.
# They are used inside simp.pcall to construct the distributed protocols.

makeshares_p = el.Primitive[tuple[el.Object, ...]]("spu.makeshares")
reconstruct_p = el.Primitive[el.Object]("spu.reconstruct")
exec_p = el.Primitive[Any]("spu.exec")


@makeshares_p.def_abstract_eval
def _makeshares_ae(
    data: elt.TensorType, *, count: int, config: SPUConfig
) -> tuple[elt.SSType, ...]:
    """Split a tensor into `count` secret shares."""
    if not isinstance(data, elt.TensorType):
        raise TypeError(f"makeshares expects TensorType, got {data}")
    # Shares have same shape/dtype as data (simplified additive sharing)
    # Return SS-typed shares directly
    return tuple(elt.SS(data) for _ in range(count))


@reconstruct_p.def_abstract_eval
def _reconstruct_ae(*shares: elt.SSType, config: SPUConfig) -> elt.TensorType:
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


# Visibility type for IR attrs (string-based, mapped to libspu.Visibility at runtime)
Visibility = Literal["secret", "public", "private"]


@exec_p.def_abstract_eval
def _exec_ae(
    *args: elt.SSType | elt.TensorType,
    executable: bytes,
    input_vis: list[Visibility],
    output_vis: list[Visibility],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[elt.ScalarType],
    input_names: list[str],
    output_names: list[str],
    config: SPUConfig,
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


def make_shares(
    config: SPUConfig, data: el.Object, count: int
) -> tuple[el.Object, ...]:
    """Generate shares locally (no transfer).

    This function should be called inside a `simp.pcall` region.

    Args:
        config: SPU configuration.
        data: Local TensorType object.
        count: Number of shares to generate.

    Returns:
        Tuple of SSType objects (shares).
    """
    return makeshares_p.bind(data, count=count, config=config)


def reconstruct(config: SPUConfig, shares: tuple[el.Object, ...]) -> el.Object:
    """Reconstruct data from shares locally (no transfer).

    This function should be called inside a `simp.pcall` region.

    Args:
        config: SPU configuration.
        shares: Tuple of SSType objects (shares).

    Returns:
        TensorType object (reconstructed).
    """
    return reconstruct_p.bind(*shares, config=config)


def run_jax(config: SPUConfig, fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Execute a function on SPU locally.

    This function should be called inside a `simp.pcall` region.
    It compiles the function and executes it using the SPU runtime.

    Args:
        config: SPU configuration.
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
    input_vis: list[Visibility] = []  # String-based visibility for IR

    for arg in in_vars:
        if isinstance(arg.type, elt.SSType):
            pt_type = arg.type.pt_type
            vis: Visibility = "secret"
        elif isinstance(arg.type, elt.TensorType):
            pt_type = arg.type
            vis = "public"
        else:
            raise TypeError(f"Unsupported input type: {arg.type}")

        if not isinstance(pt_type, elt.TensorType):
            raise TypeError(f"spu.run_jax inputs must be Tensor-based, got {pt_type}")

        # Map to JAX
        jax_dtype = dtypes.to_jax(cast(elt.ScalarType, pt_type.element_type))
        shape = tuple(d if d != -1 else 1 for d in pt_type.shape)

        jax_args_flat.append(ShapeDtypeStruct(shape, jax_dtype))
        input_vis.append(vis)

    # 3. Compile
    # Map string visibility to libspu.Visibility for spu_fe.compile
    # Import libspu only at compile time, not stored in IR
    import spu.libspu as libspu

    def vis_to_libspu(v: Visibility) -> libspu.Visibility:
        return (
            libspu.Visibility.VIS_SECRET
            if v == "secret"
            else libspu.Visibility.VIS_PUBLIC
        )

    # Note: normalized_fn takes a list of variables as input
    executable, output_info = spu_fe.compile(
        spu_fe.Kind.JAX,
        normalized_fn,
        [jax_args_flat],
        {},
        input_names=[f"in{i}" for i in range(len(in_vars))],
        input_vis=[vis_to_libspu(v) for v in input_vis],
        outputNameGen=lambda outs: [f"out{i}" for i in range(len(outs))],
    )

    # 4. Execute SPU Kernel
    flat_outputs_info, out_tree = tree_flatten(output_info)
    output_shapes = [out.shape for out in flat_outputs_info]

    output_dtypes = [dtypes.from_dtype(out.dtype) for out in flat_outputs_info]
    output_vis_list: list[Visibility] = ["secret"] * len(flat_outputs_info)

    res_shares = exec_p.bind(
        *in_vars,
        executable=executable.code,
        input_vis=input_vis,
        output_vis=output_vis_list,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        input_names=executable.input_names,
        output_names=executable.output_names,
        config=config,
    )

    # 5. Unflatten results
    if isinstance(res_shares, (tuple, list)):
        leaves = list(res_shares)
    else:
        leaves = [res_shares]
    final_result = tree_unflatten(out_tree, leaves)

    return final_result
