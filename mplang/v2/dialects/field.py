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

"""Field dialect: Finite Field Arithmetic.

This module defines the Intermediate Representation (IR) for field operations.
It contains:
1. Primitive Definitions (Abstract Operations)
2. Abstract Evaluation Rules (Type Inference)
3. Public API (Builder Functions)

Implementation logic (Backends) is strictly separated into `mplang/v2/backends/field_impl.py`.
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects import tensor

# =============================================================================
# Primitives
# =============================================================================

aes_expand_p = el.Primitive[el.Object]("field.aes_expand")
mul_p = el.Primitive[el.Object]("field.mul")
solve_okvs_p = el.Primitive[el.Object]("field.solve_okvs")
decode_okvs_p = el.Primitive[el.Object]("field.decode_okvs")
ldpc_encode_p = el.Primitive[el.Object]("field.ldpc_encode")

# Optimized Mega-Binning Primitives
solve_okvs_opt_p = el.Primitive[el.Object]("field.solve_okvs_opt")
decode_okvs_opt_p = el.Primitive[el.Object]("field.decode_okvs_opt")

# =============================================================================
# Abstract Evaluation (Type Inference)
# =============================================================================


@aes_expand_p.def_abstract_eval
def _aes_expand_ae(seeds_type: elt.TensorType, *, length: int) -> elt.TensorType:
    # seeds: (N, 2)
    # output: (N, length, 2) -> ALWAYS uint64
    n = seeds_type.shape[0]
    return elt.TensorType(elt.u64, (n, length, 2))


@mul_p.def_abstract_eval
def _mul_ae(a: elt.TensorType, b: elt.TensorType) -> elt.TensorType:
    return a


@solve_okvs_p.def_abstract_eval
def _solve_okvs_ae(
    key_type: elt.TensorType,
    val_type: elt.TensorType,
    seed_type: elt.TensorType,
    *,
    m: int,
) -> elt.TensorType:
    return elt.TensorType(val_type.element_type, (m, 2))


@decode_okvs_p.def_abstract_eval
def _decode_okvs_ae(
    key_type: elt.TensorType,
    store_type: elt.TensorType,
    seed_type: elt.TensorType,
) -> elt.TensorType:
    n = key_type.shape[0]
    return elt.TensorType(store_type.element_type, (n, 2))


@solve_okvs_opt_p.def_abstract_eval
def _solve_okvs_opt_ae(
    key_type: elt.TensorType,
    val_type: elt.TensorType,
    seed_type: elt.TensorType,
    *,
    m: int,
) -> elt.TensorType:
    return elt.TensorType(val_type.element_type, (m, 2))


@decode_okvs_opt_p.def_abstract_eval
def _decode_okvs_opt_ae(
    key_type: elt.TensorType,
    store_type: elt.TensorType,
    seed_type: elt.TensorType,
) -> elt.TensorType:
    n = key_type.shape[0]
    return elt.TensorType(store_type.element_type, (n, 2))


@ldpc_encode_p.def_abstract_eval
def _ldpc_encode_ae(
    message: elt.TensorType,
    indices: elt.TensorType,
    indptr: elt.TensorType,
    *,
    m: int,
    n: int,
) -> elt.TensorType:
    # message: (K, 2)
    # output: (M, 2) (usually N, 2 in silver context where M=N)
    # Wait, kernel computes (M, 2).
    return elt.TensorType(message.element_type, (m, 2))


# =============================================================================
# Public API
# =============================================================================


def aes_expand(seeds: el.Object, length: int) -> el.Object:
    """Expand seeds using AES-CTR PRG.

    Args:
        seeds: (N, 2) uint64 tensor (keys)
        length: Number of 128-bit blocks to generate per seed

    Returns:
        (N, length, 2) uint64 tensor
    """
    return aes_expand_p.bind(seeds, length=length)


def mul(a: el.Object, b: el.Object) -> el.Object:
    """GF(2^128) Multiplication."""
    return mul_p.bind(a, b)


def solve_okvs(
    keys: el.Object, values: el.Object, m: int, seed: el.Object
) -> el.Object:
    """Solve OKVS P for keys->values using C++ Kernel.
    Returns storage tensor of shape (m, 2).
    """
    return solve_okvs_p.bind(keys, values, seed, m=m)


def decode_okvs(keys: el.Object, storage: el.Object, seed: el.Object) -> el.Object:
    """Decode OKVS values from storage for keys.
    Returns decoded values of shape (N, 2).
    """
    return decode_okvs_p.bind(keys, storage, seed)


def solve_okvs_opt(
    keys: el.Object, values: el.Object, m: int, seed: el.Object
) -> el.Object:
    """Solve OKVS using Optimized Mega-Binning Kernel."""
    return solve_okvs_opt_p.bind(keys, values, seed, m=m)


def decode_okvs_opt(keys: el.Object, storage: el.Object, seed: el.Object) -> el.Object:
    """Decode OKVS using Optimized Mega-Binning Kernel."""
    return decode_okvs_opt_p.bind(keys, storage, seed)


def ldpc_encode(
    message: el.Object, h_indices: el.Object, h_indptr: el.Object, m: int, n: int
) -> el.Object:
    """Compute S = H * M using Sparse Matrix Multiplication kernel.

    Args:
        message: (N, 2) or (K, 2) input vector.
        h_indices: CSR indices.
        h_indptr: CSR indptr.
        m: Number of rows in H (Output size).
        n: Number of cols in H (Input size).

    Returns:
        (M, 2) output vector.
    """
    return ldpc_encode_p.bind(message, h_indices, h_indptr, m=m, n=n)


# =============================================================================
# Helpers (EDSL Composition)
# =============================================================================


def add(a: el.Object, b: el.Object) -> el.Object:
    """GF(2^128) Addition (XOR)."""
    return cast(el.Object, tensor.run_jax(jnp.bitwise_xor, a, b))


def sum(x: el.Object, axis: int | None = None) -> el.Object:
    """GF(2^128) Summation (XOR Sum)."""

    def _sum_impl(val: Any) -> Any:
        return jnp.bitwise_xor.reduce(val, axis=axis)

    return cast(el.Object, tensor.run_jax(_sum_impl, x))
