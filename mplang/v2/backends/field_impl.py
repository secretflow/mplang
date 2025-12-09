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

"""Field Backend Implementation.

Implements runtime execution logic for Field dialect primitives,
including bindings to C++ kernels (libmplang_kernels.so) and
NumPy fallbacks where appropriate.
"""

from __future__ import annotations

import ctypes
import os
import threading

import jax.numpy as jnp
import numpy as np

from mplang.v2.backends.tensor_impl import TensorValue, _unwrap, _wrap
from mplang.v2.dialects import field
from mplang.v2.edsl.graph import Operation
from mplang.v2.edsl.interpreter import Interpreter

# =============================================================================
# Kernel Loading
# =============================================================================

# Load Kernel Library
# In a real package, this path would be resolved robustly
_KERNEL_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "kernels", "libmplang_kernels.so"
)
_LIB = None
_LIB_LOCK = threading.Lock()


def _get_lib():
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            try:
                _LIB = ctypes.CDLL(_KERNEL_LIB_PATH)
                # Define signatures
                _LIB.gf128_mul.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                _LIB.gf128_mul_batch.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_int64,
                ]
                _LIB.solve_okvs.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),  # keys
                    ctypes.POINTER(ctypes.c_uint64),  # values
                    ctypes.POINTER(ctypes.c_uint64),  # output
                    ctypes.c_uint64,  # n
                    ctypes.c_uint64,  # m
                ]
                _LIB.decode_okvs.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),  # keys
                    ctypes.POINTER(ctypes.c_uint64),  # storage
                    ctypes.POINTER(ctypes.c_uint64),  # output
                    ctypes.c_uint64,  # n
                    ctypes.c_uint64,  # m
                ]
                _LIB.aes_128_expand.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),  # seeds
                    ctypes.POINTER(ctypes.c_uint64),  # output
                    ctypes.c_uint64,  # num_seeds
                    ctypes.c_uint64,  # length
                ]
            except OSError:
                print(f"WARNING: Could not load kernels from {_KERNEL_LIB_PATH}")
    return _LIB


# =============================================================================
# Helper Implementations (C++ Wrappers)
# =============================================================================


def _gf128_mul_impl(a, b):
    # a, b are numpy arrays (uint64) usually (N, 2)

    lib = _get_lib()
    if lib is None:
        return a ^ b

    # Enforce contiguous C-order arrays (important for ctypes)
    # Use ascontiguousarray to avoid copy if already contiguous
    a_contig = np.ascontiguousarray(a, dtype=np.uint64)
    b_contig = np.ascontiguousarray(b, dtype=np.uint64)
    out = np.zeros_like(a_contig)

    # Calculate number of elements
    # Assumes last dim is 2.
    # Total uint64 count / 2
    n_elements = a_contig.size // 2

    a_ptr = a_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    b_ptr = b_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    lib.gf128_mul_batch(a_ptr, b_ptr, out_ptr, n_elements)

    return out


def _okvs_solve_impl(keys, values, m):
    lib = _get_lib()
    if lib is None:
        raise RuntimeError("Kernel library not loaded")

    n = keys.shape[0]
    keys_c = np.ascontiguousarray(keys, dtype=np.uint64)
    values_c = np.ascontiguousarray(values, dtype=np.uint64)
    output = np.zeros((m, 2), dtype=np.uint64)

    lib.solve_okvs(
        keys_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        values_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        n,
        m,
    )
    return output


def _okvs_decode_impl(keys, storage, m):
    lib = _get_lib()
    if lib is None:
        raise RuntimeError("Kernel library not loaded")

    n = keys.shape[0]
    keys_c = np.ascontiguousarray(keys, dtype=np.uint64)
    storage_c = np.ascontiguousarray(storage, dtype=np.uint64)
    output = np.zeros((n, 2), dtype=np.uint64)

    lib.decode_okvs(
        keys_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        storage_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        n,
        m,
    )
    return output


# =============================================================================
# Primitive Implementations
# =============================================================================


@field.aes_expand_p.def_impl
def _aes_expand_impl_prim(
    interpreter: Interpreter, op: Operation, seeds_val: TensorValue
):
    length = op.attrs["length"]
    seeds = _unwrap(seeds_val)

    # JAX PRG Fallback crashed. Switching to NumPy PRG.

    # Check if bytes
    if seeds.dtype == np.uint8 and seeds.shape[-1] == 16:
        seeds = seeds.view(np.uint64)

    if seeds.shape[-1] != 2:
        seeds = seeds.reshape(-1, 2)

    num_seeds = seeds.shape[0]
    out_shape = (num_seeds, length, 2)
    output = np.zeros(out_shape, dtype=np.uint64)

    lib = _get_lib()
    if lib is not None:
        # Fast C++ Path
        seeds_c = np.ascontiguousarray(seeds, dtype=np.uint64)

        lib.aes_128_expand(
            seeds_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            num_seeds,
            length,
        )
    else:
        # Slow Python Path (Fallback)
        # Iterate and generate
        for i in range(num_seeds):
            # Seed from pair
            s0 = int(seeds[i, 0])
            s1 = int(seeds[i, 1])
            seed_val = [s0, s1]

            rng = np.random.default_rng(seed_val)
            vals = rng.integers(
                0, 0xFFFFFFFFFFFFFFFF, size=(length, 2), dtype=np.uint64
            )
            output[i] = vals

    # Return as JAX array to keep downstream happy
    res_jax = jnp.array(output)

    return _wrap(res_jax)


@field.mul_p.def_impl
def _mul_impl(
    interpreter: Interpreter, op: Operation, a_val: TensorValue, b_val: TensorValue
):
    a = a_val.unwrap()
    b = b_val.unwrap()
    res = _gf128_mul_impl(a, b)
    return TensorValue(res)


@field.solve_okvs_p.def_impl
def _solve_okvs_impl(
    interpreter: Interpreter,
    op: Operation,
    keys_val: TensorValue,
    values_val: TensorValue,
):
    m = op.attrs["m"]
    keys = _unwrap(keys_val)
    values = _unwrap(values_val)
    res = _okvs_solve_impl(keys, values, m)
    return _wrap(res)


@field.decode_okvs_p.def_impl
def _decode_okvs_impl(
    interpreter: Interpreter,
    op: Operation,
    keys_val: TensorValue,
    store_val: TensorValue,
):
    keys = _unwrap(keys_val)
    storage = _unwrap(store_val)
    m = storage.shape[0]
    res = _okvs_decode_impl(keys, storage, m)
    return _wrap(res)
