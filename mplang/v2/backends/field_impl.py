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

# print("DEBUG: Importing field_impl.py")
import jax.numpy as jnp
import numpy as np

from mplang.v2.backends.tensor_impl import TensorValue, _unwrap, _wrap
from mplang.v2.dialects import field
from mplang.v2.edsl.graph import Operation
from mplang.v2.kernels import py_kernels
from mplang.v2.runtime.interpreter import Interpreter

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


def _get_lib() -> ctypes.CDLL | None:
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
                    ctypes.POINTER(ctypes.c_uint64),  # seed
                ]
                _LIB.decode_okvs.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),  # keys
                    ctypes.POINTER(ctypes.c_uint64),  # storage
                    ctypes.POINTER(ctypes.c_uint64),  # output
                    ctypes.c_uint64,  # n
                    ctypes.c_uint64,  # m
                    ctypes.POINTER(ctypes.c_uint64),  # seed
                ]
                # Optimized Mega-Binning Versions
                _LIB.solve_okvs_opt.argtypes = _LIB.solve_okvs.argtypes
                _LIB.decode_okvs_opt.argtypes = _LIB.decode_okvs.argtypes

                _LIB.aes_128_expand.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),  # seeds
                    ctypes.POINTER(ctypes.c_uint64),  # output
                    ctypes.c_uint64,  # num_seeds
                    ctypes.c_uint64,  # length
                ]
                _LIB.ldpc_encode.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),  # message
                    ctypes.POINTER(ctypes.c_uint64),  # indices
                    ctypes.POINTER(ctypes.c_uint64),  # indptr
                    ctypes.POINTER(ctypes.c_uint64),  # output
                    ctypes.c_uint64,  # m
                    ctypes.c_uint64,  # n
                ]
            except OSError:
                print(f"WARNING: Could not load kernels from {_KERNEL_LIB_PATH}")
    return _LIB


# =============================================================================
# Helper Implementations (C++ Wrappers)
# =============================================================================


def _gf128_mul_impl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a, b are numpy arrays (uint64) usually (N, 2)

    lib = _get_lib()
    if lib is None:
        # Use pure Python fallback
        return py_kernels.gf128_mul_batch(a, b)

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


def _okvs_solve_opt_impl(
    keys: np.ndarray, values: np.ndarray, m: int, seed: np.ndarray
) -> np.ndarray:
    lib = _get_lib()
    if seed.ndim > 1:
        seed = seed.flatten()

    if lib is None:
        # Fallback to standard (no python impl for opt)
        return _okvs_solve_impl(keys, values, m, seed)

    n = keys.shape[0]

    # Heuristic: Mega-Binning is unstable < 200k.
    if n < 200_000:
        return _okvs_solve_impl(keys, values, m, seed)

    # Heuristic: Mega-Binning requires higher expansion (epsilon ~ 1.35)
    # If m/n is too tight, fallback to Naive (which works with 1.25)
    if m / n < 1.32:
        return _okvs_solve_impl(keys, values, m, seed)

    keys_c = np.ascontiguousarray(keys, dtype=np.uint64)
    values_c = np.ascontiguousarray(values, dtype=np.uint64)
    seed_c = np.ascontiguousarray(seed, dtype=np.uint64)
    output = np.zeros((m, 2), dtype=np.uint64)

    lib.solve_okvs_opt(
        keys_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        values_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        n,
        m,
        seed_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    return output


def _okvs_decode_opt_impl(
    keys: np.ndarray, storage: np.ndarray, m: int, seed: np.ndarray
) -> np.ndarray:
    lib = _get_lib()
    if seed.ndim > 1:
        seed = seed.flatten()

    if lib is None:
        return _okvs_decode_impl(keys, storage, m, seed)

    n = keys.shape[0]

    # Heuristic: Mega-Binning (1024 Bins) is unstable for small N due to variance.
    # It requires ~1000 items/bin to be efficient and stable with epsilon=1.3.
    # Threshold: 200,000 (approx 200 items/bin). Below this, Naive is fast enough (<50ms).
    if n < 200_000:
        return _okvs_decode_impl(keys, storage, m, seed)

    if m / n < 1.32:
        return _okvs_decode_impl(keys, storage, m, seed)

    keys_c = np.ascontiguousarray(keys, dtype=np.uint64)
    storage_c = np.ascontiguousarray(storage, dtype=np.uint64)
    seed_c = np.ascontiguousarray(seed, dtype=np.uint64)
    output = np.zeros((n, 2), dtype=np.uint64)

    lib.decode_okvs_opt(
        keys_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        storage_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        n,
        m,
        seed_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    return output


def _okvs_solve_impl(
    keys: np.ndarray, values: np.ndarray, m: int, seed: np.ndarray
) -> np.ndarray:
    lib = _get_lib()
    # Ensure seed is flat tuple or array
    if seed.ndim > 1:
        seed = seed.flatten()
    s_tuple = (int(seed[0]), int(seed[1]))

    if lib is None:
        # Use pure Python fallback
        keys_flat = keys.flatten() if keys.ndim > 1 else keys
        return py_kernels.okvs_solve(keys_flat, values, m, seed=s_tuple)

    n = keys.shape[0]
    keys_c = np.ascontiguousarray(keys, dtype=np.uint64)
    values_c = np.ascontiguousarray(values, dtype=np.uint64)
    seed_c = np.ascontiguousarray(seed, dtype=np.uint64)
    output = np.zeros((m, 2), dtype=np.uint64)

    lib.solve_okvs(
        keys_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        values_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        n,
        m,
        seed_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    return output


def _okvs_decode_impl(
    keys: np.ndarray, storage: np.ndarray, m: int, seed: np.ndarray
) -> np.ndarray:
    lib = _get_lib()
    # Ensure seed is flat tuple or array
    if seed.ndim > 1:
        seed = seed.flatten()
    s_tuple = (int(seed[0]), int(seed[1]))

    if lib is None:
        # Use pure Python fallback
        keys_flat = keys.flatten() if keys.ndim > 1 else keys
        return py_kernels.okvs_decode(keys_flat, storage, m, seed=s_tuple)

    n = keys.shape[0]
    keys_c = np.ascontiguousarray(keys, dtype=np.uint64)
    storage_c = np.ascontiguousarray(storage, dtype=np.uint64)
    seed_c = np.ascontiguousarray(seed, dtype=np.uint64)
    output = np.zeros((n, 2), dtype=np.uint64)

    lib.decode_okvs(
        keys_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        storage_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        n,
        m,
        seed_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    return output


def ldpc_encode_impl(
    message: np.ndarray, h_indices: np.ndarray, h_indptr: np.ndarray, m: int
) -> np.ndarray:
    lib = _get_lib()
    if lib is None:
        # Use pure Python fallback
        h_idx_flat = h_indices.flatten() if h_indices.ndim > 1 else h_indices
        h_ptr_flat = h_indptr.flatten() if h_indptr.ndim > 1 else h_indptr
        return py_kernels.ldpc_encode(message, h_idx_flat, h_ptr_flat, m)

    # Fast C++ Path
    msg_c = np.ascontiguousarray(message, dtype=np.uint64)
    idx_c = np.ascontiguousarray(h_indices, dtype=np.uint64)
    ptr_c = np.ascontiguousarray(h_indptr, dtype=np.uint64)

    output = np.zeros((m, 2), dtype=np.uint64)

    # n is inferred from message length
    n = message.shape[0]

    lib.ldpc_encode(
        msg_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        idx_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ptr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        m,
        n,
    )
    return output


# =============================================================================
# Primitive Implementations
# =============================================================================


@field.ldpc_encode_p.def_impl
def _ldpc_encode_impl_prim(
    interpreter: Interpreter,
    op: Operation,
    message_val: TensorValue,
    indices_val: TensorValue,
    indptr_val: TensorValue,
) -> TensorValue:
    m = op.attrs["m"]
    message = _unwrap(message_val)
    indices = _unwrap(indices_val)
    indptr = _unwrap(indptr_val)
    res = ldpc_encode_impl(message, indices, indptr, m)
    return _wrap(res)


@field.aes_expand_p.def_impl
def _aes_expand_impl_prim(
    interpreter: Interpreter, op: Operation, seeds_val: TensorValue
) -> TensorValue:
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
) -> TensorValue:
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
    seed_val: TensorValue,
) -> TensorValue:
    m = op.attrs["m"]
    keys = _unwrap(keys_val)
    values = _unwrap(values_val)
    seed = _unwrap(seed_val)
    res = _okvs_solve_impl(keys, values, m, seed)
    return _wrap(res)


@field.decode_okvs_p.def_impl
def _decode_okvs_impl(
    interpreter: Interpreter,
    op: Operation,
    keys_val: TensorValue,
    store_val: TensorValue,
    seed_val: TensorValue,
) -> TensorValue:
    keys = _unwrap(keys_val)
    storage = _unwrap(store_val)
    seed = _unwrap(seed_val)
    m = storage.shape[0]
    res = _okvs_decode_impl(keys, storage, m, seed)
    return _wrap(res)
    return _wrap(res)


@field.solve_okvs_opt_p.def_impl
def _solve_okvs_opt_impl_prim(
    interpreter: Interpreter,
    op: Operation,
    keys_val: TensorValue,
    values_val: TensorValue,
    seed_val: TensorValue,
) -> TensorValue:
    m = op.attrs["m"]
    keys = _unwrap(keys_val)
    values = _unwrap(values_val)
    seed = _unwrap(seed_val)
    res = _okvs_solve_opt_impl(keys, values, m, seed)
    return _wrap(res)


@field.decode_okvs_opt_p.def_impl
def _decode_okvs_opt_impl_prim(
    interpreter: Interpreter,
    op: Operation,
    keys_val: TensorValue,
    store_val: TensorValue,
    seed_val: TensorValue,
) -> TensorValue:
    keys = _unwrap(keys_val)
    storage = _unwrap(store_val)
    seed = _unwrap(seed_val)
    m = storage.shape[0]
    res = _okvs_decode_opt_impl(keys, storage, m, seed)
    return _wrap(res)
