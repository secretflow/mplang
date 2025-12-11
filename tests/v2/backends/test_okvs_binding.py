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

import ctypes
import os
import sys

import numpy as np

# Fallback to pure Python implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from mplang.v2.kernels import py_kernels

# Load Library
# Path is relative to repository root, not to this test file's location
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_KERNEL_LIB_PATH = os.path.join(_REPO_ROOT, "mplang/v2/kernels/libmplang_kernels.so")

_LIB = None
if os.path.exists(_KERNEL_LIB_PATH):
    try:
        _LIB = ctypes.CDLL(_KERNEL_LIB_PATH)

        # void solve_okvs(uint64_t* keys, uint64_t* values, uint64_t* output, uint64_t n, uint64_t m, uint64_t* seed);
        _LIB.solve_okvs.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),  # keys
            ctypes.POINTER(ctypes.c_uint64),  # values (actually pairs of u64, size 2*n)
            ctypes.POINTER(ctypes.c_uint64),  # output (size 2*m)
            ctypes.c_uint64,  # n
            ctypes.c_uint64,  # m
            ctypes.POINTER(ctypes.c_uint64),  # seed
        ]

        # Decode binding
        _LIB.decode_okvs.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),  # keys
            ctypes.POINTER(ctypes.c_uint64),  # storage
            ctypes.POINTER(ctypes.c_uint64),  # decoded_output
            ctypes.c_uint64,  # n
            ctypes.c_uint64,  # m
            ctypes.POINTER(ctypes.c_uint64),  # seed
        ]
    except OSError as e:
        print(f"Warning: Failed to load {_KERNEL_LIB_PATH}: {e}")
        print("Falling back to pure Python implementation")
        _LIB = None


def test_okvs_flow() -> None:
    # Use larger N to stress Peeling
    n = 100
    m = int(n * 1.6)  # Safer for small N (expected 1.23 for large N)

    keys = np.arange(n, dtype=np.uint64)
    # values: (n, 2)
    values = np.zeros((n, 2), dtype=np.uint64)
    for i in range(n):
        values[i, 0] = i
        values[i, 1] = i * 10

    storage = np.zeros((m, 2), dtype=np.uint64)
    seed = np.array([0xDEADBEEF, 0xCAFEBABE], dtype=np.uint64)

    if _LIB is not None:
        # Use C++ implementation
        keys_ptr = keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        vals_ptr = values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        storage_ptr = storage.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        seed_ptr = seed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

        print(f"Calling C++ solve_okvs n={n}, m={m}")
        _LIB.solve_okvs(keys_ptr, vals_ptr, storage_ptr, n, m, seed_ptr)

        # Verify
        decoded = np.zeros((n, 2), dtype=np.uint64)
        dec_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

        _LIB.decode_okvs(keys_ptr, storage_ptr, dec_ptr, n, m, seed_ptr)
    else:
        # Use Python fallback
        print(f"Using Python fallback for OKVS solve n={n}, m={m}")
        storage = py_kernels.okvs_solve(
            keys, values, m, seed=(int(seed[0]), int(seed[1]))
        )
        decoded = py_kernels.okvs_decode(
            keys, storage, m, seed=(int(seed[0]), int(seed[1]))
        )

    if np.array_equal(decoded, values):
        print("SUCCESS: Decoded values match inputs!")
    else:
        print("FAILURE: Decoded values do NOT match inputs!")
        # Debug failing indices
        diff = decoded != values
        bad_idx = np.where(diff.any(axis=1))[0]
        print(f"Failed count: {len(bad_idx)}")
        print(
            "First failure:", keys[bad_idx[0]], values[bad_idx[0]], decoded[bad_idx[0]]
        )
        raise AssertionError("OKVS decode verification failed")


if __name__ == "__main__":
    test_okvs_flow()
