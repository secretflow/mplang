import ctypes
import os
import sys

import numpy as np

# Load Library
_KERNEL_LIB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../../mplang/v2/kernels/libmplang_kernels.so"
    )
)

if not os.path.exists(_KERNEL_LIB_PATH):
    print(f"Error: {_KERNEL_LIB_PATH} not found")
    sys.exit(1)

_LIB = ctypes.CDLL(_KERNEL_LIB_PATH)

# void solve_okvs(uint64_t* keys, uint64_t* values, uint64_t* output, uint64_t n, uint64_t m);
_LIB.solve_okvs.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),  # keys
    ctypes.POINTER(ctypes.c_uint64),  # values (actually pairs of u64, size 2*n)
    ctypes.POINTER(ctypes.c_uint64),  # output (size 2*m)
    ctypes.c_uint64,  # n
    ctypes.c_uint64,  # m
]

# Decode binding
_LIB.decode_okvs.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),  # keys
    ctypes.POINTER(ctypes.c_uint64),  # storage
    ctypes.POINTER(ctypes.c_uint64),  # decoded_output
    ctypes.c_uint64,  # n
    ctypes.c_uint64,  # m
]


def test_okvs_flow():
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

    keys_ptr = keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    vals_ptr = values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    storage_ptr = storage.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    print(f"Calling solve_okvs n={n}, m={m}")
    _LIB.solve_okvs(keys_ptr, vals_ptr, storage_ptr, n, m)

    # Verify
    decoded = np.zeros((n, 2), dtype=np.uint64)
    dec_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    _LIB.decode_okvs(keys_ptr, storage_ptr, dec_ptr, n, m)

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
        sys.exit(1)


if __name__ == "__main__":
    test_okvs_flow()
