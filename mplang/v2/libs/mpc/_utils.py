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

"""Utilities for MPC protocols."""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.edsl as el
from mplang.v2.dialects import tensor


def bytes_to_bits(data: el.Object) -> el.Object:
    """Convert bytes (uint8 tensor) to bits (uint8 tensor of 0s and 1s).

    Output shape logic: (..., N) -> (..., N * 8)
    """

    def _to_bits(arr: Any) -> Any:
        # View as u8
        y_u8 = arr.view(jnp.uint8)
        # Unpack produces Big Endian bits [b7, b6, ..., b0] per byte
        bits = jnp.unpackbits(y_u8)
        # Reshape to (N, 8) and flip to get [b0, ..., b7]
        bits = bits.reshape(-1, 8)
        bits = jnp.fliplr(bits)
        return bits.reshape(-1)

    return cast(el.Object, tensor.run_jax(_to_bits, data))


def bits_to_bytes(bits: el.Object) -> el.Object:
    """Convert bits to bytes.

    Output shape logic: (..., N * 8) -> (..., N)
    """

    def _to_bytes(arr: Any) -> Any:
        return jnp.packbits(arr, axis=-1)

    return cast(el.Object, tensor.run_jax(_to_bytes, bits))


def transpose_128(matrix_bits: el.Object) -> el.Object:
    """Transpose a bit matrix.

    Just a wrapper for tensor.transpose currently.
    """
    return tensor.transpose(matrix_bits, perm=(1, 0))


class CuckooHash:
    """Simple Cuckoo Hashing simulation."""

    def __init__(self, num_bins: int, num_hash_functions: int = 3, stash_size: int = 0):
        self.num_bins = num_bins
        self.num_functions = num_hash_functions
        self.stash_size = stash_size

    def hash(self, items: el.Object, seed: int) -> el.Object:
        """Hash items to bin indices."""

        # We perform hashing.
        # Note: We return hashes for each function?
        # Usually simplest cuckoo uses 3 hash functions.
        # We can return (num_funcs, N) or (N, num_funcs)

        def _hash_fn(xs: Any, s: int) -> Any:
            # xs: array of items

            # Simple hash: (x * s + s) % bins
            # We want multiple hashes?
            # For now, let's just return one hash per seed provided (assuming call per seed)
            # Or if seed is a single int, we might mix it.

            # Let's assume this function handles one hash instance.
            res = (xs * s + s) % self.num_bins
            return res.astype(jnp.int32)

        # Passing self.num_bins as constant implementation detail inside _hash_fn closure is fine
        # if using run_jax (as it's compiled).
        # Actually run_jax recompiles if closure changes?
        # run_jax supports closures.

        return cast(el.Object, tensor.run_jax(_hash_fn, items, seed))
