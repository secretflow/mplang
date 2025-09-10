# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.typing import ArrayLike

import mplang.core.primitive as prim
from mplang import simp
from mplang.core import MPObject, Shape


@prim.function
def key_split(key: MPObject) -> tuple[MPObject, MPObject]:
    """Split the key into two keys."""

    def kernel(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        # TODO: since MPObject tensor does not implement slicing yet.
        # subkey, key = simp.run(jr.split)(key) does not work.
        # we workaround it by splitting inside tracer.
        subkey, key = jr.split(key)
        return subkey, key

    return simp.run(kernel)(key)  # type: ignore[no-any-return]


@prim.function
def ukey(seed: int | ArrayLike) -> MPObject:
    """Party uniformly generate a random key."""

    def kernel() -> jax.Array:
        key = jax.random.key(seed)
        # Note: key.dtype is jax._src.prng.KeyTy, which could not be handled by MPObject.
        return jax.random.key_data(key)

    return simp.run(kernel)()  # type: ignore[no-any-return]


@prim.function
def urandint(
    key: MPObject | ArrayLike,
    low: int,
    high: int,
    shape: Shape = (),
) -> MPObject:
    """Party uniformly generate a random integer in the range [low, high) with the given shape."""

    return simp.run(partial(jr.randint, minval=low, maxval=high, shape=shape))(key)  # type: ignore[no-any-return]


# Private(different per-party) related functions begin.


@prim.function
def prandint(low: int, high: int, shape: Shape = ()) -> MPObject:
    """Party privately generate a random integer in the range [low, high) with the given shape."""

    def kernel(rand_u64: jnp.ndarray) -> jnp.ndarray:
        range_size = high - low
        if range_size <= 0:
            raise ValueError("'high' must be greater than 'low'")

        remainder = jax.lax.rem(rand_u64, jnp.uint64(range_size))
        result = low + remainder.astype(jnp.int64)
        return result

    rand_u64 = prim.prand(shape)
    return simp.run(kernel)(rand_u64)  # type: ignore[no-any-return]


@prim.function
def pperm(key: MPObject) -> MPObject:
    """Party jointly generate a random permutation.

    That is, each party holds a random number in range(size), and all parties as a whole
    hold a random permutation of integers from 0 to size-1.

    Note: this function is NOT 'secure', that is, all parties know the permutation result.
    """

    if key.pmask is None:
        raise ValueError("dynamic pmask is not supported for pperm")

    full_mask = (1 << prim.psize()) - 1

    if key.pmask != full_mask:
        raise ValueError(
            "key must be a MPObject with mask covering all parties, "
            f"got {key.pmask} with world size {prim.psize()}"
        )

    if prim.pmask() is None or prim.pmask() != full_mask:
        raise ValueError(
            "pperm must be run with a mask covering all parties, "
            f"got {key.pmask} with world size {prim.psize()}"
        )

    size = prim.psize()

    def kernel(key: jax.Array) -> jax.Array:
        return jr.permutation(key, size)

    perm = simp.run(kernel)(key)
    rank = prim.prank()
    return simp.run(lambda perm, rank: perm[rank])(perm, rank)  # type: ignore[no-any-return]
