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

from collections.abc import Callable
from functools import partial
from typing import Any

from mplang.core.mask import Mask
from mplang.core.mpobject import MPObject
from mplang.core.mptype import Rank
from mplang.core.primitive import (
    cond,
    constant,
    pconv,
    peval,
    prand,
    prank,
    pshfl,
    pshfl_s,
    while_loop,
)
from mplang.frontend import ibis_cc, jax_cc
from mplang.frontend.base import FEOp
from mplang.simp.mpi import allgather_m, bcast_m, gather_m, p2p, scatter_m
from mplang.simp.random import key_split, pperm, prandint, ukey, urandint
from mplang.simp.smpc import reveal, revealTo, seal, sealFrom, srun

__reexport__ = [
    cond,
    constant,
    MPObject,
    pconv,
    peval,
    prank,
    prand,
    pshfl,
    pshfl_s,
    while_loop,
    scatter_m,
    gather_m,
    bcast_m,
    allgather_m,
    p2p,
    seal,
    sealFrom,
    reveal,
    revealTo,
    srun,
    key_split,
    ukey,
    urandint,
    prandint,
    pperm,
]


def run_impl(
    pmask: Mask | None,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Run a function that can be evaluated by the mplang system.

    This function provides a dispatch mechanism based on the first argument
    to route different function types to appropriate handlers.

    Args:
        pmask: The party mask of this function, None indicates auto deduce parties.
        func: The function to be dispatched and executed
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of evaluating the function through the appropriate handler

    Raises:
        ValueError: If builtin.write is called without required arguments
        TypeError: If the function compilation or evaluation fails
        RuntimeError: If the underlying peval execution encounters errors

    Examples:
        Reading data from a file:

        >>> tensor_info = TensorType(shape=(10, 10), dtype=np.float32)
        >>> attrs = {"format": "binary"}
        >>> result = run_impl(builtin.read, "data/input.bin", tensor_info, attrs)

        Writing data to a file:

        >>> run_impl(builtin.write, data, "data/output.bin")

        Running a JAX function:

        >>> def matrix_multiply(a, b):
        ...     return jnp.dot(a, b)
        >>> result = run_impl(matrix_multiply, mat_a, mat_b)

        Running a custom computation function:

        >>> def compute_statistics(data):
        ...     mean = jnp.mean(data)
        ...     std = jnp.std(data)
        ...     return {"mean": mean, "std": std}
        >>> stats = run_impl(compute_statistics, dataset)
    """

    if isinstance(func, FEOp):
        pfunc, eval_args, out_tree = func(*args, **kwargs)
    else:
        if ibis_cc.is_ibis_function(func):
            pfunc, eval_args, out_tree = ibis_cc.ibis_compile(func, *args, **kwargs)
        else:
            # unknown python callable, treat it as jax function
            pfunc, eval_args, out_tree = jax_cc.jax_compile(func, *args, **kwargs)
    results = peval(pfunc, eval_args, pmask)
    return out_tree.unflatten(results)


# run :: (a -> a) -> m a -> m a
def run(pyfn: Callable) -> Callable:
    return partial(run_impl, None, pyfn)


# runAt :: Rank -> (a -> a) -> m a -> m a
def runAt(rank: Rank, pyfn: Callable) -> Callable:
    pmask = Mask.from_ranks(rank)
    return partial(run_impl, pmask, pyfn)
