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

from mplang.core import primitive as prim
from mplang.core.base import Mask, MPObject, Rank, ScalarType, Shape, TensorLike
from mplang.core.pfunc import PFunction
from mplang.plib import jax2stablehlo, stdio


def prank() -> MPObject:
    """Get the rank of current party."""
    return prim.prank()


def prand(shape: Shape = ()) -> MPObject:
    """Generate a random number in the range [0, psize)."""
    return prim.prand(shape)


def constant(data: TensorLike | ScalarType) -> MPObject:
    return prim.constant(data)


def peval(
    pfunc: PFunction,
    args: list[MPObject],
    pmask: Mask | None = None,
) -> list[MPObject]:
    return prim.peval(pfunc, args, pmask)


# TODO(jint): move this to plib
def jax_cc(
    func: Callable, *args: Any, **kwargs: Any
) -> tuple[PFunction, list[MPObject], Any]:
    """
    JAX compilation helper function.

    Compiles a JAX function to StableHLO format and returns the PFunction
    along with variable arguments for evaluation.

    Args:
        func: The JAX function to compile
        *args: Positional arguments to the function
        **kwargs: Keyword arguments to the function

    Returns:
        tuple[PFunction, list[MPObject], Any]: The compiled PFunction, input variables, and output tree
    """

    def is_variable(arg: Any) -> bool:
        return isinstance(arg, MPObject)

    # Compile the function using JAX to StableHLO
    pfunc, in_vars, out_tree = jax2stablehlo.compile(is_variable, func, *args, **kwargs)

    return pfunc, in_vars, out_tree


def run_impl(pmask: Mask | None, func: Callable, *args: Any, **kwargs: Any) -> Any:
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
        ValueError: If stdio.write is called without required arguments
        TypeError: If the function compilation or evaluation fails
        RuntimeError: If the underlying peval execution encounters errors

    Examples:
        Reading data from a file:

        >>> tensor_info = TensorInfo(shape=(10, 10), dtype=np.float32)
        >>> attrs = {"format": "binary"}
        >>> result = run_impl(stdio.read, "data/input.bin", tensor_info, attrs)

        Writing data to a file:

        >>> run_impl(stdio.write, data, "data/output.bin")

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

    # Known function list - extensible dispatch table
    FUNC_WHITE_LIST = {
        stdio.read,
        stdio.write,
    }

    if func in FUNC_WHITE_LIST:
        fe_func = func
    else:
        # unknown python callable, treat it as jax_cc function
        fe_func = partial(jax_cc, func)

    pfunc, eval_args, out_tree = fe_func(*args, **kwargs)
    results = peval(pfunc, eval_args, pmask)
    return out_tree.unflatten(results)


# run :: (a -> a) -> m a -> m a
def run(pyfn: Callable) -> Callable:
    return partial(run_impl, None, pyfn)


# runMask :: Mask -> (a -> a) -> m a -> m a
def runMask(pmask: Mask, pyfn: Callable) -> Callable:
    return partial(run_impl, pmask, pyfn)


# runAt :: Rank -> (a -> a) -> m a -> m a
def runAt(rank: Rank, pyfn: Callable) -> Callable:
    pmask = Mask(1 << rank)
    return runMask(pmask, pyfn)


# runExcept :: Rank -> (a -> a) -> m a -> m a
def runExcept(rank: Rank, pyfn: Callable) -> Callable:
    wsize = prim.psize()
    pmask = Mask(((1 << wsize) - 1) ^ (1 << rank))
    return runMask(pmask, pyfn)


# cond :: m Bool -> (m a -> m b) -> (m a -> m b) -> m b
def cond(
    pred: MPObject,
    then_fn: Callable[..., MPObject],
    else_fn: Callable[..., MPObject],
    *args: Any,
) -> MPObject:
    return prim.cond(pred, then_fn, else_fn, *args)


# while_loop :: m a -> (m a -> m Bool) -> (m a -> m a) -> m a
def while_loop(
    cond_fn: Callable[[MPObject], MPObject],
    body_fn: Callable[[MPObject], MPObject],
    init: MPObject,
) -> MPObject:
    return prim.while_loop(cond_fn, body_fn, init)


def pshfl_s(src_val: MPObject, pmask: Mask, src_ranks: list[Rank]) -> MPObject:
    return prim.pshfl_s(src_val, pmask, src_ranks)


def pshfl(src: MPObject, index: MPObject) -> MPObject:
    """Shuffle the value from src to the index party."""
    raise NotImplementedError("TODO")


def pconv(vars: list[MPObject]) -> MPObject:
    return prim.pconv(vars)
