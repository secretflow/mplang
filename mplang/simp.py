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


def _run_impl(
    pyfn: Callable, fe_type: str, pmask: Mask | None, *args: Any, **kwargs: Any
) -> Any:
    if fe_type == "jax":
        if pmask is not None:
            return prim.run_jax_s(pyfn, pmask, *args, **kwargs)
        else:
            return prim.run_jax(pyfn, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported fe_type: {fe_type}")


# run :: (a -> a) -> m a -> m a
def run(pyfn: Callable, *, fe_type: str = "jax") -> Callable:
    return partial(_run_impl, pyfn, fe_type, None)


# runMask :: Mask -> (a -> a) -> m a -> m a
def runMask(pmask: Mask, pyfn: Callable, fe_type: str = "jax") -> Callable:
    return partial(_run_impl, pyfn, fe_type, pmask)


# runAt :: Rank -> (a -> a) -> m a -> m a
def runAt(rank: Rank, pyfn: Callable, *, fe_type: str = "jax") -> Callable:
    pmask = Mask(1 << rank)
    return runMask(pmask, pyfn, fe_type)


# runExcept :: Rank -> (a -> a) -> m a -> m a
def runExcept(rank: Rank, pyfn: Callable, *, fe_type: str = "jax") -> Callable:
    wsize = prim.psize()
    pmask = Mask(((1 << wsize) - 1) ^ (1 << rank))
    return runMask(pmask, pyfn, fe_type)


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
