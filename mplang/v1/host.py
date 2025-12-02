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
from typing import Any

from jax.tree_util import tree_map

from mplang.v1.core import (
    ClusterSpec,
    InterpContext,
    MPContext,
    MPObject,
    TraceContext,
    TracedFunction,
    trace,
)
from mplang.v1.core.context_mgr import cur_ctx, with_ctx


def evaluate(
    interp: InterpContext, mpfn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:  # type: ignore[misc]
    """Evaluate a multi-party function with the given interpreter context.

    This function accepts arbitrary types as it's designed to handle
    any multi-party computation function and arguments.

    Args:
        interp: The interpreter context for evaluating the multi-party function.
        mpfn: The multi-party function to evaluate.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Any: The result of evaluating the multi-party function, which can be
             any type depending on the function's return type.
    """
    assert isinstance(interp, InterpContext), f"Expect InterpContext, got {interp}"
    with with_ctx(interp):
        return mpfn(*args, **kwargs)


def fetch(interp: InterpContext | None, objs: Any) -> Any:  # type: ignore[misc]
    """Fetch computed results from MPObject instances in nested data structures.

    This function uses tree_map to handle arbitrary nested structures,
    so it needs to accept and return Any type.

    Args:
        interp: The interpreter context for fetching results. If None, uses the
                current context from cur_ctx().
        objs: The objects containing MPObject instances to fetch. Can be any
              nested structure.

    Returns:
        Any: The fetched results with the same structure as the input objects,
             but with MPObject instances replaced by their computed values.
    """
    ctx = interp or cur_ctx()
    assert isinstance(ctx, InterpContext), f"Expect MPExecutor, got {ctx}"

    evaluated = evaluate(ctx, lambda x: x, objs)

    def fetch_impl(arg: MPObject | Any) -> Any:
        if not isinstance(arg, MPObject):
            return arg

        return ctx.fetch(arg)

    return tree_map(fetch_impl, evaluated)


class CompileOptions(MPContext):
    """
    Lightweight ``MPContext`` used for ahead-of-time (AOT) compilation.

    Args:
        psize: Number of participating parties.
        spu_mask: Bitmask indicating which parties own an SPU device. Defaults
            to a mask that enables all parties.
    """

    def __init__(self, cluster_spec: Any) -> None:
        super().__init__(cluster_spec)

    @classmethod
    def simple(cls, world_size: int) -> CompileOptions:
        """Create a simple CompileOptions with the given party size and SPU mask.

        Args:
            world_size: Number of participating parties.

        Returns:
            A CompileOptions instance.
        """
        cluster_spec = ClusterSpec.simple(world_size)
        return cls(cluster_spec)


def compile(
    mctx: MPContext, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> TracedFunction:
    """Compile a multi-party function into a TracedFunction.

    Args:
        mctx: The multi-party context for compilation.
        fn: The function to compile.
        *args: Positional arguments to pass during compilation.
        **kwargs: Keyword arguments to pass during compilation.

    Returns:
        TracedFunction: The compiled function representation that can be
                       evaluated in multi-party contexts.
    """
    trace_ctx = TraceContext(mctx.cluster_spec)
    return trace(trace_ctx, fn, *args, **kwargs)
