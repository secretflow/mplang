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

from mplang.core.base import Mask, MPContext, MPObject
from mplang.core.context_mgr import cur_ctx, with_ctx
from mplang.core.interp import InterpContext
from mplang.core.trace import TraceContext, TracedFunction, trace


def evaluate(
    interp: InterpContext, mpfn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:  # type: ignore[misc]
    """Evaluate a multi-party function with the given interpreter context.

    This function accepts arbitrary types as it's designed to handle
    any multi-party computation function and arguments.
    """
    assert isinstance(interp, InterpContext), f"Expect InterpContext, got {interp}"
    with with_ctx(interp):
        return mpfn(*args, **kwargs)


def fetch(interp: InterpContext | None, objs: Any) -> Any:  # type: ignore[misc]
    """Fetch computed results from MPObject instances in nested data structures.

    This function uses tree_map to handle arbitrary nested structures,
    so it needs to accept and return Any type.
    """
    ctx = interp or cur_ctx()
    assert isinstance(ctx, InterpContext), f"Expect MPExecutor, got {ctx}"

    evaluated = evaluate(ctx, lambda x: x, objs)

    def fetch_impl(arg: MPObject | Any) -> Any:
        if isinstance(arg, MPObject):
            return ctx.fetch(arg)
        else:
            return arg

    return tree_map(fetch_impl, evaluated)


class CompileOptions(MPContext):
    """
    Lightweight ``MPContext`` used for ahead-of-time (AOT) compilation.

    Args:
        psize: Number of participating parties.
        spu_mask: Bitmask indicating which parties own an SPU device. Defaults
            to a mask that enables all parties.
        **attrs: Extra attributes forwarded to the underlying ``TraceContext``.
    """

    def __init__(self, psize: int, spu_mask: Mask | None = None, **attrs: Any):
        self._psize = psize
        self.spu_mask = spu_mask or Mask((1 << psize) - 1)
        # Keep user-defined attributes together with the default ones.
        self._attrs: dict[str, Any] = attrs
        self._attrs.setdefault("spu_mask", self.spu_mask)

    # ---------- MPContext interface ----------
    def psize(self) -> int:
        return self._psize

    def attrs(self) -> dict[str, Any]:
        return self._attrs


def compile(
    mctx: MPContext, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> TracedFunction:
    trace_ctx = TraceContext(mctx.psize(), attrs=mctx.attrs())
    return trace(trace_ctx, fn, *args, **kwargs)
