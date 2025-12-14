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

"""JIT Decorator: Compile and cache Graph IR."""

from collections.abc import Callable
from typing import Any

from jax.tree_util import tree_map

from mplang.v2.edsl.context import (
    AbstractInterpreter,
    get_current_context,
    get_default_context,
)
from mplang.v2.edsl.tracer import Tracer


def jit(fn: Callable) -> Callable:
    """JIT compilation decorator.

    Traces the function to Graph IR on first call, then executes the cached
    Graph on subsequent calls.

    Example:
        >>> @jit
        ... def compute(x, y):
        ...     return x + y
        >>> result = compute(x_interp, y_interp)  # First call: trace
        >>> result = compute(x_interp, y_interp)  # Subsequent: execute cached graph
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # If we are already inside a Tracer (e.g. pcall_static), just inline
        # the function to trace it into the current graph.
        cur_ctx = get_current_context()
        if isinstance(cur_ctx, Tracer):
            return fn(*args, **kwargs)

        # otherwise trace for JIT compilation
        with Tracer():
            result = fn(*args, **kwargs)

        # Use current context if available (e.g., SimpSimulator), otherwise use default
        cur_ctx = cur_ctx or get_default_context()
        assert isinstance(cur_ctx, AbstractInterpreter), (
            "JIT execution requires Interpreter context"
        )
        return tree_map(cur_ctx.lift, result)

    return wrapper
