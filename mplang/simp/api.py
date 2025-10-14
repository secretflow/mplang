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
from typing import Any, cast

from mplang.core import Mask, MPObject, Rank, peval
from mplang.core.primitive import builtin_function
from mplang.core.table import TableLike
from mplang.core.tensor import ScalarType, Shape, TensorLike
from mplang.ops import basic, ibis_cc, jax_cc, sql_cc
from mplang.ops.base import FeOperation


def run(
    pmask: Mask | None,
    fe_op: FeOperation,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run an operation in the current context."""
    pfunc, eval_args, out_tree = fe_op(*args, **kwargs)
    results = peval(pfunc, eval_args, pmask)
    return out_tree.unflatten(results)


def run_at(rank: Rank, op: Any, *args: Any, **kwargs: Any) -> Any:
    """Run an operation at a specific rank."""
    return run(Mask.from_ranks(rank), op, *args, **kwargs)


@builtin_function
def prank() -> MPObject:
    """Multi-party get the rank (party identifier) of each party.

    Returns an MPObject containing the rank of each party (0 to world_size-1).
    """
    return cast(MPObject, run(None, basic.rank))


@builtin_function
def prand(shape: Shape = ()) -> MPObject:
    """Multi-party generate a private random (uint64) tensor with the given shape.

    Each party independently generates its own local random values.
    """
    return cast(MPObject, run(None, basic.prand, shape))


def constant(data: TensorLike | ScalarType | TableLike) -> MPObject:
    """Create a constant tensor or table from data.

    The constant value is embedded into the computation graph and is available
    to all parties.
    """
    return cast(MPObject, run(None, basic.constant, data))


@builtin_function
def debug_print(obj: MPObject, prefix: str = "") -> MPObject:
    """Print local value of obj on owning parties and pass it through.

    Returns the same MPObject to support chaining and prevent DCE.
    """
    pfunc, eval_args, out_tree = basic.debug_print(obj, prefix=prefix)
    results = peval(pfunc, eval_args)
    return cast(MPObject, out_tree.unflatten(results))


def set_mask(arg: MPObject, mask: Mask) -> MPObject:
    """Set the mask of an MPObject to a new value.

    For dynamic pmask inputs: the return value's pmask will be the specified mask.
    For static pmask inputs: validates that mask is a subset of arg.pmask.
    """
    pfunc, eval_args, out_tree = basic.identity(arg)
    results = peval(pfunc, eval_args, mask)
    return cast(MPObject, out_tree.unflatten(results))


def run_jax(jax_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a JAX function.

    Args:
        jax_fn: The JAX function to be executed.
        *args: Positional arguments to pass to the JAX function.
        **kwargs: Keyword arguments to pass to the JAX function.

    Returns:
        The result of evaluating the JAX function through the mplang system.

    Raises:
        TypeError: If the function compilation or evaluation fails.
        RuntimeError: If the underlying peval execution encounters errors.

    Notes:
        Argument binding semantics with respect to JAX static arguments:

        - If an argument (or any leaf within a PyTree argument) is an
          :class:`~mplang.core.mpobject.MPObject`, it is captured as a runtime
          variable (dynamic value) in the traced program and is not treated as a
          JAX static argument.
        - If an argument contains no :class:`MPObject` leaves, it is treated as a
          constant configuration with respect to JAX; effectively it behaves
          like a static argument and may contribute to JAX compilation cache
          keys (similar to ``static_argnums`` semantics). Changing such constant
          arguments can lead to different compiled variants/cached entries.

    Examples:
        Defining and running a simple JAX function:

        >>> import jax.numpy as jnp
        >>> def add_matrices(a, b):
        ...     return jnp.add(a, b)
        >>> result = run_jax(add_matrices, matrix_a, matrix_b)

        Running a more complex JAX function:

        >>> def compute_statistics(data):
        ...     mean = jnp.mean(data)
        ...     std = jnp.std(data)
        ...     return {"mean": mean, "std": std}
        >>> stats = run_jax(compute_statistics, dataset)
    """
    return run(None, jax_cc.run_jax, jax_fn, *args, **kwargs)


def run_jax_at(rank: Rank, jax_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    return run_at(rank, jax_cc.run_jax, jax_fn, *args, **kwargs)


def run_ibis(ibis_expr: Any, *args: Any, **kwargs: Any) -> Any:
    # TODO(jint): add docstring, add type hints, describe args and kwargs constraints.
    return run(None, ibis_cc.run_ibis, ibis_expr, *args, **kwargs)


def run_ibis_at(rank: Rank, ibis_fn: Any, *args: Any, **kwargs: Any) -> Any:
    return run_at(rank, ibis_cc.run_ibis, ibis_fn, *args, **kwargs)


def run_sql(
    query: str, out_type: Any, in_tables: dict[str, MPObject] | None = None
) -> Any:
    # TODO(jint): add docstring, drop out_type.
    return run(None, sql_cc.run_sql, query, out_type, in_tables)


def run_sql_at(
    rank: Rank, query: str, out_type: Any, in_tables: dict[str, MPObject] | None = None
) -> Any:
    return run_at(rank, sql_cc.run_sql, query, out_type, in_tables)
