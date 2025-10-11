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

from mplang.core.mask import Mask
from mplang.core.mpobject import MPObject
from mplang.core.mptype import Rank
from mplang.core.primitive import (
    constant,
    pconv,
    peval,
    prand,
    prank,
    pshfl,
    pshfl_s,
    uniform_cond,
    while_loop,
)
from mplang.ops import ibis_cc, jax_cc, sql_cc
from mplang.ops.base import FeOperation
from mplang.simp.random import key_split, pperm, prandint, ukey, urandint
from mplang.simp.smpc import reveal, revealTo, seal, sealFrom, srun

# Public exports of the simplified party execution API.
# NOTE: Replaces previous internal __reexport__ (not a Python convention)
# to make star-imports explicit and tooling-friendly.
__all__ = [
    "MPObject",
    "constant",
    "key_split",
    "pconv",
    "peval",
    "pperm",
    "prand",
    "prandint",
    "prank",
    "pshfl",
    "pshfl_s",
    "reveal",
    "revealTo",
    "seal",
    "sealFrom",
    "srun",
    "ukey",
    "uniform_cond",
    "urandint",
    "while_loop",
]


def run_fe_op(
    pmask: Mask | None,
    fe_op: FeOperation,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Run a function that can be evaluated by the mplang system.

    This function provides a dispatch mechanism based on the first argument
    to route different function types to appropriate handlers.

    Args:
        pmask: The party mask of this function, None indicates auto deduce parties from args.
        func: The function to be dispatched and executed
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of evaluating the function through the appropriate handler

    Raises:
        ValueError: If basic.write is called without required arguments
        TypeError: If the function compilation or evaluation fails
        RuntimeError: If the underlying peval execution encounters errors

    Examples:
        Reading data from a file:

        >>> tensor_info = TensorType(shape=(10, 10), dtype=np.float32)
        >>> attrs = {"format": "binary"}
        >>> result = pcall(basic.read, "data/input.bin", tensor_info, attrs)

        Writing data to a file:

        >>> pcall(basic.write, data, "data/output.bin")
    """

    pfunc, eval_args, out_tree = fe_op(*args, **kwargs)
    results = peval(pfunc, eval_args, pmask)
    return out_tree.unflatten(results)


def run_jax(jax_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a JAX function within the mplang system.

    This function compiles and evaluates a JAX function using the mplang
    computation framework, allowing for secure multi-party computation.

    Args:
        jax_fn: The JAX function to be executed.
        *args: Positional arguments to pass to the JAX function.
        **kwargs: Keyword arguments to pass to the JAX function.

    Returns:
        The result of evaluating the JAX function through the mplang system.

    Raises:
        TypeError: If the function compilation or evaluation fails.
        RuntimeError: If the underlying peval execution encounters errors.

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
    return run_fe_op(None, jax_cc.run_jax, jax_fn, *args, **kwargs)


def run_jax_at(rank: Rank, jax_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    return run_fe_op(Mask.from_ranks(rank), jax_cc.run_jax, jax_fn, *args, **kwargs)


def run_ibis(ibis_expr: Any, *args: Any, **kwargs: Any) -> Any:
    """Run an Ibis function within the mplang system.

    This function compiles and evaluates an Ibis function using the mplang
    computation framework, allowing for secure multi-party computation.

    Args:
        ibis_fn: The Ibis function to be executed.
        *args: Positional arguments to pass to the Ibis function.
        **kwargs: Keyword arguments to pass to the Ibis function.

    Returns:
        The result of evaluating the Ibis function through the mplang system.

    Raises:
        TypeError: If the function compilation or evaluation fails.
        RuntimeError: If the underlying peval execution encounters errors.

    Examples:
        Defining and running a simple Ibis function:

        >>> import ibis
        >>> def filter_data(table):
        ...     return table[table["value"] > 0]
        >>> result = run_ibis(filter_data, input_table)

        Running a more complex Ibis function:

        >>> def compute_statistics(table):
        ...     mean = table["value"].mean()
        ...     std = table["value"].std()
        ...     return {"mean": mean, "std": std}
        >>> stats = run_ibis(compute_statistics, input_table)
    """
    return run_fe_op(None, ibis_cc.run_ibis, ibis_expr, *args, **kwargs)


def run_ibis_at(rank: Rank, ibis_fn: Any, *args: Any, **kwargs: Any) -> Any:
    return run_fe_op(Mask.from_ranks(rank), ibis_cc.run_ibis, ibis_fn, *args, **kwargs)


def run_sql(
    query: str, out_type: Any, in_tables: dict[str, MPObject] | None = None
) -> Any:
    """Run an SQL query within the mplang system.

    This function compiles and evaluates an SQL query using the mplang
    computation framework, allowing for secure multi-party computation.

    Args:
        sql: The SQL query to be executed.
        out_type: The expected output type of the SQL query.
        in_tables: A dictionary mapping table names to MPObject instances representing input tables.

    Returns:
        The result of evaluating the SQL query through the mplang system.

    Raises:
        TypeError: If the function compilation or evaluation fails.
        RuntimeError: If the underlying peval execution encounters errors.

    Examples:
        Running a simple SQL query:

        >>> sql_query = "SELECT * FROM input_table WHERE value > 0"
        >>> result = run_sql(
        ...     sql_query, output_table_type, {"input_table": input_mpobject}
        ... )

        Running a more complex SQL query:

        >>> sql_query = (
        ...     "SELECT AVG(value) AS mean, STDDEV(value) AS std FROM input_table"
        ... )
        >>> stats = run_sql(
        ...     sql_query, stats_table_type, {"input_table": input_mpobject}
        ... )
    """
    return run_fe_op(None, sql_cc.run_sql, query, out_type, in_tables)


def run_sql_at(
    rank: Rank, query: str, out_type: Any, in_tables: dict[str, MPObject] | None = None
) -> Any:
    return run_fe_op(Mask.from_ranks(rank), sql_cc.run_sql, query, out_type, in_tables)
