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

from mplang.v1.core import (
    Mask,
    MPObject,
    Rank,
    ScalarType,
    Shape,
    TableLike,
    TensorLike,
    builtin_function,
    peval,
)
from mplang.v1.ops import basic, jax_cc, nnx_cc, sql_cc
from mplang.v1.ops.base import FeOperation


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

    This function returns a scalar tensor containing the rank (party identifier)
    for each party in the current party mask. Each party independently produces
    its own rank value, which serves as a unique identifier within the multi-party
    computation context.

    The rank values range from 0 to world_size-1, where world_size is the total
    number of parties in the computation. Each party's rank is private to that
    party and represents its position in the multi-party protocol.

    Returns:
        MPObject: A variable representing a scalar tensor with:
                  - dtype: UINT64
                  - shape: () (scalar)

    Note:
        Each party in the current party mask independently produces its own rank value.
    """
    return cast(MPObject, run(None, basic.rank))


@builtin_function
def prand(shape: Shape = ()) -> MPObject:
    """Multi-party generate a private random (uint64) tensor with the given shape.

    This function creates a private random tensor where each party independently
    generates its own local random values. Each party's random values are private
    and unknown to other parties. The output tensor contains 64-bit unsigned
    integers, with each party holding its own privately generated values.

    Args:
        shape: The shape of the random tensor to generate.
               Must be a tuple of positive integers. Defaults to () for scalar.

    Returns:
        MPObject: A variable representing the generated private random tensor with:
                  - dtype: UINT64
                  - shape: As specified by the shape parameter

    Note:
        Each party in the current party mask independently generates its own
        private random values. The randomness is local to each party and is
        not shared or revealed to other parties.
    """
    return cast(MPObject, run(None, basic.prand, shape))


def constant(data: TensorLike | ScalarType | TableLike) -> MPObject:
    """Create a constant tensor or table from data.

    This function creates a constant that can be used in multi-party
    computations. The constant value is embedded directly into the computation
    graph and is available to all parties in the current party mask.

    Args:
        data: The constant data to embed. Can be:
              - A scalar value (int, float, bool)
              - A numpy array or other tensor-like object
              - A pandas DataFrame or other table-like object
              - Any object that can be converted to tensor

    Returns:
        MPObject: A variable representing the constant tensor or table with:
                  - dtype: Inferred from the input data
                  - shape: Inferred from the input data (for tensors)
                  - schema: Inferred from the input data (for tables)
                  - data: The embedded constant values

    Note:
        The constant data is embedded at graph construction time and is available
        to all parties during execution. Large constants may impact graph size.

        For table-like objects (e.g., pandas DataFrame), JSON serialization is used.
        Note that the constant primitive is not designed to carry large tables efficiently -
        consider using dedicated table loading mechanisms for substantial datasets.
    """
    return cast(MPObject, run(None, basic.constant, data))


@builtin_function
def debug_print(obj: MPObject, prefix: str = "") -> MPObject:
    """Print local value of obj on owning parties and pass it through.

    This function prints the value of an MPObject at runtime on each party that
    owns the value, and returns the same MPObject unchanged. This is useful for
    debugging multi-party computations without affecting the computation flow.

    Args:
        obj: The MPObject whose value should be printed.
        prefix: Optional text prefix for the printed output. Defaults to "".

    Returns:
        MPObject: The same MPObject value passed in, unchanged. This allows
                  the function to be used in chains like: x = debug_print(x, "x=")
                  and prevents dead code elimination (DCE) from removing the print.

    Note:
        The print operation occurs at runtime on each party that holds the value.
        If obj has a static pmask, only parties in that mask will print.
        If obj has a dynamic pmask, the parties are determined at runtime.
    """
    pfunc, eval_args, out_tree = basic.debug_print(obj, prefix=prefix)
    results = peval(pfunc, eval_args)
    return cast(MPObject, out_tree.unflatten(results))


def set_mask(arg: MPObject, mask: Mask) -> MPObject:
    """Set the mask of an MPObject to a new value.

    This function allows changing the party mask of an existing MPObject variable.
    The behavior depends on whether the input MPObject has a dynamic or static pmask:

    **Case 1: Dynamic pmask (arg.pmask is None)**
    - The input MPObject has a runtime-determined pmask
    - The return value's pmask will be exactly the specified mask
    - No validation is performed at compile time

    **Case 2: Static pmask (arg.pmask is not None)**
    - If mask is a subset of arg.pmask: return_var.pmask == arg.pmask (unchanged)
    - If mask is NOT a subset of arg.pmask: raises ValueError at compile time

    Args:
        arg: The MPObject whose mask needs to be changed.
        mask: The target mask to apply. Must be a valid party mask.

    Returns:
        MPObject: A new variable with the specified mask behavior:
                 - For dynamic inputs: pmask = mask
                 - For static inputs (valid subset): pmask = arg.pmask

    Raises:
        ValueError: When arg has a static pmask and mask is not a subset of arg.pmask.
                   This validation occurs at compile time during graph construction.

    Examples:
        **Example 1: Dynamic pmask - mask assignment**
                     P0   P1   P2
                     --   --   --
            Input:   ?    ?    ?     (pmask=None, runtime-determined)
            mask:    [0,2]            (target mask)
        -----------------------------------------------------------
            Output:  x0   -    x2    (pmask=[0,2])

        **Example 2: Static pmask - valid subset**
                     P0   P1   P2
                     --   --   --
            Input:   x0   x1   x2    (pmask=[0,1,2])
            mask:    [0,2]            (subset of input pmask)
        -----------------------------------------------------------
            Output:  x0   -    x2    (pmask=[0,2])

        **Example 3: Static pmask - invalid subset (compile error)**
                     P0   P1   P2
                     --   --   --
            Input:   x0   -    x2     (pmask=[0,2])
            mask:    [1,2]            (NOT subset of [0,2])
        -----------------------------------------------------------
            Result:  ValueError at compile time

    Note:
        This function is typically used for constraining the execution scope
        of variables or for type casting between different pmask contexts.
        The underlying implementation uses JAX identity function with the
        specified execution mask.
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


def run_sql(
    query: str, out_type: Any, in_tables: dict[str, MPObject] | None = None
) -> Any:
    # TODO(jint): add docstring, drop out_type.
    return run(None, sql_cc.run_sql_raw, query, out_type, in_tables)


def run_sql_at(
    rank: Rank, query: str, out_type: Any, in_tables: dict[str, MPObject] | None = None
) -> Any:
    return run_at(rank, sql_cc.run_sql_raw, query, out_type, in_tables)


def run_nnx(nnx_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run an NNX function.

    Args:
        nnx_fn: The NNX function to be executed.
        *args: Positional arguments to pass to the NNX function.
        **kwargs: Keyword arguments to pass to the NNX function.

    Returns:
        The result of evaluating the NNX function through the mplang system.

    Raises:
        TypeError: If the function compilation or evaluation fails.
        RuntimeError: If the underlying peval execution encounters errors.

    Notes:
        Argument binding semantics with respect to NNX static arguments:

        - If an argument (or any leaf within a PyTree argument) is an
          :class:`~mplana.v1.core.mpobject.MPObject`, it is captured as a runtime
          variable (dynamic value) in the traced program and is not treated as a
          NNX static argument.
        - If an argument contains no :class:`MPObject` leaves, it is treated as a
          constant configuration with respect to NNX; effectively it behaves
          like a static argument and may contribute to NNX compilation cache
          keys (similar to ``static_argnums`` semantics). Changing such constant
          arguments can lead to different compiled variants/cached entries.

    Examples:
        Defining and running a simple NNX function:

        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>> def nnx_linear(inputs, weights, bias):
        ...     return jnp.dot(inputs, weights) + bias
        >>> result = run_nnx(nnx_linear, inputs, weights, bias)

        Running an NNX model:

        >>> class LinearModel(nnx.Module):
        ...     def __init__(self, features: int, rngs: nnx.Rngs):
        ...         self.linear = nnx.Linear(features, features, rngs=rngs)
        ...
        ...     def __call__(self, x):
        ...         return self.linear(x)
        >>> def forward_pass(model, x):
        ...     return model(x)
        >>> output = run_nnx(forward_pass, model, input_data)
    """
    return run(None, nnx_cc.run_nnx, nnx_fn, *args, **kwargs)


def run_nnx_at(rank: Rank, nnx_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run an NNX function at a specific rank.

    Args:
        rank: The rank where the NNX function should be executed.
        nnx_fn: The NNX function to be executed.
        *args: Positional arguments to pass to the NNX function.
        **kwargs: Keyword arguments to pass to the NNX function.

    Returns:
        The result of evaluating the NNX function at the specified rank.
    """
    return run_at(rank, nnx_cc.run_nnx, nnx_fn, *args, **kwargs)
