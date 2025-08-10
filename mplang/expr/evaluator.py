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

"""
Expression evaluator for executing MPLang expressions.

This module implements an expression evaluator that can execute expressions
using a fork-based execution model for clean parameter binding and isolation.
"""

from __future__ import annotations

from typing import Any

from mplang.core.base import Mask
from mplang.core.comm import ICommunicator
from mplang.core.pfunc import PFunction, PFunctionHandler
from mplang.expr.ast import (
    AccessExpr,
    CallExpr,
    CondExpr,
    ConstExpr,
    ConvExpr,
    EvalExpr,
    Expr,
    FuncDefExpr,
    RandExpr,
    RankExpr,
    ShflExpr,
    ShflSExpr,
    TupleExpr,
    VariableExpr,
    WhileExpr,
)
from mplang.expr.visitor import ExprVisitor
from mplang.utils.mask import Mask


class Evaluator(ExprVisitor):
    """Simple expression evaluator using fork-based execution model.

    This evaluator uses a fork-based design instead of explicit execution frames:
    - Each function call creates a forked evaluator with parameter bindings
    - Variable lookup is done in a simple flat environment
    - Lexical scoping is achieved through environment sharing in forks
    """

    def __init__(
        self,
        rank: int,
        env: dict[str, Any],
        comm: ICommunicator,
        pfunc_handles: list[PFunctionHandler] | None = None,
    ):
        """Initialize the evaluator.

        Args:
            rank: The rank of the current party (default 0).
            env: Variable environment for free variables (stores values referenced by pname in DAG,
                 not the evaluator's symbol table for intermediate DAG node values)
            comm: The communicator for inter-party communication.
            pfunc_handles: Optional dictionary of PFunction handlers.
        """
        self.rank = rank
        self.comm = comm
        self.env = env
        self._pfunc_handles: list[PFunctionHandler] = pfunc_handles or []
        self._cache: dict[int, Any] = {}  # Cache based on expr id

        # setup pfunction dispatch table
        self._dispatch_table = {}
        for handler in self._pfunc_handles:
            for pfunc_name in handler.list_fn_names():
                if pfunc_name not in self._dispatch_table:
                    self._dispatch_table[pfunc_name] = handler
                else:
                    raise ValueError(
                        f"Duplicate PFunction handler for type {pfunc_name}: "
                        f"{self._dispatch_table[pfunc_name]} and {handler}"
                    )

        # setup handlers for PFunction execution
        for handler in self._pfunc_handles:
            handler.setup()

    def _get_var(self, name: str) -> Any:
        """Get variable from environment."""
        if name not in self.env:
            raise ValueError(f"Variable '{name}' not found in evaluator environment")
        return self.env[name]

    def _value(self, expr: Expr) -> Any:
        """Evaluate an expression and cache the result."""
        values = self._values(expr)
        if len(expr.mptypes) != 1:
            raise ValueError(
                f"Expected single value for expression {expr}, got {len(values)} values"
            )
        return values[0]

    def _values(self, expr: Expr) -> list[Any]:
        """Evaluate an expression and return the result as a list."""
        expr_id = id(expr)
        if expr_id not in self._cache:
            self._cache[expr_id] = expr.accept(self)
        values = self._cache[expr_id]
        if not isinstance(values, list):
            raise ValueError(f"got {type(values)} for expression {expr}")
        return values

    def fork(self, sub_bindings: dict[str, Any]) -> Evaluator:
        """Create a forked evaluator with additional variables."""
        forked = Evaluator(self.rank, sub_bindings, self.comm, self._pfunc_handles)
        return forked

    def visit_rank(self, expr: RankExpr) -> Any:
        """Evaluate rank expression."""
        # Return the current party's rank
        return [self.rank]

    def visit_const(self, expr: ConstExpr) -> Any:
        """Evaluate constant expression."""
        import numpy as np

        # Reconstruct the constant value from bytes
        shape, dtype = expr.typ.shape, expr.typ.dtype.numpy_dtype()
        if shape == ():
            # Scalar
            data = np.frombuffer(expr.data_bytes, dtype=dtype)
            return [data[0]]  # Return numpy scalar, not Python scalar
        else:
            # Tensor
            data = np.frombuffer(expr.data_bytes, dtype=dtype).reshape(shape)
            return [data]

    def visit_rand(self, expr: RandExpr) -> Any:
        """Evaluate random expression."""
        import numpy as np

        # Generate random values with the specified shape
        shape = expr.typ.shape
        dtype = expr.typ.dtype.numpy_dtype()

        rng = np.random.default_rng()
        if dtype == np.uint64:
            info = np.iinfo(np.uint64)
            data = rng.integers(
                low=info.min,
                high=info.max,
                size=shape,
                dtype=np.uint64,
                endpoint=True,  # includes the high value in the possible results
            )
        else:
            data = rng.random(size=shape).astype(dtype)
        return [data]

    def visit_eval(self, expr: EvalExpr) -> Any:
        """Evaluate function call expression."""
        # Evaluate arguments
        args = [self._value(arg) for arg in expr.args]

        assert isinstance(expr.pfunc, PFunction)

        rmask: Mask | None = expr.rmask
        if rmask is not None:
            # if rmask is provided, we check if the current rank is in the mask.
            should_run = Mask(rmask).is_rank_in(self.comm.rank)
        else:
            # deduce from runtime arguments.
            should_run = all(arg is not None for arg in args)

        if not should_run:
            # If the current rank is not in the mask, return None.
            return [None] * len(expr.mptypes)
        else:
            pfunc = expr.pfunc
            if pfunc.fn_type in self._dispatch_table:
                handler = self._dispatch_table[pfunc.fn_type]
                return handler.execute(pfunc, args)
            else:
                raise NotImplementedError(
                    f"PFunction type {pfunc.fn_type} not supported"
                )

    def visit_variable(self, expr: VariableExpr) -> Any:
        """Evaluate variable expression - just look up in environment.

        No distinction between captured variables and parameters at this level.
        All variables are just names to be resolved in the current environment.
        """
        value = self._get_var(expr.name)
        # Ensure consistency: all visit methods should return a list
        return [value]

    def visit_tuple(self, expr: TupleExpr) -> Any:
        """Evaluate tuple expression."""
        results = [self._value(arg) for arg in expr.args]
        return results

    def visit_cond(self, expr: CondExpr) -> Any:
        """Evaluate conditional expression."""
        pred = self._value(expr.pred)

        # Execute then_fn if party local pred is True, else execute else_fn
        if pred is None:
            # self party is masked out, just return None
            return [None] * len(expr.mptypes)
        else:
            if pred:
                then_call = CallExpr(expr.then_fn, expr.args)
                return self._values(then_call)
            else:
                else_call = CallExpr(expr.else_fn, expr.args)
                return self._values(else_call)

    def visit_call(self, expr: CallExpr) -> Any:
        args = [self._value(arg) for arg in expr.args]

        assert isinstance(expr.fn, FuncDefExpr)

        sub_env = dict(zip(expr.fn.params, args, strict=True))
        sub_evaluator = self.fork(sub_env)

        return expr.fn.body.accept(sub_evaluator)

    def visit_while(self, expr: WhileExpr) -> Any:
        """Evaluate while loop expression."""
        # Start with initial state
        state = [self._value(arg) for arg in expr.args]

        while True:
            # Call condition function
            cond_env = dict(zip(expr.cond_fn.params, state, strict=True))
            cond_evaluator = self.fork(cond_env)
            cond_result = expr.cond_fn.body.accept(cond_evaluator)

            assert len(cond_result) == 1, (
                f"Condition function must return a single value, got {cond_result}"
            )

            if not cond_result[0]:
                break

            # Call body function with same arguments
            body_env = dict(zip(expr.body_fn.params, state, strict=True))
            body_evaluator = self.fork(body_env)
            new_state = expr.body_fn.body.accept(body_evaluator)

            assert len(new_state) == len(expr.body_fn.mptypes)
            assert len(new_state) <= len(state)

            state = new_state + state[len(new_state) :]

        # Return in the same format as original arguments
        return state[0 : len(expr.body_fn.mptypes)]

    def visit_conv(self, expr: ConvExpr) -> Any:
        """Evaluate converge expression."""
        vars = [self._value(arg) for arg in expr.vars]

        assert len(vars) > 0, "pconv called with empty vars list."
        vars = list(filter(lambda x: x is not None, vars))

        if len(vars) == 0:
            # if all vars are None, means self is not converged, return None
            return [None]
        elif len(vars) == 1:
            # if only one var is provided, we return it directly.
            return vars
        else:
            raise ValueError(f"pconv called with multiple vars={vars}.")

    def visit_shfl_s(self, expr: ShflSExpr) -> Any:
        """Evaluate static shuffle expression."""
        pmask = expr.pmask
        src_ranks = expr.src_ranks
        value = self._value(expr.src_val)

        dst_ranks = list(Mask(pmask).enum())
        assert len(src_ranks) == len(dst_ranks)

        # do the shuffle.
        cid = self.comm.new_id()

        result = []
        for src, dst in zip(src_ranks, dst_ranks, strict=False):
            if self.comm.rank == src:
                self.comm.send(dst, cid, value)

        for src, dst in zip(src_ranks, dst_ranks, strict=False):
            if self.comm.rank == dst:
                result.append(self.comm.recv(src, cid))

        if self.comm.rank in dst_ranks:
            assert len(result) == 1, f"Expected one result, got {len(result)}"
            return result
        else:
            assert len(result) == 0, f"Expected no result, got {len(result)}"
            return [None]

    def visit_shfl(self, expr: ShflExpr) -> Any:
        """Evaluate dynamic shuffle expression."""
        data = self._value(expr.src)
        index = self._value(expr.index)

        # The algorithm
        # r = pshfl(x, i)
        #    P0  P1  P2  P3
        # x  A   B   C   _
        # i  _   2   0   1
        # r  _   C   A   B

        # First, gather all indices using send/recv
        # Note: index is runtime determined, but not 'protected'.
        # All parties participate in allgather, even if their index is None

        # Use send/recv to implement allgather logic
        indices = [None] * self.comm.world_size
        cid = self.comm.new_id()

        # Each party sends its index to all other parties
        for dst_rank in range(self.comm.world_size):
            if dst_rank != self.comm.rank:
                self.comm.send(dst_rank, cid, index)

        # Each party receives indices from all other parties
        for src_rank in range(self.comm.world_size):
            if src_rank != self.comm.rank:
                indices[src_rank] = self.comm.recv(src_rank, cid)
            else:
                indices[src_rank] = index

        assert all(val.ndim == 0 for val in indices if val is not None)
        indices = [val if val is None else int(val.item()) for val in indices]

        # Build source-to-destination mapping from the indices
        # indices[dst] = src means data from src should go to dst
        # So we need to reverse this to get src -> dst mapping
        send_pairs = []  # List of (src, dst) pairs
        for dst_rank, src_rank in enumerate(indices):
            if src_rank is not None:
                send_pairs.append((src_rank, dst_rank))

        # Sort pairs to ensure deterministic order across all parties
        send_pairs.sort()

        # Execute point-to-point communications using send/recv
        # Use separate loops to avoid deadlock if send is asynchronous
        cid = self.comm.new_id()
        received_data = None

        # First loop: all sends
        for src_rank, dst_rank in send_pairs:
            if self.comm.rank == src_rank:
                self.comm.send(dst_rank, cid, data)

        # Second loop: all receives
        for src_rank, dst_rank in send_pairs:
            if self.comm.rank == dst_rank:
                received_data = self.comm.recv(src_rank, cid)

        return [received_data]

    def visit_access(self, expr: AccessExpr) -> Any:
        """Evaluate access expression."""
        # Evaluate the expression and access the specified index
        result = self._values(expr.src)

        if expr.index < 0 or expr.index >= len(result):
            raise IndexError(
                f"Index {expr.index} out of range for list of length {len(result)}"
            )
        return [result[expr.index]]  # Ensure we return a list

    def visit_func_def(self, expr: FuncDefExpr) -> Any:
        # Definition expressions should not be evaluated directly.
        raise RuntimeError("FuncDefExpr should not be directly evaluated")
