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
Expression evaluation engines for MPLang expressions.

- IterativeEvaluator: non-recursive dataflow executor.
- RecursiveEvaluator: visitor-based executor.
- EvalSemantic: shared helpers for both engines.
- IEvaluator: minimal evaluation interface.
- evaluator(kind, ...): factory returning an IEvaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mplang.v1.core.comm import ICommunicator
from mplang.v1.core.expr.ast import (
    AccessExpr,
    CallExpr,
    CondExpr,
    ConvExpr,
    EvalExpr,
    Expr,
    FuncDefExpr,
    ShflExpr,
    ShflSExpr,
    TupleExpr,
    VariableExpr,
    WhileExpr,
)
from mplang.v1.core.expr.visitor import ExprVisitor
from mplang.v1.core.expr.walk import walk_dataflow
from mplang.v1.core.mask import Mask
from mplang.v1.core.pfunc import PFunction
from mplang.v1.kernels.context import RuntimeContext
from mplang.v1.kernels.value import Value


class IEvaluator(Protocol):
    """Public evaluator protocol.

    Added 'runtime' attribute so callers (simulation/resource) can seed
    backend state via evaluator.runtime.run_kernel(...).
    """

    runtime: RuntimeContext

    def evaluate(self, root: Expr, env: dict[str, Any] | None = None) -> list[Any]: ...


@dataclass
class EvalSemantic:
    """Shared evaluation semantics and utilities for evaluators.

    Minimal dataclass carrying runtime execution context (rank/env/comm/runtime).
    """

    rank: int
    env: dict[str, Any]
    comm: ICommunicator
    runtime: RuntimeContext

    # ------------------------------ Shared helpers (semantics) ------------------------------
    def _should_run(self, rmask: Mask | None, args: list[Any]) -> bool:
        if rmask is not None:
            return self.comm.rank in Mask(rmask)
        return all(arg is not None for arg in args)

    def _exec_pfunc(self, pfunc: PFunction, args: list[Any]) -> list[Any]:
        return self.runtime.run_kernel(pfunc, args)

    def _eval_eval_node(self, expr: EvalExpr, arg_vals: list[Any]) -> list[Any]:
        assert isinstance(expr.pfunc, PFunction)
        if not self._should_run(expr.rmask, arg_vals):
            return [None] * len(expr.mptypes)
        return self._exec_pfunc(expr.pfunc, arg_vals)

    def _eval_conv_node(self, vars_vals: list[Any]) -> list[Any]:
        assert len(vars_vals) > 0, "pconv called with empty vars list."
        filtered = [v for v in vars_vals if v is not None]
        if len(filtered) == 0:
            return [None]
        if len(filtered) == 1:
            return [filtered[0]]
        raise ValueError(f"pconv called with multiple vars={filtered}.")

    def _eval_shfl_s_node(self, expr: ShflSExpr, src_value: Any) -> list[Any]:
        pmask = expr.pmask
        src_ranks = expr.src_ranks
        dst_ranks = list(Mask(pmask))
        assert len(src_ranks) == len(dst_ranks)
        cid = self.comm.new_id()
        result = []
        for src, dst in zip(src_ranks, dst_ranks, strict=True):
            if self.comm.rank == src:
                self.comm.send(dst, cid, src_value)
        for src, dst in zip(src_ranks, dst_ranks, strict=True):
            if self.comm.rank == dst:
                result.append(self.comm.recv(src, cid))
        if self.comm.rank in dst_ranks:
            assert len(result) == 1
            return result
        else:
            assert len(result) == 0
            return [None]

    def _eval_shfl_node(self, expr: ShflExpr, data: Any, index: Any) -> list[Any]:
        # allgather index via send/recv
        indices = [None] * self.comm.world_size
        cid = self.comm.new_id()
        for dst_rank in range(self.comm.world_size):
            if dst_rank != self.comm.rank:
                self.comm.send(dst_rank, cid, index)
        for src_rank in range(self.comm.world_size):
            if src_rank != self.comm.rank:
                indices[src_rank] = self.comm.recv(src_rank, cid)
            else:
                indices[src_rank] = index
        indices_int: list[int | None] = [self._as_optional_int(val) for val in indices]
        send_pairs: list[tuple[int, int]] = []
        for dst_idx, src_idx in enumerate(indices_int):
            if src_idx is not None:
                send_pairs.append((src_idx, dst_idx))
        send_pairs.sort()
        cid = self.comm.new_id()
        received_data = None
        for src_rank, dst_rank in send_pairs:
            if self.comm.rank == src_rank:
                self.comm.send(dst_rank, cid, data)
        for src_rank, dst_rank in send_pairs:
            if self.comm.rank == dst_rank:
                received_data = self.comm.recv(src_rank, cid)
        return [received_data]

    @staticmethod
    def _as_optional_int(val: Any) -> int | None:
        """Convert a value to int if possible, preserving None.

        Handles Python ints, floats, numpy scalar types (e.g., np.int32, np.float64), and None.
        Uses int(val) for conversion which works with numpy scalars via __int__().
        """
        val = EvalSemantic._unwrap_value(val)
        if val is None:
            return None
        return int(val)

    def _simple_allgather(self, value: Any) -> list[Any]:
        """All-gather emulation using only ICommunicator send/recv.

        This implements an O(P^2) pairwise exchange (each rank sends its value to all
        other ranks) and collects values in rank order. Suitable for small P (typical
        controller / simulation sizes) and control metadata like a single bool.

        Returns a list of length world_size with entries ordered by rank.
        """
        ws = self.comm.world_size
        value = self._unwrap_value(value)
        # Trivial fast-path
        if ws == 1:
            return [value]
        cid = self.comm.new_id()
        gathered: list[Any] = [None] * ws  # type: ignore
        gathered[self.comm.rank] = value
        # Fan-out
        for dst in range(ws):
            if dst != self.comm.rank:
                self.comm.send(dst, cid, value)
        # Fan-in
        for src in range(ws):
            if src != self.comm.rank:
                gathered[src] = self.comm.recv(src, cid)
        return gathered

    def _verify_uniform_predicate(self, pred: Any) -> None:
        # Runtime uniformity check (O(P^2) send/recv emulation).
        # Use Value.to_bool() if available, otherwise unwrap and convert
        if isinstance(pred, Value):
            pred_bool = pred.to_bool()
        else:
            pred_bool = bool(self._unwrap_value(pred))
        vals = self._simple_allgather(pred_bool)
        if not vals:
            raise ValueError("uniform_cond: empty gather for predicate")
        first = vals[0]
        for v in vals[1:]:
            if v != first:
                raise ValueError(
                    "uniform_cond: predicate is not uniform across parties"
                )

    # ------------------------------ While helpers ------------------------------
    def _check_while_predicate(self, cond_result: list[Any]) -> Any:
        """Validate while_loop predicate evaluation result.

        Ensures the condition function returns exactly one value and that value
        is non-None. Returns the boolean predicate value for convenience.

        Raises:
            AssertionError: If condition function returns != 1 value.
            RuntimeError: If the single predicate value is None.
        """
        assert len(cond_result) == 1, (
            f"Condition function must return a single value, got {cond_result}"
        )
        cond_val = cond_result[0]
        if cond_val is None:
            raise RuntimeError(
                "while_loop condition produced None on rank "
                f"{self.rank}; ensure the predicate yields a boolean for every party."
            )
        # Use Value.to_bool() if available for cleaner conversion
        if isinstance(cond_val, Value):
            return cond_val.to_bool()
        return bool(self._unwrap_value(cond_val))

    @staticmethod
    def _unwrap_value(value: Any) -> Any:
        """Convert Value payloads to numpy/python equivalents when possible."""
        if value is None:
            return None

        if isinstance(value, Value):
            # Try to_numpy first for broader compatibility
            to_numpy = getattr(value, "to_numpy", None)
            if callable(to_numpy):
                arr = to_numpy()
                import numpy as np

                if isinstance(arr, np.ndarray):
                    if arr.size == 1:
                        return arr.item()
                    return arr
                return arr
        return value


class RecursiveEvaluator(EvalSemantic, ExprVisitor):
    """Recursive visitor-based evaluator."""

    def __init__(
        self,
        rank: int,
        env: dict[str, Any],
        comm: ICommunicator,
        runtime: RuntimeContext,
    ) -> None:
        super().__init__(rank, env, comm, runtime)
        self._cache: dict[int, Any] = {}  # Cache based on expr id

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

    # Internal helper to create a new evaluator with extended env for nested regions
    def _fork(self, sub_bindings: dict[str, Any]) -> RecursiveEvaluator:
        merged_env = {**self.env, **sub_bindings}
        # Create a child evaluator sharing the same runtime (no new backend state).
        return RecursiveEvaluator(self.rank, merged_env, self.comm, self.runtime)

    def visit_eval(self, expr: EvalExpr) -> Any:
        """Evaluate function call expression."""
        args = [self._value(arg) for arg in expr.args]
        return self._eval_eval_node(expr, args)

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
        """Evaluate conditional expression (uniform/global semantics).

        Current behavior:
          * Assumes predicate is already uniform (same value on every enabled party).
          * Only the selected branch is executed locally.
          * If this party is masked out for outputs, returns [None] placeholders.

        Future optimization notes:
          * Current uniform verification uses an O(P^2) manual all-gather. Replace
            with a communicator-level boolean all-reduce (AND + broadcast) when available.
          * Add optional static uniform inference (data provenance) to elide the
            runtime check when predicate uniformity is provable at trace time.
        """
        pred_val = self._value(expr.pred)
        if pred_val is None:
            return [None] * len(expr.mptypes)

        if expr.verify_uniform:
            self._verify_uniform_predicate(pred_val)

        # Convert to bool using Value.to_bool() if available
        if isinstance(pred_val, Value):
            pred = pred_val.to_bool()
        else:
            pred = bool(self._unwrap_value(pred_val))

        # Only evaluate selected branch locally
        if bool(pred):
            then_call = CallExpr("then", expr.then_fn, expr.args)
            return self._values(then_call)
        else:
            else_call = CallExpr("else", expr.else_fn, expr.args)
            return self._values(else_call)

    def visit_call(self, expr: CallExpr) -> Any:
        args = [self._value(arg) for arg in expr.args]
        assert isinstance(expr.fn, FuncDefExpr)
        sub_env = dict(zip(expr.fn.params, args, strict=True))
        sub_evaluator = self._fork(sub_env)
        return expr.fn.body.accept(sub_evaluator)

    def visit_while(self, expr: WhileExpr) -> Any:
        """Evaluate while loop expression."""
        # Start with initial state
        state = [self._value(arg) for arg in expr.args]

        while True:
            # Call condition function
            cond_env = dict(zip(expr.cond_fn.params, state, strict=True))
            cond_evaluator = self._fork(cond_env)
            cond_result = expr.cond_fn.body.accept(cond_evaluator)
            cond_value = self._check_while_predicate(cond_result)
            if not cond_value:
                break

            # Call body function with same arguments
            body_env = dict(zip(expr.body_fn.params, state, strict=True))
            body_evaluator = self._fork(body_env)
            new_state = expr.body_fn.body.accept(body_evaluator)

            assert len(new_state) == len(expr.body_fn.mptypes)
            assert len(new_state) <= len(state)

            state = new_state + state[len(new_state) :]

        # Return in the same format as original arguments
        return state[0 : len(expr.body_fn.mptypes)]

    def visit_conv(self, expr: ConvExpr) -> Any:
        """Evaluate converge expression."""
        vals = [self._value(arg) for arg in expr.vars]
        return self._eval_conv_node(vals)

    def visit_shfl_s(self, expr: ShflSExpr) -> Any:
        """Evaluate static shuffle expression."""
        value = self._value(expr.src_val)
        return self._eval_shfl_s_node(expr, value)

    def visit_shfl(self, expr: ShflExpr) -> Any:
        """Evaluate dynamic shuffle expression."""
        data = self._value(expr.src)
        index = self._value(expr.index)
        return self._eval_shfl_node(expr, data, index)

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
        raise RuntimeError("FuncDefExpr should not be directly evaluated")

    # IEvaluator API: return list of values
    def evaluate(self, root: Expr, env: dict[str, Any] | None = None) -> list[Any]:
        if env is None:
            res = root.accept(self)
        else:
            # Spawn a sibling evaluator with override env but same runtime.
            res = root.accept(
                RecursiveEvaluator(self.rank, env, self.comm, self.runtime)
            )
        if not isinstance(res, list):
            raise ValueError(f"got {type(res)} for expression {root}")
        return res


class IterativeEvaluator(EvalSemantic):
    """Iterative (non-recursive) evaluator using dataflow traversal."""

    def __init__(
        self,
        rank: int,
        env: dict[str, Any],
        comm: ICommunicator,
        runtime: RuntimeContext,
    ) -> None:
        super().__init__(rank, env, comm, runtime)

    @staticmethod
    def _first(vals: list[Any]) -> Any:
        if not isinstance(vals, list):
            return vals
        if len(vals) == 0:
            return None
        return vals[0]

    def _merge_state(self, old: list[Any], new: list[Any]) -> list[Any]:
        assert len(new) <= len(old)
        return new + old[len(new) :]

    def _iter_eval_graph(self, root: Expr, env: dict[str, Any]) -> list[Any]:
        symbols: dict[int, list[Any]] = {}
        for node in walk_dataflow(root, traversal="dfs_post_iter"):
            if isinstance(node, VariableExpr):
                if node.name not in env:
                    raise ValueError(
                        f"Variable '{node.name}' not found in evaluator environment"
                    )
                symbols[id(node)] = [env[node.name]]
            elif isinstance(node, TupleExpr):
                vals = [self._first(symbols[id(a)]) for a in node.args]
                symbols[id(node)] = vals
            elif isinstance(node, AccessExpr):
                src_vals = symbols[id(node.src)]
                symbols[id(node)] = [src_vals[node.index]]
            elif isinstance(node, CallExpr):
                arg_vals = [self._first(symbols[id(a)]) for a in node.args]
                assert isinstance(node.fn, FuncDefExpr)
                sub_env = dict(zip(node.fn.params, arg_vals, strict=True))
                res = self._iter_eval_graph(node.fn.body, {**env, **sub_env})
                symbols[id(node)] = res
            elif isinstance(node, CondExpr):
                pred_val = self._first(symbols[id(node.pred)])
                arg_vals = [self._first(symbols[id(a)]) for a in node.args]
                if pred_val is None:
                    symbols[id(node)] = [None] * len(node.mptypes)
                else:
                    # Optional uniform verification identical to recursive evaluator (DRY helper).
                    if node.verify_uniform:
                        self._verify_uniform_predicate(pred_val)
                    # Convert to bool using Value.to_bool() if available
                    if isinstance(pred_val, Value):
                        pred = pred_val.to_bool()
                    else:
                        pred = bool(self._unwrap_value(pred_val))
                    if pred:
                        sub_env = dict(zip(node.then_fn.params, arg_vals, strict=True))
                        res = self._iter_eval_graph(
                            node.then_fn.body, {**env, **sub_env}
                        )
                        symbols[id(node)] = res
                    else:
                        sub_env = dict(zip(node.else_fn.params, arg_vals, strict=True))
                        res = self._iter_eval_graph(
                            node.else_fn.body, {**env, **sub_env}
                        )
                        symbols[id(node)] = res
            elif isinstance(node, WhileExpr):
                state = [self._first(symbols[id(a)]) for a in node.args]
                while True:
                    cond_env = dict(zip(node.cond_fn.params, state, strict=True))
                    cond_vals = self._iter_eval_graph(
                        node.cond_fn.body, {**env, **cond_env}
                    )
                    cond_val = self._check_while_predicate(cond_vals)
                    if not bool(cond_val):
                        break
                    body_env = dict(zip(node.body_fn.params, state, strict=True))
                    new_state = self._iter_eval_graph(
                        node.body_fn.body, {**env, **body_env}
                    )
                    state = self._merge_state(state, new_state)
                symbols[id(node)] = state[0 : len(node.body_fn.mptypes)]
            elif isinstance(node, EvalExpr):
                arg_vals = [self._first(symbols[id(a)]) for a in node.args]
                symbols[id(node)] = self._eval_eval_node(node, arg_vals)
            elif isinstance(node, ConvExpr):
                vars_vals = [self._first(symbols[id(v)]) for v in node.vars]
                symbols[id(node)] = self._eval_conv_node(vars_vals)
            elif isinstance(node, ShflSExpr):
                value = self._first(symbols[id(node.src_val)])
                symbols[id(node)] = self._eval_shfl_s_node(node, value)
            elif isinstance(node, ShflExpr):
                data = self._first(symbols[id(node.src)])
                index = self._first(symbols[id(node.index)])
                symbols[id(node)] = self._eval_shfl_node(node, data, index)
            elif isinstance(node, FuncDefExpr):
                # Definition nodes are not evaluated; placeholder to satisfy walkers
                symbols[id(node)] = node.body.mptypes
            else:
                raise NotImplementedError(
                    f"Unsupported expr in iterative eval: {type(node)}"
                )
        res = symbols[id(root)]
        if not isinstance(res, list):
            raise ValueError(f"got {type(res)} for expression {root}")
        return res

    def evaluate(self, root: Expr, env: dict[str, Any] | None = None) -> list[Any]:
        """Evaluate an expression graph iteratively (no Python recursion).

        - Traverses dataflow using iterative DFS-postorder to compute ready nodes.
        - For control flow/functional regions (Call/Cond/While), performs a
          localized iterative evaluation of the region body with a child environment.

        Args:
            root: The root expression to evaluate.
            env: Optional environment override for VariableExpr lookups.

        Returns:
            A list of computed output values for the root expression.
        """
        cur_env = self.env if env is None else env
        return self._iter_eval_graph(root, cur_env)


def create_evaluator(
    rank: int,
    env: dict[str, Any],
    comm: ICommunicator,
    runtime: RuntimeContext,
    kind: str | None = "iterative",
) -> IEvaluator:
    """Factory to create an evaluator engine.

    Args:
        rank: Party rank.
        env: Initial variable environment.
        comm: Communicator for this party.
        kind: Evaluator implementation ("iterative" or "recursive").

    Returns:
        An IEvaluator instance of the requested kind.
    """
    # Backward compatibility: treat kind=None as default iterative implementation.
    if kind is None or kind == "iterative":
        return IterativeEvaluator(rank, env, comm, runtime)
    if kind == "recursive":
        return RecursiveEvaluator(rank, env, comm, runtime)
    raise ValueError(f"Unknown evaluator kind: {kind}")
