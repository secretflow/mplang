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

import asyncio
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

from mplang.v1.core.async_comm import IAsyncCommunicator
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
from mplang.v1.core.expr.evaluator import EvalSemantic
from mplang.v1.core.expr.visitor import AsyncExprVisitor
from mplang.v1.core.expr.walk import walk_dataflow
from mplang.v1.core.mask import Mask
from mplang.v1.core.pfunc import PFunction
from mplang.v1.kernels.context import RuntimeContext
from mplang.v1.kernels.value import Value


@dataclass
class AsyncEvalSemantic(EvalSemantic):
    """Async version of EvalSemantic.

    Reuses pure computation logic from EvalSemantic but overrides I/O bound methods
    to use IAsyncCommunicator.
    """

    comm: IAsyncCommunicator  # Override type hint
    executor: Executor | None = None

    async def _exec_pfunc_async(self, pfunc: PFunction, args: list[Any]) -> list[Any]:
        # Check if any args are None - if so, this rank shouldn't participate
        # This prevents None values from reaching kernel validation
        if any(arg is None for arg in args):
            return [None] * len(pfunc.outs_info)

        if self.executor:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor, self._exec_pfunc, pfunc, args
            )
        else:
            return self._exec_pfunc(pfunc, args)

    async def _eval_eval_node_async(
        self, expr: EvalExpr, arg_vals: list[Any]
    ) -> list[Any]:
        assert isinstance(expr.pfunc, PFunction)
        if not self._should_run(expr.rmask, arg_vals):
            return [None] * len(expr.mptypes)
        return await self._exec_pfunc_async(expr.pfunc, arg_vals)

    async def _eval_shfl_s_node_async(
        self, expr: ShflSExpr, src_value: Any
    ) -> list[Any]:
        pmask = expr.pmask
        src_ranks = expr.src_ranks
        dst_ranks = list(Mask(pmask))
        assert len(src_ranks) == len(dst_ranks)
        cid = self.comm.new_id()

        # Prepare send and recv operations separately
        send_tasks = []
        recv_futures = []

        # Send phase
        for src, dst in zip(src_ranks, dst_ranks, strict=True):
            if self.comm.rank == src:
                send_tasks.append(self.comm.async_send(dst, cid, src_value))

        # Recv phase
        for src, dst in zip(src_ranks, dst_ranks, strict=True):
            if self.comm.rank == dst:
                recv_futures.append(self.comm.async_recv(src, cid))

        # Execute all operations concurrently to avoid deadlock
        all_tasks = send_tasks + recv_futures
        if all_tasks:
            results = await asyncio.gather(*all_tasks)
            # Return only the recv results
            recv_results = results[len(send_tasks) :]
            if self.comm.rank in dst_ranks:
                assert len(recv_results) == 1
                return recv_results
            else:
                # Should not happen, but handle gracefully
                return [None]
        else:
            # This party is neither sending nor receiving
            if self.comm.rank in dst_ranks:
                # Destination rank but no src_ranks match?
                return [None]
            else:
                # Not involved in this shuffle
                return [None]

    async def _eval_shfl_node_async(
        self, expr: ShflExpr, data: Any, idx: Any
    ) -> list[Any]:
        # Async version of shuffle implementation
        # allgather index via send/recv
        indices = [None] * self.comm.world_size
        cid = self.comm.new_id()

        # Send index to all other ranks
        send_tasks = []
        for dst_rank in range(self.comm.world_size):
            if dst_rank != self.comm.rank:
                send_tasks.append(self.comm.async_send(dst_rank, cid, idx))

        # Receive index from all ranks
        recv_tasks = []
        for src_rank in range(self.comm.world_size):
            if src_rank != self.comm.rank:
                recv_tasks.append(self.comm.async_recv(src_rank, cid))

        # Wait for all operations
        if send_tasks:
            await asyncio.gather(*send_tasks)
        if recv_tasks:
            recv_results = await asyncio.gather(*recv_tasks)
            for i, src_rank in enumerate([
                r for r in range(self.comm.world_size) if r != self.comm.rank
            ]):
                indices[src_rank] = recv_results[i]

        # Set own index
        indices[self.comm.rank] = idx

        # Process indices
        indices_int: list[int | None] = [self._as_optional_int(val) for val in indices]
        send_pairs: list[tuple[int, int]] = []
        for dst_idx, src_idx in enumerate(indices_int):
            if src_idx is not None:
                send_pairs.append((src_idx, dst_idx))
        send_pairs.sort()

        # Second phase: send data according to pairs
        cid = self.comm.new_id()
        received_data = None

        # Send data
        data_send_tasks = []
        for src_rank, dst_rank in send_pairs:
            if self.comm.rank == src_rank:
                data_send_tasks.append(self.comm.async_send(dst_rank, cid, data))

        # Receive data
        data_recv_tasks = []
        for src_rank, dst_rank in send_pairs:
            if self.comm.rank == dst_rank:
                data_recv_tasks.append(self.comm.async_recv(src_rank, cid))

        # Wait for data operations
        if data_send_tasks:
            await asyncio.gather(*data_send_tasks)
        if data_recv_tasks:
            recv_data = await asyncio.gather(*data_recv_tasks)
            # Should receive exactly one data item
            received_data = recv_data[0]

        return [received_data]

    def _as_optional_int(self, val: Any) -> int | None:
        """Convert a value to int if possible, preserving None."""
        val = EvalSemantic._unwrap_value(val)
        if val is None:
            return None
        return int(val)

    async def _verify_uniform_predicate_async(self, pred: Any) -> None:
        # For now, just pass
        # Would need proper async implementation for uniform verification
        pass

    @staticmethod
    def _as_optional_int(val: Any) -> int | None:
        if isinstance(val, int):
            return val
        if isinstance(val, Value):
            if hasattr(val, "value"):
                return int(val.value)
            # Try to convert TensorValue using to_numpy
            to_numpy = getattr(val, "to_numpy", None)
            if callable(to_numpy):
                arr = to_numpy()
                import numpy as np

                if isinstance(arr, np.ndarray) and arr.size == 1:
                    return int(arr.item())
        return None


class AsyncRecursiveEvaluator(AsyncExprVisitor):
    """Original async evaluator using recursive visitor pattern.

    This evaluator can cause stack overflow with deeply nested control flow.
    Kept for reference and fallback.
    """

    def __init__(self, semantic: AsyncEvalSemantic):
        self.semantic = semantic

    def _first(self, vals: list[Any]) -> Any:
        if not isinstance(vals, list):
            return vals
        if len(vals) == 0:
            return None
        return vals[0]

    async def evaluate(self, expr: Expr, env: dict[str, Any] | None = None) -> Any:
        evaluation_env = env if env is not None else self.semantic.env
        return await expr.accept_async(self, evaluation_env)

    async def visit_cond(self, expr: CondExpr, env: dict[str, Any]) -> Any:
        pred_res = await expr.pred.accept_async(self, env)
        pred = self._first(pred_res)

        args_results = await self._spawn_and_gather(expr.args, env)
        flat_args = [self._first(res) for res in args_results]

        if expr.verify_uniform:
            await self.semantic._verify_uniform_predicate_async(pred)

        if isinstance(pred, Value):
            pred_bool = pred.to_bool()
        else:
            pred_bool = bool(self.semantic._unwrap_value(pred))

        if pred_bool:
            new_env = {**env, **dict(zip(expr.then_fn.params, flat_args, strict=True))}
            res = await expr.then_fn.body.accept_async(self, new_env)
        else:
            new_env = {**env, **dict(zip(expr.else_fn.params, flat_args, strict=True))}
            res = await expr.else_fn.body.accept_async(self, new_env)
        return res

    async def visit_call(self, expr: CallExpr, env: dict[str, Any]) -> Any:
        args_results = await self._spawn_and_gather(expr.args, env)
        flat_args = [self._first(res) for res in args_results]
        # Bind arguments
        new_env = {**env, **dict(zip(expr.fn.params, flat_args, strict=True))}
        res = await expr.fn.body.accept_async(self, new_env)
        return res

    async def visit_while(self, expr: WhileExpr, env: dict[str, Any]) -> Any:
        curr_vals_results = await self._spawn_and_gather(expr.args, env)
        curr_vals = [self._first(res) for res in curr_vals_results]

        # Determine split between state and captures
        num_state = expr.body_fn.num_outputs

        # Initial state and captures
        curr_state = curr_vals[:num_state]
        captures = curr_vals[num_state:]

        while True:
            # Reconstruct full arguments: state + captures
            full_args = curr_state + captures

            # Check condition
            cond_env = {**env, **dict(zip(expr.cond_fn.params, full_args, strict=True))}
            cond_res = await expr.cond_fn.body.accept_async(self, cond_env)

            # Validate condition
            cond_val = self.semantic._check_while_predicate(cond_res)

            if not cond_val:
                break

            # Execute body
            body_env = {**env, **dict(zip(expr.body_fn.params, full_args, strict=True))}
            body_res = await expr.body_fn.body.accept_async(self, body_env)

            # Update state - body_res is already a list
            curr_state = body_res

        return curr_state

    async def _spawn_and_gather(
        self, exprs: list[Expr], env: dict[str, Any]
    ) -> list[Any]:
        """Spawn async tasks for multiple expressions and gather results."""
        tasks = [expr.accept_async(self, env) for expr in exprs]
        return await asyncio.gather(*tasks)


class AsyncIterativeEvaluator(AsyncEvalSemantic):
    """Async evaluator using iterative traversal to avoid stack overflow.

    This evaluator follows the same pattern as the synchronous IterativeEvaluator:
    1. Uses local symbols dictionary instead of instance state
    2. Directly recurses via method calls (not Python call stack)
    3. Processes nodes in dependency order
    """

    def __init__(
        self,
        rank: int,
        env: dict[str, Any],
        comm: IAsyncCommunicator,
        runtime: RuntimeContext,
        executor: Executor,
    ):
        super().__init__(rank, env, comm, runtime, executor)

    async def evaluate(self, expr: Expr, env: dict[str, Any] | None = None) -> Any:
        """Entry point for evaluation."""
        evaluation_env = env if env is not None else self.env
        result = await self._iter_eval_graph(expr, evaluation_env)
        return result

    async def _iter_eval_graph(self, root: Expr, env: dict[str, Any]) -> list[Any]:
        """Main evaluation loop using iterative traversal (async version of sync pattern)."""
        symbols: dict[int, list[Any]] = {}

        # Process all nodes in dependency order
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
                # Recursive method call - not Python call stack recursion!
                res = await self._iter_eval_graph(node.fn.body, {**env, **sub_env})
                symbols[id(node)] = res

            elif isinstance(node, CondExpr):
                pred_val = self._first(symbols[id(node.pred)])
                arg_vals = [self._first(symbols[id(a)]) for a in node.args]

                if pred_val is None:
                    symbols[id(node)] = [None] * len(node.mptypes)
                else:
                    # Optional uniform verification
                    if node.verify_uniform:
                        await self._verify_uniform_predicate_async(pred_val)

                    # Convert to bool
                    if isinstance(pred_val, Value):
                        pred = pred_val.to_bool()
                    else:
                        pred = bool(self._unwrap_value(pred_val))

                    if pred:
                        sub_env = dict(zip(node.then_fn.params, arg_vals, strict=True))
                        # Recursive method call
                        res = await self._iter_eval_graph(
                            node.then_fn.body, {**env, **sub_env}
                        )
                        symbols[id(node)] = res
                    else:
                        sub_env = dict(zip(node.else_fn.params, arg_vals, strict=True))
                        # Recursive method call
                        res = await self._iter_eval_graph(
                            node.else_fn.body, {**env, **sub_env}
                        )
                        symbols[id(node)] = res

            elif isinstance(node, WhileExpr):
                state = [self._first(symbols[id(a)]) for a in node.args]
                while True:
                    cond_env = dict(zip(node.cond_fn.params, state, strict=True))
                    # Recursive method call for condition
                    cond_vals = await self._iter_eval_graph(
                        node.cond_fn.body, {**env, **cond_env}
                    )
                    cond_val = self._check_while_predicate(cond_vals)
                    if not bool(cond_val):
                        break

                    body_env = dict(zip(node.body_fn.params, state, strict=True))
                    # Recursive method call for body
                    new_state = await self._iter_eval_graph(
                        node.body_fn.body, {**env, **body_env}
                    )
                    state = self._merge_state(state, new_state)
                symbols[id(node)] = state[0 : len(node.body_fn.mptypes)]

            elif isinstance(node, EvalExpr):
                arg_vals = [self._first(symbols[id(a)]) for a in node.args]
                symbols[id(node)] = await self._eval_eval_node_async(node, arg_vals)

            elif isinstance(node, ConvExpr):
                vars_vals = [self._first(symbols[id(v)]) for v in node.vars]
                # ConvExpr needs async implementation
                symbols[id(node)] = await self._eval_conv_node_async(node, vars_vals)

            elif isinstance(node, ShflSExpr):
                value = self._first(symbols[id(node.src_val)])
                symbols[id(node)] = await self._eval_shfl_s_node_async(node, value)

            elif isinstance(node, ShflExpr):
                data = self._first(symbols[id(node.src)])
                index = self._first(symbols[id(node.index)])
                symbols[id(node)] = await self._eval_shfl_node_async(node, data, index)

            elif isinstance(node, FuncDefExpr):
                # FuncDefExpr should not be directly evaluated
                raise RuntimeError("FuncDefExpr should not be directly evaluated")
            else:
                raise NotImplementedError(f"Unsupported expression type: {type(node)}")

        return symbols[id(root)]

    @staticmethod
    def _first(vals: list[Any]) -> Any:
        """Get first value from list (matches sync evaluator)."""
        if not isinstance(vals, list):
            return vals
        if len(vals) == 0:
            return None
        return vals[0]

    def _merge_state(self, old: list[Any], new: list[Any]) -> list[Any]:
        """Merge state for while loops (matches sync evaluator)."""
        assert len(new) <= len(old)
        return new + old[len(new) :]

    async def _eval_conv_node_async(
        self, expr: ConvExpr, vars_vals: list[Any]
    ) -> list[Any]:
        """Async version of conv node evaluation."""
        # Implement the same logic as sync _eval_conv_node
        assert len(vars_vals) > 0, "pconv called with empty vars list."
        filtered = [v for v in vars_vals if v is not None]
        if len(filtered) == 0:
            return [None]
        if len(filtered) == 1:
            return [filtered[0]]
        raise ValueError(f"pconv called with multiple vars={filtered}.")
