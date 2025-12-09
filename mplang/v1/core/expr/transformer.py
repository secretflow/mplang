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
Expression transformer based on visitor pattern.
"""

from collections.abc import Callable

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


class ExprTransformer(ExprVisitor):
    """Transformer that transforms expressions based on rules."""

    def __init__(self, trans_rules: dict[str, Callable[[Expr], Expr]] | None = None):
        self.trans_rules = trans_rules or {}

    def visit_eval(self, expr: EvalExpr) -> Expr:
        # Transform child expressions first
        transformed_args = [arg.accept(self) for arg in expr.args]
        new_expr = EvalExpr(expr.pfunc, transformed_args, expr.rmask)

        if "eval" in self.trans_rules:
            return self.trans_rules["eval"](new_expr)
        return new_expr

    def visit_variable(self, expr: VariableExpr) -> Expr:
        if "name" in self.trans_rules:
            return self.trans_rules["name"](expr)
        return expr

    def visit_tuple(self, expr: TupleExpr) -> Expr:
        # Transform child expressions first
        transformed_args = [arg.accept(self) for arg in expr.args]
        new_expr = TupleExpr(transformed_args)

        if "tuple" in self.trans_rules:
            return self.trans_rules["tuple"](new_expr)
        return new_expr

    def visit_cond(self, expr: CondExpr) -> Expr:
        # Transform child expressions first
        transformed_pred = expr.pred.accept(self)
        transformed_args = [arg.accept(self) for arg in expr.args]
        new_expr = CondExpr(
            transformed_pred, expr.then_fn, expr.else_fn, transformed_args
        )

        if "cond" in self.trans_rules:
            return self.trans_rules["cond"](new_expr)
        return new_expr

    def visit_call(self, expr: CallExpr) -> Expr:
        # Transform child expressions first
        transformed_args = [arg.accept(self) for arg in expr.args]
        new_expr = CallExpr(expr.name, expr.fn, transformed_args)

        if "call" in self.trans_rules:
            return self.trans_rules["call"](new_expr)
        return new_expr

    def visit_while(self, expr: WhileExpr) -> Expr:
        # Transform all arguments
        transformed_args = [arg.accept(self) for arg in expr.args]
        new_expr = WhileExpr(expr.cond_fn, expr.body_fn, transformed_args)

        if "while" in self.trans_rules:
            return self.trans_rules["while"](new_expr)
        return new_expr

    def visit_conv(self, expr: ConvExpr) -> Expr:
        # Transform child expressions first
        transformed_vars = [var.accept(self) for var in expr.vars]
        new_expr = ConvExpr(transformed_vars)

        if "conv" in self.trans_rules:
            return self.trans_rules["conv"](new_expr)
        return new_expr

    def visit_shfl_s(self, expr: ShflSExpr) -> Expr:
        # Transform child expression first
        transformed_src_val = expr.src_val.accept(self)
        new_expr = ShflSExpr(transformed_src_val, expr.pmask, expr.src_ranks)

        if "shfl_s" in self.trans_rules:
            return self.trans_rules["shfl_s"](new_expr)
        return new_expr

    def visit_shfl(self, expr: ShflExpr) -> Expr:
        # Transform child expressions first
        transformed_src = expr.src.accept(self)
        transformed_index = expr.index.accept(self)
        new_expr = ShflExpr(transformed_src, transformed_index)

        if "shfl" in self.trans_rules:
            return self.trans_rules["shfl"](new_expr)
        return new_expr

    def visit_access(self, expr: AccessExpr) -> Expr:
        # Transform child expression first
        transformed_expr = expr.src.accept(self)
        new_expr = AccessExpr(transformed_expr, expr.index)

        if "access" in self.trans_rules:
            return self.trans_rules["access"](new_expr)
        return new_expr

    def visit_func_def(self, expr: FuncDefExpr) -> Expr:
        # Transform body only, params are just strings now
        transformed_body = expr.body.accept(self)
        new_expr = FuncDefExpr(expr.params, transformed_body)

        if "func_def" in self.trans_rules:
            return self.trans_rules["func_def"](new_expr)
        return new_expr
