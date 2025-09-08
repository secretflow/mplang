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
Visitor pattern interface for expression system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mplang.core.expr.ast import (
        AccessExpr,
        CallExpr,
        CondExpr,
        ConvExpr,
        EvalExpr,
        FuncDefExpr,
        ShflExpr,
        ShflSExpr,
        TupleExpr,
        VariableExpr,
        WhileExpr,
    )


class ExprVisitor(ABC):
    """Base visitor interface for expression types."""

    @abstractmethod
    def visit_eval(self, expr: EvalExpr) -> Any:
        pass

    @abstractmethod
    def visit_variable(self, expr: VariableExpr) -> Any:
        pass

    @abstractmethod
    def visit_tuple(self, expr: TupleExpr) -> Any:
        pass

    @abstractmethod
    def visit_cond(self, expr: CondExpr) -> Any:
        pass

    @abstractmethod
    def visit_call(self, expr: CallExpr) -> Any:
        pass

    @abstractmethod
    def visit_while(self, expr: WhileExpr) -> Any:
        pass

    @abstractmethod
    def visit_conv(self, expr: ConvExpr) -> Any:
        pass

    @abstractmethod
    def visit_shfl_s(self, expr: ShflSExpr) -> Any:
        pass

    @abstractmethod
    def visit_shfl(self, expr: ShflExpr) -> Any:
        pass

    @abstractmethod
    def visit_access(self, expr: AccessExpr) -> Any:
        pass

    @abstractmethod
    def visit_func_def(self, expr: FuncDefExpr) -> Any:
        pass
