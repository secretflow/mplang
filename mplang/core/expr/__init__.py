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
Expression system for multi-party computation graph construction.

This package provides a modern, extensible expression-based architecture for building
multi-party computation graphs using the visitor pattern.
"""

# Core expression types
from mplang.core.expr.ast import (
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

# Built-in visitor implementations
from mplang.core.expr.evaluator import Evaluator
from mplang.core.expr.printer import Printer
from mplang.core.expr.transformer import ExprTransformer

# Utility functions
from mplang.core.expr.utils import (
    deduce_mask,
    ensure_scalar,
    ensure_tensorlist_equal,
    type_equal,
)

# Visitor pattern interface
from mplang.core.expr.visitor import ExprVisitor

__all__ = [
    "AccessExpr",
    "CallExpr",
    "CondExpr",
    "ConvExpr",
    "EvalExpr",
    "Evaluator",
    "Expr",
    "ExprTransformer",
    "ExprVisitor",
    "FuncDefExpr",
    "Printer",
    "ShflExpr",
    "ShflSExpr",
    "TupleExpr",
    "VariableExpr",
    "WhileExpr",
    "deduce_mask",
    "ensure_scalar",
    "ensure_tensorlist_equal",
    "type_equal",
]
