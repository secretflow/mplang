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
Interpreter context and InterpVar implementation.

This module provides the interpreter context for eager evaluation and InterpVar
which references computed values in an interpreter.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, cast

from mplang.core.cluster import ClusterSpec
from mplang.core.expr.ast import Expr, VariableExpr
from mplang.core.mpobject import MPContext, MPObject
from mplang.core.mptype import MPType, TensorLike
from mplang.core.tracer import TracedFunction
from mplang.utils.func_utils import var_demorph, var_morph


# TODO(jint): Should we use inheritance or composition here?
class InterpContext(MPContext):
    """Context for eager evaluation using an interpreter.

    InterpContext executes computations immediately and stores results
    in an underlying interpreter.
    """

    def __init__(
        self,
        cluster_spec: ClusterSpec,
    ):
        super().__init__(cluster_spec)

    @abstractmethod
    def evaluate(self, expr: Expr, bindings: dict[str, MPObject]) -> Sequence[MPObject]:
        """Evaluate an expression in this context.

        Args:
            expr: The expression to evaluate.
            bindings: A dictionary of variable bindings.

        Returns:
            The result of the evaluation as an MPObject.
        """
        raise NotImplementedError("Should be overridden in subclasses.")

    @abstractmethod
    def fetch(self, obj: MPObject) -> list[TensorLike]:
        """Fetch the value of an MPObject from this InterpContext to the current Python interpreter.

        The MPObject must have been created by this InterpContext. If the object
        was not produced by this context, a ValueError will be raised.

        Args:
            obj: The MPObject to fetch. Must be produced by this InterpContext.

        Returns:
            A list of tensor-like values with length equal to psize(). For each party i,
            if the i-th bit of obj.pmask is 0 (indicating party i does not hold this value),
            the i-th element in the returned list will be None. Otherwise, it contains
            the actual tensor value held by party i.

        Raises:
            ValueError: If obj was not produced by this InterpContext.
        """
        raise NotImplementedError("Should be overridden in subclasses.")


class InterpVar(MPObject):
    """A variable that references a value in an interpreter.

    InterpVar represents a value that has been computed and exists
    in the interpreter's variable store.
    """

    def __init__(self, ctx: InterpContext, mptype: MPType):
        self._ctx = ctx
        self._mptype = mptype

    @property
    def ctx(self) -> MPContext:
        """The context this variable belongs to."""
        return self._ctx

    @property
    def mptype(self) -> MPType:
        """The type of this variable."""
        # TODO: fetch type from the Interpreter and cache it.
        return self._mptype

    def __repr__(self) -> str:
        return f"InterpVar(mptype={self.mptype})"


def apply(ctx: InterpContext, fn: TracedFunction, *args: Any, **kwargs: Any) -> Any:
    is_mpobj = lambda x: isinstance(x, MPObject)
    in_args, in_imms, in_struct = var_morph((args, kwargs), is_mpobj)

    # All variables must be in the same context as the function.
    if not all(isinstance(var, InterpVar) and var.ctx is ctx for var in in_args):
        raise ValueError("All input variables must be InterpVars in the same context.")

    # Check if the function signature matches the input types.
    if fn.in_struct != in_struct:
        raise ValueError(f"Input structure mismatch: {fn.in_struct} != {in_struct}")
    if fn.in_imms != in_imms:
        # Should trigger re-trace in JAX
        raise ValueError(f"Input immutables mismatch: {fn.in_imms} != {in_imms}")
    if len(fn.in_vars) != len(in_args):
        raise ValueError(f"Input types mismatch: {fn.in_vars} != {in_args}")
    # check parameter type match
    for param, arg in zip(fn.in_vars, in_args, strict=False):
        if param.mptype != arg.mptype:
            raise ValueError(
                f"Input variable type mismatch: {param.mptype} != {arg.mptype}"
            )

    # Prepare for the captured variables, which should also be in the same context.
    for captured, _traced in fn.capture_map.items():
        if not isinstance(captured, InterpVar) or captured.ctx is not ctx:
            raise ValueError(
                f"Capture {captured} must be in this({ctx}) context, got {captured.ctx}."
            )

    arg_binding: dict[str, MPObject] = {
        cast(VariableExpr, var.expr).name: obj
        for var, obj in zip(fn.in_vars, in_args, strict=False)
    }
    capture_binding = {
        cast(VariableExpr, var.expr).name: captured
        for captured, var in fn.capture_map.items()
    }

    if len(fn.out_vars) == 0:
        out_vars: list[MPObject] = []
    else:
        func_expr = fn.make_expr()
        assert func_expr is not None, "Function expression should not be None."
        out_vars = list(
            ctx.evaluate(func_expr.body, {**arg_binding, **capture_binding})
        )

    assert isinstance(out_vars, list), f"Expected list, got {type(out_vars)}"
    return var_demorph(out_vars, fn.out_imms, fn.out_struct)
