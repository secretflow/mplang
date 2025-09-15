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
Abstract Syntax Tree (AST) nodes for multi-party computation expressions.

This module defines the AST nodes for representing multi-party computation expressions.
Each node type represents a different kind of operation or construct in the multi-party
computation language, following the visitor pattern for extensible processing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mplang.core.expr.utils import deduce_mask
from mplang.core.mask import Mask
from mplang.core.mptype import MPType, Rank
from mplang.core.pfunc import PFunction
from mplang.core.table import TableType
from mplang.core.tensor import TensorType

if TYPE_CHECKING:
    from mplang.core.expr.visitor import ExprVisitor


class Expr(ABC):
    """Base class for all expression types in the multi-party computation graph.

    This expression system is designed to be Multi-Input Multi-Output (MIMO),
    meaning each expression node can conceptually have multiple outputs. This is
    fundamental to supporting multi-output PFunctions and constructing complex
    dataflow graphs efficiently.

    Attributes:
        mptypes (list[MPType]): The list of output types for this expression. This
            is the core property that enables MIMO capabilities. It's computed
            lazily and cached.
        mptype (MPType): A convenience property for the common case of a single-output
            expression. It raises a ValueError if the expression does not have
            exactly one output, providing a useful runtime check.
    """

    def __init__(self) -> None:
        self._mptypes: list[MPType] | None = None

    @property
    def num_outputs(self) -> int:
        """Return the number of outputs this expression produces."""
        return len(self.mptypes)

    @property
    def mptypes(self) -> list[MPType]:
        if self._mptypes is None:
            self._mptypes = self._compute_mptypes()
        return self._mptypes

    @property
    def mptype(self) -> MPType:
        """Convenience property for single-output expressions."""
        types = self.mptypes
        if len(types) != 1:
            raise ValueError(f"Expression has {len(types)} outputs, expected 1")
        return types[0]

    @abstractmethod
    def _compute_mptypes(self) -> list[MPType]:
        """Computes the types of the expression's outputs."""

    @abstractmethod
    def accept(self, visitor: ExprVisitor) -> Any:
        """Accept a visitor for the visitor pattern."""


# ============================================================================
# Concrete Expression Classes
# ============================================================================


class EvalExpr(Expr):
    """Expression for multi-party function evaluation."""

    def __init__(
        self, pfunc: PFunction, args: list[Expr], rmask: Mask | int | None = None
    ):
        super().__init__()
        # Type checking - basic validation that we have the right number of inputs
        if len(args) != len(pfunc.ins_info):
            raise ValueError(
                f"Expected {len(pfunc.ins_info)} arguments, got {len(args)}"
            )
        rmask = Mask(rmask) if rmask is not None else None

        self.pfunc = pfunc
        self.args = args
        self.rmask = rmask

    def _compute_mptypes(self) -> list[MPType]:
        """Compute output MPTypes based on PFunction and mask deduction logic.

        The logic follows these steps:
        1. Determine output TensorType (dtype + shape) from PFunction
        2. If rmask is explicitly provided (caller has strong mask knowledge):
            2.1 Deduce pmask from args (intersection of all arg pmasks)
                2.1.1 If deduced pmask is not None (trace time known):
                    - If rmask is subset of deduced pmask: use rmask
                    - If rmask is not subset of deduced pmask: raise error
               2.1.2 If deduced pmask is None (trace time unknown): force use rmask
        3. If rmask is not provided (caller lets expr deduce it): use deduced pmask from args
        """
        # Deduce pmask from arguments (including None values - if any arg has None, result is None)
        arg_pmasks = [arg.mptype.pmask for arg in self.args]
        deduced_pmask = deduce_mask(*arg_pmasks)

        # Determine effective output pmask
        effective_pmask: Mask | None
        if self.rmask is not None:
            # rmask is explicitly provided - caller has strong mask knowledge
            if deduced_pmask is not None:
                # pmask is known at trace time - validate subset relationship
                if not Mask(self.rmask).is_subset(deduced_pmask):
                    raise ValueError(
                        f"Specified rmask {self.rmask} is not a subset of deduced pmask {deduced_pmask}."
                    )
                effective_pmask = self.rmask
            else:
                # pmask is unknown at trace time - force use rmask
                effective_pmask = self.rmask
        else:
            # rmask not provided - use deduced pmask from args
            effective_pmask = deduced_pmask

        # Create result MPTypes based on PFunction output info
        result_types = []
        for out_info in self.pfunc.outs_info:
            if isinstance(out_info, TensorType):
                # Tensor type
                result_types.append(
                    MPType.tensor(out_info.dtype, out_info.shape, effective_pmask)
                )
            elif isinstance(out_info, TableType):
                # Table type
                result_types.append(MPType.table(out_info, effective_pmask))
            else:
                raise TypeError(f"Unsupported output type: {type(out_info)}")
        return result_types

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_eval(self)


class TupleExpr(Expr):
    """Expression for creating a tuple from multiple single-output expressions.

    In a Multi-Input Multi-Output (MIMO) expression system, this primitive
    creates a logical tuple from multiple single-output expressions. Unlike
    the previous FlattenExpr, TupleExpr requires all input expressions to
    have exactly one output each.

    This expression acts as a "tuple construction" primitive. It takes a list
    of single-output expressions and produces a new logical expression whose
    outputs are the list of all input expression outputs.

    For example, if expr1 has output [A] and expr2 has output [B],
    TupleExpr([expr1, expr2]) will have outputs [A, B].

    This is the opposite of AccessExpr, which extracts a single element
    from a multi-output expression.
    """

    def __init__(self, args: list[Expr]):
        super().__init__()
        # Validate that all arguments are single-output expressions
        for i, arg in enumerate(args):
            if arg.num_outputs != 1:
                raise ValueError(
                    f"TupleExpr requires all arguments to be single-output expressions, "
                    f"but argument {i} has {arg.num_outputs} outputs"
                )
        self.args = args

    def _compute_mptypes(self) -> list[MPType]:
        # TupleExpr creates a tuple from single-output expressions
        result_types = []
        for arg in self.args:
            result_types.append(
                arg.mptype
            )  # Use mptype since we validated single output
        return result_types

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_tuple(self)


class CondExpr(Expr):
    """Expression for conditional execution.

    Added fields:
        verify_uniform: whether runtime should assert the predicate is uniform across parties.
    """

    def __init__(
        self,
        pred: Expr,
        then_fn: FuncDefExpr,
        else_fn: FuncDefExpr,
        args: list[Expr],
        verify_uniform: bool = False,
    ):
        super().__init__()
        self.pred = pred
        self.then_fn = then_fn
        self.else_fn = else_fn
        self.args = args
        self.verify_uniform = verify_uniform

    def _compute_mptypes(self) -> list[MPType]:
        for t_type, e_type in zip(
            self.then_fn.mptypes, self.else_fn.mptypes, strict=False
        ):
            if t_type != e_type:
                raise TypeError(
                    f"Then branch type {t_type} does not match else branch type {e_type}"
                )
        return self.then_fn.mptypes

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_cond(self)


class WhileExpr(Expr):
    """Expression for while loop."""

    def __init__(
        self,
        cond_fn: FuncDefExpr,
        body_fn: FuncDefExpr,
        args: list[Expr],
    ):
        super().__init__()
        if not args:
            raise ValueError("WhileExpr requires at least one argument (init value)")
        self.cond_fn = cond_fn
        self.body_fn = body_fn
        self.args = args

    def _compute_mptypes(self) -> list[MPType]:
        # The result types of a while loop are the same as the body function's outputs.
        # This supports multi-value loop-carried state (PyTree leaves) and ensures
        # evaluator can determine how many values are produced by the loop.
        return self.body_fn.mptypes

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_while(self)


class ConvExpr(Expr):
    """Expression for convergence of multiple variables."""

    def __init__(self, vars: list[Expr]):
        super().__init__()

        # Validate all vars have identical out-length.
        for v in vars:
            if v.num_outputs != 1:
                raise ValueError("All variables in ConvExpr must have the same arity.")

        self.vars = vars

    def _compute_mptypes(self) -> list[MPType]:
        # Collect the idx-th mptype from every var.
        types = [v.mptype for v in self.vars]
        # Validate dtype / shape consistency.
        first = types[0]
        for c in types[1:]:
            if (c.dtype, c.shape) != (first.dtype, first.shape):
                raise TypeError(f"Inconsistent dtype/shape in pconv: {c} vs {first}")

        # Deduce the pmask by intersecting all pmasks.
        pmasks = [t.pmask for t in types]
        dynamic_pmask = False
        if any(pmask is None for pmask in pmasks):
            logging.warning("pconv called with None pmask.")
            dynamic_pmask = True

        non_none_pmasks = [pmask for pmask in pmasks if pmask is not None]
        for i, mask1 in enumerate(non_none_pmasks):
            for mask2 in non_none_pmasks[i + 1 :]:
                if not Mask(mask1).is_disjoint(mask2):
                    raise ValueError(
                        f"pconv called with non-disjoint pmasks: {pmasks}."
                    )

        # deduce output pmask.
        if dynamic_pmask:
            out_pmask = None
        else:
            valid_pmasks = [pmask for pmask in pmasks if pmask is not None]
            if valid_pmasks:
                out_pmask = Mask(valid_pmasks[0])
                for mask in valid_pmasks[1:]:
                    out_pmask = out_pmask.union(mask)
            else:
                out_pmask = None

        return [MPType.tensor(first.dtype, first.shape, out_pmask, **first.attrs)]

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_conv(self)


class ShflSExpr(Expr):
    """Expression for static shuffle operation.

    Redistributes data from source ranks to target ranks based on a specified
    mapping. Each party in the output mask (`pmask`) receives data from a
    corresponding source rank specified in `src_ranks`.

    Rationale for Design (Pull vs. Push Model):
        This operation uses a "pull" model, where each receiving party explicitly
        states its data source (`src_ranks`). This contrasts with a "push" model,
        where each sending party would specify a destination.

        The pull model is chosen because it guarantees that every party in the
        output `pmask` receives exactly one value, upholding the semantic
        integrity of the computation graph.

        A push model, on the other hand, would be semantically ambiguous. For
        example, two different source parties could attempt to send data to the
        same destination, or some parties might receive no data at all. This
        would break the Single Instruction, Multiple Programs (SIMP) paradigm by
        creating an unpredictable number of outputs at each party.

        While the pull model might have performance implications if multiple
        receivers pull from the same source (potentially creating a network
        bottleneck at that source), this is a performance consideration rather
        than a correctness issue. The chosen design prioritizes semantic
        predictability and correctness.
    """

    def __init__(self, src_val: Expr, pmask: Mask, src_ranks: list[Rank]):
        """Initialize static shuffle expression.

        Args:
            src_val (Expr): The input tensor to be shuffled.
            pmask (Mask): The mask indicating which parties will hold the output.
                         Only parties with non-zero bits in pmask will receive output.
            src_ranks (list[Rank]): List of source ranks. The i-th output party
                                   (i-th non-zero bit in pmask) receives data from
                                   src_ranks[i].

        Raises:
            ValueError: If src_val has multiple outputs, if src_ranks length doesn't
                       match pmask bit count, or if any rank in src_ranks is not
                       present in src_val.pmask.

        Example:
            If pmask indicates parties [0, 2] should receive output and src_ranks = [1, 3], then:
            - Party 0 receives data from rank 1
            - Party 2 receives data from rank 3
        """
        super().__init__()
        if src_val.num_outputs != 1:
            raise ValueError(
                f"ShflSExpr requires a single output source, got {src_val.num_outputs}"
            )

        # Assign values first before validation
        self.src_val = src_val
        self.pmask = pmask
        self.src_ranks = src_ranks

        # Now do validation using the assigned values
        if len(self.src_ranks) != Mask(self.pmask).num_parties():
            raise ValueError(
                f"src_ranks length ({len(self.src_ranks)}) not match {self.pmask}"
            )
        for i, rank in enumerate(self.src_ranks):
            src_pmask = self.src_val.mptype.pmask
            if src_pmask is not None and rank not in Mask(src_pmask):
                raise ValueError(
                    f"Source rank {rank} at index {i} is not present in src {Mask(src_pmask)}"
                )

    def _compute_mptypes(self) -> list[MPType]:
        # The types are the same as the source value, but with a new pmask.
        src_type = self.src_val.mptype
        return [
            MPType.tensor(src_type.dtype, src_type.shape, self.pmask, **src_type.attrs)
        ]

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_shfl_s(self)


class ShflExpr(Expr):
    """Expression for dynamic shuffle operation."""

    def __init__(self, src: Expr, index: Expr):
        super().__init__()
        self.src = src
        self.index = index

    def _compute_mptypes(self) -> list[MPType]:
        # Dynamic shuffle is complex. The resulting pmask is often unknown
        # at compile time. We'll assume the tensor types remain the same
        # but the pmask becomes None (runtime-determined).
        src_types = self.src.mptypes
        result_types = []
        for src_type in src_types:
            result_types.append(
                MPType.tensor(src_type.dtype, src_type.shape, None, **src_type.attrs)
            )
        return result_types

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_shfl(self)


class AccessExpr(Expr):
    """Expression for accessing a specific output of a multi-output expression.

    As the counterpart to TupleExpr, AccessExpr is the "un-packing" or "selection"
    primitive in the MIMO system. It takes a (potentially multi-output) expression
    and an index, and produces a new single-output expression representing just
    the selected output.

    This is essential for routing specific outputs from a multi-output function
    or a flattened stream to subsequent operations that expect single inputs.
    """

    def __init__(self, src: Expr, index: int):
        super().__init__()
        self.src = src
        self.index = index

    def _compute_mptypes(self) -> list[MPType]:
        # Access a specific output from the expression's output list
        expr_types = self.src.mptypes
        if self.index < 0 or self.index >= len(expr_types):
            raise IndexError(
                f"Index {self.index} out of range for expression with {len(expr_types)} outputs"
            )
        return [expr_types[self.index]]

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_access(self)


class VariableExpr(Expr):
    """Expression for variable reference/lookup."""

    def __init__(self, name: str, mptype: MPType):
        super().__init__()
        self.name = name
        self.mptype_value = mptype

    def _compute_mptypes(self) -> list[MPType]:
        # Return the explicitly provided type for this variable.
        return [self.mptype_value]

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_variable(self)


class FuncDefExpr(Expr):
    """Expression representing a function definition with parameters and body.

    This class captures the essence of lambda abstraction in functional programming.
    The body expression tree may contain free variables (VariableExpr nodes) that
    reference parameter names. When the function is called, arguments are bound
    to parameters positionally, resolving these free variables.

    Example:
        Consider a function that adds two variables:
        ```
        # Body expression tree contains free variables "x" and "y"
        body = EvalExpr(
            add_pfunc, [VariableExpr("x", int_type), VariableExpr("y", int_type)]
        )

        # Parameters define the binding order - note "y" comes before "x"
        params = ["z", "y", "x"]  # extra parameter "z", different order

        func_def = FuncDefExpr(params, body)

        # When called with [expr0, expr1, expr2]:
        # - "z" binds to expr0 (unused in body, but valid)
        # - "y" binds to expr1 (resolves VariableExpr("y") in body)
        # - "x" binds to expr2 (resolves VariableExpr("x") in body)
        call = CallExpr(func_def, [expr0, expr1, expr2])
        ```

    Key insights:
    - Free variables in the body are placeholders waiting for concrete expressions
    - Parameters act as a "binding contract" - they define which arguments map to which variables
    - Parameter order matters for positional binding, not alphabetical or usage order
    - Parameters can include names not used in the body (dead parameters)
    - All free variables in the body should have corresponding parameters for well-formed functions
    """

    def __init__(self, params: list[str], body: Expr):
        super().__init__()
        self.params = params
        self.body = body

    def _compute_mptypes(self) -> list[MPType]:
        # The types of a function are the types of its body.
        return self.body.mptypes

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_func_def(self)


class CallExpr(Expr):
    """Expression for function call."""

    def __init__(self, fn: FuncDefExpr, args: list[Expr]):
        super().__init__()
        self.fn = fn
        self.args = args

    def _compute_mptypes(self) -> list[MPType]:
        # The result types are the types of the function's body, with parameter
        # types substituted. For simplicity, we return the function's declared
        # return types. A full implementation would require substitution logic.
        return self.fn.mptypes

    def accept(self, visitor: ExprVisitor) -> Any:
        return visitor.visit_call(self)
