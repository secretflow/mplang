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
Trace context and TraceVar implementation.

This module provides the trace context for lazy evaluation and TraceVar
which stores expressions for deferred computation.

Design Philosophy (inspired by JAX):
====================================
The tracing mechanism converts Python functions operating on data into a static,
dataflow graph representation (Expr) for analysis and multi-party execution.
This follows a "closed-world" design, similar to JAX's JIT, with a core
principle: functions are for data transformation ("Tensor in, Tensor out").

This imposes several intentional limitations:
- **Data-Centric Boundaries**: Only MPObjects (tensors or their pytrees) and
  immediate values can be passed as arguments to or be returned from a traced
  function.
- **No Function Outputs**: A traced function cannot return a Python function that
  has captured tracers, as this would violate the static nature of the graph.
- **Limited Function Inputs**: Arbitrary Python functions are not supported as
  arguments. However, for structured control flow (e.g., `cond`, `while_loop`),
  `mplang` allows passing Python functions. These are not true first-class
  functions; they are immediately traced into sub-graphs (`FuncDefExpr`) and
  embedded into the IR, never existing as runtime values within the graph.

Rationale for TracedFunction vs. First-Class Functions:
-------------------------------------------------------
Instead of representing functions as `TraceVar(expr=FuncDefExpr)`, a dedicated
`TracedFunction` class is used. This is crucial for:

1.  **Type Safety & Clear Boundaries**: `TracedFunction` represents a callable
    computation, while `TraceVar` represents data. This separation prevents
    treating computation as data within the graph.
2.  **Preserving Metadata**: It holds essential metadata for marshalling arguments
    and results, such as pytree structures (`in_struct`/`out_struct`) and
    captured variables, which a simple `Expr` would not retain.

This design avoids the complexities of dynamic dispatch and higher-order functions
in the IR, making the resulting graph simpler, more analyzable, and easier to
compile for a multi-party setting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from mplang.core.cluster import ClusterSpec
from mplang.core.context_mgr import with_ctx
from mplang.core.expr.ast import Expr, FuncDefExpr, TupleExpr, VariableExpr
from mplang.core.expr.printer import Printer
from mplang.core.mask import Mask
from mplang.core.mpobject import MPContext, MPObject
from mplang.core.mptype import MPType
from mplang.core.pfunc import get_fn_name
from mplang.utils.func_utils import MorphStruct, var_demorph, var_morph


class VarNamer:
    """Utility class to generate unique variable names in a trace context."""

    def __init__(self, prefix: str = "$"):
        self._counter = 0
        self._prefix = prefix

    def next_name(self) -> str:
        """Generate a new unique variable name."""
        name = f"{self._prefix}{self._counter}"
        self._counter += 1
        return name


class TraceContext(MPContext):
    """Context for lazy evaluation using expressions.

    TraceContext builds computation graphs by creating TraceVar objects
    that store expressions instead of executing them immediately.
    """

    def __init__(
        self,
        cluster_spec: ClusterSpec,
        *,
        mask: Mask | None = None,
        capture_namer: VarNamer | None = None,
        parent: MPContext | None = None,
    ):
        """Initialize TraceContext with a cluster specification.

        Args:
            cluster_spec: The cluster specification defining the physical nodes
                          and logical devices available for computation.
            mask: The default mask for this context. If None, defaults to all parties.
            capture_namer: Optional VarNamer for naming captured variables.
        """
        super().__init__(cluster_spec, parent=parent)

        self._mask = mask or Mask.all(self.world_size())
        self._capture_namer = capture_namer or VarNamer()

        self._var_namer = VarNamer(prefix="%")
        self._captures: dict[MPObject, TraceVar] = {}

    @property
    def mask(self) -> Mask:
        """The default mask for this context."""
        return self._mask

    def _gen_name(self) -> str:
        """Generate a unique variable name."""
        return self._capture_namer.next_name()

    def fork(self, mask: Mask | None = None) -> TraceContext:
        """Create a new TraceContext with the same attributes."""
        if mask is None:
            mask = self._mask
        else:
            # ensure mask is subset of the current mask
            if not Mask(mask).is_subset(self._mask):
                raise ValueError(
                    f"New mask {mask} must be a subset of the current mask {self._mask}"
                )

        return TraceContext(
            cluster_spec=self.cluster_spec,
            mask=mask,
            parent=self._parent,
            # capture_namer=self._capture_namer,
        )

    def capture(self, obj: MPObject) -> TraceVar:
        """Create or reuse a variable that represents a captured MPObject.

        This method ensures that the same captured object always maps to
        the same variable in the traced function.

        Args:
            obj: The MPObject being captured from another context

        Returns:
            TraceVar representing the captured variable in this context
        """
        # If we've seen this object before, return the existing variable
        if obj in self._captures:
            return self._captures[obj]

        # Use the object's name directly if available, otherwise generate a name
        capture_name = self._gen_name()
        var = TraceVar(self, VariableExpr(capture_name, obj.mptype))
        self._captures[obj] = var

        return var

    def get_captures(self) -> dict[MPObject, TraceVar]:
        return self._captures


class TraceVar(MPObject):
    """A variable that stores an expression for lazy evaluation.

    TraceVar represents a computation that has not yet been executed.
    It stores the expression tree that would produce the value when evaluated.
    The expression must be single-output (checked at construction time).
    """

    def __init__(self, ctx: TraceContext, expr: Expr):
        # Ensure the expression is single-output
        if len(expr.mptypes) != 1:
            raise ValueError(
                f"TraceVar requires single-output expression, "
                f"but expression has {len(expr.mptypes)} outputs"
            )

        self._ctx = ctx
        self._expr = expr

    @property
    def ctx(self) -> MPContext:
        """The context this variable belongs to."""
        return self._ctx

    @property
    def expr(self) -> Expr:
        """The expression that this variable represents."""
        return self._expr

    @property
    def mptype(self) -> MPType:
        """The type of this variable, derived from the expression."""
        return self._expr.mptype

    def __repr__(self) -> str:
        return f"TraceVar(expr={self.expr.__class__.__name__})"


@dataclass
class TracedFunction:
    func_name: str
    """The name of the traced function."""

    in_vars: list[TraceVar]
    """List of free (input) variables in the traced function."""
    in_struct: MorphStruct
    in_imms: list[Any]

    capture_map: dict[MPObject, TraceVar]
    """Map of captured MPObjects to their traced values."""

    out_vars: list[TraceVar]
    """List of output TraceVars."""
    out_struct: MorphStruct
    out_imms: list[Any]

    def in_names(self) -> list[str]:
        """Get the parameter names of the traced function."""
        return [cast(VariableExpr, var.expr).name for var in self.in_vars]

    def capture_names(self, captures: list[MPObject] | None = None) -> list[str]:
        if captures is None:
            captures = list(self.capture_map.keys())

        def var_name(var: TraceVar | None) -> str:
            return cast(VariableExpr, var.expr).name if var is not None else ""

        return [var_name(self.capture_map.get(var, None)) for var in captures]

    def make_expr(self, freevar_names: list[str] | None = None) -> FuncDefExpr:
        """Create a FuncDefExpr from the traced function data."""
        arg_names = [cast(VariableExpr, var.expr).name for var in self.in_vars]
        capture_names = [
            cast(VariableExpr, var.expr).name for var in self.capture_map.values()
        ]
        if freevar_names is None:
            # If no freevar_names provided, use default names
            freevar_names = arg_names + capture_names
        else:
            # Ensure freevar_names is superset of arg_names and capture_names
            if not set(arg_names).issubset(freevar_names):
                raise ValueError(
                    f"Provided freevar_names {freevar_names} must include all input variable names {arg_names}"
                )
            if not set(capture_names).issubset(freevar_names):
                raise ValueError(
                    f"Provided freevar_names {freevar_names} must include all capture variable names {capture_names}"
                )

        if len(self.out_vars) == 0:
            # No outputs - use empty tuple
            body_expr: Expr = TupleExpr([])
            return FuncDefExpr(freevar_names, body_expr)
        elif len(self.out_vars) == 1:
            body_expr = self.out_vars[0].expr
            return FuncDefExpr(freevar_names, body_expr)
        else:
            # Multiple outputs - use tuple (ensures all vars are single-output)
            body_expr = TupleExpr([var.expr for var in self.out_vars])
            return FuncDefExpr(freevar_names, body_expr)

    def is_signature_match(
        self,
        other: TracedFunction,
        check_captures: bool = True,
    ) -> bool:
        """Check if this function's signature matches another."""
        if not isinstance(other, TracedFunction):
            return False
        # Check input structures and immutables
        if (
            self.in_struct != other.in_struct
            or self.in_imms != other.in_imms
            or self.out_struct != other.out_struct
            or self.out_imms != other.out_imms
        ):
            return False

        # Check input type match
        if len(self.in_vars) != len(other.in_vars):
            return False
        for var, other_var in zip(self.in_vars, other.in_vars, strict=False):
            if var.mptype != other_var.mptype:
                return False

        # Check captures if required
        if check_captures:
            if len(self.capture_map) != len(other.capture_map):
                return False
            for key, var in self.capture_map.items():
                if (
                    key not in other.capture_map
                    or var.mptype != other.capture_map[key].mptype
                ):
                    return False

        # check output type match
        if len(self.out_vars) != len(other.out_vars):
            return False
        for var, other_var in zip(self.out_vars, other.out_vars, strict=False):
            if var.mptype != other_var.mptype:
                return False

        return True

    def compiler_ir(self, verbose_peval: bool = False) -> str:
        """Get the compiler IR representation of this traced function."""
        printer = Printer(verbose_peval=verbose_peval)
        func_expr = self.make_expr()
        return printer.print_expr(func_expr)


def trace(
    tracer: TraceContext,
    mpfn: Callable,
    *args: Any,
    **kwargs: Any,
) -> TracedFunction:
    """Trace a Python function into an expression representation.

    This converts a Python function into a FuncDefExpr that can be executed
    in multi-party computation contexts. It handles:
    - Function arguments (including pytree structures)
    - Captured variables from outer scopes
    - Output structures

    Args:
        tracer: The tracing context
        fn: The Python function to trace
        *args, **kwargs: Arguments to the function

    Returns:
        A TracedFunction containing a FuncDefExpr representing the function
    """
    assert isinstance(tracer, TraceContext), f"Expect TraceContext, got {tracer}"

    # Separate MPObjects from immediate values in inputs
    is_mpobj = lambda x: isinstance(x, MPObject)
    in_params, in_imms, in_struct = var_morph((args, kwargs), is_mpobj)

    param_names = [tracer._gen_name() for _ in range(len(in_params))]
    in_vars = [
        TraceVar(tracer, VariableExpr(name, var.mptype))
        for name, var in zip(param_names, in_params, strict=False)
    ]

    with with_ctx(tracer):
        # Prepare formal parameters for the function
        vargs, vkwargs = var_demorph(in_vars, in_imms, in_struct)

        # Execute the function - this will capture any external variables through switch_ctx
        outs = mpfn(*vargs, **vkwargs)

        # Extract output MPObjects and structure
        out_vars, out_imms, out_struct = var_morph(outs, is_mpobj)
        # Each MPObject represents a single tensor, so this assertion is redundant
        # assert all(len(out.mptypes) == 1 for out in out_vars), out_vars

    captures = tracer.get_captures()

    # Return TracedFunction with all the necessary information
    return TracedFunction(
        func_name=get_fn_name(mpfn),
        in_vars=in_vars,
        in_struct=in_struct,
        in_imms=in_imms,
        capture_map=captures,
        out_vars=out_vars,
        out_struct=out_struct,
        out_imms=out_imms,
    )
