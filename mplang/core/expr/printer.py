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
Expression printer for debugging and visualization.
"""

from __future__ import annotations

from typing import Any

from mplang.core.dtype import DType
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
from mplang.core.expr.visitor import ExprVisitor
from mplang.core.mptype import MPType
from mplang.core.pfunc import PFunction
from mplang.core.tensor import Shape, TensorType


class Printer(ExprVisitor):
    """Printer that prints Expression DAG in IR style."""

    def __init__(
        self,
        indent_size: int = 2,
        compact_format: bool = True,
        *,
        verbose_peval: bool = False,
    ):
        super().__init__()  # Initialize MemorizedVisitor
        self.indent_size = indent_size
        self.compact_format = compact_format
        self.verbose_peval = verbose_peval
        self._cur_indent = 0
        self._output: list[str] = []
        self._visited: dict[Expr, str] = {}
        self._counter = 0

    def _write(self, text: str) -> None:
        """Write a line with current indentation."""
        indent = " " * (self._cur_indent * self.indent_size)
        for line in text.split("\n"):
            self._output.append(f"{indent}{line}")

    def _do_print(
        self,
        op_name: str,
        op_args: list[str],
        attrs: dict | None = None,
        regions: dict[str, FuncDefExpr] | None = None,
        mptypes: list | None = None,
    ) -> str:
        """A generic node printer that prints in the MLIR style."""
        ret_name = f"%{self._counter}"
        self._counter += 1

        args_str = f"({', '.join(op_args)})"
        attrs_str = ""
        if attrs:
            attr_parts = [f"{k}={v}" for k, v in attrs.items()]
            attrs_str = f" {{{', '.join(attr_parts)}}}"

        regions_str = ""
        if regions:
            regions_str = " {\n"
            indent = " " * self.indent_size
            for r_name, func_def_expr in regions.items():
                body_printer = Printer(
                    indent_size=self.indent_size,
                    compact_format=self.compact_format,
                )
                func_def_expr.accept(body_printer)
                regions_str += f"{indent}{r_name}: "
                body_content = ("\n" + indent).join(body_printer._output)
                regions_str += f"{body_content}\n"
            regions_str += "}"

        type_str = ""
        if mptypes:
            type_parts = [str(mptype) for mptype in mptypes]
            if len(type_parts) == 1:
                type_str = f" : {type_parts[0]}"
            else:
                type_str = f" : ({', '.join(type_parts)})"

        self._write(
            f"{ret_name} = {op_name}{args_str}{attrs_str}{regions_str}{type_str}"
        )
        return ret_name

    def _var_name(self, expr: Expr) -> str:
        key = expr
        if key not in self._visited:
            self._visited[key] = expr.accept(self)
        return self._visited[key]

    def print_expr(self, expr: Expr) -> str:
        """Print an expression and return the formatted string."""
        self._output = []
        self._visited = {}
        self._cache: dict[str, Any] = {}  # Reset memorized visitor cache
        self._counter = 0
        expr.accept(self)
        return "\n".join(self._output)

    def _get_const_data(self, dtype: DType, shape: Shape, data_bytes: bytes) -> str:
        # Get dtype and shape from output info (following party.py implementation)
        import numpy as np

        np_array = np.frombuffer(data_bytes, dtype=dtype.to_numpy()).reshape(shape)

        # Format the display based on array size
        if np_array.size <= 10:
            # Small arrays - show full content
            if np_array.size == 1:
                # Scalar value
                value_str = str(np_array.item())
            else:
                value_str = str(np_array.tolist())
        else:
            # Large arrays - use numpy's default string representation which handles truncation
            value_str = str(np_array)
        return value_str

    def _print_const(self, pfunc: PFunction, mptypes: list[MPType]) -> str:
        assert len(pfunc.outs_info) == 1
        out_type = pfunc.outs_info[0]
        assert isinstance(out_type, TensorType)
        attrs = {
            "data": self._get_const_data(
                out_type.dtype, out_type.shape, pfunc.attrs["data_bytes"]
            )
        }
        return self._do_print("pconst", [], attrs=attrs, mptypes=mptypes)

    def visit_eval(self, expr: EvalExpr) -> str:
        arg_names = [self._var_name(arg) for arg in expr.args]
        fn_type = expr.pfunc.fn_type

        # for well known builtin functions
        if fn_type == "builtin.constant":
            return self._print_const(expr.pfunc, expr.mptypes)
        elif fn_type == "builtin.rank":
            return self._do_print("prank", [], mptypes=expr.mptypes)
        elif fn_type == "builtin.prand":
            return self._do_print("prand", [], mptypes=expr.mptypes)

        attrs = {"fn_type": fn_type}
        if expr.pfunc.fn_name:
            attrs["fn_name"] = str(expr.pfunc.fn_name)
        if self.verbose_peval:
            attrs["fn_text"] = str(expr.pfunc.fn_text)

        if expr.rmask is not None:
            attrs["rmask"] = f"0x{expr.rmask.value:x}"
        return self._do_print("peval", arg_names, attrs=attrs, mptypes=expr.mptypes)

    def visit_variable(self, expr: VariableExpr) -> str:
        if self.compact_format:
            # Use $param format and don't print the variable definition
            return f"{expr.name}"
        else:
            return self._do_print(
                "pname", [f'"{expr.name}"'], attrs={}, mptypes=expr.mptypes
            )

    def visit_tuple(self, expr: TupleExpr) -> str:
        arg_names = [self._var_name(arg) for arg in expr.args]
        return self._do_print("tuple", arg_names, mptypes=expr.mptypes)

    def visit_cond(self, expr: CondExpr) -> str:
        pred_name = self._var_name(expr.pred)
        arg_names = [self._var_name(arg) for arg in expr.args]

        # Directly pass FuncDefExpr objects
        return self._do_print(
            "pcond",
            [pred_name, *arg_names],
            regions={
                "then_fn": expr.then_fn,
                "else_fn": expr.else_fn,
            },
            mptypes=expr.mptypes,
        )

    def visit_call(self, expr: CallExpr) -> str:
        arg_names = [self._var_name(arg) for arg in expr.args]
        return self._do_print(
            "pcall",
            arg_names,
            regions={"fn": expr.fn},
            mptypes=expr.mptypes,
        )

    def visit_while(self, expr: WhileExpr) -> str:
        arg_names = [self._var_name(arg) for arg in expr.args]

        return self._do_print(
            "pwhile",
            arg_names,
            regions={
                "cond_fn": expr.cond_fn,
                "body_fn": expr.body_fn,
            },
            mptypes=expr.mptypes,
        )

    def visit_conv(self, expr: ConvExpr) -> str:
        var_names = [self._var_name(var) for var in expr.vars]
        return self._do_print("pconv", var_names, mptypes=expr.mptypes)

    def visit_shfl_s(self, expr: ShflSExpr) -> str:
        src_val_name = self._var_name(expr.src_val)
        attrs = {"pmask": expr.pmask, "src_ranks": expr.src_ranks}
        return self._do_print(
            "pshfl_s", [src_val_name], attrs=attrs, mptypes=expr.mptypes
        )

    def visit_shfl(self, expr: ShflExpr) -> str:
        src_name = self._var_name(expr.src)
        index_name = self._var_name(expr.index)
        return self._do_print("pshfl", [src_name, index_name], mptypes=expr.mptypes)

    def visit_access(self, expr: AccessExpr) -> str:
        expr_name = self._var_name(expr.src)
        if self.compact_format:
            # Original:
            #   %x = ...
            #   %y = %x[0]
            #   %z = some_fn(%y)
            # Single output(optimized):
            #   %x = ...
            #   %z = some_fn(%x)
            # Multiple outputs (optimized):
            #   %x = ...
            #   %z = some_fn(%x:0, %x:1)
            if len(expr.src.mptypes) > 1:
                return f"{expr_name}:{expr.index}"
            else:
                return expr_name
        else:
            attrs = {"index": str(expr.index)}
            return self._do_print(
                "access", [expr_name], attrs=attrs, mptypes=expr.mptypes
            )

    def visit_func_def(self, expr: FuncDefExpr) -> str:
        param_names = expr.params
        self._write(f"({', '.join(param_names)}) {{")
        self._cur_indent += 1
        body_name = expr.body.accept(self)
        self._write(f"return {body_name}")
        self._cur_indent -= 1
        self._write("}")
        return ""
