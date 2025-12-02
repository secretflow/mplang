# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for expression printer module.
"""

import pytest

from mplang.v1.core.dtypes import FLOAT32, UINT64
from mplang.v1.core.expr import (
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
from mplang.v1.core.expr.printer import Printer
from mplang.v1.core.mask import Mask
from mplang.v1.core.mptype import MPType, Rank
from mplang.v1.core.pfunc import PFunction
from mplang.v1.core.tensor import TensorType


class TestPrinter:
    """Test the Printer class for expression visualization."""

    def test_printer_initialization(self):
        """Test basic printer initialization."""
        printer = Printer()
        assert printer.indent_size == 2
        assert printer._cur_indent == 0
        assert printer._output == []
        assert printer._visited == {}
        assert printer._counter == 0

    def test_printer_custom_indent(self):
        """Test printer with custom indent size."""
        printer = Printer(indent_size=4)
        assert printer.indent_size == 4

    def test_write_functionality(self):
        """Test write functionality with indentation."""
        printer = Printer(indent_size=2)

        # Test writing without indentation
        printer._write("test line")
        assert printer._output == ["test line"]

        # Test writing with indentation
        printer._cur_indent = 1
        printer._write("indented line")
        assert printer._output == ["test line", "  indented line"]


class TestPrinterExpressions:
    """Test printer for different expression types."""

    def test_variable_expr_printing(self, pmask_2p):
        """Test printing of VariableExpr."""
        printer = Printer(compact_format=False)
        mptype = MPType.tensor(FLOAT32, (2, 2), pmask_2p)
        expr = VariableExpr("test_var", mptype)

        result = printer.print_expr(expr)

        # Expected output format with exact literal comparison
        expected = '%0 = pname("test_var") : f32[2, 2]<3>'

        assert result == expected

    def test_eval_expr_printing(self, pmask_2p, pfunc_1i1o):
        """Test printing of EvalExpr."""
        printer = Printer()
        mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        arg1 = VariableExpr("input", mptype)
        expr = EvalExpr(pfunc_1i1o, [arg1])

        result = printer.print_expr(expr)
        lines = result.split("\n")

        # Should contain the eval expression (in compact mode, no pname for variables)
        assert any("peval(" in line for line in lines)
        # Should contain fn_name since verbose=false still shows it
        assert any("fn_name=" in line for line in lines)

    def test_eval_expr_with_rmask_printing(self, pmask_2p, pfunc_1i1o):
        """Test printing of EvalExpr with rmask."""
        printer = Printer()
        mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        arg = VariableExpr("input", mptype)
        rmask = Mask(3)  # 0b11
        expr = EvalExpr(pfunc_1i1o, [arg], rmask=rmask)

        result = printer.print_expr(expr)
        assert "rmask=" in result

    def test_eval_expr_verbose_peval_printing(self, pmask_2p):
        """Test printing of EvalExpr with verbose_peval enabled."""

        # Create a PFunction with actual fn_text and fn_type
        pfunc_with_text = PFunction(
            fn_type="jax",
            ins_info=[TensorType(FLOAT32, (2, 3))],
            outs_info=[TensorType(FLOAT32, (2, 3))],
            fn_name="add_one",
            fn_text="lambda x: x + 1",
        )

        # Test without verbose_peval (default)
        printer_normal = Printer()
        mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        arg = VariableExpr("input", mptype)
        expr = EvalExpr(pfunc_with_text, [arg])

        result_normal = printer_normal.print_expr(expr)
        assert "fn_type=jax" in result_normal
        assert "fn_text=" not in result_normal
        assert "fn_name=add_one" in result_normal

        # Test with verbose_peval enabled
        printer_verbose = Printer(verbose_peval=True)
        result_verbose = printer_verbose.print_expr(expr)
        assert "fn_name=add_one" in result_verbose
        assert "fn_text=lambda x: x + 1" in result_verbose
        assert "fn_type=jax" in result_verbose

    def test_tuple_expr_printing(self, pmask_2p):
        """Test printing of TupleExpr."""
        printer = Printer()
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        arg1 = VariableExpr("x", mptype)
        arg2 = VariableExpr("y", mptype)
        expr = TupleExpr([arg1, arg2])

        result = printer.print_expr(expr)
        lines = result.split("\n")

        # Should contain tuple expression (in compact mode, uses variable names)
        assert any("tuple(x, y)" in line for line in lines)

    def test_conv_expr_printing(self):
        """Test printing of ConvExpr."""
        printer = Printer()
        # Use disjoint pmasks: party 0 and party 1
        mptype1 = MPType.tensor(UINT64, (), Mask(1))  # 0b01 - party 0
        mptype2 = MPType.tensor(UINT64, (), Mask(2))  # 0b10 - party 1
        var1 = VariableExpr("p0_var", mptype1)
        var2 = VariableExpr("p1_var", mptype2)
        expr = ConvExpr([var1, var2])

        result = printer.print_expr(expr)
        lines = result.split("\n")

        # Should contain conv expression (in compact mode, uses variable names)
        assert any("pconv(p0_var, p1_var)" in line for line in lines)

    def test_multiple_expressions_same_name(self, pmask_2p):
        """Test that multiple expressions of the same type get different names."""
        printer = Printer(compact_format=False)
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        # Two different expression instances with the same name
        expr1 = VariableExpr("x", mptype)
        expr2 = VariableExpr("x", mptype)

        # Create a tuple expression with both
        tuple_expr = TupleExpr([expr1, expr2])

        result = printer.print_expr(tuple_expr)

        # Expected output format with exact literal comparison
        # The printer should assign different SSA names (%0, %1) to the two different expressions.
        expected = """
%0 = pname("x") : u64<3>
%1 = pname("x") : u64<3>
%2 = tuple(%0, %1) : (u64<3>, u64<3>)
""".strip()

        assert result == expected

    def test_shfl_s_expr_printing(self, pmask_2p):
        """Test printing of ShflSExpr."""
        printer = Printer()
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        src_val = VariableExpr("src", mptype)
        # Use a larger target pmask that requires 3 src_ranks
        target_pmask = Mask(7)  # 0b111 for 3 parties
        # Need 3 src_ranks since target_pmask has 3 bits set
        # Source has pmask_2p (parties 0,1), so src_ranks must be from those parties
        src_ranks = [Rank(0), Rank(1), Rank(0)]  # Party 2 gets data from party 0
        expr = ShflSExpr(src_val, target_pmask, src_ranks)

        result = printer.print_expr(expr)
        lines = result.split("\n")

        # Should contain pshfl_s expression
        assert any("pshfl_s(" in line for line in lines)
        assert any("pmask=" in line for line in lines)
        assert any("src_ranks=" in line for line in lines)

    def test_shfl_expr_printing(self, pmask_2p):
        """Test printing of ShflExpr."""
        printer = Printer()
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        src = VariableExpr("src", mptype)
        index = VariableExpr("idx", mptype)
        expr = ShflExpr(src, index)

        result = printer.print_expr(expr)
        lines = result.split("\n")

        # Should contain pshfl expression
        assert any("pshfl(" in line for line in lines)

    def test_access_expr_printing(self, pmask_2p):
        """Test printing of AccessExpr."""
        printer = Printer(compact_format=False)
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        base_expr = VariableExpr("base", mptype)
        expr = AccessExpr(base_expr, 0)

        result = printer.print_expr(expr)
        lines = result.split("\n")

        # Should contain access expression with index attribute
        assert any("access(" in line for line in lines)
        assert any("index=0" in line for line in lines)

    def test_func_def_expr_printing(self, pmask_2p):
        """Test printing of FuncDefExpr with parameter usage."""
        printer = Printer(compact_format=False)

        # Create a function body that actually uses the parameters
        x_mptype = MPType.tensor(FLOAT32, (1,), pmask_2p)
        y_mptype = MPType.tensor(FLOAT32, (1,), pmask_2p)

        x_var = VariableExpr("x", x_mptype)
        y_var = VariableExpr("y", y_mptype)

        # Tuple the two parameters together
        body = TupleExpr([x_var, y_var])
        func_def = FuncDefExpr(["x", "y"], body)

        result = printer.print_expr(func_def)

        # Expected output format with exact literal comparison
        expected = """
(x, y) {
  %0 = pname("x") : f32[1]<3>
  %1 = pname("y") : f32[1]<3>
  %2 = tuple(%0, %1) : (f32[1]<3>, f32[1]<3>)
  return %2
}
"""

        assert result == expected.strip()


class TestPrinterComplexExpressions:
    """Test printer for complex nested expressions."""

    def test_cond_expr_printing(self, pmask_2p):
        """Test printing of CondExpr with nested functions that actually use their parameters."""
        printer = Printer(compact_format=False)

        # Create predicate
        pred_mptype = MPType.tensor(UINT64, (), pmask_2p)
        pred = VariableExpr("pred", pred_mptype)

        # Create then function that actually uses the parameter 'x'
        x_mptype = MPType.tensor(FLOAT32, (1,), pmask_2p)
        x_var = VariableExpr("x", x_mptype)  # Reference the parameter
        then_fn = FuncDefExpr(["x"], x_var)

        # Create else function that also uses the parameter 'x'
        else_fn = FuncDefExpr(["x"], x_var)

        # Create arguments
        arg_mptype = MPType.tensor(FLOAT32, (1,), pmask_2p)
        arg = VariableExpr("arg", arg_mptype)

        # Create conditional expression
        expr = CondExpr(pred, then_fn, else_fn, [arg])

        result = printer.print_expr(expr)

        # Expected output format with exact literal comparison
        expected = """
%0 = pname("pred") : u64<3>
%1 = pname("arg") : f32[1]<3>
%2 = pcond(%0, %1) {
  then_fn: (x) {
    %0 = pname("x") : f32[1]<3>
    return %0
  }
  else_fn: (x) {
    %0 = pname("x") : f32[1]<3>
    return %0
  }
} : f32[1]<3>
"""

        assert result == expected.strip()

    def test_all_expr_types_printing(self, pmask_2p, pfunc_1i1o):
        """Test printing of a complex expression involving all Expr types with meaningful parameter usage."""
        printer = Printer(compact_format=False, inline_pcall=False)

        # 1. Variable expressions
        var1_mptype = MPType.tensor(FLOAT32, (2,), pmask_2p)
        var1 = VariableExpr("var1", var1_mptype)

        var2_mptype = MPType.tensor(UINT64, (2,), pmask_2p)
        var2 = VariableExpr("var2", var2_mptype)

        var3_mptype = MPType.tensor(UINT64, (), pmask_2p)
        var3 = VariableExpr("var3", var3_mptype)

        # 2. Access expression
        access_expr = AccessExpr(var1, 0)

        # 3. Conv expression (requires disjoint masks)
        var_p0 = VariableExpr("p0", MPType.tensor(UINT64, (), Mask(1)))
        var_p1 = VariableExpr("p1", MPType.tensor(UINT64, (), Mask(2)))
        conv_expr = ConvExpr([var_p0, var_p1])

        # 4. Eval expression
        eval_expr = EvalExpr(pfunc_1i1o, [var1])

        # 5. Shuffle expressions
        shfl_expr = ShflExpr(var3, var3)
        shfl_s_expr = ShflSExpr(
            var3,
            Mask(7),  # target 3 parties
            [Rank(0), Rank(1), Rank(0)],  # src ranks for each target party
        )

        # 6. Function definition and call
        param_type = MPType.tensor(FLOAT32, (2,), pmask_2p)
        var_expr = VariableExpr("input_data", param_type)
        func_body = TupleExpr([var_expr, access_expr])
        func_def = FuncDefExpr(["input_data"], func_body)
        call_expr = CallExpr("test", func_def, [var1])

        # Access the first output of the call expression to get a single-output expr
        call_expr_first = AccessExpr(call_expr, 0)

        # 7. Build final comprehensive expression
        final_expr = TupleExpr([
            var1,  # VariableExpr
            var2,  # VariableExpr
            var3,  # VariableExpr
            access_expr,  # AccessExpr
            conv_expr,  # ConvExpr
            eval_expr,  # EvalExpr
            shfl_expr,  # ShflExpr
            shfl_s_expr,  # ShflSExpr
            call_expr_first,  # AccessExpr of CallExpr (includes FuncDefExpr, VariableExpr, TupleExpr)
        ])

        result = printer.print_expr(final_expr)

        # Expected output with exact format (SSA semantics - no duplicate definitions)
        expected = """
%0 = pname("var1") : f32[2]<3>
%1 = pname("var2") : u64[2]<3>
%2 = pname("var3") : u64<3>
%3 = access(%0) {index=0} : f32[2]<3>
%4 = pname("p0") : u64<1>
%5 = pname("p1") : u64<2>
%6 = pconv(%4, %5) : u64<3>
%7 = peval(%0) {fn_type=mock, fn_name=mock_unary} : f32[2, 3]<3>
%8 = pshfl(%2, %2) : u64
%9 = pshfl_s(%2) {pmask=7, src_ranks=[0, 1, 0]} : u64<7>
%10 = pcall(%0) {
  fn: (input_data) {
    %0 = pname("input_data") : f32[2]<3>
    %1 = pname("var1") : f32[2]<3>
    %2 = access(%1) {index=0} : f32[2]<3>
    %3 = tuple(%0, %2) : (f32[2]<3>, f32[2]<3>)
    return %3
  }
} : (f32[2]<3>, f32[2]<3>)
%11 = access(%10) {index=0} : f32[2]<3>
%12 = tuple(%0, %1, %2, %3, %6, %7, %8, %9, %11) : (f32[2]<3>, u64[2]<3>, u64<3>, f32[2]<3>, u64<3>, f32[2, 3]<3>, u64, u64<7>, f32[2]<3>)
"""

        assert result == expected.strip()

        # Test with optimize_variables=True for comparison
        printer_optimized = Printer(compact_format=True, inline_pcall=False)
        result_optimized = printer_optimized.print_expr(final_expr)

        # Expected output with variable optimization (compact mode uses variable names directly)
        expected_optimized = """
%0 = pconv(p0, p1) : u64<3>
%1 = peval(var1) {fn_type=mock, fn_name=mock_unary} : f32[2, 3]<3>
%2 = pshfl(var3, var3) : u64
%3 = pshfl_s(var3) {pmask=7, src_ranks=[0, 1, 0]} : u64<7>
%4 = pcall(var1) {
  fn: (input_data) {
    %0 = tuple(input_data, var1) : (f32[2]<3>, f32[2]<3>)
    return %0
  }
} : (f32[2]<3>, f32[2]<3>)
%5 = tuple(var1, var2, var3, var1, %0, %1, %2, %3, %4:0) : (f32[2]<3>, u64[2]<3>, u64<3>, f32[2]<3>, u64<3>, f32[2, 3]<3>, u64, u64<7>, f32[2]<3>)
"""

        assert result_optimized == expected_optimized.strip()

    def test_while_expr_printing(self, pmask_2p):
        """Test printing of WhileExpr with nested functions that actually use their parameters."""
        printer = Printer(compact_format=False)

        # Create condition function that uses the 'state' parameter
        state_mptype = MPType.tensor(FLOAT32, (1,), pmask_2p)
        state_var = VariableExpr("state", state_mptype)
        # Access the first element of state to check if we should continue
        cond_body = AccessExpr(state_var, 0)
        cond_fn = FuncDefExpr(["state"], cond_body)

        # Create body function that also uses the 'state' parameter
        # Return state itself (identity transformation)
        body_fn = FuncDefExpr(["state"], state_var)

        # Create initial value
        init_mptype = MPType.tensor(UINT64, (), pmask_2p)
        init = VariableExpr("init", init_mptype)

        # Create while expression
        expr = WhileExpr(cond_fn, body_fn, [init])

        result = printer.print_expr(expr)

        # Expected output format with exact literal comparison
        expected = """
%0 = pname("init") : u64<3>
%1 = pwhile(%0) {
  cond_fn: (state) {
    %0 = pname("state") : f32[1]<3>
    %1 = access(%0) {index=0} : f32[1]<3>
    return %1
  }
  body_fn: (state) {
    %0 = pname("state") : f32[1]<3>
    return %0
  }
} : f32[1]<3>
"""

        assert result == expected.strip()

    def test_call_expr_printing(self, pmask_2p):
        """Test printing of CallExpr with nested function that uses its parameters."""
        printer = Printer(compact_format=False, inline_pcall=False)

        # Create function body that actually uses the parameters
        x_mptype = MPType.tensor(FLOAT32, (1,), pmask_2p)
        y_mptype = MPType.tensor(FLOAT32, (1,), pmask_2p)

        x_var = VariableExpr("x", x_mptype)
        y_var = VariableExpr("y", y_mptype)

        # Create a function that tuples its arguments together
        fn_body = TupleExpr([x_var, y_var])
        fn = FuncDefExpr(["x", "y"], fn_body)

        # Create arguments
        arg1_mptype = MPType.tensor(UINT64, (), pmask_2p)
        arg2_mptype = MPType.tensor(UINT64, (), pmask_2p)
        arg1 = VariableExpr("arg1", arg1_mptype)
        arg2 = VariableExpr("arg2", arg2_mptype)

        # Create call expression
        expr = CallExpr("test", fn, [arg1, arg2])

        result = printer.print_expr(expr)

        # Expected output format with exact literal comparison
        expected = """
%0 = pname("arg1") : u64<3>
%1 = pname("arg2") : u64<3>
%2 = pcall(%0, %1) {
  fn: (x, y) {
    %0 = pname("x") : f32[1]<3>
    %1 = pname("y") : f32[1]<3>
    %2 = tuple(%0, %1) : (f32[1]<3>, f32[1]<3>)
    return %2
  }
} : (f32[1]<3>, f32[1]<3>)
"""

        assert result == expected.strip()

    def test_nested_expressions_printing(self, pmask_2p):
        """Test printing of deeply nested expressions."""
        printer = Printer(compact_format=False)

        # Create a complex nested structure
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        inner = VariableExpr("inner", mptype)
        access1 = AccessExpr(inner, 0)
        access2 = AccessExpr(access1, 0)

        result = printer.print_expr(access2)

        # Expected output format with exact literal comparison
        expected = """
%0 = pname("inner") : u64<3>
%1 = access(%0) {index=0} : u64<3>
%2 = access(%1) {index=0} : u64<3>
"""

        assert result == expected.strip()


class TestPrinterIndentation:
    """Test printer indentation behavior."""

    def test_indentation_reset_between_prints(self, pmask_2p):
        """Test that indentation is reset between print calls."""
        printer = Printer(compact_format=False)  # Use non-compact to see pname output

        # Create a complex expression that modifies indentation
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        body = VariableExpr("x", mptype)
        func_def = FuncDefExpr(["x"], body)

        # Print first expression
        printer.print_expr(func_def)

        # Print second expression - indentation should be reset
        simple_expr = VariableExpr("y", mptype)
        result2 = printer.print_expr(simple_expr)

        # Second result should not have indentation from first
        assert result2.strip() == '%0 = pname("y") : u64<3>'

    def test_counter_reset_between_prints(self, pmask_2p):
        """Test that expression counter is reset between print calls."""
        printer = Printer(compact_format=False)  # Use non-compact to see pname output

        # Print first expression
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        expr1 = VariableExpr("x", mptype)
        result1 = printer.print_expr(expr1)
        assert '%0 = pname("x")' in result1

        # Print second expression - counter should reset
        expr2 = VariableExpr("y", mptype)
        result2 = printer.print_expr(expr2)
        assert '%0 = pname("y")' in result2

    def test_expression_names_reset_between_prints(self, pmask_2p):
        """Test that expression names are reset between print calls."""
        printer = Printer(compact_format=False)  # Use non-compact to see pname output

        # Print first expression
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        expr1 = VariableExpr("x", mptype)
        result1 = printer.print_expr(expr1)
        # After print_expr, _visited is reset so we can't check it
        # But we can verify the output shows %0
        assert '%0 = pname("x")' in result1

        # Print second expression - names should reset
        expr2 = VariableExpr("y", mptype)
        result2 = printer.print_expr(expr2)
        # Both should start with %0 because counter resets
        assert '%0 = pname("y")' in result2


class TestPrinterEdgeCases:
    """Test printer edge cases and error conditions."""

    def test_empty_tuple_expr(self):
        """Test printing of TupleExpr with no arguments."""
        printer = Printer()
        expr = TupleExpr([])

        result = printer.print_expr(expr)
        assert "%0 = tuple()" in result

    def test_printer_output_format(self, pmask_2p):
        """Test that printer output format is consistent."""
        printer = Printer(compact_format=False)  # Use non-compact to see pname output

        # Test multiple expressions to verify format consistency
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        expr1 = VariableExpr("x", mptype)
        result1 = printer.print_expr(expr1)

        # Should start with %0 and be properly formatted
        assert result1.strip() == '%0 = pname("x") : u64<3>'

        # Test that each new print starts fresh
        expr2 = VariableExpr("y", mptype)
        result2 = printer.print_expr(expr2)
        assert result2.strip() == '%0 = pname("y") : u64<3>'

    def test_deep_nesting_indentation(self, pmask_2p):
        """Test printer with deeply nested expressions that use parameters meaningfully."""
        printer = Printer(compact_format=False, inline_pcall=False)

        # Create nested function definitions with meaningful parameter usage
        inner_param_type = MPType.tensor(FLOAT32, (1,), pmask_2p)
        middle_param_type = MPType.tensor(FLOAT32, (1,), pmask_2p)

        # Inner function: takes parameter and accesses its first element
        inner_param = VariableExpr("inner_param", inner_param_type)
        inner_body = AccessExpr(inner_param, 0)
        inner_fn = FuncDefExpr(["inner_param"], inner_body)

        # Middle function: takes parameter and calls inner function with it
        middle_param = VariableExpr("middle_param", middle_param_type)
        middle_body = CallExpr("middle", inner_fn, [middle_param])
        middle_fn = FuncDefExpr(["middle_param"], middle_body)

        # Outer expression: call middle function with a variable expression
        arg_mptype = MPType.tensor(UINT64, (), pmask_2p)
        arg = VariableExpr("arg", arg_mptype)
        outer_expr = CallExpr("outer", middle_fn, [arg])

        result = printer.print_expr(outer_expr)

        # Expected output format with exact literal comparison
        # This shows the deep nesting with proper indentation and meaningful parameter usage
        expected = """
%0 = pname("arg") : u64<3>
%1 = pcall(%0) {
  fn: (middle_param) {
    %0 = pname("middle_param") : f32[1]<3>
    %1 = pcall(%0) {
      fn: (inner_param) {
        %0 = pname("inner_param") : f32[1]<3>
        %1 = access(%0) {index=0} : f32[1]<3>
        return %1
      }
    } : f32[1]<3>
    return %1
  }
} : f32[1]<3>
"""

        assert result == expected.strip()

    def test_function_def_params_handling(self, pmask_2p):
        """Test function definition parameter handling."""
        printer = Printer(compact_format=False)

        # Test with multiple parameters
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        body = VariableExpr("param1", mptype)
        func_def = FuncDefExpr(["param1", "param2", "param3"], body)

        result = printer.print_expr(func_def)
        lines = result.split("\n")

        # Should contain all parameters
        func_line = next(line for line in lines if "(" in line and "param1" in line)
        assert "param1, param2, param3" in func_line

    def test_access_expr_with_different_indices(self, pmask_2p):
        """Test AccessExpr with various index values."""
        printer = Printer(compact_format=False)

        mptype = MPType.tensor(UINT64, (), pmask_2p)
        base_expr = VariableExpr("base", mptype)

        # Test index 0 (the only valid index for single output expressions)
        access_expr = AccessExpr(base_expr, 0)
        result = printer.print_expr(access_expr)
        assert "index=0" in result
        assert "access(" in result

    def test_printer_with_single_expression_multiple_outputs(self, pmask_2p):
        """Test printer behavior with expressions that might have multiple outputs."""
        printer = Printer(compact_format=False)

        # Create a complex expression tree
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        expr1 = VariableExpr("x", mptype)
        expr2 = VariableExpr("y", mptype)
        tuple_expr = TupleExpr([expr1, expr2])
        access_expr = AccessExpr(tuple_expr, 0)

        result = printer.print_expr(access_expr)
        lines = result.split("\n")

        # Should show the dependency chain clearly
        assert len(lines) >= 3
        assert any("pname(" in line for line in lines)
        assert any("tuple(%0, %1)" in line for line in lines)
        assert any("access(" in line and "index=0" in line for line in lines)


class TestPrinterStringMethods:
    """Test printer internal string handling methods."""

    def test_write_with_multiline_content(self):
        """Test _write method behavior."""
        printer = Printer()

        # Test single line
        printer._write("single line")
        assert printer._output == ["single line"]

        # Test with indentation
        printer._cur_indent = 1
        printer._write("indented line")
        assert printer._output == ["single line", "  indented line"]

        # Test multiple writes
        printer._write("another line")
        assert len(printer._output) == 3
        assert printer._output[-1] == "  another line"


class TestPrinterMeaningfulParameterUsage:
    """Test printer with meaningful parameter usage scenarios."""

    def test_meaningful_parameter_usage_scenarios(self, pmask_2p):
        """Test various scenarios where function parameters are meaningfully used."""
        printer = Printer(compact_format=False)

        # Scenario 1: Parameter used in conditional predicate
        param_type = MPType.tensor(FLOAT32, (2,), pmask_2p)
        param_var = VariableExpr("data", param_type)

        # Use parameter in both branches - return the same parameter
        then_body = param_var  # Return parameter directly
        else_body = param_var  # Return parameter directly

        pred_mptype = MPType.tensor(UINT64, (), pmask_2p)
        pred = VariableExpr("pred", pred_mptype)
        then_fn = FuncDefExpr(["data"], then_body)
        else_fn = FuncDefExpr(["data"], else_body)

        arg_mptype = MPType.tensor(FLOAT32, (2,), pmask_2p)
        arg = VariableExpr("arg", arg_mptype)
        cond_expr = CondExpr(pred, then_fn, else_fn, [arg])

        result = printer.print_expr(cond_expr)

        expected = """
%0 = pname("pred") : u64<3>
%1 = pname("arg") : f32[2]<3>
%2 = pcond(%0, %1) {
  then_fn: (data) {
    %0 = pname("data") : f32[2]<3>
    return %0
  }
  else_fn: (data) {
    %0 = pname("data") : f32[2]<3>
    return %0
  }
} : f32[2]<3>
"""

        assert result == expected.strip()

    def test_parameter_reuse_in_complex_expressions(self, pmask_2p):
        """Test parameter being used multiple times in the same function."""
        printer = Printer(compact_format=False)

        # Create a function that uses the same parameter multiple times
        param_type = MPType.tensor(FLOAT32, (1,), pmask_2p)
        x_var = VariableExpr("x", param_type)

        # Use x multiple times: tuple x with itself
        func_body = TupleExpr([x_var, x_var])
        func_def = FuncDefExpr(["x"], func_body)

        result = printer.print_expr(func_def)

        # The same variable expression is now properly cached (SSA semantics)
        expected = """
(x) {
  %0 = pname("x") : f32[1]<3>
  %1 = tuple(%0, %0) : (f32[1]<3>, f32[1]<3>)
  return %1
}
"""

        assert result == expected.strip()

    def test_while_loop_with_meaningful_state_usage(self, pmask_2p):
        """Test while loop where state parameter is meaningfully used."""
        printer = Printer(compact_format=False)

        # Create a while loop that actually processes the state
        state_type = MPType.tensor(FLOAT32, (3,), pmask_2p)
        state_var = VariableExpr("state", state_type)

        # Condition: check if state[0] is non-zero (simplified)
        cond_body = AccessExpr(state_var, 0)
        cond_fn = FuncDefExpr(["state"], cond_body)

        # Body: return the same state (using index 0)
        body_body = AccessExpr(state_var, 0)
        body_fn = FuncDefExpr(["state"], body_body)

        # Initial state
        init_mptype = MPType.tensor(UINT64, (), pmask_2p)
        init = VariableExpr("init", init_mptype)

        while_expr = WhileExpr(cond_fn, body_fn, [init])
        result = printer.print_expr(while_expr)

        expected = """
%0 = pname("init") : u64<3>
%1 = pwhile(%0) {
  cond_fn: (state) {
    %0 = pname("state") : f32[3]<3>
    %1 = access(%0) {index=0} : f32[3]<3>
    return %1
  }
  body_fn: (state) {
    %0 = pname("state") : f32[3]<3>
    %1 = access(%0) {index=0} : f32[3]<3>
    return %1
  }
} : f32[3]<3>
"""

        assert result == expected.strip()


if __name__ == "__main__":
    pytest.main([__file__])
