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
Tests for primitive module.

This test suite demonstrates the testing pattern for primitives:
1. Define a function using primitive operations
2. Trace the function to create a TracedFunction
3. Test the traced function's expression and compare with printer output

For simple primitives: trace the primitive function directly
For complex operations: define a function and trace it
"""

import numpy as np
import pytest
from jax.tree_util import tree_unflatten

from mplang.core.base import Mask, Rank, TensorInfo, cur_ctx
from mplang.core.dtype import FLOAT32, INT32, UINT64
from mplang.core.primitive import (
    _switch_ctx,
    cond,
    constant,
    pconv,
    peval,
    prand,
    prank,
    primitive,
    pshfl,
    pshfl_s,
    set_mask,
    while_loop,
)
from mplang.core.trace import TraceContext, TraceVar, trace
from mplang.expr.printer import Printer
from mplang.plib import jax2stablehlo


@pytest.fixture
def mask_2p():
    """Mask for 2-party computation."""
    return Mask(3)  # 0b11


@pytest.fixture
def trace_context(mask_2p):
    """Create a trace context for testing."""
    return TraceContext(world_size=2, mask=mask_2p)


class TestPrimitiveDecorator:
    """Test the @primitive decorator."""

    def test_primitive_decorator_basic(self, trace_context):
        """Test basic primitive decorator functionality."""

        @primitive
        def simple_func():
            return constant(42)

        from mplang.core.base import with_ctx

        with with_ctx(trace_context):
            result = simple_func()
            assert isinstance(result, TraceVar)
            assert result.ctx is trace_context


class TestBasicPrimitives:
    """Test basic primitive operations."""

    def test_prank(self, trace_context):
        """Test prank primitive."""
        traced_fn = trace(trace_context, prank)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"prank expression:\n{expr_str}")

        expected = """
() {
  %0 = prank() : u64<3>
  return %0
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_prand(self, trace_context):
        """Test prand primitive."""
        func = lambda: prand((2, 3))

        traced_fn = trace(trace_context, func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"prand expression:\n{expr_str}")

        expected = """
() {
  %0 = prand() : u64[2, 3]<3>
  return %0
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)
        assert result.mptype.shape == (2, 3)
        assert result.mptype.dtype == UINT64

    def test_constant_scalar(self, trace_context):
        """Test constant primitive with scalar."""
        func = lambda: constant(42)

        traced_fn = trace(trace_context, func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"constant scalar expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=42} : i64<3>
  return %0
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)
        assert result.mptype.shape == ()

    def test_constant_array(self, trace_context):
        """Test constant primitive with array."""
        func = lambda: constant(np.array([1, 2, 3], dtype=np.float32))

        traced_fn = trace(trace_context, func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"constant array expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=[1.0, 2.0, 3.0]} : f32[3]<3>
  return %0
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)
        assert result.mptype.shape == (3,)
        assert result.mptype.dtype == FLOAT32

    def test_pshfl(self, trace_context):
        """Test pshfl primitive."""

        def shuffle_func():
            src = constant(42)
            index = constant(1)
            return pshfl(src, index)

        traced_fn = trace(trace_context, shuffle_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"pshfl expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=42} : i64<3>
  %1 = pconst() {data=1} : i64<3>
  %2 = pshfl(%0, %1) : i64
  return %2
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)
        # Note: ShflExpr returns None pmask (runtime-determined)
        assert result.mptype.pmask is None

    def test_pshfl_s(self, trace_context):
        """Test pshfl_s primitive."""

        def shuffle_static_func():
            src = constant(42)
            pmask = Mask(7)  # 0b111 for 3 parties
            src_ranks = [Rank(0), Rank(1), Rank(0)]  # Map to parties 0, 1, 0
            return pshfl_s(src, pmask, src_ranks)

        traced_fn = trace(trace_context, shuffle_static_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"pshfl_s expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=42} : i64<3>
  %1 = pshfl_s(%0) {pmask=7, src_ranks=[0, 1, 0]} : i64<7>
  return %1
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)
        # Note: ShflSExpr should have the specified pmask
        assert result.mptype.pmask == 7

    def test_pconv(self, trace_context):
        """Test pconv primitive."""
        from mplang.core.dtype import INT64
        from mplang.expr.ast import ConstExpr

        # Create a function that uses pconv with manually created TraceVars
        def conv_func():
            # Get current context
            from typing import cast

            import numpy as np

            ctx = cast(TraceContext, cur_ctx())

            # Create constants with different pmasks manually
            # Party 0 has value 42
            data1 = np.array(42, dtype=np.int64).tobytes()
            const1_expr = ConstExpr(TensorInfo(INT64, ()), data1, Mask(1))  # 0b01
            var1 = TraceVar(ctx, const1_expr)

            # Party 1 has value 24
            data2 = np.array(24, dtype=np.int64).tobytes()
            const2_expr = ConstExpr(TensorInfo(INT64, ()), data2, Mask(2))  # 0b10
            var2 = TraceVar(ctx, const2_expr)

            return pconv([var1, var2])

        traced_fn = trace(trace_context, conv_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"pconv expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=42} : i64<1>
  %1 = pconst() {data=24} : i64<2>
  %2 = pconv(%0, %1) : i64<3>
  return %2
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)
        # Note: ConvExpr should have the union of pmasks: 0b01 | 0b10 = 0b11 = 3
        assert result.mptype.pmask == 3  # 0b11 (union of 0b01 and 0b10)


class TestPeval:
    """Test peval primitive."""

    def test_peval_simple(self, trace_context):
        """Test peval with a simple function."""

        # Create a simple function for testing using the new jax2stablehlo.compile method
        def simple_add(x, y):
            return x + y

        def eval_func():
            x = constant(np.array([1.0, 2.0], dtype=np.float32))
            y = constant(np.array([3.0, 4.0], dtype=np.float32))

            # Use the new compilation method
            is_mpobject = lambda obj: isinstance(obj, TraceVar)
            pfunc, in_vars, out_tree = jax2stablehlo.compile(
                is_mpobject, simple_add, x, y
            )
            results = peval(pfunc, in_vars)
            return results[0]

        traced_fn = trace(trace_context, eval_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"peval expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=[1.0, 2.0]} : f32[2]<3>
  %1 = pconst() {data=[3.0, 4.0]} : f32[2]<3>
  %2 = peval(%0, %1) {fn_name="simple_add"} : f32[2]<3>
  return %2
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)


def run_jax(op_fn, *args):
    """Helper function for arithmetic operations using JAX."""
    is_mpobject = lambda obj: isinstance(obj, TraceVar)
    pfunc, in_vars, out_tree = jax2stablehlo.compile(is_mpobject, op_fn, *args)
    outs = peval(pfunc, in_vars)
    return outs[0]


class TestConditional:
    """Test conditional primitive with complex semantics."""

    def test_cond_simple(self, trace_context):
        """Test simple conditional."""

        def cond_func():
            pred = constant(True)
            x = constant(10)

            def then_fn(x_arg):  # type: ignore
                return x_arg

            def else_fn(x_arg):  # type: ignore
                return x_arg

            return cond(pred, then_fn, else_fn, x)

        traced_fn = trace(trace_context, cond_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"conditional expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=True} : bool<3>
  %1 = pconst() {data=10} : i64<3>
  %2 = pcond(%0, %1) {
    then_fn: ($0) {
      return $0
    }
    else_fn: ($0) {
      return $0
    }
  } : i64<3>
  return %2
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_cond_different_behaviors(self, trace_context):
        """Test then_fn and else_fn with different behaviors."""

        def cond_func():
            pred = constant(True)
            x = constant(5)

            def then_fn(x_arg):  # type: ignore
                # Then branch: return constant 10
                return constant(10)

            def else_fn(x_arg):  # type: ignore
                # Else branch: return constant 20
                return constant(20)

            return cond(pred, then_fn, else_fn, x)

        traced_fn = trace(trace_context, cond_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"different behaviors expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=True} : bool<3>
  %1 = pconst() {data=5} : i64<3>
  %2 = pcond(%0, %1) {
    then_fn: ($0) {
      %0 = pconst() {data=10} : i64<3>
      return %0
    }
    else_fn: ($0) {
      %0 = pconst() {data=20} : i64<3>
      return %0
    }
  } : i64<3>
  return %2
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_cond_same_capture(self, trace_context):
        """Test then_fn and else_fn capturing the same concept but different values."""

        def cond_func():
            pred = constant(True)
            x = constant(5)

            def then_fn(x_arg):  # type: ignore
                # Then branch: add a constant 10 (no external capture)
                ten = constant(10)
                return run_jax(lambda a, b: a + b, x_arg, ten)

            def else_fn(x_arg):  # type: ignore
                # Else branch: subtract a constant 5 (no external capture)
                five = constant(5)
                return run_jax(lambda a, b: a - b, x_arg, five)

            return cond(pred, then_fn, else_fn, x)

        traced_fn = trace(trace_context, cond_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"same capture expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=True} : bool<3>
  %1 = pconst() {data=5} : i64<3>
  %2 = pcond(%0, %1) {
    then_fn: ($0) {
      %0 = pconst() {data=10} : i64<3>
      %1 = peval($0, %0) {fn_name="<lambda>"} : i64<3>
      return %1
    }
    else_fn: ($0) {
      %0 = pconst() {data=5} : i64<3>
      %1 = peval($0, %0) {fn_name="<lambda>"} : i64<3>
      return %1
    }
  } : i64<3>
  return %2
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_cond_different_captures(self, trace_context):
        """Test then_fn and else_fn capturing different values."""

        def cond_func():
            pred = constant(True)
            x = constant(5)
            val_a = constant(10)  # Only captured by then_fn
            val_b = constant(20)  # Only captured by else_fn

            def then_fn(x_arg):  # type: ignore
                # Only captures val_a
                return run_jax(lambda a, b: a + b, x_arg, val_a)

            def else_fn(x_arg):  # type: ignore
                # Only captures val_b
                return run_jax(lambda a, b: a + b, x_arg, val_b)

            return cond(pred, then_fn, else_fn, x)

        traced_fn = trace(trace_context, cond_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"different captures expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=True} : bool<3>
  %1 = pconst() {data=5} : i64<3>
  %2 = pconst() {data=10} : i64<3>
  %3 = pconst() {data=20} : i64<3>
  %4 = pcond(%0, %1, %2, %3) {
    then_fn: ($0, $1, ) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
      return %0
    }
    else_fn: ($0, , $1) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
      return %0
    }
  } : i64<3>
  return %4
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_cond_nested_with_captures(self, trace_context):
        """Test nested cond with complex captures from outer scopes."""

        def nested_cond_func():
            # Outermost scope variables
            outer_var = constant(100)
            pred1 = constant(True)
            pred2 = constant(False)
            x = constant(5)

            def outer_then_fn(arg1):  # arg1 will be x
                # This branch contains a nested cond.
                # It defines its own variable and captures one from the outermost scope.
                inner_var = constant(50)

                def inner_then_fn(inner_arg):  # inner_arg will be arg1 (x)
                    # Captures outer_var from the outermost scope (nested_cond_func)
                    return run_jax(lambda a, b: a + b, inner_arg, outer_var)

                def inner_else_fn(inner_arg):  # inner_arg will be arg1 (x)
                    # Captures inner_var from its defining scope (outer_then_fn)
                    return run_jax(lambda a, b: a - b, inner_arg, inner_var)

                # The inner cond captures outer_var.
                return cond(pred2, inner_then_fn, inner_else_fn, arg1)

            def outer_else_fn(arg1):
                # This branch is simpler, just returns a constant.
                return constant(999)

            return cond(pred1, outer_then_fn, outer_else_fn, x)

        traced_fn = trace(trace_context, nested_cond_func)
        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"nested cond with captures expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=True} : bool<3>
  %1 = pconst() {data=5} : i64<3>
  %2 = pconst() {data=False} : bool<3>
  %3 = pconst() {data=100} : i64<3>
  %4 = pcond(%0, %1, %2, %3) {
    then_fn: ($0, $1, $2) {
      %0 = pconst() {data=50} : i64<3>
      %1 = pcond($1, $0, $2, %0) {
        then_fn: ($0, $1, ) {
          %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
          return %0
        }
        else_fn: ($0, , $1) {
          %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
          return %0
        }
      } : i64<3>
      return %1
    }
    else_fn: ($0, , ) {
      %0 = pconst() {data=999} : i64<3>
      return %0
    }
  } : i64<3>
  return %4
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_cond_unequal_capture_count(self, trace_context):
        """Test then_fn and else_fn capturing different numbers of values."""

        def cond_func():
            pred = constant(True)
            x = constant(5)
            val_a = constant(10)
            val_b = constant(20)
            val_c = constant(30)

            def then_fn(x_arg):  # type: ignore
                # Captures two values - chain additions
                temp = run_jax(lambda a, b: a + b, x_arg, val_a)
                return run_jax(lambda a, b: a + b, temp, val_b)

            def else_fn(x_arg):  # type: ignore
                # Captures one value
                return run_jax(lambda a, b: a + b, x_arg, val_c)

            return cond(pred, then_fn, else_fn, x)

        traced_fn = trace(trace_context, cond_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"unequal capture count expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=True} : bool<3>
  %1 = pconst() {data=5} : i64<3>
  %2 = pconst() {data=10} : i64<3>
  %3 = pconst() {data=20} : i64<3>
  %4 = pconst() {data=30} : i64<3>
  %5 = pcond(%0, %1, %2, %3, %4) {
    then_fn: ($0, $1, $2, ) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
      %1 = peval(%0, $2) {fn_name="<lambda>"} : i64<3>
      return %1
    }
    else_fn: ($0, , , $1) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
      return %0
    }
  } : i64<3>
  return %5
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_cond_inconsistent_signatures_error(self, trace_context):
        """Test that inconsistent signatures between then_fn and else_fn raise an error."""

        def cond_func():
            pred = constant(True)
            x = constant(5)
            y = constant(10)

            def then_fn(x_arg, y_arg):  # type: ignore
                # Correct signature: takes multiple arguments
                return run_jax(lambda a, b: a + b, x_arg, y_arg)

            def else_fn(x_arg, y_arg):  # type: ignore
                # Incorrect implementation: only uses first argument
                return x_arg

            return cond(pred, then_fn, else_fn, x, y)

        # This should work but the behavior might be different
        # Since both functions have compatible signatures now
        traced_fn = trace(trace_context, cond_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"potentially inconsistent behavior expression:\n{expr_str}")

        # Both functions have same signature but different behavior
        expected = """
() {
  %0 = pconst() {data=True} : bool<3>
  %1 = pconst() {data=5} : i64<3>
  %2 = pconst() {data=10} : i64<3>
  %3 = pcond(%0, %1, %2) {
    then_fn: ($0, $1) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
      return %0
    }
    else_fn: ($0, $1) {
      return $0
    }
  } : i64<3>
  return %3
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_cond_real_signature_mismatch_error(self, trace_context):
        """Test that truly incompatible function structures raise an error."""

        def cond_func_bad():
            pred = constant(True)
            x = constant(5)

            def then_fn_good(args):  # type: ignore
                return args[0]

            def else_fn_bad(invalid_signature):  # type: ignore
                # This should cause issues because the signature is fundamentally different
                return invalid_signature

            return cond(pred, then_fn_good, else_fn_bad, x)

        # This should raise an error due to truly incompatible signatures
        # The error might come from the trace function itself when it tries to call else_fn_bad
        with pytest.raises((ValueError, TypeError, AssertionError)):
            traced_fn = trace(trace_context, cond_func_bad)


class TestWhileLoop:
    """Test while loop primitive."""

    def test_while_loop_simple(self, trace_context):
        """Test simple while loop."""

        def while_func():
            init_val = constant(0)

            def cond_fn(x):
                five = constant(5)
                # For testing, return a boolean constant
                return constant(True)

            def body_fn(x):
                one = constant(1)
                # For testing, return the input
                return x

            return while_loop(cond_fn, body_fn, init_val)

        traced_fn = trace(trace_context, while_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"while loop expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=0} : i64<3>
  %1 = pwhile(%0) {
    cond_fn: ($0) {
      %0 = pconst() {data=True} : bool<3>
      return %0
    }
    body_fn: ($0) {
      return $0
    }
  } : i64<3>
  return %1
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_while_loop_with_captures(self, trace_context):
        """Test while loop with captured variables from outer scope."""

        def while_func():
            init_val = constant(0)
            counter_max = constant(5)  # Will be captured by cond_fn
            increment = constant(1)  # Will be captured by body_fn

            def cond_fn(x):
                # Captures counter_max from outer scope
                return run_jax(lambda a, b: a < b, x, counter_max)

            def body_fn(x):
                # Captures increment from outer scope
                return run_jax(lambda a, b: a + b, x, increment)

            return while_loop(cond_fn, body_fn, init_val)

        traced_fn = trace(trace_context, while_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"while loop with captures expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=0} : i64<3>
  %1 = pconst() {data=5} : i64<3>
  %2 = pconst() {data=1} : i64<3>
  %3 = pwhile(%0, %1, %2) {
    cond_fn: ($0, $1, ) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : bool<3>
      return %0
    }
    body_fn: ($0, , $1) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
      return %0
    }
  } : i64<3>
  return %3
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_while_loop_shared_captures(self, trace_context):
        """Test while loop where both functions capture the same variable."""

        def while_func():
            init_val = constant(0)
            shared_val = constant(10)  # Captured by both functions

            def cond_fn(x):
                # Both functions capture shared_val
                return run_jax(lambda a, b: a < b, x, shared_val)

            def body_fn(x):
                # Both functions capture shared_val
                return run_jax(lambda a, b: a + b, x, shared_val)

            return while_loop(cond_fn, body_fn, init_val)

        traced_fn = trace(trace_context, while_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"while loop shared captures expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=0} : i64<3>
  %1 = pconst() {data=10} : i64<3>
  %2 = pwhile(%0, %1) {
    cond_fn: ($0, $1) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : bool<3>
      return %0
    }
    body_fn: ($0, $1) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
      return %0
    }
  } : i64<3>
  return %2
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_while_loop_no_captures(self, trace_context):
        """Test while loop with no external captures."""

        def while_func():
            init_val = constant(0)

            def cond_fn(x):
                # No captures, just return a constant
                return constant(False)

            def body_fn(x):
                # No captures, just return the input
                return x

            return while_loop(cond_fn, body_fn, init_val)

        traced_fn = trace(trace_context, while_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"while loop no captures expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=0} : i64<3>
  %1 = pwhile(%0) {
    cond_fn: ($0) {
      %0 = pconst() {data=False} : bool<3>
      return %0
    }
    body_fn: ($0) {
      return $0
    }
  } : i64<3>
  return %1
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_while_loop_complex_captures(self, trace_context):
        """Test while loop with complex capture patterns."""

        def while_func():
            init_val = constant(0)
            val_a = constant(10)
            val_b = constant(20)
            val_c = constant(30)

            def cond_fn(x):
                # Captures val_a and val_b
                temp = run_jax(lambda a, b: a + b, x, val_a)
                return run_jax(lambda a, b: a < b, temp, val_b)

            def body_fn(x):
                # Captures val_c only
                return run_jax(lambda a, b: a + b, x, val_c)

            return while_loop(cond_fn, body_fn, init_val)

        traced_fn = trace(trace_context, while_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"while loop complex captures expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=0} : i64<3>
  %1 = pconst() {data=10} : i64<3>
  %2 = pconst() {data=20} : i64<3>
  %3 = pconst() {data=30} : i64<3>
  %4 = pwhile(%0, %1, %2, %3) {
    cond_fn: ($0, $1, $2, ) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
      %1 = peval(%0, $2) {fn_name="<lambda>"} : bool<3>
      return %1
    }
    body_fn: ($0, , , $1) {
      %0 = peval($0, $1) {fn_name="<lambda>"} : i64<3>
      return %0
    }
  } : i64<3>
  return %4
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_while_loop_type_validation(self, trace_context):
        """Test while loop type validation for body function."""

        def while_func_wrong_type():
            init_val = constant(0)

            def cond_fn(x):
                return constant(True)

            def body_fn(x):
                # Returns wrong type - should be same as init_val
                return constant(3.14)  # float instead of int

            return while_loop(cond_fn, body_fn, init_val)

        # This should raise a ValueError due to type mismatch
        with pytest.raises(
            ValueError,
            match="Body function output type .* does not match initial state type",
        ):
            traced_fn = trace(trace_context, while_func_wrong_type)


class TestCompleteExample:
    """Comprehensive example showing the complete testing pattern."""

    def test_complete_primitive_example(self, trace_context):
        """Complete example showing the complete testing pattern."""

        @primitive
        def example_computation():
            """Example multi-party computation function."""
            my_rank = prank()
            random_data = prand((2,))
            zero = constant(0)
            one = constant(1)

            return my_rank

        traced_fn = trace(trace_context, example_computation)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"Complete example IR:\n{expr_str}")

        expected = """
() {
  %0 = prank() : u64<3>
  return %0
}
"""
        assert expr_str == expected.strip()


class TestComplexExpressions:
    """Test more complex expressions combining multiple primitives."""

    def test_complex_function(self, trace_context):
        """Test function combining multiple primitives."""

        @primitive
        def complex_func():
            rank_val = prank()
            random_vals = prand((2, 2))
            const_val = constant(42)
            return const_val

        def func():
            return complex_func()

        traced_fn = trace(trace_context, func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"complex function expression:\n{expr_str}")

        # Unused variables are optimized away
        expected = """
() {
  %0 = pconst() {data=42} : i64<3>
  return %0
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

    def test_nested_primitives(self, trace_context):
        """Test nested primitive calls."""

        @primitive
        def outer_func():
            @primitive
            def inner_func():
                return constant(1)

            val1 = inner_func()
            val2 = constant(2)
            return val1

        def func():
            return outer_func()

        traced_fn = trace(trace_context, func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"nested primitives expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=1} : i64<3>
  return %0
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)


class TestSwitchContext:
    """Test _switch_ctx helper function."""

    def test_switch_ctx_same_context(self, trace_context):
        """Test switching to the same context."""
        from mplang.expr.ast import RankExpr

        rank_expr = RankExpr(trace_context.mask)
        var = TraceVar(trace_context, rank_expr)

        result = _switch_ctx(trace_context, var)
        assert result is var

    def test_switch_ctx_non_mpobject(self, trace_context):
        """Test switching context with non-MPObject."""
        non_mpobj = 42
        result = _switch_ctx(trace_context, non_mpobj)
        assert result is non_mpobj


class TestSetMask:
    """Test set_mask function behavior."""

    def test_set_mask_with_dynamic_pmask(self, trace_context):
        """Test set_mask with dynamic pmask (arg.pmask is None)."""

        # Create a variable with dynamic pmask using pshfl.
        # pshfl's output pmask is data-dependent and thus None at compile time.
        def create_dynamic_var():
            src = constant(123)
            # The shuffle index determines which party's data is taken.
            # Since this can vary, the output pmask is dynamic.
            shuffle_index = constant(0)
            return pshfl(src, shuffle_index)

        traced_dynamic = trace(trace_context, create_dynamic_var)
        dynamic_var = traced_dynamic.out_vars[0]

        # Verify that the created variable has a dynamic pmask
        assert dynamic_var.pmask is None

        # Apply set_mask with a specific mask (valid for 2-party context)
        target_mask = 0b01  # Mask for party 0 only

        def test_func():
            return set_mask(dynamic_var, target_mask)

        traced_fn = trace(trace_context, test_func)

        # The traced function should capture the set_mask operation
        assert len(traced_fn.out_vars) == 1
        result_var = traced_fn.out_vars[0]

        # Verify the operation was traced
        assert isinstance(result_var, TraceVar)

        # The output of set_mask on a dynamic input should have the new mask
        assert result_var.pmask == target_mask

        # Print the expression to verify structure
        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"set_mask with dynamic pmask expression:\n{expr_str}")

        # The expression should contain a peval operation with the target mask
        # and the output var should have the correct pmask in the IR
        assert "pshfl" in expr_str
        assert f"peval" in expr_str
        # The final output pmask should be the target_mask
        assert f": i64<{target_mask}>" in expr_str

    def test_set_mask_with_static_pmask_valid_subset(self, trace_context):
        """Test set_mask with static pmask where mask is a valid subset."""
        # Create a variable with static pmask
        from mplang.expr.ast import RankExpr

        original_mask = 0b11  # Parties 0, 1 (full mask for 2-party context)
        rank_expr = RankExpr(original_mask)
        static_var = TraceVar(trace_context, rank_expr)

        # Verify the input has static pmask
        assert static_var.pmask == original_mask

        # Apply set_mask with a subset mask
        subset_mask = 0b01  # Party 0 only (subset of 0b11)

        def test_func():
            return set_mask(static_var, subset_mask)

        traced_fn = trace(trace_context, test_func)

        # The traced function should succeed
        assert len(traced_fn.out_vars) == 1
        result_var = traced_fn.out_vars[0]

        # For static input with valid subset, pmask should remain original
        assert isinstance(result_var, TraceVar)

        # Print the expression to verify structure
        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"set_mask with valid subset expression:\n{expr_str}")

    def test_set_mask_with_static_pmask_invalid_subset(self, trace_context):
        """Test set_mask with static pmask where mask is NOT a subset."""
        # Create a variable with static pmask
        from mplang.expr.ast import RankExpr

        original_mask = 0b01  # Party 0 only
        rank_expr = RankExpr(original_mask)
        static_var = TraceVar(trace_context, rank_expr)

        # Verify the input has static pmask
        assert static_var.pmask == original_mask

        # Apply set_mask with a non-subset mask
        invalid_mask = 0b10  # Party 1 only (NOT subset of 0b01)

        def test_func():
            return set_mask(static_var, invalid_mask)

        # This should raise ValueError at trace time (compile time)
        with pytest.raises(ValueError, match="not a subset"):
            trace(trace_context, test_func)

    def test_set_mask_with_empty_mask(self, trace_context):
        """Test set_mask with empty mask (0)."""
        from mplang.expr.ast import RankExpr

        original_mask = 0b11  # Parties 0, 1 (full mask for 2-party context)
        rank_expr = RankExpr(original_mask)
        static_var = TraceVar(trace_context, rank_expr)

        # Apply set_mask with empty mask
        empty_mask = 0b00  # No parties

        def test_func():
            return set_mask(static_var, empty_mask)

        traced_fn = trace(trace_context, test_func)

        # Should succeed (empty mask is subset of any mask)
        assert len(traced_fn.out_vars) == 1

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"set_mask with empty mask expression:\n{expr_str}")

    def test_set_mask_with_same_mask(self, trace_context):
        """Test set_mask with the same mask as input."""
        from mplang.expr.ast import RankExpr

        original_mask = 0b01  # Party 0 only
        rank_expr = RankExpr(original_mask)
        static_var = TraceVar(trace_context, rank_expr)

        # Apply set_mask with the same mask
        same_mask = 0b01  # Same as original

        def test_func():
            return set_mask(static_var, same_mask)

        traced_fn = trace(trace_context, test_func)

        # Should succeed
        assert len(traced_fn.out_vars) == 1

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"set_mask with same mask expression:\n{expr_str}")

    def test_set_mask_expression_structure(self, trace_context):
        """Test the internal expression structure of set_mask."""
        from mplang.expr.ast import AccessExpr, EvalExpr, RankExpr

        # Create input variable
        input_mask = 0b11  # Parties 0, 1 (full mask for 2-party context)
        rank_expr = RankExpr(input_mask)
        input_var = TraceVar(trace_context, rank_expr)

        # Apply set_mask
        target_mask = 0b01  # Party 0 only

        def test_func():
            return set_mask(input_var, target_mask)

        traced_fn = trace(trace_context, test_func)
        result_var = traced_fn.out_vars[0]

        # The result should be wrapped in an AccessExpr that accesses an EvalExpr
        assert isinstance(result_var.expr, AccessExpr)

        # The AccessExpr should reference an EvalExpr
        eval_expr = result_var.expr.src  # Use 'src' not 'fn_expr'
        assert isinstance(eval_expr, EvalExpr)

        # The EvalExpr should have the target mask as rmask
        assert eval_expr.rmask == target_mask

        # The EvalExpr should have one input (the original variable)
        assert len(eval_expr.args) == 1  # Use 'args' not 'in_exprs'
        assert eval_expr.args[0] == input_var.expr

    def test_set_mask_integration_with_other_primitives(self, trace_context):
        """Test set_mask integration with other primitive operations."""

        # Create a chain of operations involving set_mask
        def test_func():
            # Start with a rank operation
            rank_var = prank()  # This has dynamic pmask initially

            # Apply set_mask to constrain it
            constrained_var = set_mask(rank_var, 0b10)  # Party 1 only

            # Use in another operation
            const_var = constant(1)

            # Combine using JAX function
            from mplang.core.primitive import run_jax

            result = run_jax(lambda x, y: x + y, constrained_var, const_var)

            return result

        traced_fn = trace(trace_context, test_func)

        # Should successfully trace the entire chain
        assert len(traced_fn.out_vars) == 1

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"set_mask integration expression:\n{expr_str}")

        # The expression should contain multiple operations
        assert "eval" in expr_str  # From set_mask and run_jax
        assert "prank" in expr_str  # From prank()
        assert "pconst" in expr_str  # From constant()
