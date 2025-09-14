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

from mplang import simp
from mplang.core.cluster import ClusterSpec
from mplang.core.context_mgr import with_ctx
from mplang.core.dtype import FLOAT32, UINT64
from mplang.core.expr.printer import Printer
from mplang.core.mask import Mask
from mplang.core.mptype import Rank
from mplang.core.primitive import (
    _switch_ctx,
    constant,
    pconv,
    peval,
    prand,
    prank,
    primitive,
    pshfl,
    pshfl_s,
    set_mask,
    uniform_cond,
    while_loop,
)
from mplang.core.tracer import TraceContext, TraceVar, trace
from mplang.frontend import jax_cc


@pytest.fixture
def mask_2p():
    """Mask for 2-party computation."""
    return Mask(3)  # 0b11


@pytest.fixture
def cluster_spec_2p():
    """Create a 2-party cluster spec for testing."""
    return ClusterSpec.simple(2)


@pytest.fixture
def trace_context(mask_2p, cluster_spec_2p):
    """Create a trace context for testing."""
    return TraceContext(cluster_spec_2p, mask=mask_2p)


class TestPrimitiveDecorator:
    """Test the @primitive decorator."""

    def test_primitive_decorator_basic(self, trace_context):
        """Test basic primitive decorator functionality."""

        @primitive
        def simple_func():
            return constant(42)

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

    def test_constant_dataframe(self, trace_context):
        """Test constant primitive with pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        # Create a simple DataFrame for testing
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.2, 92.8],
        })

        func = lambda: constant(df)
        traced_fn = trace(trace_context, func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        # Check that we get a TraceVar with table type
        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)

        # Verify it's a table type
        from mplang.core.table import TableType

        assert isinstance(result.mptype._type, TableType)

        # Verify the schema is correct
        table_type = result.mptype._type
        assert table_type.has_column("id")
        assert table_type.has_column("name")
        assert table_type.has_column("score")
        assert table_type.num_columns() == 3

        # Test semantic behavior - constant should work and preserve the table structure
        # We don't need to check the internal expression structure
        # Since constant now uses builtin function, we can't directly access data_bytes
        # But we can verify it's an EvalExpr with the right function

    def test_constant_dataframe_empty(self, trace_context):
        """Test constant primitive with empty pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        # Create an empty DataFrame with schema
        df = pd.DataFrame(columns=["id", "name"], dtype=object)
        df = df.astype({"id": "int64", "name": "string"})

        func = lambda: constant(df)
        traced_fn = trace(trace_context, func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        assert len(traced_fn.out_vars) == 1
        result = traced_fn.out_vars[0]
        assert isinstance(result, TraceVar)

        # Verify it's a table type with correct schema
        from mplang.core.table import TableType

        assert isinstance(result.mptype._type, TableType)
        table_type = result.mptype._type
        assert table_type.has_column("id")
        assert table_type.has_column("name")
        assert table_type.num_columns() == 2

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

        # Create a function that uses pconv with primitive functions
        def conv_func():
            import numpy as np

            # Create constants using primitive.constant and set their masks
            # Party 0 has value 42
            const1 = constant(np.array(42, dtype=np.int64))
            var1 = set_mask(const1, Mask(1))  # Party 0 only

            # Party 1 has value 24
            const2 = constant(np.array(24, dtype=np.int64))
            var2 = set_mask(const2, Mask(2))  # Party 1 only

            return pconv([var1, var2])

        traced_fn = trace(trace_context, conv_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"pconv expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=42} : i64<3>
  %1 = peval(%0) {fn_type=builtin.identity, rmask=0x1} : i64<1>
  %2 = pconst() {data=24} : i64<3>
  %3 = peval(%2) {fn_type=builtin.identity, rmask=0x2} : i64<2>
  %4 = pconv(%1, %3) : i64<3>
  return %4
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

        # Create a simple function for testing using the new jax_cc.compile method
        def simple_add(x, y):
            return x + y

        def eval_func():
            x = constant(np.array([1.0, 2.0], dtype=np.float32))
            y = constant(np.array([3.0, 4.0], dtype=np.float32))

            # Use the new compilation method
            is_mpobject = lambda obj: isinstance(obj, TraceVar)
            pfunc, in_vars, _out_tree = jax_cc.jax2stablehlo(
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
  %2 = peval(%0, %1) {fn_type=mlir.stablehlo, fn_name=simple_add} : f32[2]<3>
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
    pfunc, in_vars, _out_tree = jax_cc.jax2stablehlo(is_mpobject, op_fn, *args)
    outs = peval(pfunc, in_vars)
    return outs[0]


class TestConditional:
    """Test conditional primitive with complex semantics."""

    def test_cond_simple(self, trace_context):
        """Test simple conditional with default verify_uniform disabled (placeholder)."""

        def cond_func():
            pred = constant(True)
            x = constant(10)

            def then_fn(x_arg):
                return x_arg

            def else_fn(x_arg):
                return x_arg

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=False)

        traced_fn = trace(trace_context, cond_func)
        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        assert "pcond" in expr_str

    def test_uniform_cond_traces_with_verify(self, trace_context):
        """uniform_cond with verify_uniform=True traces successfully for uniform bool predicate."""

        def cond_func():
            pred = constant(True)
            x = constant(1)

            def then_fn(a):
                return a

            def else_fn(a):
                return a

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=True)

        traced_fn = trace(trace_context, cond_func)
        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        assert "pcond" in expr_str

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

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=False)

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

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=False)

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
      %1 = peval($0, %0) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
      return %1
    }
    else_fn: ($0) {
      %0 = pconst() {data=5} : i64<3>
      %1 = peval($0, %0) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
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

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=False)

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
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
      return %0
    }
    else_fn: ($0, , $1) {
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
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
                return uniform_cond(
                    pred2, inner_then_fn, inner_else_fn, arg1, verify_uniform=False
                )

            def outer_else_fn(arg1):
                # This branch is simpler, just returns a constant.
                return constant(999)

            return uniform_cond(
                pred1, outer_then_fn, outer_else_fn, x, verify_uniform=False
            )

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
          %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
          return %0
        }
        else_fn: ($0, , $1) {
          %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
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

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=False)

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
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
      %1 = peval(%0, $2) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
      return %1
    }
    else_fn: ($0, , , $1) {
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
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

            return uniform_cond(pred, then_fn, else_fn, x, y, verify_uniform=False)

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
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
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

            return uniform_cond(
                pred, then_fn_good, else_fn_bad, x, verify_uniform=False
            )

        # This should raise an error due to truly incompatible signatures
        # The error might come from the trace function itself when it tries to call else_fn_bad
        with pytest.raises((ValueError, TypeError, AssertionError)):
            trace(trace_context, cond_func_bad)


class TestWhileLoop:
    """Test while loop primitive."""

    def test_while_loop_simple(self, trace_context):
        """Test simple while loop."""

        def while_func():
            init_val = constant(0)

            def cond_fn(x):
                constant(5)
                # For testing, return a boolean constant
                return constant(True)

            def body_fn(x):
                constant(1)
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
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : bool<3>
      return %0
    }
    body_fn: ($0, , $1) {
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
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
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : bool<3>
      return %0
    }
    body_fn: ($0, $1) {
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
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
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
      %1 = peval(%0, $2) {fn_type=mlir.stablehlo, fn_name=<lambda>} : bool<3>
      return %1
    }
    body_fn: ($0, , , $1) {
      %0 = peval($0, $1) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
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

        # This should raise a ValueError/TypeError due to type mismatch
        with pytest.raises(
            (ValueError, TypeError), match=r"Body output leaf 0 type mismatch: .*"
        ):
            trace(trace_context, while_func_wrong_type)

    def test_while_loop_pytree_params_complex(self, trace_context):
        """Test while loop with a complex PyTree init and multiple leaf updates."""

        def while_func():
            init = {
                "left": (constant(0), [constant(1), constant(2)]),
                "right": {"x": constant(10)},
            }

            inc = constant(3)
            limit = constant(20)

            def cond_fn(s):
                i, _arr = s["left"]
                # Keep it simple: loop while i < limit
                return run_jax(lambda a, b: a < b, i, limit)

            def body_fn(s):
                i, arr = s["left"]
                x = s["right"]["x"]
                new_i = run_jax(lambda a, b: a + b, i, inc)
                new_arr0 = run_jax(lambda a, b: a + b, arr[0], inc)
                return {"left": (new_i, [new_arr0, arr[1]]), "right": {"x": x}}

            final_state = while_loop(cond_fn, body_fn, init)
            # Return a single leaf to keep a single output var for tracing
            return final_state["left"][0]

        traced_fn = trace(trace_context, while_func)

        func_expr = traced_fn.make_expr()
        assert func_expr is not None
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"while loop complex PyTree expression:\n{expr_str}")

        expected = """
() {
  %0 = pconst() {data=0} : i64<3>
  %1 = pconst() {data=1} : i64<3>
  %2 = pconst() {data=2} : i64<3>
  %3 = pconst() {data=10} : i64<3>
  %4 = pconst() {data=20} : i64<3>
  %5 = pconst() {data=3} : i64<3>
  %6 = pwhile(%0, %1, %2, %3, %4, %5) {
    cond_fn: ($0, $1, $2, $3, $4, ) {
      %0 = peval($0, $4) {fn_type=mlir.stablehlo, fn_name=<lambda>} : bool<3>
      return %0
    }
    body_fn: ($0, $1, $2, $3, , $4) {
      %0 = peval($0, $4) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
      %1 = peval($1, $4) {fn_type=mlir.stablehlo, fn_name=<lambda>} : i64<3>
      %2 = tuple(%0, %1, $2, $3) : (i64<3>, i64<3>, i64<3>, i64<3>)
      return %2
    }
  } : (i64<3>, i64<3>, i64<3>, i64<3>)
  return %6:0
}
"""
        assert expr_str == expected.strip()
        assert len(traced_fn.out_vars) == 1
        assert isinstance(traced_fn.out_vars[0], TraceVar)


class TestCompleteExample:
    """Comprehensive example showing the complete testing pattern."""

    def test_complete_primitive_example(self, trace_context):
        """Complete example showing the complete testing pattern."""

        @primitive
        def example_computation():
            """Example multi-party computation function."""
            my_rank = prank()
            prand((2,))
            constant(0)
            constant(1)

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
            prank()
            prand((2, 2))
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
            constant(2)
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

        # Use primitive function to create a rank variable
        def rank_func():
            return prank()

        traced_fn = trace(trace_context, rank_func)
        rank_var = traced_fn.out_vars[0]

        result = _switch_ctx(trace_context, rank_var)
        assert result is rank_var

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
        target_mask = Mask(0b01)  # Mask for party 0 only

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
        assert "peval" in expr_str
        # The final output pmask should be the target_mask
        assert f": i64<{target_mask}>" in expr_str

    def test_set_mask_with_static_pmask_valid_subset(
        self, trace_context, cluster_spec_2p
    ):
        """Test set_mask with static pmask where mask is a valid subset."""

        # Create a function that uses prank to get rank in specific context
        def rank_func():
            return prank()

        # Create a new trace context with specific mask
        original_mask = Mask(0b11)  # Parties 0, 1 (full mask for 2-party context)
        specific_context = TraceContext(cluster_spec_2p, mask=original_mask)
        traced_fn = trace(specific_context, rank_func)
        static_var = traced_fn.out_vars[0]

        # Verify the input has static pmask
        assert static_var.pmask == original_mask

        # Apply set_mask with a subset mask
        subset_mask = Mask(0b01)  # Party 0 only (subset of 0b11)

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

        # Create a function that sets mask to subset first, then tries invalid subset
        def test_func():
            static_var = constant(42)
            static_var_party0 = set_mask(static_var, Mask(0b01))  # Party 0 only
            # Now try to set it to party 1 only (NOT subset of 0b01)
            return set_mask(static_var_party0, Mask(0b10))  # Party 1 only

        # This should raise ValueError at trace time (compile time)
        with pytest.raises(ValueError, match="not a subset"):
            trace(trace_context, test_func)

    def test_set_mask_with_empty_mask(self, trace_context, cluster_spec_2p):
        """Test set_mask with empty mask (0)."""

        # Create a function that uses prank to get rank in specific context
        def rank_func():
            return prank()

        # Create a new trace context with specific mask
        original_mask = Mask(0b11)  # Parties 0, 1 (full mask for 2-party context)
        specific_context = TraceContext(cluster_spec_2p, mask=original_mask)
        traced_fn = trace(specific_context, rank_func)
        static_var = traced_fn.out_vars[0]

        # Apply set_mask with empty mask
        empty_mask = Mask(0b00)  # No parties

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

        # Create a function that sets mask and then sets the same mask again
        def test_func():
            static_var = constant(42)
            static_var_party0 = set_mask(static_var, Mask(0b01))  # Party 0 only
            # Set it to the same mask again
            return set_mask(static_var_party0, Mask(0b01))  # Same mask

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
        from mplang.core.expr.ast import AccessExpr, EvalExpr

        # Create input variable using constant
        def const_func():
            return constant(42)

        traced_fn = trace(trace_context, const_func)
        input_var = traced_fn.out_vars[0]

        # Apply set_mask
        target_mask = Mask(0b01)  # Party 0 only

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
        """Test set_mask integration with other primitive operations using the new simp.run API."""

        # Create a chain of operations involving set_mask
        def test_func():
            # Start with a rank operation
            rank_var = prank()  # This has dynamic pmask initially

            # Apply set_mask to constrain it
            constrained_var = set_mask(rank_var, Mask(0b10))  # Party 1 only

            # Use in another operation
            const_var = constant(1)

            # Combine using the new simp.run API
            result = simp.run(lambda x, y: x + y)(constrained_var, const_var)

            return result

        traced_fn = trace(trace_context, test_func)

        # Should successfully trace the entire chain
        assert len(traced_fn.out_vars) == 1

        func_expr = traced_fn.make_expr()
        assert func_expr is not None

        from mplang.core.expr.printer import Printer

        printer = Printer()
        expr_str = printer.print_expr(func_expr)
        print(f"set_mask integration expression:\n{expr_str}")

        # The expression should contain multiple operations
        assert "eval" in expr_str  # From set_mask and simp.run
        assert "prank" in expr_str  # From prank()
        assert "pconst" in expr_str  # From constant()


class TestUniformCondValidation:
    """Focused tests for uniform_cond validation edge cases."""

    def test_uniform_cond_predicate_non_scalar_error(self, trace_context):
        """Predicate must be scalar: supplying a vector should raise TypeError."""
        import numpy as np

        def cond_func():
            # Shape (2,) predicate (invalid)
            pred = constant(np.array([True, False]))  # non-scalar
            x = constant(1)

            def then_fn(a):  # type: ignore
                return a

            def else_fn(a):  # type: ignore
                return a

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=False)

        with pytest.raises(TypeError):
            trace(trace_context, cond_func)

    def test_uniform_cond_predicate_non_bool_error(self, trace_context):
        """Predicate must be boolean: supplying int scalar should raise TypeError."""

        def cond_func():
            pred = constant(1)  # int, not bool
            x = constant(1)

            def then_fn(a):  # type: ignore
                return a

            def else_fn(a):  # type: ignore
                return a

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=False)

        with pytest.raises(TypeError):
            trace(trace_context, cond_func)

    def test_uniform_cond_branch_output_mismatch(self, trace_context):
        """Branches returning different shapes should raise TypeError."""
        import numpy as np

        def cond_func():
            pred = constant(True)
            x = constant(np.array([1, 2, 3], dtype=np.int64))

            def then_fn(v):  # type: ignore
                return v  # shape (3,)

            def else_fn(v):  # type: ignore
                return constant(np.array([[1, 2, 3]], dtype=np.int64))  # shape (1,3)

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=False)

        with pytest.raises(TypeError):
            trace(trace_context, cond_func)

    def test_uniform_cond_divergent_predicate_no_verification(self, trace_context):
        """Divergent predicate allowed when verify_uniform=False (no error)."""

        def divergent_predicate():
            # We simulate divergent predicate by constructing a boolean whose logical
            # value could differ per party if produced via local computation. For now
            # we just use a uniform True (cannot fabricate divergence without runtime),
            # the key point is verify_uniform=False does not raise.
            pred = constant(True)
            x = constant(5)

            def then_fn(v):  # type: ignore
                return v

            def else_fn(v):  # type: ignore
                return v

            return uniform_cond(pred, then_fn, else_fn, x, verify_uniform=False)

        traced = trace(trace_context, divergent_predicate)
        assert traced.make_expr() is not None
