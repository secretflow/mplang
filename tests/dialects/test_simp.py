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

"""Tests for SIMP dialect (EDSL-based primitives).

Note: uniform_cond and while_loop are bound methods from Primitive instances.
Call them directly like functions: uniform_cond(...), while_loop(...)
"""

import pytest

from mplang.dialects.simp import uniform_cond, while_loop


class TestUniformCond:
    """Test suite for uniform_cond primitive."""

    def test_with_callables_true_branch(self):
        """Test uniform_cond executes then_fn when pred is True."""

        def then_fn(x):
            return x * 2

        def else_fn(x):
            return x + 1

        result = uniform_cond(True, then_fn, else_fn, 5)
        assert result == 10  # 5 * 2

    def test_with_callables_false_branch(self):
        """Test uniform_cond executes else_fn when pred is False."""

        def then_fn(x):
            return x * 2

        def else_fn(x):
            return x + 1

        result = uniform_cond(False, then_fn, else_fn, 5)
        assert result == 6  # 5 + 1

    def test_with_precomputed_values(self):
        """Test uniform_cond with pre-computed values (jax.where-like)."""
        result_true = uniform_cond(True, 100, 200)
        assert result_true == 100

        result_false = uniform_cond(False, 100, 200)
        assert result_false == 200

    def test_side_effects_only_selected_branch(self):
        """Test that only the selected branch is executed (lazy evaluation)."""
        side_effects = []

        def then_fn(x):
            side_effects.append("then")
            return x * 2

        def else_fn(x):
            side_effects.append("else")
            return x + 1

        result = uniform_cond(True, then_fn, else_fn, 5)
        assert result == 10
        assert side_effects == ["then"]  # Only then_fn executed

        side_effects.clear()
        result = uniform_cond(False, then_fn, else_fn, 5)
        assert result == 6
        assert side_effects == ["else"]  # Only else_fn executed

    def test_with_multiple_args(self):
        """Test uniform_cond with multiple arguments."""

        def then_fn(x, y):
            return x + y

        def else_fn(x, y):
            return x - y

        result = uniform_cond(True, then_fn, else_fn, 10, 3)
        assert result == 13

        result = uniform_cond(False, then_fn, else_fn, 10, 3)
        assert result == 7

    def test_invalid_predicate_type(self):
        """Test that non-boolean predicate raises TypeError."""
        with pytest.raises(TypeError, match="must be boolean scalar"):
            uniform_cond(42, lambda x: x, lambda x: x, 5)

        with pytest.raises(TypeError, match="must be boolean scalar"):
            uniform_cond("true", lambda x: x, lambda x: x, 5)


class TestWhileLoop:
    """Test suite for while_loop primitive."""

    def test_simple_counter(self):
        """Test while_loop with simple counter."""
        result = while_loop(lambda x: x < 10, lambda x: x + 1, 0)
        assert result == 10

    def test_fibonacci(self):
        """Test while_loop computing Fibonacci numbers."""

        def cond_fn(state):
            i, _a, _b = state
            return i < 10

        def body_fn(state):
            i, a, b = state
            return (i + 1, b, a + b)

        i, _a, b = while_loop(cond_fn, body_fn, (0, 0, 1))
        assert i == 10
        assert b == 89  # 11th Fibonacci number (F(11) = 89)

    def test_zero_iterations(self):
        """Test while_loop that doesn't execute body (condition false initially)."""
        result = while_loop(lambda x: x < 5, lambda x: x + 1, 10)
        assert result == 10  # Body never executed

    def test_accumulation(self):
        """Test while_loop accumulating values."""

        def cond_fn(state):
            i, _acc = state
            return i < 5

        def body_fn(state):
            i, acc = state
            return (i + 1, acc + i)

        i, acc = while_loop(cond_fn, body_fn, (0, 0))
        assert i == 5
        assert acc == 10  # 0 + 1 + 2 + 3 + 4

    def test_side_effects_tracking(self):
        """Test that body is called correct number of times."""
        call_count = []

        def cond_fn(x):
            return x < 3

        def body_fn(x):
            call_count.append(x)
            return x + 1

        result = while_loop(cond_fn, body_fn, 0)
        assert result == 3
        assert call_count == [0, 1, 2]  # Body called 3 times


class TestPrimitiveBinding:
    """Test primitive binding API (uniform_cond_p.bind, while_loop_p.bind)."""

    def test_uniform_cond_p_bind(self):
        """Test that uniform_cond_p.bind works."""
        from mplang.dialects.simp import uniform_cond_p

        result = uniform_cond_p.bind(True, 100, 200)
        assert result == 100

        result = uniform_cond_p.bind(False, 100, 200)
        assert result == 200

    def test_while_loop_p_bind(self):
        """Test that while_loop_p.bind works."""
        from mplang.dialects.simp import while_loop_p

        result = while_loop_p.bind(lambda x: x < 5, lambda x: x + 1, 0)
        assert result == 5
