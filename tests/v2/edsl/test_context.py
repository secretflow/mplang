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

"""Tests for mplang.edsl.context module."""

import numpy as np
import pytest

import mplang.v2 as mp
import mplang.v2.edsl.context as ctx_mod
from mplang.v2.dialects import simp
from mplang.v2.dialects.simp import make_simulator
from mplang.v2.edsl.context import (
    find_interpreter,
    get_current_context,
    is_tracing,
)
from mplang.v2.edsl.primitive import Primitive
from mplang.v2.edsl.tracer import TraceObject, Tracer
from mplang.v2.edsl.typing import Tensor, f32
from mplang.v2.runtime.interpreter import InterpObject


class TestContextManager:
    """Test Context as context manager."""

    def test_context_with_statement(self):
        """Test using context directly with 'with' statement."""
        tracer = Tracer()
        assert get_current_context() is None

        with tracer:
            assert get_current_context() is tracer

        assert get_current_context() is None

    def test_context_exception_safety(self):
        """Test that context is cleaned up even when exception occurs."""
        tracer = Tracer()
        assert get_current_context() is None

        with pytest.raises(ValueError):
            with tracer:
                assert get_current_context() is tracer
                raise ValueError("Test exception")

        # Context should still be popped
        assert get_current_context() is None

    def test_context_nested(self):
        """Test nested contexts."""
        tracer1 = Tracer()
        tracer2 = Tracer()
        assert get_current_context() is None

        with tracer1:
            assert get_current_context() is tracer1

            with tracer2:
                assert get_current_context() is tracer2

            assert get_current_context() is tracer1

        assert get_current_context() is None

    def test_context_with_operations(self):
        """Test context with actual primitive operations."""
        add_p = Primitive("add")

        @add_p.def_abstract_eval
        def add_abstract(x_type, y_type):
            return x_type

        tracer = Tracer()
        x = InterpObject(np.array([1.0, 2.0]), Tensor[f32, (2,)])
        y = InterpObject(np.array([3.0, 4.0]), Tensor[f32, (2,)])

        with tracer:
            z = add_p.bind(x, y)
            assert isinstance(z, TraceObject)
            assert len(tracer.graph.operations) == 1

        # Context properly cleaned up
        assert get_current_context() is None


class TestContextStack:
    """Test context stack management."""

    def setup_method(self):
        """Clean context stack before each test."""
        ctx_mod._context_stack.clear()

    def teardown_method(self):
        """Clean context stack after each test."""
        ctx_mod._context_stack.clear()

    def test_tracer_interaction(self):
        """Verify that Tracers sit on top of Interpreter in the stack."""
        sim = make_simulator(3)
        with sim:
            # stack=[sim] (Interpreter)
            assert find_interpreter() is sim
            assert not is_tracing()

            @mp.compile
            def job():
                # Inside job: stack=[sim, Tracer]
                # find_interpreter should still find sim
                found_interp = find_interpreter()
                curr = get_current_context()

                assert found_interp is sim
                assert curr is not sim  # curr is a Tracer
                assert is_tracing()

                mp.device("P0")
                return simp.constant((0,), 1)

            # Note: @mp.compile executes tracing immediately.

    def test_compile_auto_context(self):
        """Test mp.compile uses the context provided in args or set_root_context."""
        sim = make_simulator(3)

        # Case 1: set_root_context
        mp.set_root_context(sim, force=True)

        captured_interp = None

        @mp.compile
        def job1():
            nonlocal captured_interp
            captured_interp = find_interpreter()
            return simp.constant((0,), 1)

        # check side effect immediately
        assert captured_interp is sim

        # Case 2: context arg to compile
        sim2 = make_simulator(3)
        captured_cluster = None

        def func():
            nonlocal captured_cluster
            from mplang.v2.libs.device.api import _resolve_cluster

            captured_cluster = _resolve_cluster()
            return simp.constant((0,), 1)

        mp.compile(func, context=sim2)
        # compile() executes trace(). sim2's cluster should be resolved.
        assert captured_cluster is sim2._cluster_spec
