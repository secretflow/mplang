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

from mplang.v2.edsl.context import get_current_context
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
