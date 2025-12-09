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

"""Tests for func dialect: function definition and calls."""

import numpy as np
import pytest

from mplang.v2.edsl.tracer import TracedFunction, trace
from mplang.v2.edsl.typing import TensorType, f32
from mplang.v2.runtime.interpreter import InterpObject, Interpreter


class TestMakeGraph:
    """Test func.func primitive for tracing arbitrary Python functions."""

    @pytest.fixture
    def interpreter(self):
        """Create an interpreter context."""
        return Interpreter()

    @pytest.fixture
    def sample_inputs(self, interpreter):
        """Create sample tensor inputs."""
        x_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        x_obj = InterpObject(x_val, TensorType(f32, (3,)), interpreter)
        y_obj = InterpObject(y_val, TensorType(f32, (3,)), interpreter)

        return x_obj, y_obj

    def test_simple_function_args_only(self, sample_inputs):
        """Test tracing a simple function with positional args."""
        x_obj, y_obj = sample_inputs

        def add(x, y):
            # Just return tuple for now (no actual ops)
            return x, y

        traced = trace(add, x_obj, y_obj)

        assert isinstance(traced, TracedFunction)
        assert traced.name == "add"
        # 2 variables (x, y), no constants
        assert len(traced.in_var_pos) == 2
        assert len(traced.in_imms) == 0
        assert traced.in_var_pos == [0, 1]
        # 2 variable outputs (x, y), no constants
        assert len(traced.out_var_pos) == 2
        assert len(traced.out_imms) == 0
        assert len(traced.graph.inputs) == 2
        assert len(traced.graph.outputs) == 2

    def test_same_object_multiple_params(self, sample_inputs):
        """Test that same object passed multiple times creates independent graph inputs.

        This is critical for correct semantics: trace(fn, x, x) should create
        two independent graph inputs, not share the same one.
        """
        x_obj, _ = sample_inputs

        def use_twice(a, b):
            # Even though a and b are the same Python object,
            # they should map to different graph inputs
            return a, b

        traced = trace(use_twice, x_obj, x_obj)

        assert traced.name == "use_twice"
        # Should have 2 independent inputs, not 1
        assert len(traced.graph.inputs) == 2
        assert len(traced.in_var_pos) == 2
        # The two inputs should be distinct graph Values
        assert traced.graph.inputs[0] is not traced.graph.inputs[1]
        assert traced.graph.inputs[0].name == "%arg0"
        assert traced.graph.inputs[1].name == "%arg1"
        # Should have 2 outputs (a, b)
        assert len(traced.graph.outputs) == 2
        # Outputs should reference different inputs
        assert traced.graph.outputs[0] is traced.graph.inputs[0]
        assert traced.graph.outputs[1] is traced.graph.inputs[1]

    def test_same_object_as_param_and_capture(self, sample_inputs):
        """Test object used both as param and captured creates independent inputs.

        When the same object is both passed as a parameter AND captured from
        closure, they should be treated independently.
        """
        x_obj, _ = sample_inputs
        captured = x_obj  # Same object as param

        def use_as_both(a):
            # a is a param, captured is from closure (same Python object)
            return a, captured

        traced = trace(use_as_both, x_obj)

        # Should have 2 inputs: 1 param + 1 capture
        assert len(traced.graph.inputs) == 2
        assert len(traced.in_var_pos) == 1  # Only 1 explicit param
        assert len(traced.captured) == 1  # 1 captured variable
        # Both inputs should be distinct
        assert traced.graph.inputs[0] is not traced.graph.inputs[1]

    def test_function_with_kwargs(self, sample_inputs):
        """Test tracing function with keyword arguments."""
        x_obj, _ = sample_inputs

        def scale(x, *, factor):
            # Just return for now
            return x

        traced = trace(scale, x_obj, factor=2.0)

        assert traced.name == "scale"
        # 1 variable (x), 1 constant (factor=2.0)
        assert len(traced.in_var_pos) == 1
        assert len(traced.in_imms) == 1
        assert traced.in_var_pos == [0]  # x at position 0
        assert traced.in_imms == [2.0]  # factor is constant
        # 1 variable output (x)
        assert len(traced.out_var_pos) == 1
        assert len(traced.out_imms) == 0

    def test_tuple_input_output(self, sample_inputs):
        """Test function with tuple inputs and outputs."""
        x_obj, y_obj = sample_inputs

        def swap(data):
            x, y = data
            return (y, x)

        traced = trace(swap, (x_obj, y_obj))

        assert traced.name == "swap"
        # Input: ((x, y),) flattened = [x, y], both variables
        assert len(traced.in_var_pos) == 2
        assert len(traced.in_imms) == 0
        # Output: (y, x) flattened = [y, x], both variables
        assert len(traced.out_var_pos) == 2
        assert len(traced.out_imms) == 0

    def test_dict_output(self, sample_inputs):
        """Test function with dict output."""
        x_obj, y_obj = sample_inputs

        def make_dict(x, y):
            return {"first": x, "second": y}

        traced = trace(make_dict, x_obj, y_obj)

        assert traced.name == "make_dict"
        # 2 variables (x, y), no constants
        assert len(traced.in_var_pos) == 2
        assert len(traced.in_imms) == 0
        # Dict with 2 variable values
        assert len(traced.out_var_pos) == 2
        assert len(traced.out_imms) == 0

    def test_nested_pytree(self, sample_inputs):
        """Test function with nested PyTree structure."""
        x_obj, y_obj = sample_inputs

        def complex_fn(data, config):
            x, y = data
            threshold = config["threshold"]
            # Return nested structure
            return {"result": (x, y), "metadata": threshold}

        config = {"threshold": 0.5}
        traced = trace(complex_fn, (x_obj, y_obj), config)

        assert traced.name == "complex_fn"
        # Input: ((x, y), {"threshold": 0.5}) flattened = [x, y, 0.5]
        # x, y are variables at pos 0, 1; 0.5 is constant at pos 2
        assert len(traced.in_var_pos) == 2
        assert traced.in_var_pos == [0, 1]
        assert len(traced.in_imms) == 1
        assert traced.in_imms == [0.5]
        # Output: {"result": (x, y), "metadata": 0.5}
        # JAX flattens dicts in sorted key order: "metadata" < "result"
        # So flattened = [0.5, x, y] (metadata first, then result tuple)
        # 0.5 is constant at pos 0; x, y are variables at pos 1, 2
        assert len(traced.out_var_pos) == 2
        assert traced.out_var_pos == [1, 2]
        assert len(traced.out_imms) == 1
        assert traced.out_imms == [0.5]

    def test_input_output_tree_preservation(self, sample_inputs):
        """Test that input/output tree structures are preserved."""
        x_obj, y_obj = sample_inputs

        def identity(x, y):
            return {"x": x, "y": y}

        traced = trace(identity, x_obj, y_obj)

        # Check that we can reconstruct the structures
        assert traced.in_tree is not None
        assert traced.out_tree is not None

        # Verify metadata is stored
        assert traced.graph is not None
        assert traced.name == "identity"
        # 2 variables in, 2 variables out
        assert len(traced.in_var_pos) == 2
        assert len(traced.out_var_pos) == 2

    def test_single_output(self, sample_inputs):
        """Test function with single output (not tuple)."""
        x_obj, _ = sample_inputs

        def passthrough(x):
            return x

        traced = trace(passthrough, x_obj)

        assert traced.name == "passthrough"
        # 1 variable in, 1 variable out
        assert len(traced.in_var_pos) == 1
        assert len(traced.out_var_pos) == 1
        assert len(traced.in_imms) == 0
        assert len(traced.out_imms) == 0

    def test_no_args_function(self):
        """Test function with no arguments."""

        def constant():
            # Note: This will create a graph with no inputs
            return 42

        traced = trace(constant)

        assert traced.name == "constant"
        # No variables in
        assert len(traced.in_var_pos) == 0
        assert len(traced.in_imms) == 0
        # Output is a constant (42)
        assert len(traced.out_var_pos) == 0
        assert len(traced.out_imms) == 1
        assert traced.out_imms == [42]

    def test_capture_only_output(self, sample_inputs):
        """Returning a captured InterpObject should register it as capture input."""
        x_obj, y_obj = sample_inputs
        captured = y_obj

        def return_capture(_):
            return captured

        traced = trace(return_capture, x_obj)

        assert traced.captured == [captured]
        assert len(traced.graph.inputs) == 2  # arg + capture
        assert traced.graph.inputs[0].type == x_obj.type
        assert traced.graph.inputs[1].type == captured.type
        assert traced.out_var_pos == [0]
        assert traced.out_imms == []

    def test_mixed_param_and_capture_outputs(self, sample_inputs):
        """Outputs mixing parameters and captures should preserve ordering."""
        x_obj, y_obj = sample_inputs
        captured = y_obj

        def mixed_outputs(x):
            return x, captured

        traced = trace(mixed_outputs, x_obj)

        assert traced.captured == [captured]
        assert len(traced.graph.inputs) == 2
        assert [inp.type for inp in traced.graph.inputs] == [
            x_obj.type,
            captured.type,
        ]
        assert traced.out_var_pos == [0, 1]
        assert traced.out_imms == []

    def test_duplicate_captures_deduped(self, sample_inputs, interpreter):
        """Capturing the same object multiple times should only add one capture."""
        x_obj, _ = sample_inputs
        captured = InterpObject(
            np.array([9.0, 9.0, 9.0], dtype=np.float32),
            TensorType(f32, (3,)),
            interpreter,
        )

        def capture_twice(x):
            del x
            return captured, captured

        traced = trace(capture_twice, x_obj)

        assert traced.captured == [captured]
        assert len(traced.graph.inputs) == 2  # arg + single capture
        assert traced.out_var_pos == [0, 1]
        assert traced.out_imms == []
