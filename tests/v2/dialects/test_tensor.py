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

"""Tests for tensor dialect ops."""

import numpy as np

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects.tensor import elementwise, reshape, run_jax, transpose
from mplang.v2.runtime.interpreter import InterpObject


def _add_fn(x):
    return x + 2


def test_tensor_run_jax_op_emitted():
    value = InterpObject(np.array(1.0), elt.TensorType(elt.f32, ()))

    def wrapper(x):
        return run_jax(_add_fn, x)

    traced = el.trace(wrapper, value)
    graph = traced.graph

    assert len(graph.operations) == 1
    op = graph.operations[0]
    assert op.opcode == "tensor.run_jax"


def test_tensor_transpose_op():
    """Test transpose operation with ranked tensors."""
    x = InterpObject(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        elt.TensorType(elt.f32, (2, 3)),
    )

    def wrapper(tensor):
        return transpose(tensor, (1, 0))

    traced = el.trace(wrapper, x)
    graph = traced.graph

    # Check transpose op was emitted
    ops = [op for op in graph.operations if op.opcode == "tensor.transpose"]
    assert len(ops) == 1
    op = ops[0]

    # Check perm is in attrs
    assert "perm" in op.attrs
    assert op.attrs["perm"] == (1, 0)

    # Check output shape is transposed
    output_type = op.outputs[0].type
    assert isinstance(output_type, elt.TensorType)
    assert output_type.shape == (3, 2)


def test_tensor_reshape_op():
    """Test reshape operation with ranked tensors."""
    x = InterpObject(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        elt.TensorType(elt.f32, (2, 3)),
    )

    def wrapper(tensor):
        return reshape(tensor, (6,))

    traced = el.trace(wrapper, x)
    graph = traced.graph

    # Check reshape op was emitted
    ops = [op for op in graph.operations if op.opcode == "tensor.reshape"]
    assert len(ops) == 1
    op = ops[0]

    # Check new_shape is in attrs
    assert "new_shape" in op.attrs
    assert op.attrs["new_shape"] == (6,)

    # Check output shape is reshaped
    output_type = op.outputs[0].type
    assert isinstance(output_type, elt.TensorType)
    assert output_type.shape == (6,)


def test_tensor_transpose_with_dynamic_dims():
    """Test transpose works with dynamic dimensions."""
    x_dyn = InterpObject(None, elt.TensorType(elt.f32, (-1, 3)))

    def wrapper(tensor):
        return transpose(tensor, (1, 0))

    traced = el.trace(wrapper, x_dyn)
    graph = traced.graph

    ops = [op for op in graph.operations if op.opcode == "tensor.transpose"]
    assert len(ops) == 1

    # Dynamic dims preserved after transpose
    output_type = ops[0].outputs[0].type
    assert isinstance(output_type, elt.TensorType)
    assert output_type.shape == (3, -1)


def test_tensor_reshape_infers_minus_one():
    """Test reshape infers -1 dimension when input is fully static."""
    x = InterpObject(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        elt.TensorType(elt.f32, (2, 3)),
    )

    def wrapper(tensor):
        # Total size is 6, so (2, -1) should infer to (2, 3)
        return reshape(tensor, (2, -1))

    traced = el.trace(wrapper, x)
    graph = traced.graph

    ops = [op for op in graph.operations if op.opcode == "tensor.reshape"]
    assert len(ops) == 1
    op = ops[0]

    # new_shape in attrs keeps original -1
    assert op.attrs["new_shape"] == (2, -1)

    # But output type should have inferred dimension
    output_type = op.outputs[0].type
    assert isinstance(output_type, elt.TensorType)
    assert output_type.shape == (2, 3)


def test_tensor_reshape_keeps_minus_one_with_dynamic_input():
    """Test reshape preserves -1 when input has dynamic dims."""
    x_dyn = InterpObject(None, elt.TensorType(elt.f32, (-1, 3)))

    def wrapper(tensor):
        return reshape(tensor, (2, -1))

    traced = el.trace(wrapper, x_dyn)
    graph = traced.graph

    ops = [op for op in graph.operations if op.opcode == "tensor.reshape"]
    assert len(ops) == 1

    # Output keeps -1 since we can't infer at trace time
    output_type = ops[0].outputs[0].type
    assert isinstance(output_type, elt.TensorType)
    assert output_type.shape == (2, -1)


def test_tensor_elementwise_supports_kwargs_arguments():
    tensor = InterpObject(
        np.array([1.0, 2.0], dtype=np.float32), elt.TensorType(elt.f32, (2,))
    )
    scalar = InterpObject(np.array(0.5, dtype=np.float32), elt.f32)

    def wrapper(x, bias):
        return elementwise(lambda elem, *, offset: elem + offset, x, offset=bias)

    traced = el.trace(wrapper, tensor, scalar)
    ops = [op for op in traced.graph.operations if op.opcode == "tensor.elementwise"]
    assert len(ops) == 1
    op = ops[0]
    assert len(op.inputs) == 2  # tensor + kwarg scalar
    assert len(op.regions) == 1
    assert len(op.regions[0].inputs) == 2  # region sees both arguments


def test_tensor_elementwise_emits_pytree_outputs():
    tensor = InterpObject(
        np.array([3.0, 4.0], dtype=np.float32), elt.TensorType(elt.f32, (2,))
    )

    def wrapper(x):
        return elementwise(lambda elem: (elem + 1, {"neg": elem - 1}), x)

    traced = el.trace(wrapper, tensor)
    ops = [op for op in traced.graph.operations if op.opcode == "tensor.elementwise"]
    assert len(ops) == 1
    op = ops[0]
    assert len(op.outputs) == 2
    assert all(isinstance(val.type, elt.TensorType) for val in op.outputs)
    assert len(traced.graph.outputs) == 2  # wrapper returns two tensors
