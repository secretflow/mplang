"""Tensor Runtime Implementation.

Implements execution logic for Tensor primitives.
"""

from typing import Any

import numpy as np

from mplang2.dialects import tensor
from mplang2.edsl.graph import Operation
from mplang2.edsl.interpreter import Interpreter, interpret


@tensor.constant_p.def_impl
def constant_impl(interpreter: Interpreter, op: Operation) -> Any:
    return np.array(op.attrs["data"])


@tensor.concat_p.def_impl
def concat_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    axis = op.attrs.get("axis", 0)
    return np.concatenate(args, axis=axis)


@tensor.elementwise_p.def_impl
def elementwise_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Execute elementwise operation by iterating over tensor elements."""
    # args are the input tensors (or scalars)
    # op.regions[0] is the scalar computation graph

    # 1. Determine shape and broadcast
    # For simplicity, assume all tensor args have same shape (validated by tracer)
    # and scalars are broadcasted.
    shape = None
    for arg in args:
        if hasattr(arg, "shape") and arg.shape:
            shape = arg.shape
            break

    if shape is None:
        # All scalars? Should not happen for elementwise usually, but handle it
        shape = ()

    # 2. Construct output container
    # We need to know the output type/dtype.
    # op.outputs[0].type should give us a hint, but here we are in runtime.
    # Let's just use a list or numpy array of objects for flexibility.
    # Since we might be mixing types (e.g. Encrypted objects), object array is safest.
    results = np.empty(shape, dtype=object)

    # 3. Iterate and execute
    # Use np.ndindex for multi-dimensional iteration
    subgraph = op.regions[0]

    if shape == ():
        # Scalar case
        scalar_inputs = {}
        for inp_val, arg in zip(subgraph.inputs, args, strict=True):
            scalar_inputs[inp_val] = arg
        return interpret(subgraph, scalar_inputs, interpreter)

    for index in np.ndindex(shape):
        # Prepare inputs for this element
        scalar_inputs = {}
        for inp_val, arg in zip(subgraph.inputs, args, strict=True):
            if hasattr(arg, "shape") and arg.shape == shape:
                # Tensor argument: pick element
                scalar_inputs[inp_val] = arg[index]
            else:
                # Scalar/Broadcast argument: use as is
                scalar_inputs[inp_val] = arg

        # Recursive execution
        scalar_out = interpret(subgraph, scalar_inputs, interpreter)

        results[index] = scalar_out

    return results


@tensor.run_jax_p.def_impl
def run_jax_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Execute JAX function."""
    fn = op.attrs.get("fn")
    if fn is None:
        raise NotImplementedError(
            "run_jax execution requires 'fn' attribute (simulation mode)"
        )

    # args are the runtime values for the dynamic inputs
    return fn(list(args))


@tensor.slice_p.def_impl
def slice_impl(interpreter: Interpreter, op: Operation, tensor_data: Any) -> Any:
    starts = op.attrs["starts"]
    ends = op.attrs["ends"]
    strides = op.attrs.get("strides")

    slices = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        stride = strides[i] if strides else 1
        slices.append(slice(start, end, stride))

    return tensor_data[tuple(slices)]


@tensor.reshape_p.def_impl
def reshape_impl(interpreter: Interpreter, op: Operation, tensor_data: Any) -> Any:
    new_shape = op.attrs["new_shape"]
    return tensor_data.reshape(new_shape)
