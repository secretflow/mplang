"""Tensor Runtime Implementation.

Implements execution logic for Tensor primitives.
"""

import base64
from typing import Any

import jax
import jax.extend as jxt
import jax.numpy as jnp
import numpy as np
from jax._src import compiler

import mplang2.edsl.typing as elt
from mplang2.dialects import tensor
from mplang2.edsl.graph import Operation
from mplang2.edsl.interpreter import Interpreter, interpret


@tensor.constant_p.def_impl
def constant_impl(interpreter: Interpreter, op: Operation) -> Any:
    # Recover dtype and shape from IR type
    output_type = op.outputs[0].type
    if not isinstance(output_type, elt.TensorType):
        raise TypeError(f"Expected TensorType, got {output_type}")

    scalar_str = str(output_type.element_type)
    dtype = tensor._SCALAR_TO_NP_DTYPE.get(scalar_str)
    if dtype is None:
        raise ValueError(f"Unsupported scalar type {output_type.element_type}")

    shape = output_type.shape

    # Decode data
    data_b64 = op.attrs["value_b64"]
    data_bytes = base64.b64decode(data_b64)

    # Create array
    return np.frombuffer(data_bytes, dtype=dtype).reshape(shape).copy()


@tensor.concat_p.def_impl
def concat_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    axis = op.attrs.get("axis", 0)
    return np.concatenate(args, axis=axis)


@tensor.elementwise_p.def_impl
def elementwise_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Execute elementwise operation by iterating over tensor elements."""
    # args are the input tensors (or scalars)
    # op.regions[0] is the scalar computation graph

    # 1. Determine shape from IR types and runtime args
    shape = ()
    for i, inp_val in enumerate(op.inputs):
        if isinstance(inp_val.type, elt.TensorType):
            if inp_val.type.shape != ():
                # Found a non-scalar tensor input. Use its runtime shape.
                # We assume the tracer ensured all non-scalar tensors have compatible shapes.
                arg = args[i]
                if hasattr(arg, "shape"):
                    shape = arg.shape
                break

    # 2. Construct output container
    # We need to know the output type/dtype.
    # op.outputs[0].type should give us a hint, but here we are in runtime.
    # Let's just use a list or numpy array of objects for flexibility.
    # Since we might be mixing types (e.g. Encrypted objects), object array is safest.
    num_outputs = len(op.outputs)
    if num_outputs > 1:
        results = [np.empty(shape, dtype=object) for _ in range(num_outputs)]
    else:
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
        for i, (inp_val, arg) in enumerate(zip(subgraph.inputs, args, strict=True)):
            outer_val = op.inputs[i]
            # Check if this argument should be iterated based on OUTER IR type
            if (
                isinstance(outer_val.type, elt.TensorType)
                and outer_val.type.shape != ()
            ):
                # Tensor argument: pick element
                scalar_inputs[inp_val] = arg[index]
            else:
                # Scalar/Broadcast argument: use as is
                scalar_inputs[inp_val] = arg

        # Recursive execution
        scalar_out = interpret(subgraph, scalar_inputs, interpreter)

        if num_outputs > 1:
            for i, val in enumerate(scalar_out):
                results[i][index] = val
        else:
            results[index] = scalar_out

    return results


def _enforce_jax_types(args: tuple[Any, ...], op_inputs: list[Any]) -> list[Any]:
    """Ensure runtime values match the IR types expected by JAX/StableHLO.

    This handles implicit casting from Python types (int, float) to strict Numpy types
    required by the compiled executable.
    """
    casted_args = []
    for i, arg in enumerate(args):
        if i < len(op_inputs):
            input_type = op_inputs[i].type
            if isinstance(input_type, elt.TensorType):
                scalar_str = str(input_type.element_type)
                dtype = tensor._SCALAR_TO_NP_DTYPE.get(scalar_str)
                if dtype is not None:
                    # Only cast if strictly necessary to avoid overhead
                    # np.asarray handles scalar->array and dtype conversion
                    casted_args.append(np.asarray(arg, dtype=dtype))
                else:
                    casted_args.append(arg)
            else:
                casted_args.append(arg)
        else:
            casted_args.append(arg)
    return casted_args


@tensor.run_jax_p.def_impl
def run_jax_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Execute JAX function."""
    # Execute via StableHLO
    stablehlo_code = op.attrs.get("stablehlo_code")
    if stablehlo_code is None:
        raise NotImplementedError(
            "run_jax execution requires 'stablehlo_code' attribute"
        )

    # Compile StableHLO
    # TODO: Cache compilation based on stablehlo_code hash
    client = jxt.backend.get_backend()
    compile_options = compiler.get_compile_options(num_replicas=1, num_partitions=1)
    try:
        compiled = client.compile(stablehlo_code, compile_options)
    except Exception as e:
        raise RuntimeError(f"StableHLO compile failed: {e}") from e

    # Cast inputs to expected types (Boundary Type Guard)
    # This allows users to pass Python ints/floats to functions expecting f32/i32
    jax_input_args = _enforce_jax_types(args, op.inputs)

    # Handle JAX's unused parameter elimination via arg_keep_map
    arg_keep_map = op.attrs.get("arg_keep_map")
    if arg_keep_map is not None:
        # Filter out arguments that were eliminated by JAX during compilation
        jax_input_args = [jax_input_args[i] for i in arg_keep_map]

    # Convert args to JAX arrays
    jax_args = [jax.device_put(jnp.asarray(arg)) for arg in jax_input_args]

    try:
        result = compiled.execute_sharded(jax_args)
        arrays = result.disassemble_into_single_device_arrays()
        flat: list[Any] = []
        for lst in arrays:
            if isinstance(lst, list) and len(lst) == 1:
                flat.append(np.asarray(lst[0]))
            else:
                flat.extend(np.asarray(a) for a in lst)

        # If single output, return it directly (but run_jax usually returns list of vars)
        # The primitive expects a list of results matching outputs.
        # If op has 1 output, flat should have 1 element.
        if len(op.outputs) == 1 and len(flat) == 1:
            return flat[0]
        return flat
    except Exception as e:
        raise RuntimeError(f"StableHLO execute failed: {e}") from e


@tensor.gather_p.def_impl
def gather_impl(
    interpreter: Interpreter, op: Operation, operand: Any, indices: Any
) -> Any:
    axis = op.attrs.get("axis", 0)
    # Ensure indices are integers (they might be JAX arrays or numpy arrays)
    if hasattr(indices, "astype"):
        indices = indices.astype(int)
    return np.take(operand, indices, axis=axis)


@tensor.slice_p.def_impl
def slice_impl(interpreter: Interpreter, op: Operation, operand: Any) -> Any:
    starts = op.attrs["starts"]
    ends = op.attrs["ends"]
    strides = op.attrs.get("strides")

    slices = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        stride = strides[i] if strides else 1
        slices.append(slice(start, end, stride))

    # If operand is numpy array, we can slice directly
    # If operand has more dimensions than slices provided, we assume full slice for remaining
    if hasattr(operand, "ndim") and len(slices) < operand.ndim:
        slices.append(Ellipsis)

    return operand[tuple(slices)]


@tensor.reshape_p.def_impl
def reshape_impl(interpreter: Interpreter, op: Operation, tensor_data: Any) -> Any:
    new_shape = op.attrs["new_shape"]
    return tensor_data.reshape(new_shape)
