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

"""Tensor Runtime Implementation.

Implements execution logic for Tensor primitives.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from typing import Any, ClassVar, cast

import jax
import jax.extend as jxt
import jax.numpy as jnp
import numpy as np
from jax._src import compiler
from numpy.typing import ArrayLike

import mplang.v2.edsl.typing as elt
from mplang.v2.dialects import dtypes, tensor
from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import Interpreter
from mplang.v2.runtime.value import Value, WrapValue

# =============================================================================
# TensorValue Wrapper
# =============================================================================


@serde.register_class
class TensorValue(WrapValue[Any]):
    """Runtime value wrapping a numpy array or JAX array.

    Handles numpy arrays, JAX arrays, and other numpy-like objects via duck typing.
    Serialization uses base64-encoded raw bytes for efficiency.

    Note: This is for numeric tensors only. Object dtype arrays (containing
    encrypted values, etc.) should NOT be wrapped - they are handled separately
    by elementwise_impl which returns raw np.ndarray(dtype=object).
    """

    _serde_kind: ClassVar[str] = "tensor_impl.TensorValue"

    # Expose common array properties for convenience
    @property
    def shape(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self._data.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self._data.dtype)  # type: ignore[no-any-return]

    @property
    def ndim(self) -> int:
        return cast(int, self._data.ndim)

    def __getitem__(self, key: Any) -> Any:
        """Allow indexing into the underlying array."""
        return self._data[key]

    # =========== Wrap/Unwrap ===========

    def _convert(self, data: Any) -> Any:
        """Convert input data to numpy array or JAX array."""
        if isinstance(data, TensorValue):
            return data._data

        # Allow JAX arrays to pass through
        if hasattr(data, "__jax_array__"):
            return data

        # Handle other numpy-like objects via np.asarray
        if (
            hasattr(data, "__module__")
            and data.__module__ is not None
            and "jax" in data.__module__
        ):
            return data

        if isinstance(data, np.ndarray):
            return data
        # Try converting other array-like objects
        return np.asarray(data)

    def unwrap(self) -> np.ndarray:
        """Get the underlying data as a numpy array.

        If the data is a JAX array, it will be transferred to host.
        """
        return np.asarray(self._data)

    def as_jax(self) -> Any:
        """Get the underlying data as a JAX array.

        If the data is a numpy array, it will be transferred to device.
        """
        if hasattr(self._data, "__jax_array__"):
            return self._data

        # Handle object arrays that might contain numbers (e.g. from elementwise)
        if isinstance(self._data, np.ndarray) and self._data.dtype == object:
            try:
                # Attempt to convert to numeric numpy array first
                # This handles cases where elementwise returned object array of numbers
                val_numeric = np.array(self._data.tolist())
                if val_numeric.dtype != object:
                    return jax.device_put(jnp.asarray(val_numeric))
            except Exception:
                # If conversion fails, proceed with original (which will likely fail in jax)
                pass

        return jax.device_put(jnp.asarray(self._data))

    # =========== Serialization ===========

    def to_json(self) -> dict[str, Any]:
        # Ensure we have numpy data for serialization
        # This forces synchronization if data is on device
        data_np = np.asarray(self._data)

        # Handle object dtype arrays - serialize element by element
        if data_np.dtype == np.object_:
            return {
                "kind": "object",
                "shape": list(data_np.shape),
                "items": [serde.to_json(item) for item in data_np.flat],
            }
        # Standard numeric arrays - use raw bytes
        return {
            "kind": "numeric",
            "dtype": str(data_np.dtype),
            "shape": list(data_np.shape),
            "data": base64.b64encode(data_np.tobytes()).decode("ascii"),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TensorValue:
        kind = data.get("kind", "numeric")
        shape = tuple(data["shape"])

        if kind == "object":
            items = [serde.from_json(item) for item in data["items"]]
            arr = np.empty(len(items), dtype=object)
            for i, item in enumerate(items):
                arr[i] = item
            return cls(arr.reshape(shape))
        else:
            arr = np.frombuffer(
                base64.b64decode(data["data"]),
                dtype=np.dtype(data["dtype"]),
            )
            return cls(arr.reshape(shape).copy())


# Module-level helpers for convenience (delegate to class methods)
def _wrap(val: ArrayLike | TensorValue) -> TensorValue:
    """Wrap an array-like value into TensorValue."""
    return TensorValue.wrap(val)


def _unwrap(val: TensorValue | np.ndarray | ArrayLike) -> np.ndarray:
    """Unwrap TensorValue to np.ndarray, also accepts raw arrays."""
    if isinstance(val, TensorValue):
        return val.unwrap()
    if isinstance(val, np.ndarray):
        return val
    # Handle JAX arrays
    if hasattr(val, "__jax_array__"):
        return np.asarray(val)
    return np.asarray(val)


# _ensure_tensor_value removed - callers should unwrap InterpObject before calling impls


# =============================================================================
# Tensor Primitive Implementations
# =============================================================================


@tensor.constant_p.def_impl
def constant_impl(interpreter: Interpreter, op: Operation) -> TensorValue:
    # Recover dtype and shape from IR type
    output_type = op.outputs[0].type
    if not isinstance(output_type, elt.TensorType):
        raise TypeError(f"Expected TensorType, got {output_type}")

    dtype = dtypes.to_jax(cast(elt.ScalarType, output_type.element_type))
    if dtype is None:
        raise ValueError(f"Unsupported scalar type {output_type.element_type}")

    shape = output_type.shape

    # Decode data
    data_b64 = op.attrs["value_b64"]
    data_bytes = base64.b64decode(data_b64)

    # Create array
    arr = np.frombuffer(data_bytes, dtype=cast(Any, dtype)).reshape(shape).copy()
    return _wrap(arr)


@tensor.concat_p.def_impl
def concat_impl(
    interpreter: Interpreter, op: Operation, *args: TensorValue
) -> TensorValue:
    axis = op.attrs.get("axis", 0)
    unwrapped = [_unwrap(a) for a in args]
    return _wrap(np.concatenate(unwrapped, axis=axis))


@tensor.elementwise_p.def_impl
def elementwise_impl(interpreter: Interpreter, op: Operation, *args: Value) -> Any:
    """Execute elementwise operation by iterating over tensor elements.

    Note: args typed as Value (base class) because elementwise handles polymorphic
    inputs - TensorValue for numeric tensors, or np.ndarray with dtype=object
    containing encrypted values (BFVValue, etc.) that are processed element-wise.
    """
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
    results: Any
    if num_outputs > 1:
        results = [np.empty(shape, dtype=object) for _ in range(num_outputs)]
    else:
        results = np.empty(shape, dtype=object)

    # 3. Iterate and execute
    # Use np.ndindex for multi-dimensional iteration
    subgraph = op.regions[0]

    if shape == ():
        # Scalar case - return first element from result list
        result = interpreter.evaluate_graph(subgraph, list(args))
        return result[0] if len(result) == 1 else result

    for index in np.ndindex(shape):
        # Prepare inputs for this element (list ordered by subgraph.inputs)
        scalar_inputs = []
        for i, arg in enumerate(args):
            outer_val = op.inputs[i]
            # Check if this argument should be iterated based on OUTER IR type
            if (
                isinstance(outer_val.type, elt.TensorType)
                and outer_val.type.shape != ()
            ):
                # Tensor argument: pick element (arg is array-like at runtime)
                # Wrap scalar in TensorValue to maintain Value-only contract
                elem = cast(Any, arg)[index]
                if isinstance(elem, Value):
                    scalar_inputs.append(elem)
                else:
                    scalar_inputs.append(_wrap(np.array(elem)))  # type: ignore[index]
            else:
                # Scalar/Broadcast argument: use as is
                # Ensure it is wrapped (it should be, but double check)
                if not isinstance(arg, Value):
                    scalar_inputs.append(_wrap(np.array(arg)))
                else:
                    scalar_inputs.append(arg)

        # Recursive execution
        scalar_out_list = interpreter.evaluate_graph(subgraph, scalar_inputs)
        scalar_out = (
            scalar_out_list[0] if len(scalar_out_list) == 1 else scalar_out_list
        )

        # Unwrap result if it's a TensorValue (to store in numpy array)
        # We store raw values in the object array for now, but will wrap the final array
        if isinstance(scalar_out, TensorValue):
            scalar_out = scalar_out.unwrap()
            if scalar_out.shape == ():
                scalar_out = scalar_out.item()

        if num_outputs > 1:
            for i, val in enumerate(scalar_out):
                results[i][index] = val
        else:
            results[index] = scalar_out

    # Wrap results in TensorValue if possible
    if num_outputs > 1:
        return [_wrap(res) for res in results]
    else:
        return _wrap(results)


# Global cache for compiled StableHLO executables
_STABLEHLO_CACHE: dict[str, Any] = {}


@tensor.run_jax_p.def_impl
def run_jax_impl(
    interpreter: Interpreter, op: Operation, *args: TensorValue
) -> TensorValue | list[TensorValue]:
    """Execute JAX function."""
    t0 = time.time()

    # Execute via StableHLO
    stablehlo_code = op.attrs.get("stablehlo_code")
    if stablehlo_code is None:
        raise NotImplementedError(
            "run_jax execution requires 'stablehlo_code' attribute"
        )

    # Compile StableHLO
    client = jxt.backend.get_backend()

    # Use SHA256 of code as cache key for stability across runs
    # Note: We assume compile_options are constant (num_replicas=1, num_partitions=1)
    code_hash = hashlib.sha256(stablehlo_code.encode("utf-8")).hexdigest()

    if code_hash in _STABLEHLO_CACHE:
        compiled = _STABLEHLO_CACHE[code_hash]
    else:
        compile_options = compiler.get_compile_options(num_replicas=1, num_partitions=1)

        # Try disk cache
        cache_dir = interpreter.root_dir / "cache" / "jax"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / f"{code_hash}.pjrt")
        loaded_from_disk = False

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    serialized = f.read()
                compiled = client.deserialize_executable(
                    serialized, client.devices(), compile_options
                )
                loaded_from_disk = True
                # print(f"[JAX] Loaded compiled executable from {cache_path}")
            except Exception as e:
                print(f"[JAX] Failed to load from disk cache: {e}")

        if not loaded_from_disk:
            try:
                compiled = client.compile_and_load(
                    stablehlo_code, client.devices(), compile_options
                )
                # Save to disk
                try:
                    # Directory creation handled above
                    with open(cache_path, "wb") as f:
                        f.write(client.serialize_executable(compiled))
                    # print(f"[JAX] Saved compiled executable to {cache_path}")
                except Exception as e:
                    print(f"[JAX] Failed to save to disk cache: {e}")
            except Exception as e:
                raise RuntimeError(f"StableHLO compile failed: {e}") from e

        _STABLEHLO_CACHE[code_hash] = compiled

    # Cast inputs to expected types (Boundary Type Guard)
    # This allows users to pass Python ints/floats to functions expecting f32/i32
    t1 = time.time()

    jax_input_args = []
    for i, arg in enumerate(args):
        # arg is TensorValue
        if i < len(op.inputs):
            input_type = op.inputs[i].type
            # Check if we need casting
            if isinstance(input_type, elt.TensorType):
                dtype = dtypes.to_jax(cast(elt.ScalarType, input_type.element_type))
                # Get as JAX array
                if isinstance(arg, TensorValue):
                    val = arg.as_jax()
                else:
                    val = jnp.asarray(arg)

                if (
                    dtype is not None
                    and isinstance(val, (jnp.ndarray, np.ndarray))
                    and val.dtype != dtype
                ):
                    val = val.astype(dtype)
                jax_input_args.append(val)
            else:
                if isinstance(arg, TensorValue):
                    jax_input_args.append(arg.as_jax())
                else:
                    jax_input_args.append(jnp.asarray(arg))
        else:
            if isinstance(arg, TensorValue):
                jax_input_args.append(arg.as_jax())
            else:
                jax_input_args.append(jnp.asarray(arg))

    # Handle JAX's unused parameter elimination via arg_keep_map
    arg_keep_map = op.attrs.get("arg_keep_map")
    if arg_keep_map is not None:
        # Filter out arguments that were eliminated by JAX during compilation
        jax_input_args = [jax_input_args[i] for i in arg_keep_map]

    # Convert args to JAX arrays
    t2 = time.time()
    # jax_input_args are already JAX arrays (or will be handled by execute_sharded if not)
    jax_args = jax_input_args

    try:
        t3 = time.time()
        result = compiled.execute_sharded(jax_args)
        t4 = time.time()
        arrays = result.disassemble_into_single_device_arrays()
        flat: list[TensorValue] = []
        for lst in arrays:
            if isinstance(lst, list) and len(lst) == 1:
                # Wrap JAX array directly, avoiding np.asarray
                flat.append(_wrap(lst[0]))
            else:
                flat.extend(_wrap(a) for a in lst)
        t5 = time.time()

        if interpreter.tracer:
            p = interpreter.tracer
            p.log_custom_event("JAX Compile/Cache", t0, t1, cat="jax")
            p.log_custom_event("JAX Prep", t1, t2, cat="jax")
            p.log_custom_event("JAX Transfer In", t2, t3, cat="jax")
            p.log_custom_event("JAX Exec", t3, t4, cat="jax")
            p.log_custom_event("JAX Transfer Out", t4, t5, cat="jax")

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
    interpreter: Interpreter, op: Operation, operand: TensorValue, indices: TensorValue
) -> TensorValue:
    axis = op.attrs.get("axis", 0)
    operand_arr = _unwrap(operand)
    indices_arr = _unwrap(indices)
    # Ensure indices are integers (they might be JAX arrays or numpy arrays)
    if hasattr(indices_arr, "astype"):
        indices_arr = indices_arr.astype(int)
    return _wrap(np.take(operand_arr, indices_arr, axis=axis))


@tensor.slice_p.def_impl
def slice_impl(
    interpreter: Interpreter, op: Operation, operand: TensorValue
) -> TensorValue:
    starts = op.attrs["starts"]
    ends = op.attrs["ends"]
    strides = op.attrs.get("strides")

    slices: list[Any] = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        stride = strides[i] if strides else 1
        slices.append(slice(start, end, stride))

    operand_arr = _unwrap(operand)
    # If operand is numpy array, we can slice directly
    # If operand has more dimensions than slices provided, we assume full slice for remaining
    if len(slices) < operand_arr.ndim:
        slices.append(Ellipsis)

    return _wrap(operand_arr[tuple(slices)])


@tensor.reshape_p.def_impl
def reshape_impl(
    interpreter: Interpreter, op: Operation, tensor_data: TensorValue
) -> TensorValue:
    new_shape = op.attrs["new_shape"]
    return _wrap(_unwrap(tensor_data).reshape(new_shape))


@tensor.transpose_p.def_impl
def transpose_impl(
    interpreter: Interpreter, op: Operation, tensor_data: TensorValue
) -> TensorValue:
    perm = op.attrs.get("perm")
    return _wrap(np.transpose(_unwrap(tensor_data), axes=perm))
