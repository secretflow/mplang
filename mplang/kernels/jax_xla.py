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

from __future__ import annotations

import base64
from typing import Any

import jax.export as jax_export
import jax.numpy as jnp
import numpy as np

from mplang.core import PFunction
from mplang.kernels.base import kernel_def
from mplang.kernels.value import TensorValue


@kernel_def("jax.exec")
def _jax_exec(pfunc: PFunction, *args: Any) -> Any:
    """Execute a JAX exported function.

    Args:
        pfunc: PFunction containing serialized JAX export data
        *args: Input arguments for the function execution

    Returns:
        The result of executing the JAX function with the provided arguments
    """
    if pfunc.fn_type != "jax.exec":
        raise ValueError(f"jax exec kernel received wrong fn_type: {pfunc.fn_type}")

    export_text = pfunc.fn_text
    if export_text is None:
        raise ValueError("jax exec kernel missing fn_text")

    try:
        export_bytes = base64.b64decode(export_text)
    except ValueError as e:
        raise ValueError(f"Failed to decode base64 export data: {e}") from e

    try:
        exported = jax_export.deserialize(bytearray(export_bytes))
    except Exception as e:
        raise ValueError(f"Failed to deserialize JAX export: {e}") from e

    # Convert TensorValue arguments to JAX arrays
    jax_args = []
    for i, arg in enumerate(args):
        value_to_convert = arg
        if isinstance(arg, TensorValue):
            value_to_convert = arg.to_numpy()

        try:
            jax_args.append(jnp.array(value_to_convert))
        except Exception as e:
            raise ValueError(
                f"Cannot convert argument {i} of type {type(arg)} to JAX array: {e}"
            ) from e

    # Execute the exported function
    # The normalized function expects a single list argument containing all variables
    try:
        result = exported.call(jax_args)
    except Exception as e:
        raise RuntimeError(f"Failed to execute JAX exported function: {e}") from e

    # Convert result back to TensorValue if it's a JAX array
    if isinstance(result, (jnp.ndarray, np.ndarray)):
        return TensorValue(np.array(result))
    elif isinstance(result, (tuple, list)):
        # Handle multiple outputs
        converted_result = []
        for item in result:
            if isinstance(item, (jnp.ndarray, np.ndarray)):
                converted_result.append(TensorValue(np.array(item)))
            else:
                converted_result.append(item)
        return type(result)(converted_result)
    else:
        return result
