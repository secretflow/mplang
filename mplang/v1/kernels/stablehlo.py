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

from typing import Any

import jax
import jax.extend as jxt
import jax.numpy as jnp
import numpy as np
from jax._src import compiler

from mplang.v1.core import PFunction
from mplang.v1.kernels.base import cur_kctx, kernel_def
from mplang.v1.kernels.value import TensorValue


@kernel_def("mlir.stablehlo")
def _stablehlo_exec(pfunc: PFunction, *args: Any) -> Any:
    if pfunc.fn_type != "mlir.stablehlo":
        raise ValueError("stablehlo kernel received wrong fn_type")

    mlir_text = pfunc.fn_text
    if mlir_text is None:
        raise ValueError("StableHLO kernel missing fn_text")
    if isinstance(mlir_text, bytes):
        mlir_text = mlir_text.decode("utf-8")

    # Flat-key compile cache: stablehlo.compile_cache.<hash>
    ctx = cur_kctx()
    rt = ctx.runtime
    import hashlib

    h = hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()[:16]
    key = f"stablehlo.compile_cache.{h}"
    compiled = rt.get_state(key)
    if compiled is None:
        client = jxt.backend.get_backend()
        compile_options = compiler.get_compile_options(num_replicas=1, num_partitions=1)

        try:
            compiled = client.compile_and_load(
                mlir_text, client.devices(), compile_options
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"StableHLO compile failed: {e}") from e
        rt.set_state(key, compiled)

    # Handle JAX's unused parameter elimination via arg_keep_map
    runtime_args = args
    if "arg_keep_map" in pfunc.attrs:
        keep_indices = pfunc.attrs["arg_keep_map"]
        # Filter out arguments that were eliminated by JAX during compilation
        runtime_args = tuple(args[i] for i in keep_indices)

    tensor_args: list[TensorValue] = []
    for idx, arg in enumerate(runtime_args):
        if not isinstance(arg, TensorValue):
            raise TypeError(
                f"StableHLO kernel expects TensorValue inputs, got {type(arg).__name__} at position {idx}"
            )
        tensor_args.append(arg)

    jax_args = [
        jax.device_put(jnp.asarray(tensor.to_numpy())) for tensor in tensor_args
    ]

    try:
        # Execute with the new LoadedExecutable interface
        result = compiled.execute(jax_args)

        # Use jax.tree_util.tree_flatten to robustly handle any PyTree structure
        flat_results, _ = jax.tree_util.tree_flatten(result)
        flat = [TensorValue(np.asarray(item)) for item in flat_results]

        return tuple(flat)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"StableHLO execute failed: {e}") from e
