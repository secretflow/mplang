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
import jax.numpy as jnp
from jax._src import xla_bridge
from jax.lib import xla_client as xc

from mplang.backend.base import cur_kctx, kernel_def
from mplang.core.pfunc import PFunction


@kernel_def("mlir.stablehlo")
def _stablehlo_exec(pfunc: PFunction, *args: Any) -> Any:
    if pfunc.fn_type != "mlir.stablehlo":
        raise ValueError("stablehlo kernel received wrong fn_type")

    mlir_text = pfunc.fn_text
    if mlir_text is None:
        raise ValueError("StableHLO kernel missing fn_text")
    if isinstance(mlir_text, bytes):
        mlir_text = mlir_text.decode("utf-8")

    # Simple compile cache per runtime (state pocket per backend namespace)
    ctx = cur_kctx()
    pocket = ctx.state.setdefault("stablehlo", {})
    cache = pocket.setdefault("compile_cache", {})
    compiled = cache.get(mlir_text)
    if compiled is None:
        backend = jax.default_backend()
        client = xla_bridge.get_backend(backend)
        compile_options = xc.CompileOptions()
        try:
            compiled = client.compile(mlir_text, compile_options)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"StableHLO compile failed: {e}") from e
        cache[mlir_text] = compiled

    jax_args = []
    for arg in args:
        if hasattr(arg, "numpy"):
            jax_arg = jnp.array(arg.numpy())  # type: ignore
        else:
            jax_arg = jnp.array(arg)
        jax_args.append(jax.device_put(jax_arg))

    try:
        result = compiled.execute_sharded(jax_args)
        arrays = result.disassemble_into_single_device_arrays()
        flat: list[Any] = []
        for lst in arrays:
            if isinstance(lst, list) and len(lst) == 1:
                flat.append(jnp.array(lst[0]))
            else:
                flat.extend([jnp.array(a) for a in lst])
        return tuple(flat)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"StableHLO execute failed: {e}") from e
