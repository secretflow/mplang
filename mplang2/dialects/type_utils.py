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

import jax.numpy as jnp
import numpy as np

import mplang2.edsl.typing as elt


def elt_to_jax_dtype(dtype: elt.ScalarType) -> Any:
    """Convert MPLang scalar type to JAX dtype."""
    if dtype == elt.f32:
        return jnp.float32
    elif dtype == elt.f64:
        return jnp.float64
    elif dtype == elt.i32:
        return jnp.int32
    elif dtype == elt.i64:
        return jnp.int64
    elif isinstance(dtype, elt.IntegerType) and dtype.bitwidth == 1:
        return jnp.bool_
    else:
        # Default fallback
        return jnp.float32


def jax_to_elt_dtype(dtype: Any) -> elt.ScalarType:
    """Convert JAX/NumPy dtype to MPLang scalar type."""
    # Ensure we have a numpy/jax dtype object
    if not isinstance(dtype, (np.dtype, type(jnp.dtype("float32")))):
        try:
            dtype = jnp.dtype(dtype)
        except Exception:
            pass

    if dtype == jnp.float32 or dtype == np.float32:
        return elt.f32
    elif dtype == jnp.float64 or dtype == np.float64:
        return elt.f64
    elif dtype == jnp.int32 or dtype == np.int32:
        return elt.i32
    elif dtype == jnp.int64 or dtype == np.int64:
        return elt.i64
    elif dtype == jnp.bool_ or dtype == np.bool_:
        return elt.IntegerType(bitwidth=1, signed=True)

    # Fallback based on name if direct comparison fails
    name = getattr(dtype, "name", str(dtype))
    if "float32" in name:
        return elt.f32
    elif "float64" in name:
        return elt.f64
    elif "int32" in name:
        return elt.i32
    elif "int64" in name:
        return elt.i64
    elif "bool" in name:
        return elt.IntegerType(bitwidth=1, signed=True)

    return elt.f32
