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

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mplang.v1.core.dtypes import DType

# basic type aliases
Shape = tuple[int, ...]
ScalarType = int | float | bool | complex

__all__ = ["ScalarType", "Shape", "TensorLike", "TensorType"]


@runtime_checkable
class TensorLike(Protocol):
    """
    Protocol for objects structurally resembling tensors from common libraries
    (NumPy, PyTorch, JAX), focusing on dtype and shape attributes.
    """

    @property
    def dtype(self) -> Any: ...

    @property
    def shape(self) -> Shape: ...


@dataclass(frozen=True)
class TensorType:
    """A data class that describes the type information of a tensor."""

    dtype: DType
    shape: Shape

    def __init__(self, dtype: DType | Any, shape: Shape):
        # Convert dtype to DType if needed
        if not isinstance(dtype, DType):
            dtype = DType.from_any(dtype)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)

    @classmethod
    def from_obj(cls, obj: TensorLike | ScalarType) -> TensorType:
        if isinstance(obj, ScalarType):
            return cls(DType.from_python_type(type(obj)), ())
        elif isinstance(obj, TensorLike):
            return cls(DType.from_any(obj.dtype), obj.shape)
        else:
            raise TypeError(f"Unsupported type: {type(obj)}.")

    def to_numpy(self) -> np.dtype:
        """Convert to NumPy dtype for compatibility."""
        return self.dtype.to_numpy()

    def __repr__(self) -> str:
        shape_str = "x".join(str(d) for d in self.shape)
        dtype_name = str(self.dtype)
        return f"Tensor<{shape_str}x{dtype_name}>" if shape_str else f"{dtype_name}"
