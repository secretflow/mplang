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

"""
Core components for multi-party computation.

This package provides the fundamental building blocks for multi-party computation,
including type systems, tracing mechanisms, and interpreter contexts.
"""

# Core type system
# Communication interfaces & core symbols
from mplang.v1.core.cluster import ClusterSpec, Device, Node, RuntimeInfo
from mplang.v1.core.comm import (
    CollectiveMixin,
    CommunicatorBase,
    ICollective,
    ICommunicator,
)
from mplang.v1.core.context_mgr import cur_ctx, set_ctx, with_ctx
from mplang.v1.core.dtypes import (
    BINARY,
    BOOL,
    COMPLEX64,
    COMPLEX128,
    DATE,
    DECIMAL,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    INTERVAL,
    JSON,
    STRING,
    TIME,
    TIMESTAMP,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    UUID,
    DType,
)
from mplang.v1.core.interp import InterpContext, InterpVar
from mplang.v1.core.mask import Mask
from mplang.v1.core.mpir import IrReader, IrWriter
from mplang.v1.core.mpobject import MPContext, MPObject
from mplang.v1.core.mptype import MPType, Rank, Shape
from mplang.v1.core.pfunc import PFunction, get_fn_name

# Import primitive-dependent symbols at the end to avoid cycles when frontend ops
# import from the core facade during package initialization.
from mplang.v1.core.primitive import (
    builtin_function,
    function,
    pconv,
    peval,
    pmask,
    pshfl,
    pshfl_s,
    psize,
    uniform_cond,
    while_loop,
)
from mplang.v1.core.table import TableLike, TableType
from mplang.v1.core.tensor import ScalarType, TensorLike, TensorType
from mplang.v1.core.tracer import (
    TraceContext,
    TracedFunction,
    TraceVar,
    VarNamer,
    trace,
)

__all__ = [
    "BINARY",
    "BOOL",
    "COMPLEX64",
    "COMPLEX128",
    "DATE",
    "DECIMAL",
    "FLOAT16",
    "FLOAT32",
    "FLOAT64",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "INTERVAL",
    "JSON",
    "STRING",
    "TIME",
    "TIMESTAMP",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "UUID",
    "ClusterSpec",
    "CollectiveMixin",
    "CommunicatorBase",
    "DType",
    "Device",
    "ICollective",
    "ICommunicator",
    "InterpContext",
    "InterpVar",
    "IrReader",
    "IrWriter",
    "MPContext",
    "MPObject",
    "MPType",
    "Mask",
    "Node",
    "PFunction",
    "Rank",
    "RuntimeInfo",
    "ScalarType",
    "Shape",
    "TableLike",
    "TableType",
    "TensorLike",
    "TensorType",
    "TraceContext",
    "TraceVar",
    "TracedFunction",
    "VarNamer",
    "builtin_function",
    "cur_ctx",
    "function",
    "get_fn_name",
    "pconv",
    "peval",
    "pmask",
    "pshfl",
    "pshfl_s",
    "psize",
    "set_ctx",
    "trace",
    "uniform_cond",
    "while_loop",
    "with_ctx",
]
