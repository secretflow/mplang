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
# Communication interfaces
from mplang.core.comm import (
    CollectiveMixin,
    CommunicatorBase,
    ICollective,
    ICommunicator,
)
from mplang.core.dtype import DType

# Interpreter system
from mplang.core.interp import InterpContext, InterpVar
from mplang.core.mask import Mask

# Core object model
from mplang.core.mpobject import MPContext, MPObject
from mplang.core.mptype import MPType

# Function handling
from mplang.core.pfunc import PFunction

# Primitive operations
from mplang.core.primitive import (
    cond,
    constant,
    function,
    pconv,
    peval,
    prand,
    prank,
    pshfl,
    pshfl_s,
    psize,
    set_mask,
    while_loop,
)
from mplang.core.table import TableLike, TableType

# Protocol types
from mplang.core.tensor import TensorLike, TensorType

# Tracing system
from mplang.core.tracer import TraceContext, TracedFunction, TraceVar, VarNamer, trace

__all__ = [
    "CollectiveMixin",
    "CommunicatorBase",
    "DType",
    "ICollective",
    "ICommunicator",
    "InterpContext",
    "InterpVar",
    "MPContext",
    "MPObject",
    "MPType",
    "Mask",
    "PFunction",
    "TableLike",
    "TableType",
    "TensorLike",
    "TensorType",
    "TraceContext",
    "TraceVar",
    "TracedFunction",
    "VarNamer",
    "cond",
    "constant",
    "function",
    "pconv",
    "peval",
    "prand",
    "prank",
    "pshfl",
    "pshfl_s",
    "psize",
    "set_mask",
    "trace",
    "while_loop",
]
