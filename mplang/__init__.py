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

"""Multi-Party Programming Language for Secure Computation."""

# Version is managed by hatch-vcs and available after package installation
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mplang")
except PackageNotFoundError:
    # Fallback for development/editable installs when package is not installed
    __version__ = "0.0.0-dev"

from mplang import analysis
from mplang.api import CompileOptions, compile, evaluate, fetch
from mplang.core import (
    DType,
    InterpContext,
    Mask,
    MPContext,
    MPObject,
    MPType,
    TableType,
    TensorType,
    function,
)
from mplang.core.cluster import ClusterSpec, Device, Node, RuntimeInfo
from mplang.core.context_mgr import cur_ctx, set_ctx, with_ctx
from mplang.runtime.driver import Driver
from mplang.runtime.simulation import Simulator

# Public API
__all__ = [
    "ClusterSpec",
    "CompileOptions",
    "DType",
    "Device",
    "Driver",
    "InterpContext",
    "MPContext",
    "MPObject",
    "MPType",
    "Mask",
    "Node",
    "RuntimeInfo",
    "Simulator",
    "TableType",
    "TensorType",
    "__version__",
    "analysis",
    "compile",
    "cur_ctx",
    "evaluate",
    "fetch",
    "function",
    "set_ctx",
    "with_ctx",
]
