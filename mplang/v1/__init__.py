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

from mplang.v1 import analysis
from mplang.v1._device import device, get_dev_attr, is_device_obj, put, set_dev_attr
from mplang.v1.core import (
    ClusterSpec,
    Device,
    DType,
    InterpContext,
    IrReader,
    IrWriter,
    Mask,
    MPContext,
    MPObject,
    MPType,
    Node,
    Rank,
    RuntimeInfo,
    Shape,
    TableType,
    TensorType,
    TraceContext,
    TracedFunction,
    cur_ctx,
    function,
    pconv,
    peval,
    pshfl,
    pshfl_s,
    set_ctx,
    trace,
    uniform_cond,
    while_loop,
    with_ctx,
)
from mplang.v1.host import CompileOptions, compile, evaluate, fetch
from mplang.v1.runtime.driver import Driver
from mplang.v1.runtime.simulation import Simulator
from mplang.v1.simp.api import (
    constant,
    debug_print,
    prand,
    prank,
    run,
    run_at,
    run_jax,
    run_jax_at,
    run_sql,
    run_sql_at,
    set_mask,
)
from mplang.v1.simp.mpi import allgather_m, bcast_m, gather_m, p2p, scatter_m
from mplang.v1.simp.party import P0, P1, P2, P2P, P, Party, load_module
from mplang.v1.simp.random import key_split, pperm, prandint, ukey, urandint
from mplang.v1.simp.smpc import reveal, reveal_to, seal, seal_from, srun_jax

# Public API
__all__ = [
    "P0",
    "P1",
    "P2",
    "P2P",
    "ClusterSpec",
    "CompileOptions",
    "DType",
    "Device",
    "Driver",
    "InterpContext",
    "IrReader",
    "IrWriter",
    "MPContext",
    "MPObject",
    "MPType",
    "Mask",
    "Node",
    "P",
    "Party",
    "Rank",
    "RuntimeInfo",
    "Shape",
    "Simulator",
    "TableType",
    "TensorType",
    "TraceContext",
    "TracedFunction",
    "__version__",
    "allgather_m",
    "analysis",
    "bcast_m",
    "compile",
    "constant",
    "cur_ctx",
    "debug_print",
    "device",
    "evaluate",
    "fetch",
    "function",
    "gather_m",
    "get_dev_attr",
    "is_device_obj",
    "key_split",
    "load_module",
    "p2p",
    "pconv",
    "peval",
    "pperm",
    "prand",
    "prandint",
    "prank",
    "pshfl",
    "pshfl_s",
    "put",
    "reveal",
    "reveal_to",
    "run",
    "run_at",
    "run_jax",
    "run_jax_at",
    "run_sql",
    "run_sql_at",
    "scatter_m",
    "seal",
    "seal_from",
    "set_ctx",
    "set_dev_attr",
    "set_mask",
    "srun_jax",
    "trace",
    "ukey",
    "uniform_cond",
    "urandint",
    "while_loop",
    "with_ctx",
]
