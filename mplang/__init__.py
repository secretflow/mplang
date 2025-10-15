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
from mplang.core import (
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
from mplang.host import CompileOptions, compile, evaluate, fetch
from mplang.runtime.driver import Driver
from mplang.runtime.simulation import Simulator
from mplang.simp.api import (
    constant,
    debug_print,
    prand,
    prank,
    run,
    run_at,
    run_ibis,
    run_ibis_at,
    run_jax,
    run_jax_at,
    run_sql,
    run_sql_at,
    set_mask,
)
from mplang.simp.mpi import allgather_m, bcast_m, gather_m, p2p, scatter_m
from mplang.simp.party import P0, P1, P2, P2P, P, Party, load_module
from mplang.simp.random import key_split, pperm, prandint, ukey, urandint
from mplang.simp.smpc import reveal, revealTo, seal, sealFrom, srun

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
    "evaluate",
    "fetch",
    "function",
    "gather_m",
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
    "reveal",
    "revealTo",
    "run",
    "run_at",
    "run_ibis",
    "run_ibis_at",
    "run_jax",
    "run_jax_at",
    "run_sql",
    "run_sql_at",
    "scatter_m",
    "seal",
    "sealFrom",
    "set_ctx",
    "set_mask",
    "srun",
    "trace",
    "ukey",
    "uniform_cond",
    "urandint",
    "while_loop",
    "with_ctx",
]
