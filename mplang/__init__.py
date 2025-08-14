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

# Version of the mplang package
__version__ = "0.1.0"

# Core API functions
from mplang.api import CompileOptions, compile, evaluate, fetch
from mplang.core import primitive as prim
from mplang.core.context_mgr import cur_ctx, set_ctx, with_ctx
from mplang.runtime.simulation import Simulator

function = prim.primitive

# Public API
__all__ = [
    "CompileOptions",
    "Simulator",
    "__version__",
    "compile",
    "cur_ctx",
    "evaluate",
    "fetch",
    "function",
    "set_ctx",
    "with_ctx",
]
