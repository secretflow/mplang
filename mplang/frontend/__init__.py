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
Frontend module for mplang.

This module contains compilers that transform high-level functions into
portable, serializable intermediate representations.
"""

from mplang.frontend import builtin as builtin
from mplang.frontend import crypto as crypto
from mplang.frontend import ibis_cc as ibis_cc
from mplang.frontend import jax_cc as jax_cc
from mplang.frontend import phe as phe
from mplang.frontend import spu as spu
from mplang.frontend import tee as tee
from mplang.frontend.base import FeOperation as FeOperation
from mplang.frontend.ibis_cc import ibis_compile as ibis_compile
from mplang.frontend.jax_cc import jax_compile as jax_compile
from mplang.frontend.sql import sql_run as sql_run

__all__ = [
    "FeOperation",
    "builtin",
    "crypto",
    "ibis_cc",
    "ibis_compile",
    "jax_cc",
    "jax_compile",
    "phe",
    "spu",
    "sql_run",
    "tee",
]
