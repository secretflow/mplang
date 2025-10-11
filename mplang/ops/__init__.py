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

from mplang.ops import builtin as builtin
from mplang.ops import crypto as crypto
from mplang.ops import ibis_cc as ibis_cc
from mplang.ops import jax_cc as jax_cc
from mplang.ops import phe as phe
from mplang.ops import spu as spu
from mplang.ops import tee as tee
from mplang.ops.base import FeOperation as FeOperation
from mplang.ops.ibis_cc import run_ibis as run_ibis
from mplang.ops.jax_cc import run_jax as run_jax
from mplang.ops.sql import run_sql as run_sql

__all__ = [
    "FeOperation",
    "builtin",
    "crypto",
    "ibis_cc",
    "jax_cc",
    "phe",
    "run_ibis",
    "run_jax",
    "run_sql",
    "spu",
    "tee",
]
