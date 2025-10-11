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

from mplang.ops import basic, crypto, ibis_cc, jax_cc, phe, spu, sql_cc, tee
from mplang.ops.base import FeOperation as FeOperation

__all__ = [
    "FeOperation",
    "basic",
    "crypto",
    "ibis_cc",
    "jax_cc",
    "phe",
    "spu",
    "sql_cc",
    "tee",
]
