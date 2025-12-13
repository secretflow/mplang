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

"""Simp Driver package.

Provides Driver-side state, values, and ops for the simp dialect.
"""

from mplang.v2.backends.simp_driver.base import SimpDriver
from mplang.v2.backends.simp_driver.http import SimpHttpDriver, make_driver
from mplang.v2.backends.simp_driver.mem import (
    LocalCluster,
    MemCluster,
    SimpMemDriver,
    make_simulator,
)
from mplang.v2.backends.simp_driver.ops import HOST_HANDLERS
from mplang.v2.backends.simp_driver.values import HostVar

__all__ = [
    "HOST_HANDLERS",
    "HostVar",
    "LocalCluster",
    "MemCluster",
    "SimpDriver",
    "SimpHttpDriver",
    "SimpMemDriver",
    "make_driver",
    "make_simulator",
]
