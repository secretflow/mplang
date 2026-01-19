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

"""MPC (Multi-Party Computation) library for MPLang2.

Subpackages:
- ot: Oblivious Transfer protocols
- vole: Vector OLE protocols
- psi: Private Set Intersection
- analytics: Privacy-preserving analytics

Example usage:
    from mplang.libs.mpc import ot_transfer, apply_permutation
    from mplang.libs.mpc.vole import silver_vole
    from mplang.libs.mpc.psi import psi_intersect
"""

from .analytics.aggregation import rotate_and_sum
from .analytics.groupby import oblivious_groupby_sum_bfv, oblivious_groupby_sum_shuffle
from .analytics.permutation import apply_permutation, secure_switch
from .ot.base import transfer as ot_transfer

__all__ = [
    "apply_permutation",
    "oblivious_groupby_sum_bfv",
    "oblivious_groupby_sum_shuffle",
    "ot_transfer",
    "rotate_and_sum",
    "secure_switch",
]
