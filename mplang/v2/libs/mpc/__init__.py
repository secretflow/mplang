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

This module provides cryptographic building blocks and privacy-preserving algorithms:

- ot: Oblivious Transfer protocols
- permutation: Secure shuffle using OT
- aggregation: BFV homomorphic aggregation
- groupby: Oblivious group-by operations
"""

from .aggregation import rotate_and_sum
from .groupby import oblivious_groupby_sum_bfv, oblivious_groupby_sum_shuffle
from .ot import transfer as ot_transfer
from .permutation import apply_permutation, secure_switch

__all__ = [
    "apply_permutation",
    "oblivious_groupby_sum_bfv",
    "oblivious_groupby_sum_shuffle",
    "ot_transfer",
    "rotate_and_sum",
    "secure_switch",
]
