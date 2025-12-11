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

"""Privacy-preserving analytics operations.

Submodules:
- aggregation: BFV homomorphic aggregation
- groupby: Oblivious Group-By operations
- permutation: Secure permutation (Bitonic Sort)
"""

from .aggregation import aggregate_sparse, batch_bucket_aggregate, rotate_and_sum
from .groupby import oblivious_groupby_sum_bfv, oblivious_groupby_sum_shuffle
from .permutation import apply_permutation, secure_switch

__all__ = [
    "aggregate_sparse",
    "apply_permutation",
    "batch_bucket_aggregate",
    "oblivious_groupby_sum_bfv",
    "oblivious_groupby_sum_shuffle",
    "rotate_and_sum",
    "secure_switch",
]
