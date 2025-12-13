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

"""Sparse OKVS (Oblivious Key-Value Store) Implementation.

This module provides the core data structures and algorithms for Sparse OKVS,
which is a critical component in unbalanced Private Set Intersection (PSI).
"""

import mplang.v2.edsl as el
from mplang.v2.dialects import field
from mplang.v2.libs.mpc.psi.okvs import OKVS

# ============================================================================
# Constants
# ============================================================================

# Number of hash functions for Cuckoo hashing
NUM_HASHES = 3


def get_okvs_expansion(n: int) -> float:
    """Get optimal OKVS expansion factor based on dataset size.

    The 3-hash Garbled Cuckoo Table algorithm requires table size M > N for
    the peeling algorithm to successfully solve the system. The minimum safe
    expansion factor ε (where M = (1+ε)*N) depends on N:

    - For N → ∞: Theoretical minimum is ε ≈ 0.23 (M = 1.23N)
    - For finite N: Larger ε needed due to variance in random hash collisions

    Empirical safe thresholds (failure probability < 0.01%):
    - N < 1,000:    ε = 4.5  (M = 5.5N)  - very small sets need extra wide margin
                                           to handle worst-case hash collisions
    - N < 10,000:   ε = 0.4  (M = 1.4N)
    - N < 100,000:  ε = 0.3  (M = 1.3N)
    - N ≥ 100,000:  ε = 0.35 (M = 1.35N) - large sets converge near theory

    Args:
        n: Number of key-value pairs to encode

    Returns:
        Expansion factor ε such that M = (1+ε)*N is safe for peeling
    """
    if n < 1000:
        return 5.5  # Small scale: need very wide safety margin for stability
    elif n <= 10000:
        return 1.4  # Medium scale
    elif n <= 100000:
        return 1.3  # Large scale
    else:
        # Mega-Binning requires ~1.35 for stability with 1024 bins
        return 1.35


class SparseOKVS(OKVS):
    """Sparse OKVS Implementation using 3-Hash Garbled Cuckoo Table."""

    def __init__(self, m: int):
        self.m = m

    def encode(self, keys: el.Object, values: el.Object, seed: el.Object) -> el.Object:
        """Encode items into OKVS storage using C++ Kernel."""
        return field.solve_okvs(keys, values, self.m, seed)

    def decode(self, keys: el.Object, storage: el.Object, seed: el.Object) -> el.Object:
        """Decode items from OKVS storage using C++ Kernel."""
        return field.decode_okvs(keys, storage, seed)
