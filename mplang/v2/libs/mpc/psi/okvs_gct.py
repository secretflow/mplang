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

    Empirical safe thresholds (failure probability < 0.001%):
    - N ≤ 200:      ε = 24.0 (M = 25.0N)  - extremely small sets need very wide margin
    - N < 1,000:    ε = 11.0 (M = 12.0N)  - small sets need extra wide safety margin
    - N < 10,000:   ε = 0.6  (M = 1.6N)
    - N < 100,000:  ε = 0.4  (M = 1.4N)
    - N ≥ 100,000:  ε = 0.35 (M = 1.35N)  - large sets converge near theory

    Note: These expansion factors account for the 128-byte alignment requirement
    in the OKVS implementation. The factors are intentionally conservative to
    ensure high success rates (>99.9%) for the probabilistic peeling algorithm.

    Args:
        n: Number of key-value pairs to encode

    Returns:
        Expansion factor ε such that M = (1+ε)*N is safe for peeling
    """
    if n <= 200:
        return 25.0  # Extremely small scale: need very wide margin for stability
    elif n < 1000:
        return 12.0  # Small scale: need wide safety margin for stability
    elif n <= 10000:
        return 1.6  # Medium scale
    elif n <= 100000:
        return 1.4  # Large scale
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
