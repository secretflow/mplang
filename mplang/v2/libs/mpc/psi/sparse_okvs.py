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

from typing import cast

import jax.numpy as jnp

import mplang.v2.dialects.field as field
import mplang.v2.dialects.tensor as tensor
import mplang.v2.edsl as el
from mplang.v2.libs.mpc.psi.okvs_base import OKVS

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

    Empirical safe thresholds (failure probability < 0.1%):
    - N < 1,000:    ε = 0.6  (M = 1.6N)  - small sets need wide margin
    - N < 10,000:   ε = 0.4  (M = 1.4N)
    - N < 100,000:  ε = 0.3  (M = 1.3N)
    - N ≥ 100,000:  ε = 0.25 (M = 1.25N) - large sets converge to theory

    Args:
        n: Number of key-value pairs to encode

    Returns:
        Expansion factor ε such that M = (1+ε)*N is safe for peeling
    """
    if n < 1000:
        return 3.0  # Small scale: need very wide safety margin for stability
    elif n < 10000:
        return 1.4  # Medium scale
    elif n < 100000:
        return 1.3  # Large scale
    else:
        return 1.25  # Very large scale: near theoretical minimum


class SparseOKVS(OKVS):
    """Sparse OKVS Implementation using 3-Hash Garbled Cuckoo Table."""

    def __init__(self, m: int):
        self.m = m

    def encode(
        self, keys: el.Object, values: el.Object, seed: el.Object
    ) -> el.Object:
        """Encode items into OKVS storage using C++ Kernel."""
        return field.solve_okvs(keys, values, self.m, seed)

    def decode(
        self, keys: el.Object, storage: el.Object, seed: el.Object
    ) -> el.Object:
        """Decode items from OKVS storage using C++ Kernel."""
        return field.decode_okvs(keys, storage, seed)

