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
