"""Oblivious Transfer (OT) library.

This module implements OT logic using existing SIMP primitives.
Currently, it provides a 'Trusted Simulation' implementation where data is
transferred via `simp.shuffle` and selection happens locally.

Security Note:
    This implementation is NOT cryptographically secure against the receiver
    (who receives both messages). It is intended for:
    1. Verifying the correctness of higher-level protocols (like Permutation Networks).
    2. Environments where a Trusted Third Party or secure channel assumptions hold.

    For real MPC security, this should be replaced by a backend that implements
    actual OT protocols (e.g., Naor-Pinkas, IKNP).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from mplang2.dialects import simp, tensor


def transfer(m0: Any, m1: Any, choice: Any, sender: int, receiver: int) -> Any:
    """Perform 1-out-of-2 Oblivious Transfer.

    Args:
        m0: Message 0 (on Sender).
        m1: Message 1 (on Sender).
        choice: Selection bit 0 or 1 (on Receiver).
        sender: Rank of the sender.
        receiver: Rank of the receiver.

    Returns:
        The selected message (on Receiver).
    """

    # 1. Local Computation (Sender): Pack messages
    # We use pcall_static to ensure this runs on the sender
    def pack_fn(a, b):
        # Stack them into a single array [2, ...]
        return tensor.run_jax(lambda x, y: jnp.stack([x, y]), a, b)

    # We assume m0, m1 are available on sender (or will be moved there if not)
    # But strictly, they should be on sender.
    # The `simp` dialect handles data movement if needed, but let's be explicit.
    packed_msgs = simp.pcall_static((sender,), pack_fn, m0, m1)

    # 2. Transmission: Shuffle from Sender to Receiver
    # We use shuffle_static (or just shuffle if available)
    # routing: {receiver_rank: sender_rank}
    routing = {receiver: sender}
    received_msgs = simp.shuffle_static(packed_msgs, routing=routing)

    # 3. Local Computation (Receiver): Select message
    def select_fn(msgs, c):
        # msgs is [2, ...], c is 0 or 1
        # We use jax.numpy to select
        return tensor.run_jax(lambda m, i: m[i], msgs, c)

    result = simp.pcall_static((receiver,), select_fn, received_msgs, choice)

    return result
