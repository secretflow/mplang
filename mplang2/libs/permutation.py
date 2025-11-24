"""Secure Permutation Network library.

This module implements secure permutation (shuffling) using Oblivious Transfer (OT).
It allows a Sender (holding data) and a Receiver (holding a permutation) to
cooperatively shuffle the data such that:
1. The Receiver obtains the shuffled data.
2. The Sender learns nothing about the permutation.
3. The Receiver learns nothing about the original data order (beyond the result).

The implementation uses a Waksman network decomposition of the permutation into
a series of 2x2 "Secure Switches".
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import mplang2.edsl.typing as elt
from mplang2.dialects import crypto, simp, tensor
from mplang2.libs import ot


def secure_switch(
    x0: Any, x1: Any, control_bit: Any, sender: int, receiver: int
) -> tuple[Any, Any]:
    """A 2x2 Secure Switch using OT.

    Args:
        x0: Input 0 (on Sender).
        x1: Input 1 (on Sender).
        control_bit: Control bit (on Receiver). 0 = straight, 1 = swap.
        sender: Rank of the sender.
        receiver: Rank of the receiver.

    Returns:
        (y0, y1) on Receiver, where:
        y0 = x0 if c=0 else x1
        y1 = x1 if c=0 else x0
    """
    # y0 = select(x0, x1, c)
    y0 = ot.transfer(x0, x1, control_bit, sender, receiver)

    # y1 = select(x0, x1, 1-c)
    # Compute 1-c locally on receiver
    def invert_bit(c: Any) -> Any:
        return tensor.run_jax(lambda x: 1 - x, c)

    inv_control_bit = simp.pcall_static((receiver,), invert_bit, control_bit)
    y1 = ot.transfer(x0, x1, inv_control_bit, sender, receiver)

    return y0, y1


def _compute_benes_network_controls(permutation: list[int] | Any) -> list[list[int]]:
    """Compute control bits for Benes network to realize the permutation.

    Args:
        permutation: List of source indices. permutation[i] = src means
                     dest i comes from src.

    Returns:
        List of lists of control bits (0 or 1).
        Each inner list corresponds to a stage of switches.
    """
    # Ensure permutation is a list of ints
    if hasattr(permutation, "tolist"):
        permutation = permutation.tolist()
    permutation = [int(x) for x in permutation]

    n = len(permutation)
    if n <= 1:
        return []
    if n == 2:
        # Single switch.
        # If perm=[0, 1], ctrl=0. If perm=[1, 0], ctrl=1.
        return [[0]] if permutation[0] == 0 else [[1]]

    # Benes network recursive construction.
    # We need to decompose the permutation into two sub-permutations
    # for the upper and lower sub-networks.
    raise NotImplementedError("Benes network control computation not implemented")


def apply_permutation(data: Any, permutation: Any, sender: int, receiver: int) -> Any:
    """Apply a secure permutation using a Benes network.

    Args:
        data: Data items (on Sender). Must be a Tensor.
        permutation: List of indices (on Receiver). e.g. [2, 0, 1, 3]
        sender: Rank of sender.
        receiver: Rank of receiver.

    Returns:
        Shuffled data (on Receiver).
    """
    target_type = data.type
    if isinstance(target_type, elt.MPType):
        target_type = target_type.value_type
    n = target_type.shape[0]

    # Compute controls (on Receiver)
    def compute_flat_controls(perm: Any) -> Any:
        controls = _compute_benes_network_controls(perm)
        flat = []
        for stage in controls:
            flat.extend(stage)
        return jnp.array(flat, dtype=jnp.int64)

    flat_controls = simp.pcall_static((receiver,), compute_flat_controls, permutation)

    # Helper functions to simplify the recursion and optimize control inversion
    def split_evens_odds(x: Any) -> tuple[Any, Any]:
        typ = x.type
        parties = typ.parties if isinstance(typ, elt.MPType) else None
        if parties is not None:
            evens = simp.pcall_static(
                parties, lambda t: tensor.run_jax(lambda a: a[0::2], t), x
            )
            odds = simp.pcall_static(
                parties, lambda t: tensor.run_jax(lambda a: a[1::2], t), x
            )
        else:
            evens = tensor.run_jax(lambda a: a[0::2], x)
            odds = tensor.run_jax(lambda a: a[1::2], x)
        return evens, odds

    def interleave(a: Any, b: Any) -> Any:
        def _impl(x: Any, y: Any) -> Any:
            return tensor.run_jax(
                lambda u, v: jnp.ravel(jnp.column_stack((u, v))), x, y
            )

        typ = a.type
        parties = typ.parties if isinstance(typ, elt.MPType) else None
        if parties is not None:
            return simp.pcall_static(parties, _impl, a, b)
        else:
            return _impl(a, b)

    def apply_switch(x0: Any, x1: Any, ctrls: Any) -> tuple[Any, Any]:
        # Optimization: Compute inverse controls on device to avoid
        # transferring a second array from Python.
        def invert(c: Any) -> Any:
            return 1 - c

        inv_ctrls = simp.pcall_static(
            (receiver,), lambda c: tensor.run_jax(invert, c), ctrls
        )

        typ = x0.type
        parties = typ.parties if isinstance(typ, elt.MPType) else None
        is_on_sender = parties == (sender,)

        if is_on_sender:
            # Use OT if data is on Sender
            y0 = ot.transfer(x0, x1, ctrls, sender, receiver)
            y1 = ot.transfer(x0, x1, inv_ctrls, sender, receiver)
        else:
            # Use local select if data is already on Receiver
            def switch_fn(u: Any, v: Any, c: Any) -> Any:
                return tensor.elementwise(crypto.select, c, v, u)

            y0 = simp.pcall_static((receiver,), switch_fn, x0, x1, ctrls)
            y1 = simp.pcall_static((receiver,), switch_fn, x0, x1, inv_ctrls)
        return y0, y1

    # Mutable offset to track position in flat_controls
    offset = [0]

    def get_next_controls(count: int) -> Any:
        start = offset[0]
        offset[0] += count

        def slice_fn(arr: Any, s: Any, c: Any) -> Any:
            return arr[s : s + c]

        return simp.pcall_static((receiver,), slice_fn, flat_controls, start, count)

    def recursive_network_batch(inputs: Any, depth: int) -> Any:
        # Calculate local size based on depth.
        # Benes network structure is deterministic based on N.
        n_local = n // (2**depth)

        if n_local == 1:
            return inputs  # Base case for Benes: n=2 is a single switch

        # Collect controls for the current stage
        # For n_local=2, we need 1 control. For n_local>2, we need n_local/2.
        count = n_local // 2
        current_ctrls = get_next_controls(count)

        if n_local == 2:
            # Base case: Single switch
            x_evens, x_odds = split_evens_odds(inputs)
            y0, y1 = apply_switch(x_evens, x_odds, current_ctrls)
            return interleave(y0, y1)

        # General case: Input Stage -> Recurse -> Output Stage

        # Input Stage
        x_evens, x_odds = split_evens_odds(inputs)
        y0, y1 = apply_switch(x_evens, x_odds, current_ctrls)

        # Recurse
        upper_outputs = recursive_network_batch(y0, depth + 1)
        lower_outputs = recursive_network_batch(y1, depth + 1)

        # Output Stage
        out_ctrls = get_next_controls(n_local // 2)

        z0, z1 = apply_switch(upper_outputs, lower_outputs, out_ctrls)
        return interleave(z0, z1)

    return recursive_network_batch(data, 0)
