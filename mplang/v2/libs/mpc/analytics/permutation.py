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

"""Secure Permutation Network library.

This module implements secure permutation (shuffling) using Oblivious Transfer (OT).
It allows a Sender (holding data) and a Receiver (holding a permutation) to
cooperatively shuffle the data such that:
1. The Receiver obtains the shuffled data.
2. The Sender learns nothing about the permutation.
3. The Receiver learns nothing about the original data order (beyond the result).

The implementation uses a Bitonic sorting network to achieve oblivious permutation.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import mplang.v2.edsl.typing as elt
from mplang.v2.dialects import simp, tensor
from mplang.v2.libs.mpc.ot import base as ot


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


def _compute_bitonic_sort_controls(permutation: Any, n: int) -> Any:
    """Compute control bits for bitonic sorting network using JAX.

    This uses a simple approach: track permutation values through
    bitonic sort and record comparison results.

    Args:
        permutation: JAX array of target indices.
        n: Size (must be power of 2).

    Returns:
        JAX array of control bits.
    """

    def impl(perm: Any) -> Any:
        # Convert from gather-style permutation (output[i] = input[perm[i]])
        # to destination positions for each input index: inv[src] = dest
        perm_i64 = perm.astype(jnp.int64)
        indices = jnp.arange(n, dtype=jnp.int64)
        current = jnp.zeros_like(perm_i64)
        current = current.at[perm_i64].set(indices)

        controls = []
        num_stages = int(math.log2(n))

        for stage in range(num_stages):
            for step in range(stage + 1):
                step_dist = 2 ** (stage - step)

                # Vectorized calculation using numpy for static indices
                indices_np = np.arange(n, dtype=np.int64)
                partners_np = indices_np ^ step_dist
                mask_np = partners_np > indices_np

                idx_i_np = indices_np[mask_np]
                idx_j_np = partners_np[mask_np]

                # Determine sort direction
                block_size = 2 ** (stage + 1)
                block_ids = indices_np // block_size
                ascending_np = (block_ids % 2) == 0
                asc_mask_np = ascending_np[mask_np]

                # Extract values
                v_i = current[idx_i_np]
                v_j = current[idx_j_np]

                swap_asc = v_i > v_j
                swap_desc = v_i < v_j

                should_swap = jnp.where(asc_mask_np, swap_asc, swap_desc)

                controls.append(should_swap)

                # Update current
                new_i = jnp.where(should_swap, v_j, v_i)
                new_j = jnp.where(should_swap, v_i, v_j)

                current = current.at[idx_i_np].set(new_i)
                current = current.at[idx_j_np].set(new_j)

        return jnp.concatenate(controls)

    return tensor.run_jax(impl, permutation)


def apply_permutation(data: Any, permutation: Any, sender: int, receiver: int) -> Any:
    """Apply a secure permutation using a Bitonic sorting network.

    Args:
        data: Data items (on Sender). Can be a Tensor or list of Objects.
        permutation: Tensor of indices (on Receiver). e.g. [2, 0, 1, 3]
                     permutation[i] = src means output[i] comes from input[src].
        sender: Rank of sender.
        receiver: Rank of receiver.

    Returns:
        Shuffled data (on Receiver). Returns a list if input was a list.
    """
    # Remember if input was a list
    is_list_input = isinstance(data, list)

    # Handle list input - convert to tensor
    if is_list_input:
        if len(data) == 0:
            return []

        # Stack list elements into a tensor
        def stack_elements(*args: Any) -> Any:
            return tensor.run_jax(lambda *xs: jnp.stack(xs), *args)

        data = simp.pcall_static((sender,), stack_elements, *data)

    target_type = data.type
    if isinstance(target_type, elt.MPType):
        target_type = target_type.value_type
    if not isinstance(target_type, elt.TensorType):
        raise TypeError("apply_permutation expects tensor inputs")
    n = target_type.shape[0]
    original_n = n

    # Bitonic sort requires power-of-2 size - pad if necessary
    n_padded = 2 ** math.ceil(math.log2(max(n, 2)))

    if n_padded != n:
        # Pad data with zeros
        def pad_data(d: Any, pad_n: int) -> Any:
            return tensor.run_jax(
                lambda x: jnp.pad(x, (0, pad_n - x.shape[0]), mode="constant"), d
            )

        data = simp.pcall_static((sender,), pad_data, data, n_padded)

        # Pad permutation with identity mapping for extra elements
        def pad_perm(p: Any, orig: int, pad_n: int) -> Any:
            extra = jnp.arange(orig, pad_n, dtype=jnp.int64)
            return tensor.run_jax(lambda x: jnp.concatenate([x, extra]), p)

        permutation = simp.pcall_static((receiver,), pad_perm, permutation, n, n_padded)
        n = n_padded

    # Compute control bits for bitonic sort (on Receiver)
    controls = simp.pcall_static(
        (receiver,), lambda p: _compute_bitonic_sort_controls(p, n), permutation
    )

    # Apply bitonic sorting network
    # Strategy:
    # - Iterate through stages/steps.
    # - For each step, identify pairs (i, j).
    # - If data is on Sender (first step), use Vectorized OT (secure_switch).
    # - If data is on Receiver (subsequent steps), use local select.

    current = data
    ctrl_offset = 0
    num_stages = int(math.log2(n))

    # Helper to extract a slice of controls
    def get_step_ctrls(all_ctrls: Any, off: int, count: int) -> Any:
        def impl(c: Any, o: int, n: int) -> Any:
            # Convert offset to tensor to avoid recompilation (dynamic slice start)
            o_tensor = tensor.constant(np.array(o, dtype=np.int64))
            # n (slice size) must be static for dynamic_slice
            return tensor.run_jax(
                lambda x, start, size: jax.lax.dynamic_slice(x, (start,), (size,)),
                c,
                o_tensor,
                n,
            )

        return simp.pcall_static((receiver,), impl, all_ctrls, off, count)

    for stage in range(num_stages):
        for step in range(stage + 1):
            step_dist = 2 ** (stage - step)

            # Vectorized step application
            # Construct indices for all pairs in this step
            indices = np.arange(n, dtype=np.int64)
            partners = indices ^ step_dist
            mask = partners > indices
            idx_i_np = indices[mask]
            idx_j_np = partners[mask]

            # Number of pairs
            num_pairs = len(idx_i_np)

            # Check where data is
            typ = current.type
            is_on_sender = isinstance(typ, elt.MPType) and typ.parties == (sender,)

            # Helper to create index tensors on specific parties
            def make_indices(party: tuple[int, ...], idx: Any) -> Any:
                return simp.pcall_static(party, lambda: tensor.constant(idx))

            if is_on_sender:
                # Get controls for this step (on Receiver)
                step_ctrls = get_step_ctrls(controls, ctrl_offset, num_pairs)

                # Data on Sender: Use OT
                idx_i_sender = make_indices((sender,), idx_i_np)
                idx_j_sender = make_indices((sender,), idx_j_np)

                # Extract pairs on Sender
                def extract_pairs_sender(
                    d: Any, idx_i: Any, idx_j: Any
                ) -> tuple[Any, Any]:
                    return (
                        tensor.run_jax(lambda x, i: x[i], d, idx_i),
                        tensor.run_jax(lambda x, j: x[j], d, idx_j),
                    )

                val_i, val_j = simp.pcall_static(
                    (sender,),
                    extract_pairs_sender,
                    current,
                    idx_i_sender,
                    idx_j_sender,
                )

                # Secure Switch (OT) -> Result on Receiver
                res_i, res_j = secure_switch(val_i, val_j, step_ctrls, sender, receiver)

                # Reconstruct full array on Receiver
                idx_i_recv = make_indices((receiver,), idx_i_np)
                idx_j_recv = make_indices((receiver,), idx_j_np)

                # We need to scatter res_i and res_j back to their positions
                def scatter_results(
                    vi: Any, vj: Any, ii: Any, ij: Any, size: int
                ) -> Any:
                    # Initialize with zeros (or dummy)
                    # Note: We assume we cover all indices.
                    # Bitonic sort step covers all indices exactly once.
                    # So we can just scatter.

                    # We need a template for the result.
                    # vi is the type of elements.
                    # We can use jnp.zeros_like(vi) but expanded?
                    # Or just allocate.

                    def impl(
                        v_i: jnp.ndarray,
                        v_j: jnp.ndarray,
                        idx_i: jnp.ndarray,
                        idx_j: jnp.ndarray,
                    ) -> jnp.ndarray:
                        # v_i shape (N/2, ...), idx_i shape (N/2,)
                        # We want output shape (N, ...)

                        # Infer shape from v_i
                        out_shape = (size, *v_i.shape[1:])
                        out = jnp.zeros(out_shape, dtype=v_i.dtype)

                        out = out.at[idx_i].set(v_i)
                        out = out.at[idx_j].set(v_j)
                        return out

                    return tensor.run_jax(impl, vi, vj, ii, ij)

                current = simp.pcall_static(
                    (receiver,),
                    scatter_results,
                    res_i,
                    res_j,
                    idx_i_recv,
                    idx_j_recv,
                    n,
                )
                ctrl_offset += num_pairs

            else:
                # Data on Receiver: Execute locally
                step_ctrls = get_step_ctrls(controls, ctrl_offset, num_pairs)

                # Construct indices on Receiver
                idx_i_recv = make_indices((receiver,), idx_i_np)
                idx_j_recv = make_indices((receiver,), idx_j_np)

                def apply_local_step(d: Any, c: Any, ii: Any, ij: Any) -> Any:
                    def impl(
                        curr: jnp.ndarray,
                        ctrls: jnp.ndarray,
                        idx_i: jnp.ndarray,
                        idx_j: jnp.ndarray,
                    ) -> jnp.ndarray:
                        val_i = curr[idx_i]
                        val_j = curr[idx_j]

                        new_i = jnp.where(ctrls, val_j, val_i)
                        new_j = jnp.where(ctrls, val_i, val_j)

                        curr = curr.at[idx_i].set(new_i)
                        curr = curr.at[idx_j].set(new_j)
                        return curr

                    return tensor.run_jax(impl, d, c, ii, ij)

                current = simp.pcall_static(
                    (receiver,),
                    apply_local_step,
                    current,
                    step_ctrls,
                    idx_i_recv,
                    idx_j_recv,
                )
                ctrl_offset += num_pairs

    # Unpad if necessary
    if n_padded != original_n:

        def unpad(d: Any, orig: int) -> Any:
            return tensor.run_jax(lambda x: x[:orig], d)

        final_parties = (
            current.type.parties if isinstance(current.type, elt.MPType) else None
        )
        if final_parties is not None:
            current = simp.pcall_static(final_parties, unpad, current, original_n)
        else:
            current = unpad(current, original_n)

    # Convert back to list if input was a list
    if is_list_input:

        def unstack_to_list(d: Any, n: int) -> list:
            results = []
            for i in range(n):
                elem = tensor.run_jax(lambda x, idx=i: x[idx], d)
                results.append(elem)
            return results

        return simp.pcall_static((receiver,), unstack_to_list, current, original_n)

    return current
