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

"""Silent OT (Random VOLE) Implementation.

Implements "Silent Random VOLE" via Linear Expansion (LPN-like).
This provides O(N) local computation but O(k) communication.
"""

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
import mplang.v2.libs.mpc.vole.gilboa as vole
from mplang.v2.dialects import crypto, field, simp, tensor


def silent_vole_random_u(
    sender: int,
    receiver: int,
    n: int,
    base_k: int = 1024,
) -> tuple[el.Object, el.Object, el.Object, el.Object]:
    """Execute Silent Random VOLE (Linear Expansion).

    Args:
        sender: Rank of Sender.
        receiver: Rank of Receiver.
        n: Target vector length (e.g. 10^9).
        base_k: Size of Base VOLE (LPN parameter).

    Returns:
        v, w, u, delta
        Where w = v + u * delta.
        u is RANDOM.
    """

    # 1. Base VOLE (Standard Gilboa)
    # We need providers for base_u and base_delta.

    def _base_u_provider() -> el.Object:
        # Random U_base (base_k, 2) using new API
        return crypto.random_tensor((base_k, 2), elt.u64)

    def _base_delta_provider() -> el.Object:
        # Random Delta (2,) using new API
        return crypto.random_tensor((2,), elt.u64)

    # v_base: (k, 2), w_base: (k, 2)
    # The return type is a Union, mypy complains about unpacking.
    # We ignore the type error here as we know return_secrets=True returns 4 values.
    v_base, w_base, u_base, delta = vole.vole(  # type: ignore
        sender,
        receiver,
        base_k,
        _base_u_provider,
        _base_delta_provider,
        return_secrets=True,
    )

    # 2. Linear Expansion
    # We rely on a public seed for the mixing matrix M.
    seed = simp.pcall_static((sender,), lambda: crypto.random_bytes(32))
    # Share seed (Receiver needs it too)
    seed_recv = simp.shuffle_static(seed, {receiver: sender})  # S -> R

    # Expansion Logic
    # We process in chunks to avoid massive implementation limit or memory issues.
    # But EDSL graph optimization might handle loops?
    # For safe side, let's implement a loop over chunks in Python if N is large.
    # However, N is dynamic usually? Here N is int param.

    # Chunk size
    CHUNK_SIZE = 100_000  # 100k items per chunk

    # We need to broadcast delta and bases to expansion function?
    # Actually, we can just expand v_base -> v_long, w_base -> w_long.
    # u_long is implicit (u_base * M).
    # Since we need to return u, we expand u_base too.

    # Define expansion op
    def _expand_chunk(base_vec: Any, mask_packed: Any, chunk_len: int) -> Any:
        # base_vec: (K, 2) u64
        # mask_packed: (K, blocks, 2) u64 (AES output)
        # chunk_len: number of bits to extract

        # 1. Unpack bits from mask_packed
        # mask_packed is (K, blocks, 2) u64.
        # View as u8: (K, blocks, 16)
        mask_u8 = mask_packed.view(jnp.uint8)

        # Unpack bits: (K, blocks, 16, 8)
        bits = jnp.unpackbits(mask_u8, bitorder="little")

        # Flatten to (K, total_bits)
        bits_flat = bits.reshape(base_k, -1)

        # Slice to chunk_len
        mask_mat = bits_flat[:, :chunk_len].astype(jnp.uint64)

        # Broadcast base_vec (K, 2)
        # We want: out[c] = XOR_sum_j (base[j] * mask[j, c])

        base_shuffled = base_vec.reshape(base_k, 1, 2)
        mask_expanded = mask_mat.reshape(base_k, chunk_len, 1)

        # term[j, c] = base[j] * mask[j, c]
        # mask is 0 or 1 (uint64). Multiplication works as selection.
        terms = base_shuffled * mask_expanded  # (K, chunk, 2)

        # XOR Reduce over K
        # Use simple loop or scan.
        # terms: (K, chunk, 2)

        def _xor_scan(carry: Any, x: Any) -> tuple[Any, Any]:
            new_carry = jnp.bitwise_xor(carry, x)
            return new_carry, None

        # init: (chunk, 2) zeros
        init_val = jnp.zeros((chunk_len, 2), dtype=jnp.uint64)

        from jax import lax

        res, _ = lax.scan(_xor_scan, init_val, terms)

        return res

    # 3. Orchestration
    # We iterate chunks on Host? Or use `scan`?
    # Host loop is easier for Memory management (Streaming).
    # Return a "Lazy Object" or List of Objects?
    # The signature `silent_vole` usually returns full Tensor.
    # User requirement: "Silent OT" to reduce communications.
    # If we return a full (N,) tensor, we solved bandwidth but not RAM.
    # But for Phase 2 task "Protocol Upgrade", bandwidth is key.
    # Phase 2 task "Streaming" handles RAM.
    # So returning full Tensor is "okay" for now, although it might OOM 1B.
    # Let's implement blocked execution and stack? No, that OOMs.

    # We will implement `silent_vole_random_u` to return a `BigTensor` handle?
    # Or just `el.Object` (which might be huge).
    # Since we are in EDSL, the `el.Object` represents the *computation*.
    # If we return a graph that produces (10^9,) tensor, the Evaluator might crash trying to allocate it.

    # Let's just implement loop and return concatenated for now, assume 10^7-10^8 test case.
    # For 10^9, we rely on Streaming Refactor later.

    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    def _run_expansion(b: Any, seed_val: Any) -> el.Object:
        # b: base (K, 2)
        # seed_val: (32,) u8

        # 1. Derive K seeds from master seed using combined run_jax block
        def _view_slice_reshape(b: Any) -> Any:
            # View as u64, slice first row, then reshape for AES expand
            u64_view = b.view(jnp.uint64).reshape(-1, 2)
            master_seed = u64_view[:1]  # (1, 2)
            return master_seed

        master_seed = tensor.run_jax(_view_slice_reshape, seed_val)

        # Expand to K seeds: (1, K, 2)
        row_seeds_packed = field.aes_expand(master_seed, base_k)
        # Reshape using run_jax for XLA optimization
        row_seeds = tensor.run_jax(lambda x: x.reshape(base_k, 2), row_seeds_packed)

        # Iterate chunks
        local_res = []
        for i in range(num_chunks):
            this_len = min(CHUNK_SIZE, n - i * CHUNK_SIZE)

            # Generate mask for this chunk using AES
            # Need ceil(this_len / 128) blocks
            num_blocks = (this_len + 127) // 128
            mask_packed = field.aes_expand(row_seeds, num_blocks)

            # We must use `tensor.run_jax` so logic runs on device
            def _core(base: Any, mask: Any, this_len: int = this_len) -> Any:
                return _expand_chunk(base, mask, this_len)

            chunk_res = tensor.run_jax(_core, b, mask_packed)
            local_res.append(chunk_res)

        # Use run_jax for concat to enable XLA fusion
        if len(local_res) == 1:
            return cast(el.Object, local_res[0])

        def _concat_chunks(*chunks: Any) -> Any:
            return jnp.concatenate(chunks, axis=0)

        return cast(el.Object, tensor.run_jax(_concat_chunks, *local_res))

    # Execute on Sender
    v_long = simp.pcall_static((sender,), _run_expansion, v_base, seed)
    # Execute on Receiver
    w_long = simp.pcall_static((receiver,), _run_expansion, w_base, seed_recv)

    # U expansion
    u_long = simp.pcall_static((sender,), _run_expansion, u_base, seed)

    # Delta is scalar, reusable

    return v_long, w_long, u_long, delta
