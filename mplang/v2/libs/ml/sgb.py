# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code="no-untyped-def,no-any-return,var-annotated"

"""SecureBoost v2: Optimized implementation using mplang.v2 low-level BFV APIs.

This implementation improves upon v1 by leveraging BFV SIMD slots and the
groupby primitives for efficient histogram computation.

Key optimizations:
1. SIMD slot packing for parallel histogram bucket computation
2. Rotation-based aggregation for efficient slot summation
3. Reduced communication via packed ciphertext results

See design/sgb_v2.md for detailed architecture documentation.

Usage:
    from examples.v2.sgb import SecureBoost

    model = SecureBoost(n_estimators=10, max_depth=3)
    model.fit([X_ap, X_pp], y)
    predictions = model.predict([X_ap_test, X_pp_test])
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum

from mplang.v2.dialects import bfv, simp, tensor
from mplang.v2.libs.mpc.analytics import aggregation

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_FXP_BITS = 15  # Fixed-point scale = 2^15 = 32768
# BFV slot count (Increased for depth)
# NOTE: For 1M samples, the sum of gradients can reach ~3.2e10 (2^35).
# The default plain_modulus (1032193 ~ 2^20) will cause overflow.
# For large datasets, you MUST increase plain_modulus (e.g. to a 40-bit prime).
DEFAULT_POLY_MODULUS_DEGREE = 8192


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class Tree:
    """Single decision tree in flat array representation."""

    feature: list[Any]  # Per-party feature indices, shape (n_nodes,)
    threshold: list[Any]  # Per-party thresholds, shape (n_nodes,)
    value: Any  # Leaf values at AP, shape (n_nodes,)
    is_leaf: Any  # Leaf mask, shape (n_nodes,)
    owned_party_id: Any  # Node owner, shape (n_nodes,)


@dataclass
class TreeEnsemble:
    """XGBoost ensemble model."""

    max_depth: int
    trees: list[Tree]
    initial_prediction: Any  # Base prediction at AP


# ==============================================================================
# JAX Mathematical Functions
# ==============================================================================


@jax.jit
def compute_init_pred(y: jnp.ndarray) -> jnp.ndarray:
    """Compute initial prediction for binary classification (log-odds)."""
    p_base = jnp.clip(jnp.mean(y), 1e-15, 1 - 1e-15)
    return jnp.log(p_base / (1 - p_base))


@jax.jit
def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + jnp.exp(-x))


@jax.jit
def compute_gh(y_true: jnp.ndarray, y_pred_logits: jnp.ndarray) -> jnp.ndarray:
    """Compute gradient and hessian for log loss. Returns (m, 2) array."""
    p = sigmoid(y_pred_logits)
    g = p - y_true
    h = p * (1 - p)
    return jnp.column_stack([g, h])


@jax.jit
def quantize_gh(gh: jnp.ndarray, scale: int) -> jnp.ndarray:
    """Quantize float G/H to int64 for BFV encryption."""
    return jnp.round(gh * scale).astype(jnp.int64)


@jax.jit
def dequantize(arr: jnp.ndarray, scale: int) -> jnp.ndarray:
    """Dequantize int64 back to float."""
    return arr.astype(jnp.float32) / scale


# ==============================================================================
# Binning Functions
# ==============================================================================


def build_bins_equi_width(x: jnp.ndarray, max_bin: int) -> jnp.ndarray:
    """Build equi-width bin boundaries for a single feature."""
    n_samples = x.shape[0]
    n_splits = max_bin - 1
    inf_splits = jnp.full(n_splits, jnp.inf, dtype=x.dtype)

    def create_bins():
        min_val, max_val = jnp.min(x), jnp.max(x)
        is_constant = (max_val - min_val) < 1e-9

        def gen_splits():
            return jnp.linspace(min_val, max_val, num=max_bin + 1)[1:-1]

        return jax.lax.cond(is_constant, lambda: inf_splits, gen_splits)

    return jax.lax.cond(n_samples >= 2, create_bins, lambda: inf_splits)


@jax.jit
def compute_bin_indices(x: jnp.ndarray, bins: jnp.ndarray) -> jnp.ndarray:
    """Compute bin indices for all samples of a single feature."""
    return jnp.digitize(x, bins, right=True)


# ==============================================================================
# Local Histogram (AP, no FHE needed)
# ==============================================================================


def make_local_build_histogram(n_nodes: int, n_buckets: int):
    """Create a JIT-compiled local histogram builder with static n_nodes and n_buckets."""

    @jax.jit
    def local_build_histogram(
        gh: jnp.ndarray,
        bt_local: jnp.ndarray,
        bin_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """Build G/H histogram using segment_sum. Returns (n_features, n_nodes, n_buckets, 2)."""

        def hist_one_feature(bins_one: jnp.ndarray) -> jnp.ndarray:
            combined = bt_local * n_buckets + bins_one
            valid_mask = bt_local >= 0
            valid_gh = gh * valid_mask[:, None]
            return segment_sum(valid_gh, combined, num_segments=n_nodes * n_buckets)

        flat = jax.vmap(hist_one_feature, in_axes=1, out_axes=0)(bin_indices)
        return flat.reshape((bin_indices.shape[1], n_nodes, n_buckets, 2))

    return local_build_histogram


@jax.jit
def compute_best_split_from_hist(
    gh_hist: jnp.ndarray,  # (n_features, n_buckets, 2) for one node
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find best split for a single node from its histogram."""
    gh_total = jnp.sum(gh_hist, axis=1)  # (n_features, 2)
    gh_left = jnp.cumsum(gh_hist, axis=1)[:, :-1, :]  # (n_features, n_buckets-1, 2)

    g_total, h_total = gh_total[..., 0], gh_total[..., 1]
    G_left, H_left = gh_left[..., 0], gh_left[..., 1]
    G_right = g_total[:, None] - G_left
    H_right = h_total[:, None] - H_left

    score_parent = jnp.square(g_total) / (h_total + reg_lambda + 1e-9)
    score_left = jnp.square(G_left) / (H_left + reg_lambda + 1e-9)
    score_right = jnp.square(G_right) / (H_right + reg_lambda + 1e-9)

    gain = (score_left + score_right - score_parent[:, None]) / 2.0
    valid = (H_left >= min_child_weight) & (H_right >= min_child_weight)
    gain = jnp.where(valid, gain - gamma, -jnp.inf)

    flat_idx = jnp.argmax(gain)
    best_feat, best_thresh = jnp.unravel_index(flat_idx, gain.shape)
    return jnp.max(gain), best_feat, best_thresh


def local_compute_best_splits(
    gh_hist: jnp.ndarray,  # (n_features, n_nodes, n_buckets, 2)
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find best splits for all nodes. Returns (n_nodes,) arrays."""
    # Transpose to (n_nodes, n_features, n_buckets, 2)
    gh_trans = jnp.transpose(gh_hist, (1, 0, 2, 3))

    fn = partial(
        compute_best_split_from_hist,
        reg_lambda=reg_lambda,
        gamma=gamma,
        min_child_weight=min_child_weight,
    )
    return jax.vmap(fn)(gh_trans)


# ==============================================================================
# FHE Histogram (PP, using low-level BFV)
# ==============================================================================


def _build_packed_mask_jit(node_mask, feat_bins, n_buckets, stride, slot_count):
    valid = node_mask == 1
    bucket_onehot = (jnp.arange(n_buckets)[None, :] == feat_bins[:, None]) & valid[
        :, None
    ]
    running_counts = jnp.cumsum(bucket_onehot, axis=0)
    shifted_counts = jnp.zeros_like(running_counts)
    shifted_counts = shifted_counts.at[1:].set(running_counts[:-1])
    sample_offsets = jnp.take_along_axis(
        shifted_counts, feat_bins[:, None], axis=1
    ).squeeze(-1)

    scatter_indices = jnp.where(valid, feat_bins * stride + sample_offsets, -1)

    valid_mask = scatter_indices >= 0
    valid_indices = jnp.where(valid_mask, scatter_indices, 0).astype(jnp.int32)
    valid_ones = jnp.where(valid_mask, 1, 0).astype(jnp.int64)
    output = segment_sum(valid_ones, valid_indices, num_segments=slot_count)
    return output


def compute_all_masks(
    subgroup_map, bin_indices, n_buckets, stride, slot_count, n_chunks
):
    # subgroup_map: (n_nodes, m)
    # bin_indices: (m, n_features)

    m = bin_indices.shape[0]
    n_features = bin_indices.shape[1]
    n_nodes = subgroup_map.shape[0]

    # Pad
    pad_len = n_chunks * slot_count - m
    if pad_len > 0:
        subgroup_map = jnp.pad(subgroup_map, ((0, 0), (0, pad_len)))
        bin_indices = jnp.pad(bin_indices, ((0, pad_len), (0, 0)))

    # Reshape chunks
    # subgroup_map: (n_nodes, n_chunks, slot_count)
    sg_chunks = subgroup_map.reshape(n_nodes, n_chunks, slot_count)

    # bin_indices: (n_chunks, slot_count, n_features) -> (n_features, n_chunks, slot_count)
    bi_chunks = bin_indices.reshape(n_chunks, slot_count, n_features).transpose(2, 0, 1)

    # vmap over chunks
    def process_chunk(nm, fb):
        return _build_packed_mask_jit(nm, fb, n_buckets, stride, slot_count)

    v_chunk = jax.vmap(process_chunk, in_axes=(0, 0))

    # vmap over features (nm fixed, fb varies)
    v_feat = jax.vmap(v_chunk, in_axes=(None, 0))

    # vmap over nodes (nm varies, fb fixed)
    v_node = jax.vmap(v_feat, in_axes=(0, None))

    all_masks = v_node(sg_chunks, bi_chunks)
    # Flatten and convert to tuple of arrays
    return all_masks.reshape(-1, slot_count)


def _compute_histogram_chunk_batch(
    subgroup_map,
    bin_indices,
    g_cts,
    h_cts,
    encoder,
    relin_keys,
    galois_keys,
    n_nodes,
    n_features,
    n_chunks,
    n_buckets,
    slot_count,
    stride,
    max_samples_per_bucket,
    m,
):
    # Precompute all masks in one go
    compute_all_masks_jit = partial(
        compute_all_masks,
        n_buckets=n_buckets,
        stride=stride,
        slot_count=slot_count,
        n_chunks=n_chunks,
    )
    all_masks_tensor = tensor.run_jax(
        compute_all_masks_jit,
        subgroup_map,
        bin_indices,
    )

    # Batch encode all masks at once to avoid scheduler bottleneck
    # Pass relin_keys as context provider (it holds the SEALContext)
    all_masks_pt = bfv.batch_encode(all_masks_tensor, encoder, key=relin_keys)
    mask_iter = iter(all_masks_pt)

    # ==========================================================================
    # Optimization: Incremental Packing to reduce peak memory
    # ==========================================================================
    # Instead of accumulating all features and then packing, we pack incrementally.
    # This reduces peak memory from O(n_features) to O(stride).

    # Create mask for valid slots (0, stride, 2*stride, ...)
    m_np = np.zeros(slot_count, dtype=np.int64)
    idx_np = np.arange(n_buckets) * stride
    m_np[idx_np] = 1
    mask_arr = tensor.constant(m_np)
    mask_pt_pack = bfv.encode(mask_arr, encoder)

    g_packed_flat = []
    h_packed_flat = []

    # Optimization 2: Tree Reduction
    # Helper to sum a list of ciphertexts using a binary tree structure.
    # This reduces the dependency chain depth from O(N) to O(log N),
    # allowing the scheduler to parallelize additions.
    def tree_sum(items):
        if not items:
            return None
        if len(items) == 1:
            return items[0]

        queue = deque(items)
        while len(queue) > 1:
            # Process in pairs
            for _ in range(len(queue) // 2):
                left = queue.popleft()
                right = queue.popleft()
                queue.append(bfv.add(left, right))

        return queue[0] if queue else None

    for _node_idx in range(n_nodes):
        # Process features in batches of 'stride'
        for batch_start in range(0, n_features, stride):
            batch_end = min(batch_start + stride, n_features)

            g_rot_list = []
            h_rot_list = []

            for i, _feat_idx in enumerate(range(batch_start, batch_end)):
                # 1. Compute Histogram for this feature (across chunks)
                g_masked_list = []
                h_masked_list = []

                for chunk_idx in range(n_chunks):
                    mask_pt = next(mask_iter)
                    # mask_pt is already encoded via batch_encode

                    g_ct_chunk = g_cts[chunk_idx]
                    h_ct_chunk = h_cts[chunk_idx]

                    g_masked = bfv.relinearize(bfv.mul(g_ct_chunk, mask_pt), relin_keys)
                    h_masked = bfv.relinearize(bfv.mul(h_ct_chunk, mask_pt), relin_keys)

                    g_masked_list.append(g_masked)
                    h_masked_list.append(h_masked)

                g_masked_acc = tree_sum(g_masked_list)
                h_masked_acc = tree_sum(h_masked_list)

                # Lazy Aggregation: Aggregate once after summing all chunks
                # This reduces rotations by a factor of n_chunks
                g_feat_acc = aggregation.batch_bucket_aggregate(
                    g_masked_acc,
                    n_buckets,
                    max_samples_per_bucket,
                    galois_keys,
                    slot_count,
                )
                h_feat_acc = aggregation.batch_bucket_aggregate(
                    h_masked_acc,
                    n_buckets,
                    max_samples_per_bucket,
                    galois_keys,
                    slot_count,
                )

                assert g_feat_acc is not None
                assert h_feat_acc is not None

                # 2. Pack immediately
                # Relative offset = i
                # Mask valid slots
                g_masked_pack = bfv.relinearize(
                    bfv.mul(g_feat_acc, mask_pt_pack), relin_keys
                )
                h_masked_pack = bfv.relinearize(
                    bfv.mul(h_feat_acc, mask_pt_pack), relin_keys
                )

                # Rotate to position
                g_rot = bfv.rotate(g_masked_pack, -i, galois_keys)
                h_rot = bfv.rotate(h_masked_pack, -i, galois_keys)

                g_rot_list.append(g_rot)
                h_rot_list.append(h_rot)

            g_packed_acc = tree_sum(g_rot_list)
            h_packed_acc = tree_sum(h_rot_list)

            g_packed_flat.append(g_packed_acc)
            h_packed_flat.append(h_packed_acc)

    return g_packed_flat, h_packed_flat


def _process_decrypted_jit(
    g_vecs, h_vecs, scale, n_nodes, n_features, n_buckets, stride
):
    # g_vecs is list of packed vectors.
    # Shape of each vector: (slot_count,)
    g_stack = jnp.stack(g_vecs)
    h_stack = jnp.stack(h_vecs)

    # We need to reconstruct (n_nodes, n_features, n_buckets)
    g_unpacked = []
    h_unpacked = []

    cts_per_node = (n_features + stride - 1) // stride

    for node_i in range(n_nodes):
        for feat_i in range(n_features):
            # Which CT?
            ct_idx = node_i * cts_per_node + (feat_i // stride)
            # Which offset in CT?
            offset = feat_i % stride

            # Indices for buckets: b*stride + offset
            bucket_indices = jnp.arange(n_buckets) * stride + offset

            g_vals = g_stack[ct_idx, bucket_indices]
            h_vals = h_stack[ct_idx, bucket_indices]

            g_unpacked.append(g_vals)
            h_unpacked.append(h_vals)

    # Now we have flat list of (n_buckets,) arrays
    g_flat = jnp.stack(g_unpacked)  # (n_nodes*n_features, n_buckets)
    h_flat = jnp.stack(h_unpacked)

    g_buckets = g_flat.astype(jnp.float32) / scale
    h_buckets = h_flat.astype(jnp.float32) / scale

    g_cumsum = jnp.cumsum(g_buckets, axis=1)
    h_cumsum = jnp.cumsum(h_buckets, axis=1)

    g_reshaped = g_cumsum.reshape(n_nodes, n_features, n_buckets)
    h_reshaped = h_cumsum.reshape(n_nodes, n_features, n_buckets)

    combined = jnp.stack([g_reshaped, h_reshaped], axis=-1)
    return combined


def _decrypt_batch(
    g_enc_flat,
    h_enc_flat,
    sk,
    encoder,
    fxp_scale,
    n_nodes,
    n_features,
    n_buckets,
    stride,
):
    g_vecs = [bfv.decode(bfv.decrypt(ct, sk), encoder) for ct in g_enc_flat]
    h_vecs = [bfv.decode(bfv.decrypt(ct, sk), encoder) for ct in h_enc_flat]

    fn_jit = partial(
        _process_decrypted_jit,
        n_nodes=n_nodes,
        n_features=n_features,
        n_buckets=n_buckets,
        stride=stride,
    )
    return tensor.run_jax(
        fn_jit,
        g_vecs,
        h_vecs,
        fxp_scale,
    )


def fhe_encrypt_gh(
    qg: Any,
    qh: Any,
    pk: Any,
    encoder: Any,
    ap_rank: int,
    n_samples: int,
    slot_count: int = DEFAULT_POLY_MODULUS_DEGREE,
) -> tuple[list[Any], list[Any], int]:
    """Encrypt quantized G/H vectors at AP, splitting into chunks if m > slot_count.

    When m > slot_count, the vectors are split into ceil(m / slot_count) chunks,
    each encrypted as a separate ciphertext. This enables processing arbitrarily
    large datasets with a fixed poly_modulus_degree.

    Args:
        qg: Quantized G vector, shape (m,)
        qh: Quantized H vector, shape (m,)
        pk: BFV public key
        encoder: BFV encoder
        ap_rank: Active party rank
        n_samples: Number of samples (m)
        slot_count: Number of slots per ciphertext (default 4096)

    Returns:
        (g_cts, h_cts, n_chunks): Lists of encrypted G/H chunks and chunk count
    """
    # Calculate n_chunks at trace time (known statically)
    n_chunks = (n_samples + slot_count - 1) // slot_count

    g_cts: list[Any] = []
    h_cts: list[Any] = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * slot_count
        end = min((chunk_idx + 1) * slot_count, n_samples)
        chunk_size = end - start

        # Extract, pad, encode and encrypt both G and H chunks together
        def slice_pad_encode_encrypt(
            g_vec, h_vec, enc, key, s=start, e=end, cs=chunk_size, sc=slot_count
        ):
            # Slice and pad using JAX
            def slice_and_pad_both(gv, hv):
                g_chunk = gv[s:e]
                h_chunk = hv[s:e]
                if cs < sc:
                    g_chunk = jnp.pad(g_chunk, (0, sc - cs))
                    h_chunk = jnp.pad(h_chunk, (0, sc - cs))
                return g_chunk, h_chunk

            g_chunk, h_chunk = tensor.run_jax(slice_and_pad_both, g_vec, h_vec)
            # Encode and encrypt
            g_pt = bfv.encode(g_chunk, enc)
            h_pt = bfv.encode(h_chunk, enc)
            return bfv.encrypt(g_pt, key), bfv.encrypt(h_pt, key)

        g_ct, h_ct = simp.pcall_static(
            (ap_rank,), slice_pad_encode_encrypt, qg, qh, encoder, pk
        )

        g_cts.append(g_ct)
        h_cts.append(h_ct)

    return g_cts, h_cts, n_chunks


def fhe_histogram_optimized(
    g_cts: list[Any],  # List of encrypted G chunks at PP
    h_cts: list[Any],  # List of encrypted H chunks at PP
    subgroup_map: Any,  # (n_nodes, m) binary node membership
    bin_indices: Any,  # (m, n_features) binned features
    n_buckets: int,
    n_nodes: int,
    n_features: int,
    pp_rank: int,
    ap_rank: int,
    encoder: Any,
    relin_keys: Any,
    galois_keys: Any,
    m: int,
    n_chunks: int = 1,
    slot_count: int = DEFAULT_POLY_MODULUS_DEGREE,
) -> tuple[list[Any], list[Any]]:
    """Compute encrypted histogram sums using SIMD bucket packing.

    **Multi-CT Support**

    When m > slot_count, data is split into n_chunks ciphertexts:
    - Chunk 0: samples [0, slot_count)
    - Chunk 1: samples [slot_count, 2*slot_count)
    - ...

    For each chunk, we compute the histogram separately, then add results
    in the FHE domain.

    **SIMD Bucket Packing** (per chunk)

    1. Divide slot_count into n_buckets regions, each with `stride` slots
    2. Build scatter mask placing sample i at slot (bucket[i] * stride + offset[i])
    3. Single CT × packed_mask multiplication
    4. Single rotate_and_sum aggregates ALL buckets simultaneously
    5. Add chunk results together

    Returns:
        g_enc[node][feat]: List of packed encrypted G histograms (one CT per feature)
        h_enc[node][feat]: List of packed encrypted H histograms (one CT per feature)
    """
    stride = slot_count // n_buckets
    # Estimate max samples per bucket per chunk
    samples_per_chunk = (m + n_chunks - 1) // n_chunks
    max_samples_per_bucket = min(stride, max(samples_per_chunk // n_buckets * 2, 64))

    # Use partial to bake in static arguments (integers) so they are treated as static by JAX
    fn = partial(
        _compute_histogram_chunk_batch,
        n_nodes=n_nodes,
        n_features=n_features,
        n_chunks=n_chunks,
        n_buckets=n_buckets,
        slot_count=slot_count,
        stride=stride,
        max_samples_per_bucket=max_samples_per_bucket,
        m=m,
    )

    g_results_flat, h_results_flat = simp.pcall_static(
        (pp_rank,),
        fn,
        subgroup_map,
        bin_indices,
        g_cts,
        h_cts,
        encoder,
        relin_keys,
        galois_keys,
    )

    # Transfer final packed result to AP
    # g_results_flat is a list of Objects (one per node/feature/chunk accumulation)
    g_packed_ap = [
        simp.shuffle_static(obj, {ap_rank: pp_rank}) for obj in g_results_flat
    ]
    h_packed_ap = [
        simp.shuffle_static(obj, {ap_rank: pp_rank}) for obj in h_results_flat
    ]

    return g_packed_ap, h_packed_ap


def decrypt_histogram_results(
    g_enc_flat: Any,
    h_enc_flat: Any,
    sk: Any,
    encoder: Any,
    fxp_scale: int,
    n_nodes: int,
    n_features: int,
    n_buckets: int,
    ap_rank: int,
    slot_count: int = DEFAULT_POLY_MODULUS_DEGREE,
) -> list[Any]:
    """Decrypt and assemble histogram results at AP.

    **SIMD Bucket Packing Format**

    With SIMD bucket packing, each ciphertext contains ALL buckets for one feature:
    - g_enc_flat is a list of packed CTs (one per feature per node)
    - slot[b * stride] contains histogram[b] for bucket b
    - stride = slot_count // n_buckets

    We extract bucket results from strided positions, then compute cumulative sum.

    Returns list of (n_features, n_buckets, 2) arrays, one per node.
    The returned histograms are CUMULATIVE (sum of all bins <= bucket_idx).
    """
    stride = slot_count // n_buckets

    fn = partial(
        _decrypt_batch,
        fxp_scale=fxp_scale,
        n_nodes=n_nodes,
        n_features=n_features,
        n_buckets=n_buckets,
        stride=stride,
    )

    combined_results = simp.pcall_static(
        (ap_rank,),
        fn,
        g_enc_flat,
        h_enc_flat,
        sk,
        encoder,
    )

    # combined_results is (n_nodes, n_features, n_buckets, 2)
    # Convert to list of (n_features, n_buckets, 2)
    # Since combined_results is an Object, we can't iterate it in Python.
    # But the caller (build_tree) expects a list of Objects (one per node)
    # because it stacks them later: stacked = jnp.stack(hists, axis=0)

    # Wait, if combined_results is a single Object representing the whole tensor,
    # we can just return that single Object if we change the caller to handle it.
    # But build_tree expects a list.

    # Actually, build_tree does:
    # pp_hists = decrypt_histogram_results(...)
    # def find_splits(*hists):
    #     stacked = jnp.stack(hists, axis=0)
    # pp_gains, ... = simp.pcall_static(..., tensor.run_jax(find_splits, *pp_hists))

    # If pp_hists is a single tensor (n_nodes, ...), we can change find_splits to take it directly.

    return combined_results


# ==============================================================================
# Tree Update Functions
# ==============================================================================


def make_get_subgroup_map(n_nodes: int):
    """Create a JIT-compiled subgroup map function with static n_nodes."""

    @jax.jit
    def get_subgroup_map(bt_level: jnp.ndarray) -> jnp.ndarray:
        """Create one-hot node membership map. Returns (n_nodes, m)."""
        return (jnp.arange(n_nodes)[:, None] == bt_level).astype(jnp.int8)

    return get_subgroup_map


@jax.jit
def update_is_leaf(
    is_leaf: jnp.ndarray,
    gains: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    """Mark nodes as leaves if gain <= 0 or non-finite."""
    new_leaf = (gains <= 0.0) | (~jnp.isfinite(gains))
    return is_leaf.at[indices].set(new_leaf.astype(jnp.int64))


@jax.jit
def update_bt(
    bt: jnp.ndarray,
    bt_level: jnp.ndarray,
    is_leaf: jnp.ndarray,
    bin_indices: jnp.ndarray,
    best_feature: jnp.ndarray,
    best_thresh_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Update sample-to-node assignments after splitting."""
    m = bt.shape[0]
    feat_per_sample = best_feature[bt_level]
    thresh_per_sample = best_thresh_idx[bt_level]
    sample_bins = bin_indices[jnp.arange(m), feat_per_sample]

    go_left = sample_bins <= thresh_per_sample
    bt_next = jnp.where(go_left, 2 * bt + 1, 2 * bt + 2)
    return jnp.where(is_leaf[bt].astype(bool), bt, bt_next)


def make_compute_leaf_values(n_nodes: int):
    """Create a JIT-compiled leaf value computation with static n_nodes."""

    @jax.jit
    def compute_leaf_values(
        gh: jnp.ndarray,
        bt: jnp.ndarray,
        is_leaf: jnp.ndarray,
        reg_lambda: float,
    ) -> jnp.ndarray:
        """Compute leaf values from aggregated G/H."""
        sum_gh = segment_sum(gh, bt, num_segments=n_nodes)
        sum_g, sum_h = sum_gh[:, 0], sum_gh[:, 1]
        safe_h = jnp.where(sum_h == 0, 1.0, sum_h)
        leaf_vals = -sum_g / (safe_h + reg_lambda)

        has_samples = sum_h != 0
        return jnp.where(is_leaf.astype(bool) & has_samples, leaf_vals, 0.0)

    return compute_leaf_values


# ==============================================================================
# Tree Building Helpers
# ==============================================================================


def _find_splits_ap(
    ap_rank: int,
    n_level: int,
    n_buckets: int,
    gh: Any,
    bt_level: Any,
    bin_indices: Any,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[Any, Any, Any]:
    """Compute local histograms and find best splits at AP."""
    local_hist_fn = make_local_build_histogram(n_level, n_buckets)
    ap_hist = simp.pcall_static(
        (ap_rank,),
        lambda fn=local_hist_fn: tensor.run_jax(fn, gh, bt_level, bin_indices),
    )
    ap_gains, ap_feats, ap_threshs = simp.pcall_static(
        (ap_rank,),
        lambda rl=reg_lambda, gm=gamma, mcw=min_child_weight: tensor.run_jax(
            local_compute_best_splits, ap_hist, rl, gm, mcw
        ),
    )
    return ap_gains, ap_feats, ap_threshs


def _find_splits_pps(
    level: int,
    pp_ranks: list[int],
    ap_rank: int,
    g_cts_pps: dict[int, list[Any]],
    h_cts_pps: dict[int, list[Any]],
    bt_level: Any,
    all_bin_indices: list[Any],
    n_features_per_party: list[int],
    last_level_hists: list[Any],
    encoder: Any,
    relin_keys: Any,
    galois_keys: Any,
    sk: Any,
    fxp_scale: int,
    m: int,
    n_chunks: int,
    slot_count: int,
    n_buckets: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Compute remote histograms via FHE and find best splits at PPs."""
    pp_gains_list = []
    pp_feats_list = []
    pp_threshs_list = []

    n_level = 2**level

    for pp_idx, pp_rank in enumerate(pp_ranks):
        # Retrieve pre-transferred encrypted CT chunks
        g_cts_pp = g_cts_pps[pp_rank]
        h_cts_pp = h_cts_pps[pp_rank]

        # Transfer keys and other metadata to PP
        bt_level_pp = simp.shuffle_static(bt_level, {pp_rank: ap_rank})
        encoder_pp = simp.shuffle_static(encoder, {pp_rank: ap_rank})
        rk_pp = simp.shuffle_static(relin_keys, {pp_rank: ap_rank})
        gk_pp = simp.shuffle_static(galois_keys, {pp_rank: ap_rank})

        # Build subgroup map at PP
        subgroup_map_fn = make_get_subgroup_map(n_level)
        subgroup_map = simp.pcall_static(
            (pp_rank,),
            lambda fn=subgroup_map_fn, bt_lv=bt_level_pp: tensor.run_jax(fn, bt_lv),
        )

        n_pp_features = n_features_per_party[pp_idx + 1]

        if level == 0:
            # Root level: Compute full FHE
            g_enc, h_enc = fhe_histogram_optimized(
                g_cts_pp,
                h_cts_pp,
                subgroup_map,
                all_bin_indices[pp_idx + 1],
                n_buckets,
                n_level,
                n_pp_features,
                pp_rank,
                ap_rank,
                encoder_pp,
                rk_pp,
                gk_pp,
                m,
                n_chunks,
                slot_count,
            )

            pp_hists = decrypt_histogram_results(
                g_enc,
                h_enc,
                sk,
                encoder,
                fxp_scale,
                n_level,
                n_pp_features,
                n_buckets,
                ap_rank,
            )
            # Store for next level
            last_level_hists[pp_idx + 1] = pp_hists

        else:
            # Histogram Subtraction Optimization
            # 1. Slice subgroup_map to get Left children (even indices 0, 2, ...)
            def slice_left(sm):
                return sm[0::2]

            subgroup_map_left = simp.pcall_static(
                (pp_rank,),
                lambda sm=subgroup_map: tensor.run_jax(slice_left, sm),
            )

            # 2. Run FHE for Left children
            n_left = n_level // 2
            g_enc, h_enc = fhe_histogram_optimized(
                g_cts_pp,
                h_cts_pp,
                subgroup_map_left,
                all_bin_indices[pp_idx + 1],
                n_buckets,
                n_left,
                n_pp_features,
                pp_rank,
                ap_rank,
                encoder_pp,
                rk_pp,
                gk_pp,
                m,
                n_chunks,
                slot_count,
            )

            # 3. Decrypt Left
            left_hists = decrypt_histogram_results(
                g_enc,
                h_enc,
                sk,
                encoder,
                fxp_scale,
                n_left,
                n_pp_features,
                n_buckets,
                ap_rank,
            )

            # 4. Derive Right and Reconstruct
            parent_hists = last_level_hists[pp_idx + 1]

            def derive_right_and_combine(l_hists, p_hists):
                # l_hists: (n_left, ...)
                # p_hists: (n_left, ...) - parents correspond exactly to left children
                r_hists = p_hists - l_hists

                # Interleave [L, R]
                # Stack on new axis 1 -> (n_left, 2, ...)
                combined = jnp.stack([l_hists, r_hists], axis=1)
                # Reshape -> (2*n_left, ...)
                return combined.reshape((-1, *l_hists.shape[1:]))

            pp_hists = simp.pcall_static(
                (ap_rank,),
                lambda lh=left_hists, ph=parent_hists: tensor.run_jax(
                    derive_right_and_combine, lh, ph
                ),
            )

            # Store for next level (if needed)
            # Note: We don't know max_depth here, but storing it is harmless if not used
            last_level_hists[pp_idx + 1] = pp_hists

        # Stack and find best splits
        def find_splits(hists, rl=reg_lambda, gm=gamma, mcw=min_child_weight):
            # hists is already (n_nodes, n_feat, n_buck, 2)
            return jax.vmap(lambda h: compute_best_split_from_hist(h, rl, gm, mcw))(
                hists
            )

        pp_gains, pp_feats, pp_threshs = simp.pcall_static(
            (ap_rank,),
            lambda h=pp_hists: tensor.run_jax(find_splits, h),
        )

        pp_gains_list.append(pp_gains)
        pp_feats_list.append(pp_feats)
        pp_threshs_list.append(pp_threshs)

    return pp_gains_list, pp_feats_list, pp_threshs_list


def _update_tree_state(
    ap_rank: int,
    pp_ranks: list[int],
    all_ranks: list[int],
    all_feats: list[Any],
    all_thresholds: list[Any],
    bt: Any,
    bt_level: Any,
    is_leaf: Any,
    owned_party: Any,
    cur_indices: Any,
    best_party: Any,
    best_gains: Any,
    all_feats_level: list[Any],
    all_threshs_level: list[Any],
    all_bins: list[Any],
    all_bin_indices: list[Any],
) -> tuple[Any, Any, list[Any], list[Any], Any]:
    """Update tree structure and sample assignments based on best splits."""
    # Update is_leaf
    is_leaf = simp.pcall_static(
        (ap_rank,),
        lambda: tensor.run_jax(update_is_leaf, is_leaf, best_gains, cur_indices),
    )

    # Broadcast is_leaf to all parties (keep source, shuffle to each target, then converge)
    if pp_ranks:
        is_leaf_parts = [is_leaf]  # Start with AP's copy
        for r in pp_ranks:
            is_leaf_parts.append(simp.shuffle_static(is_leaf, {r: ap_rank}))
        is_leaf = simp.converge(*is_leaf_parts)

    # Update owned_party
    owned_party = simp.pcall_static(
        (ap_rank,),
        lambda: tensor.run_jax(
            lambda op, bp, ci: op.at[ci].set(bp),
            owned_party,
            best_party,
            cur_indices,
        ),
    )

    # Broadcast owned_party to all parties
    if pp_ranks:
        owned_party_parts = [owned_party]
        for r in pp_ranks:
            owned_party_parts.append(simp.shuffle_static(owned_party, {r: ap_rank}))
        owned_party = simp.converge(*owned_party_parts)

    # === Update features and thresholds for each party ===
    # Route best_feats/best_threshs to correct parties based on best_party
    all_tmp_bt: list[Any] = []

    for party_idx, party_rank in enumerate(all_ranks):
        # Transfer data to this party if needed
        if party_idx > 0:
            # PP's results are already at AP, send back to PP
            all_feats_level[party_idx] = simp.shuffle_static(
                all_feats_level[party_idx], {party_rank: ap_rank}
            )
            all_threshs_level[party_idx] = simp.shuffle_static(
                all_threshs_level[party_idx], {party_rank: ap_rank}
            )
            # Also need cur_indices, owned_party, is_leaf at PP
            cur_indices_party = simp.shuffle_static(cur_indices, {party_rank: ap_rank})
            owned_party_party = simp.shuffle_static(owned_party, {party_rank: ap_rank})
            is_leaf_party = simp.shuffle_static(is_leaf, {party_rank: ap_rank})
        else:
            cur_indices_party = cur_indices
            owned_party_party = owned_party
            is_leaf_party = is_leaf

        # Update this party's feature and threshold arrays
        def update_party_feats(
            feats,
            best_feat,
            indices,
            owned,
            leaf,
            pid=party_idx,
        ):
            tmp = feats.at[indices].set(best_feat)
            tmp = jnp.where(leaf.astype(bool), jnp.int64(-1), tmp)
            mask = owned == pid
            return jnp.where(mask, tmp, jnp.int64(-1))

        all_feats[party_idx] = simp.pcall_static(
            (party_rank,),
            lambda pf=all_feats[party_idx],
            bf=all_feats_level[party_idx],
            ci=cur_indices_party,
            op=owned_party_party,
            il=is_leaf_party: tensor.run_jax(update_party_feats, pf, bf, ci, op, il),
        )

        def update_party_thresholds(
            thresholds,
            bins_arr,
            best_feat,
            best_thresh_idx,
            indices,
            owned,
            leaf,
            pid=party_idx,
        ):
            # Get actual threshold values from bins
            best_thresh = bins_arr[best_feat, best_thresh_idx]
            tmp = thresholds.at[indices].set(best_thresh)
            tmp = jnp.where(leaf.astype(bool), jnp.float32(jnp.inf), tmp)
            mask = owned == pid
            return jnp.where(mask, tmp, jnp.float32(jnp.inf))

        all_thresholds[party_idx] = simp.pcall_static(
            (party_rank,),
            lambda pt=all_thresholds[party_idx],
            b=all_bins[party_idx],
            bf=all_feats_level[party_idx],
            bt_idx=all_threshs_level[party_idx],
            ci=cur_indices_party,
            op=owned_party_party,
            il=is_leaf_party: tensor.run_jax(
                update_party_thresholds,
                pt,
                b,
                bf,
                bt_idx,
                ci,
                op,
                il,
            ),
        )

        # Compute temporary bt for this party
        # Need bt and bt_level at this party too
        if party_idx > 0:
            bt_party = simp.shuffle_static(bt, {party_rank: ap_rank})
            bt_level_party = simp.shuffle_static(bt_level, {party_rank: ap_rank})
        else:
            bt_party = bt
            bt_level_party = bt_level

        tmp_bt = simp.pcall_static(
            (party_rank,),
            lambda bi=all_bin_indices[party_idx],
            bf=all_feats_level[party_idx],
            bt_idx=all_threshs_level[party_idx],
            bt_arr=bt_party,
            bt_lv=bt_level_party,
            il=is_leaf_party: tensor.run_jax(
                update_bt, bt_arr, bt_lv, il, bi, bf, bt_idx
            ),
        )

        # Transfer PP's tmp_bt to AP for merging
        if party_idx > 0:
            tmp_bt = simp.shuffle_static(tmp_bt, {ap_rank: party_rank})

        all_tmp_bt.append(tmp_bt)

    # === Merge bt updates based on best_party ===
    def merge_bt_updates(
        current_bt,
        all_tmp,
        best_party_arr,
        level_indices,
    ):
        stacked = jnp.stack(all_tmp, axis=0)  # (n_parties, m)
        updated_bt = current_bt

        def update_for_node(carry, i):
            bt_arr = carry
            node_idx = level_indices[i]
            winning_party = best_party_arr[i]
            samples_in_node = current_bt == node_idx
            winning_bt = stacked[winning_party]
            return jnp.where(samples_in_node, winning_bt, bt_arr), None

        updated_bt, _ = jax.lax.scan(
            update_for_node, updated_bt, jnp.arange(len(level_indices))
        )
        return updated_bt

    bt = simp.pcall_static(
        (ap_rank,),
        lambda: tensor.run_jax(
            merge_bt_updates, bt, all_tmp_bt, best_party, cur_indices
        ),
    )

    # Broadcast updated bt to all parties
    if pp_ranks:
        bt_parts = [bt]
        for r in pp_ranks:
            bt_parts.append(simp.shuffle_static(bt, {r: ap_rank}))
        bt = simp.converge(*bt_parts)

    return is_leaf, owned_party, all_feats, all_thresholds, bt


def build_tree(
    gh: Any,  # Plaintext G/H at AP, shape (m, 2)
    g_cts: list[Any],  # Encrypted G chunks at AP
    h_cts: list[Any],  # Encrypted H chunks at AP
    n_chunks: int,  # Number of CT chunks
    all_bins: list[Any],  # Bin boundaries per party
    all_bin_indices: list[Any],  # Binned features per party
    sk: Any,  # Secret key at AP
    pk: Any,  # Public key at AP
    encoder: Any,  # BFV encoder
    relin_keys: Any,  # Relinearization keys
    galois_keys: Any,  # Galois keys for rotation
    fxp_scale: int,
    ap_rank: int,
    pp_ranks: list[int],
    max_depth: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    n_samples: int,
    n_buckets: int,
    n_features_per_party: list[int],  # Number of features for each party
    slot_count: int = DEFAULT_POLY_MODULUS_DEGREE,
) -> Tree:
    """Build a single decision tree level by level.

    The algorithm proceeds breadth-first:
    1. For each level, compute histograms (local at AP, FHE at PPs)
    2. Find best split per node across all parties
    3. Update tree structure and sample assignments
    4. Repeat until max_depth reached

    **Multi-CT Support**: When n_samples > slot_count, data is split into
    n_chunks ciphertexts. Each chunk is processed separately and results
    are accumulated in the FHE domain.
    """
    m = n_samples
    n_nodes = 2 ** (max_depth + 1) - 1
    all_ranks = [ap_rank, *pp_ranks]

    # Initialize tree arrays
    def init_array(rank, shape, dtype, fill):
        return simp.pcall_static(
            (rank,),
            lambda: tensor.constant(np.full(shape, fill, dtype=dtype)),
        )

    all_feats = [init_array(r, n_nodes, np.int64, -1) for r in all_ranks]
    all_thresholds = [init_array(r, n_nodes, np.float32, np.inf) for r in all_ranks]
    values = init_array(ap_rank, n_nodes, np.float32, 0.0)
    is_leaf = init_array(ap_rank, n_nodes, np.int64, 0)
    owned_party = init_array(ap_rank, n_nodes, np.int64, -1)
    bt = init_array(ap_rank, m, np.int64, 0)

    # Store parent histograms for subtraction optimization
    # List of TraceObjects (JAX arrays) representing stacked histograms for previous level
    # Index 0 is AP (unused), 1..k are PPs
    last_level_hists: list[Any] = [None] * (len(pp_ranks) + 1)

    # Optimization 1: Hoist Ciphertext Transfer
    # Transfer encrypted gradients to all PPs once, before the tree building loop.
    g_cts_pps: dict[int, list[Any]] = {}
    h_cts_pps: dict[int, list[Any]] = {}

    for pp_rank in pp_ranks:
        g_cts_pps[pp_rank] = [
            simp.shuffle_static(ct, {pp_rank: ap_rank}) for ct in g_cts
        ]
        h_cts_pps[pp_rank] = [
            simp.shuffle_static(ct, {pp_rank: ap_rank}) for ct in h_cts
        ]

    for level in range(max_depth):
        n_level = 2**level
        level_offset = 2**level - 1

        cur_indices = simp.pcall_static(
            (ap_rank,),
            lambda off=level_offset, nl=n_level: tensor.constant(
                np.arange(nl, dtype=np.int64) + off
            ),
        )

        # Local bt for this level
        bt_level = simp.pcall_static(
            (ap_rank,),
            lambda off=level_offset, b=bt: tensor.run_jax(lambda x: x - off, b),
        )

        # === AP: Local histogram computation ===
        ap_gains, ap_feats, ap_threshs = _find_splits_ap(
            ap_rank,
            n_level,
            n_buckets,
            gh,
            bt_level,
            all_bin_indices[0],
            reg_lambda,
            gamma,
            min_child_weight,
        )

        all_gains = [ap_gains]
        all_feats_level = [ap_feats]
        all_threshs_level = [ap_threshs]

        # === PP: FHE histogram computation ===
        pp_gains_list, pp_feats_list, pp_threshs_list = _find_splits_pps(
            level,
            pp_ranks,
            ap_rank,
            g_cts_pps,
            h_cts_pps,
            bt_level,
            all_bin_indices,
            n_features_per_party,
            last_level_hists,
            encoder,
            relin_keys,
            galois_keys,
            sk,
            fxp_scale,
            m,
            n_chunks,
            slot_count,
            n_buckets,
            reg_lambda,
            gamma,
            min_child_weight,
        )

        all_gains.extend(pp_gains_list)
        all_feats_level.extend(pp_feats_list)
        all_threshs_level.extend(pp_threshs_list)

        # === Find global best split across all parties ===
        def find_global_best(*gains):
            stacked = jnp.stack(gains, axis=0)  # (n_parties, n_nodes)
            best_party = jnp.argmax(stacked, axis=0)
            best_gains = jnp.take_along_axis(
                stacked, best_party[None, :], axis=0
            ).squeeze(0)
            return best_gains, best_party

        best_gains, best_party = simp.pcall_static(
            (ap_rank,),
            lambda gains=all_gains: tensor.run_jax(find_global_best, *gains),
        )

        # === Update Tree State ===
        is_leaf, owned_party, all_feats, all_thresholds, bt = _update_tree_state(
            ap_rank,
            pp_ranks,
            all_ranks,
            all_feats,
            all_thresholds,
            bt,
            bt_level,
            is_leaf,
            owned_party,
            cur_indices,
            best_party,
            best_gains,
            all_feats_level,
            all_threshs_level,
            all_bins,
            all_bin_indices,
        )

    # Force final level nodes to be leaves
    final_start = 2**max_depth - 1
    final_end = 2 ** (max_depth + 1) - 1
    final_indices = simp.pcall_static(
        (ap_rank,),
        lambda: tensor.constant(np.arange(final_start, final_end, dtype=np.int64)),
    )
    is_leaf = simp.pcall_static(
        (ap_rank,),
        lambda: tensor.run_jax(
            lambda il, fi: il.at[fi].set(1),
            is_leaf,
            final_indices,
        ),
    )

    # Broadcast final is_leaf to all parties (needed for prediction)
    # Note: owned_party is already converged to all parties during the level loop
    if pp_ranks:
        is_leaf_parts = [is_leaf]
        for r in pp_ranks:
            is_leaf_parts.append(simp.shuffle_static(is_leaf, {r: ap_rank}))
        is_leaf = simp.converge(*is_leaf_parts)

    # Compute final leaf values
    leaf_val_fn = make_compute_leaf_values(n_nodes)
    values = simp.pcall_static(
        (ap_rank,),
        lambda fn=leaf_val_fn: tensor.run_jax(fn, gh, bt, is_leaf, reg_lambda),
    )

    return Tree(
        feature=all_feats,
        threshold=all_thresholds,
        value=values,
        is_leaf=is_leaf,
        owned_party_id=owned_party,
    )


# ==============================================================================
# Prediction
# ==============================================================================


def predict_tree_single_party(
    data: Any,
    feature: Any,
    threshold: Any,
    is_leaf: Any,
    owned_party_id: Any,
    party_id: int,
    n_nodes: int,
) -> Any:
    """Local tree traversal for a single party.

    Returns a location matrix (m, n_nodes) where each sample may be in multiple
    nodes if splits are owned by other parties.
    """

    def traverse_kernel(
        data_arr,
        feat_arr,
        thresh_arr,
        leaf_arr,
        owner_arr,
    ):
        n_samples = data_arr.shape[0]
        # Start all samples at root
        locations = jnp.zeros((n_samples, n_nodes), dtype=jnp.int64).at[:, 0].set(1)

        def propagate(i, locs):
            is_my_split = (leaf_arr[i] == 0) & (owner_arr[i] == party_id)

            def process_my_split(locs_inner):
                samples_here = locs_inner[:, i]
                feat_idx = feat_arr[i]
                thresh = thresh_arr[i]
                go_left = data_arr[:, feat_idx] <= thresh
                to_left = samples_here * go_left.astype(jnp.int64)
                to_right = samples_here * (1 - go_left.astype(jnp.int64))
                locs_inner = locs_inner.at[:, 2 * i + 1].add(to_left)
                locs_inner = locs_inner.at[:, 2 * i + 2].add(to_right)
                return locs_inner.at[:, i].set(0)

            def propagate_unknown(locs_inner):
                is_split = leaf_arr[i] == 0

                def propagate_both(loc):
                    samples_here = loc[:, i]
                    loc = loc.at[:, 2 * i + 1].add(samples_here)
                    loc = loc.at[:, 2 * i + 2].add(samples_here)
                    return loc.at[:, i].set(0)

                return jax.lax.cond(is_split, propagate_both, lambda x: x, locs_inner)

            return jax.lax.cond(is_my_split, process_my_split, propagate_unknown, locs)

        return jax.lax.fori_loop(0, n_nodes // 2, propagate, locations)

    return tensor.run_jax(
        traverse_kernel, data, feature, threshold, is_leaf, owned_party_id
    )


def predict_tree(
    tree: Tree,
    all_datas: list[Any],
    ap_rank: int,
    pp_ranks: list[int],
    n_nodes: int,
) -> Any:
    """Predict using a single tree by aggregating location masks from all parties."""
    all_ranks = [ap_rank, *pp_ranks]

    # Each party computes its local traversal
    all_masks: list[Any] = []

    for i, rank in enumerate(all_ranks):
        mask = simp.pcall_static(
            (rank,),
            lambda d=all_datas[i],
            f=tree.feature[i],
            t=tree.threshold[i],
            idx=i: predict_tree_single_party(
                d, f, t, tree.is_leaf, tree.owned_party_id, idx, n_nodes
            ),
        )
        # Transfer to AP
        if rank != ap_rank:
            mask = simp.shuffle_static(mask, {ap_rank: rank})
        all_masks.append(mask)

    # Aggregate masks at AP
    def aggregate_predictions(
        *masks,
        leaf_arr,
        values_arr,
    ):
        stacked = jnp.stack(masks, axis=0)  # (n_parties, m, n_nodes)
        # Consensus: sample is at node only if ALL parties agree
        consensus = jnp.all(stacked > 0, axis=0)  # (m, n_nodes)
        # Find leaf nodes
        final_leaf_mask = consensus * leaf_arr.astype(bool)
        # Get leaf index for each sample
        leaf_indices = jnp.argmax(final_leaf_mask, axis=1)
        return values_arr[leaf_indices]

    predictions = simp.pcall_static(
        (ap_rank,),
        lambda: tensor.run_jax(
            aggregate_predictions,
            *all_masks,
            leaf_arr=tree.is_leaf,
            values_arr=tree.value,
        ),
    )

    return predictions


def predict_ensemble(
    model: TreeEnsemble,
    all_datas: list[Any],
    ap_rank: int,
    pp_ranks: list[int],
    learning_rate: float,
    n_samples: int,
    n_nodes: int,
) -> Any:
    """Predict using the full ensemble."""
    m = n_samples

    # Start with initial prediction
    y_pred_logits = simp.pcall_static(
        (ap_rank,),
        lambda n=m: tensor.run_jax(
            lambda init: init * jnp.ones(n), model.initial_prediction
        ),
    )

    # Add predictions from each tree
    for tree in model.trees:
        tree_pred = predict_tree(tree, all_datas, ap_rank, pp_ranks, n_nodes)

        def update_pred(y_pred, pred, lr=learning_rate):
            return y_pred + lr * pred

        y_pred_logits = simp.pcall_static(
            (ap_rank,),
            lambda yp=y_pred_logits, tp=tree_pred: tensor.run_jax(update_pred, yp, tp),
        )

    # Convert logits to probabilities
    y_prob = simp.pcall_static(
        (ap_rank,),
        lambda: tensor.run_jax(sigmoid, y_pred_logits),
    )

    return y_prob


# ==============================================================================
# Training API
# ==============================================================================


def fit_tree_ensemble(
    all_datas: list[Any],
    y_data: Any,
    all_bins: list[Any],
    all_bin_indices: list[Any],
    initial_pred: Any,
    n_samples: int,
    n_buckets: int,
    n_features_per_party: list[int],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    ap_rank: int,
    pp_ranks: list[int],
) -> TreeEnsemble:
    """Fit a SecureBoost tree ensemble."""
    m = n_samples
    fxp_scale = 1 << DEFAULT_FXP_BITS

    y_pred = simp.pcall_static(
        (ap_rank,),
        lambda n=m: tensor.run_jax(lambda init: init * jnp.ones(n), initial_pred),
    )

    # BFV key generation at AP (only if we have passive parties)
    pk, sk, relin_keys, galois_keys, encoder = None, None, None, None, None
    if pp_ranks:

        def keygen_fn():
            pub, sec = bfv.keygen(poly_modulus_degree=DEFAULT_POLY_MODULUS_DEGREE)
            rk = bfv.make_relin_keys(sec)
            gk = bfv.make_galois_keys(sec)
            enc = bfv.create_encoder(poly_modulus_degree=DEFAULT_POLY_MODULUS_DEGREE)
            return pub, sec, rk, gk, enc

        pk, sk, relin_keys, galois_keys, encoder = simp.pcall_static(
            (ap_rank,), keygen_fn
        )

    trees: list[Tree] = []

    for _tree_idx in range(n_estimators):
        # Compute G/H, quantize, and split into qg/qh in one call
        def compute_gh_quantized(y_true, y_pred_logits, scale):
            gh = compute_gh(y_true, y_pred_logits)
            qgh = quantize_gh(gh, scale)
            return gh, qgh[:, 0], qgh[:, 1]

        gh, qg, qh = simp.pcall_static(
            (ap_rank,),
            lambda yp=y_pred: tensor.run_jax(
                compute_gh_quantized, y_data, yp, fxp_scale
            ),
        )

        # FHE encrypt only if we have passive parties
        g_cts, h_cts, n_chunks = [], [], 1
        if pp_ranks:
            g_cts, h_cts, n_chunks = fhe_encrypt_gh(
                qg, qh, pk, encoder, ap_rank, n_samples
            )

        tree = build_tree(
            gh,
            g_cts,
            h_cts,
            n_chunks,
            all_bins,
            all_bin_indices,
            sk,
            pk,
            encoder,
            relin_keys,
            galois_keys,
            fxp_scale,
            ap_rank,
            pp_ranks,
            max_depth,
            reg_lambda,
            gamma,
            min_child_weight,
            n_samples,
            n_buckets,
            n_features_per_party,
        )
        trees.append(tree)

        # Predict tree and update y_pred
        n_nodes = 2 ** (max_depth + 1) - 1
        tree_pred = predict_tree(tree, all_datas, ap_rank, pp_ranks, n_nodes)

        def update_pred_fn(curr_y, t_pred, lr=learning_rate):
            return curr_y + lr * t_pred

        y_pred = simp.pcall_static(
            (ap_rank,),
            lambda yp=y_pred, tp=tree_pred: tensor.run_jax(update_pred_fn, yp, tp),
        )

    return TreeEnsemble(
        max_depth=max_depth,
        trees=trees,
        initial_prediction=initial_pred,
    )


# ==============================================================================
# SecureBoost Class
# ==============================================================================


class SecureBoost:
    """SecureBoost classifier using mplang.v2 low-level BFV APIs.

    This is an optimized implementation that uses BFV SIMD slots for
    efficient histogram computation.

    Example:
        model = SecureBoost(n_estimators=10, max_depth=3)
        model.fit([X_ap, X_pp], y)
        predictions = model.predict([X_ap_test, X_pp_test])
    """

    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        max_bin: int = 8,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        ap_rank: int = 0,
        pp_ranks: list[int] | None = None,
    ):
        """Initialize SecureBoost model.

        Args:
            n_estimators: Number of trees to train
            learning_rate: Shrinkage factor for updates
            max_depth: Maximum tree depth
            max_bin: Maximum number of bins per feature
            reg_lambda: L2 regularization on leaf weights
            gamma: Minimum gain required to split
            min_child_weight: Minimum hessian sum in children
            ap_rank: Active party rank (holds labels)
            pp_ranks: Passive party ranks (hold features)
        """
        if max_bin < 2:
            raise ValueError(f"max_bin must be >= 2, got {max_bin}")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bin = max_bin
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.ap_rank = ap_rank
        self.pp_ranks = pp_ranks if pp_ranks is not None else [1]
        self.model: TreeEnsemble | None = None

    def fit(
        self,
        all_datas: list[Any],
        y_data: Any,
        n_samples: int,
        n_features_per_party: list[int],
    ) -> SecureBoost:
        """Fit the SecureBoost model.

        Args:
            all_datas: List of feature tensors, one per party.
                       First element is AP's features, rest are PPs'.
            y_data: Labels tensor at AP.
            n_samples: Number of training samples.
            n_features_per_party: Number of features for each party.

        Returns:
            self for method chaining
        """
        self.n_samples = n_samples
        self.n_features_per_party = n_features_per_party
        # Build bins for each party
        all_ranks = [self.ap_rank, *self.pp_ranks]

        build_bins_vmap = jax.vmap(
            partial(build_bins_equi_width, max_bin=self.max_bin), in_axes=1
        )
        compute_indices_vmap = jax.vmap(compute_bin_indices, in_axes=(1, 0), out_axes=1)

        all_bins: list[Any] = []
        all_bin_indices: list[Any] = []

        for i, rank in enumerate(all_ranks):
            data = all_datas[i]
            bins = simp.pcall_static(
                (rank,),
                lambda d=data: tensor.run_jax(build_bins_vmap, d),
            )
            indices = simp.pcall_static(
                (rank,),
                lambda d=data, b=bins: tensor.run_jax(compute_indices_vmap, d, b),
            )
            all_bins.append(bins)
            all_bin_indices.append(indices)

        # Initial prediction
        initial_pred = simp.pcall_static(
            (self.ap_rank,),
            lambda: tensor.run_jax(compute_init_pred, y_data),
        )

        # Calculate metadata
        n_buckets = self.max_bin + 1
        n_features_per_party = self.n_features_per_party

        self.model = fit_tree_ensemble(
            all_datas,
            y_data,
            all_bins,
            all_bin_indices,
            initial_pred,
            self.n_samples,
            n_buckets,
            n_features_per_party,
            self.n_estimators,
            self.learning_rate,
            self.max_depth,
            self.reg_lambda,
            self.gamma,
            self.min_child_weight,
            self.ap_rank,
            self.pp_ranks,
        )

        return self

    def predict(self, all_datas: list[Any], n_samples: int) -> Any:
        """Predict probabilities for new data.

        Args:
            all_datas: List of feature tensors, one per party.
            n_samples: Number of samples.

        Returns:
            Predicted probabilities at AP.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_nodes = 2 ** (self.max_depth + 1) - 1
        return predict_ensemble(
            self.model,
            all_datas,
            self.ap_rank,
            self.pp_ranks,
            self.learning_rate,
            n_samples,
            n_nodes,
        )

    def predict_proba(self, all_datas: list[Any], n_samples: int) -> Any:
        """Alias for predict()."""
        return self.predict(all_datas, n_samples)

    def evaluate(self, all_datas: list[Any], y_data: Any, n_samples: int) -> Any:
        """Evaluate model on test data.

        Returns:
            Accuracy tensor at AP (needs to be fetched after graph execution).
        """
        y_prob = self.predict(all_datas, n_samples)

        def compute_metrics(y_pred, y_true):
            y_class = (y_pred > 0.5).astype(jnp.float32)
            accuracy = jnp.mean(y_class == y_true)
            return accuracy

        accuracy = simp.pcall_static(
            (self.ap_rank,),
            lambda: tensor.run_jax(compute_metrics, y_prob, y_data),
        )
        return accuracy
