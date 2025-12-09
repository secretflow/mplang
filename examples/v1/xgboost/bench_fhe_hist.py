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

"""
Microbenchmark for FHE(BFV) histogram path in SecureBoost.

This script measures the time to compute PP-side cumulative bucket sums
via encrypted ct·ct dot products using TenSEAL/SEAL BFV vector backend.

It provides two modes:
- classic: separate g/h ciphertexts + ct·ct dot (current training path)
- interleaved: interleave g/h into one ct, do one ct·ct mul + two ct·pt dots (even/odd)

Usage examples:
    uv run -q python examples/v1/xgboost/bench_fhe_hist.py --world-size 2 --m 4096 --n-total 16 --n-ap 4 --k 16 --t 4 --reps 3 --mode classic
    uv run -q python examples/v1/xgboost/bench_fhe_hist.py --world-size 2 --m 4096 --n-total 16 --n-ap 4 --k 16 --t 4 --reps 3 --mode interleaved
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from examples.xgboost.sgb import (
    DEFAULT_FXP_BITS,
    batch_feature_wise_bucket_sum_fhe_vector,
    build_bins_equi_width,
    compute_bin_indices,
    compute_gh,
    compute_init_pred,
    quantize_gh,
)

import mplang.v1 as mp
from mplang.v1.ops import fhe


def _gen_data(n_samples: int, n_total_features: int, n_features_ap: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_total_features)).astype(np.float32)
    # make a simple linear label with noise
    w = rng.normal(size=(n_total_features,)).astype(np.float32)
    z = X @ w + 0.1 * rng.normal(size=(n_samples,)).astype(np.float32)
    p = 1 / (1 + np.exp(-z))
    y = (p > 0.5).astype(np.float32)

    X_ap = X[:, :n_features_ap]
    X_pp = X[:, n_features_ap:]
    return X_ap, X_pp, y


@mp.function
def _bench_once(
    ap_id: int,
    pp_ids: list[int],
    X_ap: np.ndarray,
    X_pp_splits: list[np.ndarray],
    y: np.ndarray,
    k: int,
    t: int,
    reps: int,
    mode: str,
    include_precompute: bool,
    breakdown: bool,
):
    # Place data
    X_ap_j = mp.run_jax_at(ap_id, lambda x: x, jnp.array(X_ap, dtype=jnp.float32))
    X_pp_j = [
        mp.run_jax_at(pp, lambda x: x, jnp.array(xpp, dtype=jnp.float32))
        for pp, xpp in zip(pp_ids, X_pp_splits, strict=True)
    ]
    y_j = mp.run_jax_at(ap_id, lambda x: x, jnp.array(y, dtype=jnp.float32))

    # Binning per party
    build_bins_vmapped = jax.vmap(partial(build_bins_equi_width, max_bin=k), in_axes=1)
    compute_indices_vmapped = jax.vmap(compute_bin_indices, in_axes=(1, 0), out_axes=1)

    ap_bins = mp.run_jax_at(ap_id, build_bins_vmapped, X_ap_j)
    _ = mp.run_jax_at(ap_id, compute_indices_vmapped, X_ap_j, ap_bins)

    pp_bins = [
        mp.run_jax_at(pp, build_bins_vmapped, X_pp_j[i]) for i, pp in enumerate(pp_ids)
    ]
    pp_idx = [
        mp.run_jax_at(pp, compute_indices_vmapped, X_pp_j[i], pp_bins[i])
        for i, pp in enumerate(pp_ids)
    ]

    # AP GH + quantize + encrypt
    init_pred = mp.run_jax_at(ap_id, compute_init_pred, y_j)
    logits0 = mp.run_jax_at(ap_id, lambda p, m=y_j.shape[0]: p * jnp.ones(m), init_pred)
    GH = mp.run_jax_at(ap_id, compute_gh, y_j, logits0)

    fxp_scale = 1 << DEFAULT_FXP_BITS
    Q = mp.run_jax_at(ap_id, quantize_gh, GH, fxp_scale)
    qg = mp.run_jax_at(ap_id, lambda a: a[:, 0].astype(jnp.int64), Q)
    qh = mp.run_jax_at(ap_id, lambda a: a[:, 1].astype(jnp.int64), Q)

    priv_ctx, pub_ctx, _ = mp.run_at(ap_id, fhe.keygen, scheme="BFV")

    # Prepare ciphertext(s)
    g_ct = None  # type: ignore[assignment]
    h_ct = None  # type: ignore[assignment]
    gh_ct = None  # type: ignore[assignment]
    if mode in ("classic", "classic_cached"):
        g_ct = mp.run_at(ap_id, fhe.encrypt, qg, pub_ctx)
        h_ct = mp.run_at(ap_id, fhe.encrypt, qh, pub_ctx)
    elif mode in (
        "interleaved",
        "interleaved_cached",
        "interleaved_fused",
        "interleaved_fused_cached",
    ):
        # Interleave qg and qh into one vector: [g0,h0,g1,h1,...]
        def _interleave(a, b):
            m = a.shape[0]
            out = jnp.empty((m * 2,), dtype=jnp.int64)
            out = out.at[0::2].set(a)
            out = out.at[1::2].set(b)
            return out

        qgh = mp.run_jax_at(ap_id, _interleave, qg, qh)
        gh_ct = mp.run_at(ap_id, fhe.encrypt, qgh, pub_ctx)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    rng = mp.run_jax_at(
        ap_id,
        lambda m: jnp.array(
            np.random.default_rng(0).integers(0, t, size=m), dtype=jnp.int64
        ),
        y_j.shape[0],
    )

    def mk_subgroup_map(bt_level, group_size):
        group_indices = jnp.arange(group_size)[:, None]
        return (group_indices == bt_level).astype(jnp.int8)

    # Precompute subgroup maps per-PP once (rng fixed) and parity selectors once if needed
    subgroup_maps = []
    for pp in pp_ids:
        pub_ctx_pp = mp.p2p(ap_id, pp, pub_ctx)
        rng_pp = mp.p2p(ap_id, pp, rng)
        subgroup_map_pp = mp.run_jax_at(pp, mk_subgroup_map, rng_pp, t)
        subgroup_maps.append((pp, pub_ctx_pp, subgroup_map_pp))

    even_sel = None
    odd_sel = None
    if mode in (
        "interleaved",
        "interleaved_cached",
    ):

        def _build_parity_selectors(m_samples):
            n = m_samples * 2
            even = jnp.zeros((n,), dtype=jnp.int64).at[0::2].set(1)
            odd = jnp.zeros((n,), dtype=jnp.int64).at[1::2].set(1)
            return even, odd

        even_sel, odd_sel = mp.run_jax_at(ap_id, _build_parity_selectors, y_j.shape[0])

    # Optional: precompute and encrypt all bucket masks per-PP for cached modes
    cached_masks = None
    pre_dt = mp.run_jax_at(ap_id, lambda: jnp.array(0.0, dtype=jnp.float64))
    if mode in ("interleaved_cached", "classic_cached"):
        # Helper function to duplicate mask to interleaved length (used only for interleaved mode)
        def _dup2(mask):
            n = mask.shape[0]
            out = jnp.empty((n * 2,), dtype=jnp.int64)
            out = out.at[0::2].set(mask)
            out = out.at[1::2].set(mask)
            return out

        use_interleave = mode == "interleaved_cached"

        cached_masks = []  # list per PP: [ [list per group: [list per feature: [mask_ct per bucket]]] ]
        tpre0 = mp.run_jax_at(ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64))
        for i, (pp, pub_ctx_pp, subgroup_map_pp) in enumerate(subgroup_maps):
            feature_size = pp_idx[i].shape[1]
            grp_masks = []
            for grp in range(t):
                gom = mp.run_jax_at(pp, lambda m, idx: m[idx], subgroup_map_pp, grp)

                def create_masked_order_map(m, om):
                    mask_expanded = jnp.expand_dims(m, axis=1)
                    mask_full = jnp.broadcast_to(mask_expanded, om.shape)
                    return jnp.where(mask_full == 1, om, -1)

                gom_map = mp.run_jax_at(pp, create_masked_order_map, gom, pp_idx[i])

                feat_masks = []
                for feature_idx in range(feature_size):
                    # Build all bucket masks at once: (k, M)
                    def build_bucket_masks(gom_, f_idx, num_buckets):
                        def mask_for_b(b_idx, gom_i, f_i):
                            fb = gom_i[:, f_i]
                            valid_and_in_bucket = (fb >= 0) & (fb <= b_idx)
                            return valid_and_in_bucket.astype(jnp.int64)

                        bs = jnp.arange(num_buckets, dtype=jnp.int64)
                        return jax.vmap(mask_for_b, in_axes=(0, None, None))(
                            bs, gom_, f_idx
                        )

                    bucket_masks = mp.run_jax_at(
                        pp, build_bucket_masks, gom_map, feature_idx, k
                    )
                    # Encrypt each bucket mask (with optional duplication for interleaved mode)
                    masks_ct = []
                    for b in range(k):
                        row_b = mp.run_jax_at(pp, lambda M, bi: M[bi], bucket_masks, b)
                        # Apply _dup2 transformation only for interleaved mode
                        if use_interleave:
                            mask_to_encrypt = mp.run_jax_at(pp, _dup2, row_b)
                        else:
                            mask_to_encrypt = row_b
                        mask_ct_pp = mp.run_at(
                            pp, fhe.encrypt, mask_to_encrypt, pub_ctx_pp
                        )
                        mask_ct_ap = mp.p2p(pp, ap_id, mask_ct_pp)
                        masks_ct.append(mask_ct_ap)
                    feat_masks.append(masks_ct)
                grp_masks.append(feat_masks)
            cached_masks.append(grp_masks)
        tpre1 = mp.run_jax_at(ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64))
        pre_dt = mp.run_jax_at(ap_id, lambda a, b: a - b, tpre1, tpre0)

    # Run reps and time compute + decrypt assembly across all PPs
    times_total = []
    times_comp = []
    times_dec = []
    for rep_i in range(reps):
        t0 = mp.run_jax_at(ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64))
        comp_parts = []
        dec_parts = []
        for i, (pp, pub_ctx_pp, subgroup_map_pp) in enumerate(subgroup_maps):
            if mode == "classic":
                assert g_ct is not None and h_ct is not None
                tcomp0 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                g_lists, h_lists = batch_feature_wise_bucket_sum_fhe_vector(
                    g_ct,
                    h_ct,
                    subgroup_map_pp,
                    pp_idx[i],
                    k,
                    t,
                    rank=pp,
                    ap_rank=ap_id,
                )
                tcomp1 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                comp_parts.append(
                    mp.run_jax_at(ap_id, lambda a, b: a - b, tcomp1, tcomp0)
                )
                tdec0 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                for grp in range(t):
                    enc_g_list = g_lists[grp]
                    enc_h_list = h_lists[grp]
                    dec_g = [
                        mp.run_at(ap_id, fhe.decrypt, ct, priv_ctx) for ct in enc_g_list
                    ]
                    dec_h = [
                        mp.run_at(ap_id, fhe.decrypt, ct, priv_ctx) for ct in enc_h_list
                    ]

                    def _stack(*xs):
                        return jnp.stack(xs)

                    _ = mp.run_jax_at(ap_id, _stack, *dec_g)
                    _ = mp.run_jax_at(ap_id, _stack, *dec_h)
                tdec1 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                dec_parts.append(mp.run_jax_at(ap_id, lambda a, b: a - b, tdec1, tdec0))
            elif mode == "classic_cached":
                assert g_ct is not None and h_ct is not None
                assert cached_masks is not None
                feature_size = pp_idx[i].shape[1]
                g_lists = [[] for _ in range(t)]
                h_lists = [[] for _ in range(t)]
                tcomp0 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                for grp in range(t):
                    feat_masks = cached_masks[i][grp]
                    for feature_idx in range(feature_size):
                        for bucket_idx in range(k):
                            mask_ct_ap = feat_masks[feature_idx][bucket_idx]
                            g_sum_ct = mp.run_at(ap_id, fhe.dot, g_ct, mask_ct_ap)
                            h_sum_ct = mp.run_at(ap_id, fhe.dot, h_ct, mask_ct_ap)
                            g_lists[grp].append(g_sum_ct)
                            h_lists[grp].append(h_sum_ct)
                tcomp1 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                comp_parts.append(
                    mp.run_jax_at(ap_id, lambda a, b: a - b, tcomp1, tcomp0)
                )
                tdec0 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                for grp in range(t):
                    enc_g_list = g_lists[grp]
                    enc_h_list = h_lists[grp]
                    dec_g = [
                        mp.run_at(ap_id, fhe.decrypt, ct, priv_ctx) for ct in enc_g_list
                    ]
                    dec_h = [
                        mp.run_at(ap_id, fhe.decrypt, ct, priv_ctx) for ct in enc_h_list
                    ]

                    def _stack(*xs):
                        return jnp.stack(xs)

                    _ = mp.run_jax_at(ap_id, _stack, *dec_g)
                    _ = mp.run_jax_at(ap_id, _stack, *dec_h)
                tdec1 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                dec_parts.append(mp.run_jax_at(ap_id, lambda a, b: a - b, tdec1, tdec0))
            else:
                assert gh_ct is not None
                # even_sel/odd_sel were built once before reps

                def _dup2(mask):
                    n = mask.shape[0]
                    out = jnp.empty((n * 2,), dtype=jnp.int64)
                    out = out.at[0::2].set(mask)
                    out = out.at[1::2].set(mask)
                    return out

                feature_size = pp_idx[i].shape[1]
                g_lists = [[] for _ in range(t)]
                h_lists = [[] for _ in range(t)]
                tcomp0 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                for grp in range(t):
                    if mode in ("interleaved_cached",) and cached_masks is not None:
                        # Use precomputed encrypted masks
                        feat_masks = cached_masks[i][grp]
                        for feature_idx in range(feature_size):
                            for bucket_idx in range(k):
                                mask_ct_ap = feat_masks[feature_idx][bucket_idx]
                                prod_ct = mp.run_at(ap_id, fhe.mul, gh_ct, mask_ct_ap)
                                g_sum_ct = mp.run_at(ap_id, fhe.dot, prod_ct, even_sel)
                                h_sum_ct = mp.run_at(ap_id, fhe.dot, prod_ct, odd_sel)
                                g_lists[grp].append(g_sum_ct)
                                h_lists[grp].append(h_sum_ct)
                    else:
                        # Build on the fly
                        gom = mp.run_jax_at(
                            pp, lambda m, idx: m[idx], subgroup_map_pp, grp
                        )

                        def create_masked_order_map(m, om):
                            mask_expanded = jnp.expand_dims(m, axis=1)
                            mask_full = jnp.broadcast_to(mask_expanded, om.shape)
                            return jnp.where(mask_full == 1, om, -1)

                        gom_map = mp.run_jax_at(
                            pp, create_masked_order_map, gom, pp_idx[i]
                        )
                        for feature_idx in range(feature_size):
                            for bucket_idx in range(k):

                                def create_bucket_mask(gom_, f_idx, b_idx):
                                    fb = gom_[:, f_idx]
                                    valid_and_in_bucket = (fb >= 0) & (fb <= b_idx)
                                    return valid_and_in_bucket.astype(jnp.int64)

                                bucket_mask = mp.run_jax_at(
                                    pp,
                                    create_bucket_mask,
                                    gom_map,
                                    feature_idx,
                                    bucket_idx,
                                )
                                inter_mask = mp.run_jax_at(pp, _dup2, bucket_mask)
                                mask_ct_pp = mp.run_at(
                                    pp, fhe.encrypt, inter_mask, pub_ctx_pp
                                )
                                mask_ct_ap = mp.p2p(pp, ap_id, mask_ct_pp)

                                prod_ct = mp.run_at(ap_id, fhe.mul, gh_ct, mask_ct_ap)
                                g_sum_ct = mp.run_at(ap_id, fhe.dot, prod_ct, even_sel)
                                h_sum_ct = mp.run_at(ap_id, fhe.dot, prod_ct, odd_sel)

                                g_lists[grp].append(g_sum_ct)
                                h_lists[grp].append(h_sum_ct)
                tcomp1 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                comp_parts.append(
                    mp.run_jax_at(ap_id, lambda a, b: a - b, tcomp1, tcomp0)
                )
                tdec0 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                for grp in range(t):
                    enc_g_list = g_lists[grp]
                    enc_h_list = h_lists[grp]
                    dec_g = [
                        mp.run_at(ap_id, fhe.decrypt, ct, priv_ctx) for ct in enc_g_list
                    ]
                    dec_h = [
                        mp.run_at(ap_id, fhe.decrypt, ct, priv_ctx) for ct in enc_h_list
                    ]

                    def _stack(*xs):
                        return jnp.stack(xs)

                    _ = mp.run_jax_at(ap_id, _stack, *dec_g)
                    _ = mp.run_jax_at(ap_id, _stack, *dec_h)
                tdec1 = mp.run_jax_at(
                    ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64)
                )
                dec_parts.append(mp.run_jax_at(ap_id, lambda a, b: a - b, tdec1, tdec0))

        t1 = mp.run_jax_at(ap_id, lambda: jnp.array(time.time(), dtype=jnp.float64))
        dt = mp.run_jax_at(ap_id, lambda a, b: a - b, t1, t0)
        # Optionally include precompute cost once (first repetition only)
        if include_precompute and rep_i == 0:
            dt = mp.run_jax_at(ap_id, lambda x, add: x + add, dt, pre_dt)
        # Sum parts across PPs

        def _sum_vec(*xs):
            s = xs[0]
            for x in xs[1:]:
                s = s + x
            return s

        comp_sum = (
            mp.run_jax_at(ap_id, _sum_vec, *comp_parts)
            if comp_parts
            else mp.run_jax_at(ap_id, lambda: jnp.array(0.0, dtype=jnp.float64))
        )
        dec_sum = (
            mp.run_jax_at(ap_id, _sum_vec, *dec_parts)
            if dec_parts
            else mp.run_jax_at(ap_id, lambda: jnp.array(0.0, dtype=jnp.float64))
        )
        times_total.append(dt)
        times_comp.append(comp_sum)
        times_dec.append(dec_sum)

    # Stack per-rep durations into a vector at AP for robust fetch
    def _stack_times(*xs):
        return jnp.stack(xs)

    total_vec = mp.run_jax_at(ap_id, _stack_times, *times_total)
    if not breakdown:
        return total_vec
    comp_vec = mp.run_jax_at(ap_id, _stack_times, *times_comp)
    dec_vec = mp.run_jax_at(ap_id, _stack_times, *times_dec)

    def _stack3(a, b, c):
        return jnp.stack([a, b, c], axis=0)

    return mp.run_jax_at(ap_id, _stack3, total_vec, comp_vec, dec_vec)


def main():
    parser = argparse.ArgumentParser(description="FHE histogram microbenchmark")
    parser.add_argument(
        "--world-size", type=int, default=2, help="Total parties (AP=1+PPs)"
    )
    parser.add_argument("--m", type=int, default=4096, help="Samples")
    parser.add_argument("--n-total", type=int, default=16, help="Total features")
    parser.add_argument("--n-ap", type=int, default=4, help="AP feature count")
    parser.add_argument("--k", type=int, default=16, help="Bins per feature")
    parser.add_argument("--t", type=int, default=4, help="Groups (nodes at level)")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--mode",
        type=str,
        default="classic",
        choices=[
            "classic",
            "classic_cached",
            "interleaved",
            "interleaved_cached",
        ],
        help="Histogram mode to benchmark",
    )
    parser.add_argument(
        "--include-precompute",
        action="store_true",
        help="Include precompute (mask generation+encryption) time in the first repetition",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Report timing breakdown (total, compute, decrypt)",
    )
    args = parser.parse_args()

    assert args.world_size >= 2, "world-size must be >= 2"
    pp_parties = args.world_size - 1
    assert args.n_total >= args.n_ap

    # Split PP features evenly
    n_pp_total = args.n_total - args.n_ap
    n_pp_each = [n_pp_total // pp_parties] * pp_parties
    n_pp_each[-1] += n_pp_total - sum(n_pp_each)

    X_ap, X_pp_all, y = _gen_data(args.m, args.n_total, args.n_ap, args.seed)
    offset = 0
    X_pp_splits = []
    for c in n_pp_each:
        X_pp_splits.append(X_pp_all[:, offset : offset + c])
        offset += c

    sim = mp.Simulator.simple(args.world_size)

    ap_id = 0
    pp_ids = list(range(1, args.world_size))

    print("\n=== FHE Histogram Microbenchmark ===")
    print(
        f"world-size={args.world_size} (AP+{pp_parties} PPs), m={args.m}, n_total={args.n_total} (AP={args.n_ap}, PP={n_pp_total}), k={args.k}, t={args.t}, reps={args.reps}, mode={args.mode}"
    )

    out = mp.evaluate(
        sim,
        _bench_once,
        ap_id,
        pp_ids,
        X_ap,
        X_pp_splits,
        y,
        args.k,
        args.t,
        args.reps,
        args.mode,
        args.include_precompute,
        args.breakdown,
    )
    times_raw = mp.fetch(sim, out)

    # Expected: [times_at_ap, None, ...] in 2PC; extract first non-None
    if isinstance(times_raw, list) and len(times_raw) >= 1 and times_raw[-1] is None:
        times_nodes = times_raw[0]
    else:
        times_nodes = times_raw

    if args.breakdown:
        times_arr = np.asarray(times_nodes, dtype=float)
        # Expect shape (3, reps): [total, compute, decrypt]
        if times_arr.ndim == 1:
            # Fallback if flattened; try to split into 3 roughly equal parts
            n = times_arr.size
            r = n // 3
            total = times_arr[:r]
            comp = times_arr[r : 2 * r]
            dec = times_arr[2 * r : 3 * r]
        else:
            total, comp, dec = (
                times_arr[0].ravel(),
                times_arr[1].ravel(),
                times_arr[2].ravel(),
            )

        print(f"Per-rep total (s): {', '.join(f'{t:.4f}' for t in total.tolist())}")
        print(
            f"Per-rep compute-only (s): {', '.join(f'{t:.4f}' for t in comp.tolist())}"
        )
        print(
            f"Per-rep decrypt-only (s): {', '.join(f'{t:.4f}' for t in dec.tolist())}"
        )
        print(
            f"Averages — total: {float(total.mean()):.4f}s, compute: {float(comp.mean()):.4f}s, decrypt: {float(dec.mean()):.4f}s"
        )
    else:
        # Convert to numpy array of floats (handle scalar, list, or numpy array)
        if isinstance(times_nodes, list):
            # elements are likely [val, None] pairs; take first
            times_arr = np.array(
                [
                    float(np.array(e[0]))
                    if isinstance(e, (list, tuple))
                    else float(np.array(e))
                    for e in times_nodes
                ],
                dtype=float,
            )
        else:
            times_arr = np.asarray(times_nodes, dtype=float).ravel()
        avg = float(times_arr.mean())
        print(f"Per-rep time (s): {', '.join(f'{t:.4f}' for t in times_arr.tolist())}")
        print(f"Average time (s): {avg:.4f}")


if __name__ == "__main__":
    main()
