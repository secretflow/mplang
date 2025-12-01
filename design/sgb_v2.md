# SecureBoost v2 Design: Low-Level BFV SIMD Optimization

## Status

**✅ Implemented and Working** (2025-01-XX)

- Single-party mode: 87% accuracy on synthetic data
- Multi-party FHE mode: 76% accuracy with TenSEAL/BFV backend
- Code location: `examples/mp2/sgb.py`

## Executive Summary

This document outlines the design for SecureBoost (SGB) using mplang2's low-level BFV dialect. The key insight is that **we can leverage BFV SIMD slots and the `groupby.py` primitives to achieve significantly better performance than v1**.

## Problem Analysis

### V1 SGB Performance Issues

Looking at `examples/xgboost/sgb.py`, the v1 implementation has critical bottlenecks:

1. **Scalar-by-Scalar FHE Operations**: The `batch_feature_wise_bucket_sum_fhe_vector` function iterates:

   ```python
   for group_idx in range(group_size):
       for feature_idx in range(feature_size):
           for bucket_idx in range(bucket_num):
               # Individual dot product per (group, feature, bucket)
               g_sum_ct = fhe.dot(g_ct, bucket_mask)
               h_sum_ct = fhe.dot(h_ct, bucket_mask)
   ```

   - Total operations: `O(n_nodes × n_features × n_buckets)` FHE dot products
   - Each result is a separate scalar ciphertext
   - Each must be transferred individually from PP to AP

2. **No SIMD Exploitation**: BFV supports packing ~2048 values in a single ciphertext, but v1 treats each operation independently.

3. **Redundant Transfers**: Keys and encrypted vectors are sent multiple times.

### V2 Opportunity: Oblivious GroupBy

Looking at `mplang2/libs/mpc/groupby.py`, we have two optimized primitives:

1. **`oblivious_groupby_sum_bfv`**: Uses BFV masking + rotation-sum
2. **`oblivious_groupby_sum_shuffle`**: Uses secret sharing + permutation

**Key Insight**: Histogram computation IS group-by sum!

- Data: encrypted G/H vectors
- Bins: combined (node × feature × bucket) assignments
- Result: cumulative histogram sums

## Proposed Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ACTIVE PARTY (AP)                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   G/H       │    │  Encrypt    │    │   Decrypt   │                  │
│  │ (plaintext) │───▶│  (BFV)      │───▶│   Results   │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│         │                 │                   ▲                          │
│         │                 │                   │                          │
│         ▼                 ▼                   │                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │ Local Hist  │    │ Enc(G), Enc(H) │ │ Enc(Sums)   │                  │
│  │  (JAX)      │    │  + Keys     │    │ from PP     │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│                           │                   ▲                          │
└───────────────────────────│───────────────────│──────────────────────────┘
                            │                   │
                            ▼                   │
┌───────────────────────────│───────────────────│──────────────────────────┐
│                           │  PASSIVE PARTY (PP) │                        │
│                           ▼                   │                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  Features   │    │ Build Masks │    │  CT × PT    │                  │
│  │  (binned)   │───▶│  (SIMD)     │───▶│  + RotSum   │─────────────────▶│
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### SIMD Slot Packing Strategy

Instead of processing one (feature, bucket) at a time, we pack multiple:

```
BFV Ciphertext (4096 slots):
┌────────────────────────────────────────────────────────────────┐
│ Slot 0-m: Sample 0..m-1 values for (feat=0, bucket=0)          │
│ ... (repeat pattern for each chunk)                            │
└────────────────────────────────────────────────────────────────┘

After masking and rotation-sum:
┌────────────────────────────────────────────────────────────────┐
│ Slot 0: Sum for (feat=0, bucket=0)                             │
│ Slot 1: Sum for (feat=0, bucket=1)                             │
│ ...                                                             │
│ Slot K: Sum for (feat=1, bucket=0)                             │
└────────────────────────────────────────────────────────────────┘
```

### Complexity Comparison

| Metric | V1 | V2 (Optimized) |
|--------|-----|----------------|
| FHE Multiplications | O(t × n × k) | O(t × n × k) |
| FHE Rotations | 0 | O(t × log(m) × n × k / B) |
| Ciphertext Transfers PP→AP | O(t × n × k) | O(t × ceil(n × k / B)) |
| Total Communication | High | ~B× reduction |

Where:

- t = nodes at current level
- n = features
- k = buckets
- m = samples
- B = BFV batch size (~2048)

## Implementation Plan

### Phase 1: Core Primitives (mplang2/libs/mpc/)

1. **Extend `aggregation.py`**:
   - Add `batched_rotate_and_sum` for multiple packed slots
   - Add `slot_extract` for unpacking results

2. **Extend `groupby.py`**:
   - Add `oblivious_histogram_bfv` specialized for histogram computation
   - Optimize mask building using JAX vmap

### Phase 2: SGB Implementation (examples/mp2/sgb.py)

```python
import mplang2 as mp
from mplang2.dialects import bfv, simp, tensor
from mplang2.libs.mpc import aggregation, groupby

# Key generation at AP
def setup_fhe(ap_rank):
    def keygen():
        pk, sk = bfv.keygen(poly_modulus_degree=4096)
        rk = bfv.make_relin_keys(sk)
        gk = bfv.make_galois_keys(sk)
        enc = bfv.create_encoder(poly_modulus_degree=4096)
        return pk, sk, rk, gk, enc
    return simp.pcall_static((ap_rank,), keygen)

# Optimized histogram computation
def fhe_histogram_optimized(g_ct, h_ct, node_mask, bin_indices, ...):
    """Compute histogram using SIMD batching."""
    # 1. Build all masks at once using vmap
    # 2. Pack masks into fewer ciphertexts
    # 3. Use batched CT×PT and rotate-sum
    # 4. Return packed results
    pass
```

### Phase 3: Integration

1. Replace `batch_feature_wise_bucket_sum_fhe_vector` with optimized version
2. Add benchmarks comparing v1 vs v2
3. Ensure security properties are preserved

## API Design

### User-Facing API (same as v1)

```python
from examples.mp2.sgb import SecureBoost

model = SecureBoost(
    n_estimators=10,
    max_depth=3,
    max_bin=8,
    ap_rank=0,
    pp_ranks=[1],
)

# Data loading uses device API
X_ap = mp.put("P0", X_ap_data)
X_pp = mp.put("P1", X_pp_data)
y = mp.put("P0", y_data)

model.fit([X_ap, X_pp], y)
predictions = model.predict([X_ap_test, X_pp_test])
```

### Internal Dialect Usage

```python
# Using mplang2 primitives directly
from mplang2.dialects import bfv, simp, tensor

# Party-local computation
result = simp.pcall_static((rank,), lambda: tensor.run_jax(fn, args))

# Data transfer
result_at_other = simp.shuffle_static(result, {other_rank: rank})

# BFV operations
ct = bfv.encrypt(pt, pk)
ct_sum = bfv.add(ct1, ct2)
ct_rot = bfv.rotate(ct, steps, gk)
```

## Security Considerations

1. **Same security model as v1**:
   - PP never sees decrypted G/H
   - AP never sees raw features or masks
   - Only aggregated statistics revealed to AP

2. **Additional considerations for SIMD**:
   - Packed masks don't reveal additional information
   - Rotation patterns are data-independent

## Performance Targets

- **Communication**: 10-100× reduction in ciphertext transfers
- **Computation**: Similar FHE operations but better batching
- **Memory**: Reduced peak memory from fewer intermediate ciphertexts

## Test Plan

1. Unit tests for new aggregation primitives
2. Integration tests comparing v1 vs v2 results
3. Benchmark suite for various (m, n, k, t) configurations
4. Security validation (same leakage profile as v1)

## Migration Path

1. v1 remains default, v2 opt-in via `method="v2"` parameter
2. Gradual transition after v2 validated
3. Eventually deprecate v1 histogram path

## Open Questions

1. **Optimal batch size**: Depends on (m, n, k) - need adaptive strategy
2. **Multi-PP coordination**: How to efficiently aggregate from multiple PPs
3. **Memory pressure**: Large slot vectors may exceed memory on constrained devices

## Performance Analysis (2025-12-01)

### Profiling Results (m=500, depth=3, 2 trees)

Using `registry.enable_profiling()` for automatic primitive timing:

| Operation | Time | % | Count | Mean |
|-----------|------|---|-------|------|
| `tensor.run_jax` | 8.97s | 48.9% | 997 | 9.0ms |
| `bfv.rotate` | 7.21s | 39.3% | 12,420 | 0.58ms |
| `bfv.mul` | 0.78s | 4.2% | 1,440 | 0.54ms |
| Other BFV | 2.15s | 11.8% | - | - |
| **Total (leaf ops)** | **18.33s** | 100% | - | - |

### SIMD Slot Utilization Analysis

BFV with `poly_modulus_degree=4096` provides 4096 slots per ciphertext.

| Sample Size (m) | Ciphertexts Needed | Slot Usage | SIMD Packing Effective? |
|-----------------|--------------------| -----------|-------------------------|
| 500 | 1 | 12.2% | ✅ Yes - 64x potential speedup |
| 1,000 | 1 | 24.4% | ✅ Yes - 64x potential speedup |
| 4,096 | 1 | 100% | ❌ No - slots fully used |
| 10,000 | 3 | 100% | ❌ No - need multiple CTs |
| 100,000 | 25 | 100% | ❌ No - need multiple CTs |

**Key Finding**: SIMD Packing optimization (packing multiple buckets into one CT) is only effective when `m < poly_modulus_degree`.

### Verified SIMD Behavior

Tested BFV operation times with different sample sizes:

| m | Exec Time | rotate Time | JAX Time |
|---|-----------|-------------|----------|
| 500 | 2.91s | 0.59s | 2.15s |
| 1000 | 2.33s | 0.65s | 1.72s |
| 2000 | 2.42s | 0.71s | 1.74s |

**Confirmed**: BFV rotate time is nearly constant regardless of slot usage (SIMD parallelism).

### Optimization Strategies by Scale

#### Small Scale (m < 4096) - Current Implementation Target

| Optimization | Expected Gain | Status |
|--------------|---------------|--------|
| Keygen caching | -0.34s | ✅ Implemented |
| Batch JAX calls | -1s (59% fewer calls) | ✅ Implemented |
| SIMD bucket packing | -6s (64x fewer rotates) | 📋 Planned |

#### Large Scale (m > 10,000) - Production Target

SIMD packing is **NOT effective** at this scale. Alternative strategies:

| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| Increase `poly_modulus_degree` (8192→16384) | 2-3x fewer CTs | Low |
| Reduce bucket count (64→16) | 4x fewer rotates | Low |
| Cumulative histogram algorithm | ~10x fewer ops | Medium |
| Distributed FHE (sample sharding) | Linear scaling | High |
| Parallel CT processing (multi-core) | ~Nx on N cores | Medium |

### Known Issues (Resolved)

1. **m ≥ 2049 caused hang** - ✅ **FIXED** (2025-12-01)
   - **Root Cause**: SEAL's `rotate_rows` has max step limit of `slot_count/2 - 1 = 2047`
   - When `m > 2048`, `rotate_and_sum` needs step sizes `≥ 2048` which triggered SEAL error
   - **Fix**: Added `_safe_rotate()` to decompose large rotations + `rotate_columns` for cross-row aggregation
   - **New files/functions**:
     - `mplang2/libs/mpc/aggregation.py`: `_safe_rotate()`, `_rotate_and_sum_row()`, updated `rotate_and_sum()`
     - `mplang2/dialects/bfv.py`: Added `rotate_columns_p` and `rotate_columns()`
     - `mplang2/backends/bfv_impl.py`: Added `rotate_columns_impl()`
   - **Verified**: m=3000 now runs successfully (3.127s, 81.63% accuracy)

2. **m > 4096 requires multiple ciphertexts** - 📋 Future work
   - Current implementation handles m > row_size (2048) but m ≤ slot_count (4096)
   - For m > 4096, need to pack data into multiple ciphertexts

### BFV SIMD Batching Explained

SEAL's BFV batching with `poly_modulus_degree=4096`:

- Total slots: 4096 (arranged as 2 rows × 2048 columns)
- `rotate_rows(ct, step)`: Rotates within each row (step must be in range `(-2048, 2048)`)
- `rotate_columns(ct)`: Swaps the two rows

For `rotate_and_sum(ct, k)`:

- `k ≤ 2048`: Only needs row rotations (simple case)
- `2048 < k ≤ 4096`: Needs row rotations + column rotation for cross-row aggregation

## Next Steps: Optimization Roadmap

### Phase 1: Algorithm Optimization (High Value, Low Effort) ✅ Complete

| Task | Status | Expected Gain | Actual Gain |
|------|--------|---------------|-------------|
| Keygen caching | ✅ Done | -0.34s | -0.34s |
| Batch JAX calls | ✅ Done | -1s (59% fewer) | -1s |
| **Exact bucket histogram** | ✅ Done | Avoid m>=2048 + fewer rotates | **-27% rotates, -23% exec time** |

**Exact Bucket Histogram Algorithm**:

Current approach uses **cumulative histogram** which requires aggregating ALL samples in node:

```python
# Current: cumulative mask
mask = (node_mask == 1) & (feat_bins <= bucket_idx)  # All samples with bin <= bucket
# Then rotate_and_sum(ct, m, gk)  # Sum over m samples - PROBLEM when m >= 2048
```

Optimized approach uses **exact bucket counts** then client-side cumsum:

```python
# Optimized: exact mask
mask = (node_mask == 1) & (feat_bins == bucket_idx)  # Only samples in this bucket
# Then rotate_and_sum(ct, k, gk)  # k = samples in bucket << m

# Client-side cumsum after decryption
cumsum_g = jnp.cumsum(bucket_g_sums)
```

**Benefits**:

- k = ceil(m / n_buckets) instead of m
- For m=10000, n_buckets=64: k ≈ 156 << 2048
- **Completely avoids m >= 2048 issue**
- Same FHE security (only aggregates revealed)

**Measured Results (2025-12-01)**:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| bfv.rotate count | 14,688 | 10,728 | **-27%** |
| bfv.rotate time | 8.54s | 6.26s | **-27%** |
| rotate_columns calls | 108 | 0 | **-100%** |
| Total leaf ops time | 22.23s | 19.53s | **-12%** |
| m=3000 exec time | 3.13s | 2.41s | **-23%** |
| m=500 accuracy | 86.8% | 90.2% | **+3.4%** |

### Phase 2: SIMD Bucket Packing (Medium Value, Medium Effort)

Only effective when `m < poly_modulus_degree` (4096):

| Approach | Description | Complexity |
|----------|-------------|------------|
| Pack multiple buckets | Store bucket_0..bucket_63 results in slots 0..63 | Medium |
| Single rotate_and_sum | Aggregate all at once | Low |
| Fewer CT transfers | 1 CT instead of 64 per feature | High value |

### Phase 3: Large Scale (m > 10,000) Production

| Approach | Description | When to Use |
|----------|-------------|-------------|
| Increase poly_degree | 8192→16384 doubles slots | Memory OK |
| Sample sharding | Split samples across CTs | Very large m |
| Parallel CT processing | Multi-core rotation | CPU-bound |

### Recommended Next Action

**Implement exact bucket histogram algorithm** - this provides:

1. No m >= 2048 hang issue (root cause eliminated)
2. Faster rotate_and_sum (k << m)
3. Same security guarantees
4. No API changes needed

The fix I implemented (rotate_columns for m > row_size) is a **safety net** but the algorithm should avoid triggering it.

## References

- [mplang2/libs/mpc/groupby.py](../mplang2/libs/mpc/groupby.py)
- [mplang2/dialects/bfv.py](../mplang2/dialects/bfv.py)
- [examples/xgboost/sgb.py](../examples/xgboost/sgb.py) (v1 implementation)
