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

## References

- [mplang2/libs/mpc/groupby.py](../mplang2/libs/mpc/groupby.py)
- [mplang2/dialects/bfv.py](../mplang2/dialects/bfv.py)
- [examples/xgboost/sgb.py](../examples/xgboost/sgb.py) (v1 implementation)
