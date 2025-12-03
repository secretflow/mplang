# SecureBoost v2 Optimization Log

**Date:** December 3, 2025
**Author:** GitHub Copilot & User

This document records the optimization journey for the SecureBoost (SGB) v2 implementation in `examples/v2/sgb.py`.

## 1. Baseline Performance

Initial profiling on a large dataset (`n=10000`, `features=50+50`, `depth=3`) revealed significant bottlenecks.

* **Total Time:** ~151s
* **Tracing Time:** ~55s
* **Execution Time:** ~96s
* **Accuracy:** ~59% (Low due to noise/overflow with 4096 slots)
* **Key Bottlenecks:**
    1. **Communication:** `simp.shuffle` called ~765 times, transferring individual ciphertexts for each feature.
    2. **Overhead:** `tensor.run_jax` called ~1500 times, mostly for computing small masks, causing massive tracing overhead.

## 2. Optimization Steps

### Phase 1: SIMD Feature Packing & JAX Fusion

**Goal:** Reduce communication volume and tracing overhead.

* **SIMD Feature Packing:**
  * Instead of sending one ciphertext per feature, we pack multiple features into a single BFV ciphertext.
  * Since each feature's histogram only uses `n_buckets` slots spaced by `stride`, we can interleave multiple features into the unused slots.
  * **Result:** Reduced `simp.shuffle` calls and data volume by a factor of `stride`.

* **JAX Fusion:**
  * Replaced thousands of small `tensor.run_jax` calls (for mask generation) with a single vectorized `compute_all_masks` function.
  * **Result:** Tracing time dropped from ~55s to ~18s.

### Phase 2: Parameter Tuning (The Trade-off)

**Goal:** Fix accuracy issues and enable deeper computation.

* **Change:** Increased `poly_modulus_degree` from 4096 to 8192.
* **Impact:**
  * **Pros:**
    * Accuracy restored from ~59% to ~80% (larger noise budget).
    * Fewer chunks needed for large datasets (10k samples fit in 2 chunks instead of 3).
  * **Cons:**
    * Single BFV operation (Rotate/Mul) became ~4.7x slower due to increased polynomial degree.
  * **Net Result:** Despite slower individual ops, the massive reduction in communication (from Phase 1) kept the total time faster than baseline (~79s vs 151s).

### Phase 3: Histogram Subtraction

**Goal:** Algorithmic reduction of FHE operations.

* **Algorithm:**
  * For a node split, only compute the **Left Child** histogram using expensive FHE (`Mask -> Mul -> Rotate -> Sum`).
  * Compute the **Right Child** histogram via subtraction: $H_{Right} = H_{Parent} - H_{Left}$.
  * This is valid because the sum of samples in left and right children equals the parent's samples.
* **Implementation:**
  * Modified `build_tree` to cache parent histograms.
  * Implemented `derive_right_and_combine` to perform the subtraction and interleave results.

## 3. Final Results Comparison

Benchmark Config: `n=10000`, `features=50+50`, `trees=1`, `depth=3`

| Metric | Baseline (4096 slots) | Phase 2 (8192 slots + Packing) | Phase 3 (Subtraction) | Total Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Total Time** | ~151s | ~79s | **~41s** | **3.7x Faster** |
| **Tracing Time** | ~55s | ~18s | **~6.6s** | **8.3x Faster** |
| **Execution Time** | ~96s | ~61s | **~34.5s** | **2.8x Faster** |
| **Accuracy** | ~59% | ~80.8% | **~84%** | **+25% (Usable)** |
| **Rotate Ops** | 20,952 | 16,064 | **9,224** | **56% Reduction** |

## 4. Future Work

1. **Multi-threading:** Parallelize the `(node, feature)` loops in `fhe_histogram_optimized` using `ThreadPoolExecutor` to exploit multi-core CPUs (BFV ops release GIL).
2. **GPU Acceleration:** Use a GPU-backed BFV library (e.g., TenSEAL with CUDA) to accelerate the `Rotate` bottleneck.
