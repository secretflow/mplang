# SecureBoost v2 Optimization Log

**Date:** December 3, 2025

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

### Phase 4: Non-blocking DAG Scheduler

**Goal:** Resolve thread starvation and maximize CPU utilization on high-core machines.

* **Problem:**
  * Previous interpreter used `future.result()` which blocked threads while waiting for dependencies.
  * On deep graphs (like SGB), the thread pool would fill with blocked tasks, causing starvation and capping CPU usage (e.g., ~600% on a 96-core machine).
* **Solution:**
  * Rewrote `interpreter.py` to use a **Non-blocking DAG Scheduler**.
  * Uses topological sort (in-degree counting) to track dependencies.
  * Tasks are only submitted when inputs are ready.
  * Uses callbacks (`add_done_callback`) to trigger dependent tasks, ensuring threads are never blocked waiting.
* **Result:**
  * CPU utilization unlocked (scaling to available cores).
  * Execution time for `n=10000` dropped from ~41s to ~10s.

### Phase 5: Memory & Aggregation Optimization

**Goal:** Reduce peak memory usage and further minimize FHE operations.

* **Incremental Packing (Memory):**
  * **Problem:** Previous implementation accumulated unpacked ciphertexts for all features before packing them.
    For large feature sets, this caused massive memory spikes (O(n_features)).
  * **Solution:** Refactored to process features in small batches (size `stride`).
    Pack immediately after computing a batch and release intermediate ciphertexts.
  * **Result:** Peak memory usage reduced from O(n_features) to O(stride), eliminating OOM risks on large datasets.

* **Lazy Aggregation (Compute):**
  * **Problem:** `batch_bucket_aggregate` (expensive rotations) was called inside the chunk loop.
    For $N$ chunks, we performed $N \times R$ rotations.
  * **Solution:** Exploited the linearity of BFV. We now sum the masked ciphertexts (cheap additions) across all chunks first,
    then perform aggregation **once** on the sum.
  * **Result:** Reduced rotation count by a factor of `n_chunks` (e.g., 75x reduction for 300k samples),
    significantly speeding up large-scale training.

## 3. Final Results Comparison

Benchmark Config: `n=10000`, `features=50+50`, `trees=1`, `depth=3`

| Metric | Baseline (4096 slots) | Phase 2 (8192 slots + Packing) | Phase 3 (Subtraction) | Phase 4 (Async DAG) | Total Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Time** | ~151s | ~79s | ~41s | **~10.3s** | **14.6x Faster** |
| **Tracing Time** | ~55s | ~18s | ~6.6s | **~2.7s** | **20.3x Faster** |
| **Execution Time** | ~96s | ~61s | ~34.5s | **~7.6s** | **12.6x Faster** |
| **Accuracy** | ~59% | ~80.8% | ~84% | **~84%** | **+25% (Usable)** |
| **Rotate Ops** | 20,952 | 16,064 | 9,224 | **9,224** | **56% Reduction** |

*(Note: Phase 5 improvements are most visible on larger datasets where `n_chunks > 1`)*

## 4. Operation Breakdown (Phase 4)

**Config:** `samples=10000`, `features=50+50`, `trees=1`, `depth=3`

* **bfv.rotate:** 67s (cumulative) - 43.1% of compute time.
* **bfv.mul:** 27.6s - 17.8%.
* **bfv.add:** 20.3s - 13.1%.
* **simp.pcall_static:** 16.7s - 10.7%.

**Observation:**
Even with the scheduler fix, **Rotation** remains the dominant cost (43%), but the overhead of Python/Interpreter (reflected in `simp.pcall_static`
and general execution gaps) has been drastically reduced.

## 5. Future Work

1. **GPU Acceleration:** Use a GPU-backed BFV library (e.g., TenSEAL with CUDA) to accelerate the `Rotate` bottleneck.
