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

*(Note: Phase 5 improvements are most visible on larger datasets where `n_chunks > 1`)*

### Phase 6: Scheduler Optimization & Batch Encoding

**Goal:** Eliminate scheduler bottleneck on massive datasets (1M+ samples).

* **Problem:**
  * Profiling revealed that `bfv.encode` (creating plaintext masks) was being called individually for every chunk and every feature.
  * For 1M samples (122 chunks) and 100 features, this generated ~12,000 tiny tasks.
  * The Python-based async scheduler couldn't dispatch these fast enough, causing CPU starvation (worker threads idle waiting for tasks).
* **Solution:**
  * Implemented `bfv.batch_encode` primitive.
  * Instead of 12,000 ops, we now encode all masks for a chunk in a single operation (O(1) scheduling cost).
  * This reduced the task count significantly, allowing the scheduler to saturate all cores with heavy `bfv.mul` and `bfv.rotate` ops.
* **Result:**
  * CPU usage during execution spiked from ~100% (scheduler bound) to ~2000% (fully parallel).
  * Enabled successful training on **1 Million Samples** with depth 5.

## 3. Final Results Comparison (Small Scale)

Benchmark Config: `n=10000`, `features=50+50`, `trees=1`, `depth=3`

| Metric | Baseline (4096 slots) | Phase 2 (8192 slots + Packing) | Phase 3 (Subtraction) | Phase 4 (Async DAG) | Total Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Time** | ~151s | ~79s | ~41s | **~10.3s** | **14.6x Faster** |
| **Tracing Time** | ~55s | ~18s | ~6.6s | **~2.7s** | **20.3x Faster** |
| **Execution Time** | ~96s | ~61s | ~34.5s | **~7.6s** | **12.6x Faster** |
| **Accuracy** | ~59% | ~80.8% | ~84% | **~84%** | **+25% (Usable)** |
| **Rotate Ops** | 20,952 | 16,064 | 9,224 | **9,224** | **56% Reduction** |

### Large Scale Benchmark (Phase 6)

**Config:** `samples=1,000,000`, `features=50+50`, `trees=1`, `depth=5`

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Total Time** | **500.1s** | ~8.3 minutes |
| **Tracing Time** | 60.6s | Single-threaded overhead |
| **Execution Time** | 439.5s | Fully parallel (2000% CPU) |
| **Accuracy** | **89.26%** | High fidelity |
| **Status** | **Success** | Scaled to 1M samples on CPU |

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

### Phase 7: Hoisting & Tree Reduction

**Goal:** Reduce communication frequency and break dependency chains in aggregation.

* **Hoisting Ciphertext Transfer:**
  * **Problem:** Encrypted gradients (`g_cts`, `h_cts`) were being shuffled to PPs at every tree level.
  * **Solution:** Moved the transfer outside the tree-building loop. Transferred once, used by all levels.
  * **Result:** `simp.shuffle` calls reduced from ~1586 to ~602 (for depth 3).

* **Tree Reduction (Tree Sum):**
  * **Problem:** Aggregation of chunks and features used linear accumulation (`acc = acc + x`), creating long dependency chains that blocked the scheduler.
  * **Solution:** Implemented `tree_sum` to perform binary tree reduction (`((a+b) + (c+d))`).
  * **Result:**
    * `bfv.add` time dropped from ~293s to ~181s (-38%).
    * `bfv.mul` time dropped from ~328s to ~248s (-24%) due to better scheduling.
    * **Total Leaf Op Time** dropped from ~730s to ~533s (-27%).

**Benchmark (1M samples, depth 3):**

* **Total Time:** ~94s (Wall clock) / ~533s (Leaf Ops)
* **Accuracy:** 88.79%

### Phase 8: Transparent Ciphertext Optimization

**Goal:** Further reduce FHE overhead by skipping operations on "zero" (transparent) ciphertexts.

* **Transparent Ciphertext:**
  * Implemented in `bfv_impl.py` (`mul`, `add`, `rotate`).
  * If a ciphertext has size 0 (transparent zero), operations are skipped or simplified (e.g., `x + 0 = x`, `x * 0 = 0`).
  * **Result:** Significant reduction in actual FHE operations executed, especially for sparse updates or masked-out branches.

* **Negative Results (Reverted):**
  * **Parallel `batch_encode`:** Attempted to parallelize encoding within chunks. Resulted in severe performance regression (13x slowdown for that op) due to memory bandwidth saturation and cache contention. Reverted to serial.
  * **Fused `bfv.dot`:** Implemented a fused dot product. While promising for compute-bound tasks, it introduced complexity and was held for now in favor of the simpler `tree_sum` approach which benefits from the transparent optimization.

**Benchmark (1M samples, depth 5):**

* **Total Time:** **315s** (vs 500s in Phase 6)
* **Improvement:** ~37% faster than Phase 6.

### Phase 9: JAX Integration & Encoding Optimization

**Goal:** Optimize JAX <-> Python/BFV data transfer and reduce Python overhead in BFV encoding.

* **Device-Resident Tensor (Lazy Transfer):**
  * **Problem:** `run_jax` was forcing synchronization (`Device -> Host`) for every output, causing massive overhead
    (e.g., 3.5s out of 4.8s total time for some ops).
  * **Solution:** Modified `TensorValue` to hold `jax.Array` directly. `run_jax` now accepts and returns JAX arrays without conversion.
    Data is only transferred to Host when explicitly requested (e.g., by BFV ops).
  * **Result:** `run_jax` time dropped from ~13.5s to ~3.6s. Consecutive JAX operations now run entirely on device.

* **Fast Encoding (`tolist`):**
  * **Problem:** `bfv.batch_encode` used Python list comprehension `[int(x) for x in arr]` to convert Numpy arrays to Python lists
    for the C++ binding. This was slow and CPU-intensive.
  * **Solution:** Replaced with `arr.tolist()`.
  * **Result:** Encoding throughput increased by ~3x.

**Current Status (1M samples, depth 5):**

* **Tracing:** 62.07s
* **Execution:** 261.90s
* **Total Time:** 323.96s
* **Note:** While total time is similar (data transfer cost moved from `run_jax` to `batch_encode`), the architecture is now optimized
  for future JAX-native extensions.

### Phase 10: Distributed Profiling & Driver Mode

**Goal:** Validate performance in a realistic distributed setting (Driver + Workers) and enable detailed profiling.

* **Driver Mode Benchmark:**
  * Ran the 1M sample benchmark using `mplang.v2.cli` to simulate a real cluster environment (Driver + 2 Workers).
  * **Command:**

    ```bash
    # Start cluster
    nohup uv run -m mplang.v2.cli up -w 2 > /tmp/mplang_up.log 2>&1 &

    # Run benchmark
    uv run -m mplang.v2.cli run -f examples/v2/sgb.py --entry run_sgb_bench -w 2
    ```

  * **Result:**
    * **Tracing:** 61.39s
    * **Execution:** 376.87s
    * **Total:** 438.26s
    * **Accuracy:** 89.26%
    * **Note:** Execution is slightly slower than local simulation due to HTTP communication overhead (localhost loopback), but confirms the system scales correctly in distributed mode.

* **Distributed Profiling:**
  * Implemented `job_id` propagation to link Driver and Worker traces.
  * Added `mplang.v2.cli trace merge` tool to combine multi-party traces into a single Perfetto view.
  * **Command:**

    ```bash
    uv run -m mplang.v2.cli trace merge "trace_*.json" -o sgb_1m_merged.json
    ```

  * **Visualization:** The merged trace clearly shows the interaction between Party 0 and Party 1, with communication events (`comm.send`) annotated with data size.

### Phase 11: JAX/BFV Interface Optimization

**Goal:** Eliminate the massive overhead of passing 100,000+ small tensors between JAX and BFV.

* **Problem:**
  * Profiling revealed that `tensor.run_jax` was a major bottleneck, not due to computation, but due to returning a Tuple of ~100,000 small arrays
    (one per chunk per feature).
  * This caused massive object creation overhead and bloated the IR.
* **Solution:**
  * **Batch Return:** Modified `sgb.py` to return a single large 2D tensor `(N, slot_count)` from JAX instead of a tuple of 1D tensors.
  * **Batch Encode:** Enhanced `bfv.batch_encode` to accept a single 2D tensor and perform optimized row-wise encoding in C++.
* **Result:**
  * **Total Time (Local Sim):** Dropped from ~438s (Distributed) / ~323s (Phase 9 Local) to **220.8s**.
  * **`tensor.run_jax` Time:** Reduced to negligible levels (~1.4s total).
  * **Throughput:** System is now fully compute-bound by BFV operations (`mul`, `add`, `rotate`), which is the ideal state for an FHE application.

**Benchmark (1M samples, depth 5):**

* **Total Time:** **220.8s** (Local Simulation)
* **Accuracy:** 89.26%
* **Speedup:** ~1.5x faster than Phase 9.
