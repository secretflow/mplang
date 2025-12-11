/*
 * Copyright 2025 Ant Group Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <vector>
#include <stack>
#include <random>
#include <immintrin.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <atomic>

extern "C" {

    // Number of Bins for Mega-Binning strategy.
    // 1024 bins implies ~1000 items per bin for N=1M, fitting the working set 
    // entirely in L1 cache (32KB/48KB) for maximum performance.
    static const uint64_t NUM_BINS = 1024;

    struct Indices {
        uint64_t h1, h2, h3;
    };

    // Stateless Bin Selection
    // Maps a key to a deterministic bin index [0, NUM_BINS).
    inline uint64_t get_bin_index(uint64_t key, __m128i seed) {
        __m128i k = _mm_set_epi64x(0, key);
        __m128i h = _mm_aesenc_si128(k, seed);
        h = _mm_aesenc_si128(h, seed);
        uint64_t v1 = _mm_extract_epi64(h, 0);
        return v1 % NUM_BINS;
    }

    // Generate 3 positions within a local bin of size m_local.
    inline Indices get_bin_local_indices(uint64_t key, uint64_t m_local, __m128i seed) {
        // Use a distinct seed mix to decorrelate from bin selection
        __m128i k = _mm_set_epi64x(0, key);
        __m128i s2 = _mm_add_epi64(seed, _mm_set_epi64x(1, 1)); 
        __m128i h = _mm_aesenc_si128(k, s2);
        h = _mm_aesenc_si128(h, s2);
        h = _mm_aesenc_si128(h, s2);

        uint64_t r = _mm_extract_epi64(h, 0);
        Indices idx;
        
        // Fast modulo for local indices
        idx.h1 = r % m_local;
        r = r * 6364136223846793005ULL + 1442695040888963407ULL; // LCG step
        idx.h2 = r % m_local;
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        idx.h3 = r % m_local;

        // Ensure distinct indices
        if(idx.h2 == idx.h1) idx.h2 = (idx.h2 + 1) % m_local;
        if(idx.h3 == idx.h1 || idx.h3 == idx.h2) {
            idx.h3 = (idx.h3 + 1) % m_local;
            if(idx.h3 == idx.h1 || idx.h3 == idx.h2) idx.h3 = (idx.h3 + 1) % m_local;
        }
        return idx;
    }

    // Core Peeling Solver for a single Bin
    bool solve_bin(
        const std::vector<uint64_t>& keys, 
        const std::vector<__m128i>& vals, 
        __m128i* P_local, 
        uint64_t m, 
        __m128i seed
    ) {
        uint64_t n = keys.size();
        if (n == 0) return true;

        struct Edge {
            uint64_t h1, h2, h3;
            uint64_t key_idx;
        };
        std::vector<Edge> edges(n);
        std::vector<int> col_degree(m, 0);
        
        // 1. Build Local Graph
        for(uint64_t i=0; i<n; ++i) {
             Indices idx = get_bin_local_indices(keys[i], m, seed);
             edges[i] = {idx.h1, idx.h2, idx.h3, i};
             col_degree[idx.h1]++;
             col_degree[idx.h2]++;
             col_degree[idx.h3]++;
        }

        // 2. CSR Construction
        std::vector<int> col_start(m + 1, 0);
        for(uint64_t j=0; j<m; ++j) {
            col_start[j+1] = col_start[j] + col_degree[j];
        }
        std::vector<int> flat_rows(n * 3);
        std::vector<int> fill_ptr = col_start;
        for(uint64_t i=0; i<n; ++i) {
            flat_rows[fill_ptr[edges[i].h1]++] = i;
            flat_rows[fill_ptr[edges[i].h2]++] = i;
            flat_rows[fill_ptr[edges[i].h3]++] = i;
        }

        // 3. Peeling Process
        std::vector<int> peel_stack;
        peel_stack.reserve(m);
        for(uint64_t j=0; j<m; ++j) {
            if(col_degree[j] == 1) peel_stack.push_back(j);
        }

        std::vector<bool> row_removed(n, false);
        std::vector<bool> col_removed(m, false);
        
        struct Assignment {
            int col;
            int row_idx;
        };
        std::vector<Assignment> assignment_stack;
        assignment_stack.reserve(n);
        
        int head = 0;
        while(head < peel_stack.size()) {
            int j = peel_stack[head++];
            if(col_removed[j]) continue;

            int owner_row = -1;
            for(int k=col_start[j]; k<col_start[j+1]; ++k) {
                int r = flat_rows[k];
                if(!row_removed[r]) {
                    owner_row = r;
                    break;
                }
            }
            if(owner_row == -1) {
                col_removed[j] = true;
                continue;
            }

            assignment_stack.push_back({j, owner_row});
            col_removed[j] = true;
            row_removed[owner_row] = true;

            const auto& e = edges[owner_row];
            uint64_t nbs[3] = {e.h1, e.h2, e.h3};
            for(uint64_t nb : nbs) {
                if(nb == (uint64_t)j) continue;
                if(col_removed[nb]) continue;
                col_degree[nb]--;
                if(col_degree[nb] == 1) peel_stack.push_back((int)nb);
            }
        }

        if(assignment_stack.size() != n) return false;

        // 4. Back-Substitution
        for(int i=(int)assignment_stack.size()-1; i>=0; --i) {
            auto a = assignment_stack[i];
            const auto& e = edges[a.row_idx];
            
            __m128i val1 = _mm_loadu_si128(&P_local[e.h1]);
            __m128i val2 = _mm_loadu_si128(&P_local[e.h2]);
            __m128i val3 = _mm_loadu_si128(&P_local[e.h3]);
            __m128i target = vals[e.key_idx];
            
            __m128i current = _mm_xor_si128(_mm_xor_si128(val1, val2), val3);
            __m128i diff = _mm_xor_si128(target, current);
            
            _mm_storeu_si128(&P_local[a.col], diff);
        }
        return true;
    }

    void solve_okvs_opt(uint64_t* keys, uint64_t* values, uint64_t* output, uint64_t n, uint64_t m, uint64_t* seed_ptr) {
        __m128i seed = _mm_loadu_si128((__m128i*)seed_ptr);
        
        // 1. Calculate Bin Boundaries
        // We divide M evenly among bins. The remainder is distributed to the first few bins.
        std::vector<uint64_t> bin_offsets(NUM_BINS + 1);
        std::vector<uint64_t> m_per_bin(NUM_BINS);
        
        uint64_t base_m = m / NUM_BINS;
        uint64_t remainder = m % NUM_BINS;
        
        uint64_t current_offset = 0;
        for(uint64_t b=0; b<NUM_BINS; ++b) {
            bin_offsets[b] = current_offset;
            m_per_bin[b] = base_m + (b < remainder ? 1 : 0);
            current_offset += m_per_bin[b];
        }
        bin_offsets[NUM_BINS] = m;

        // 2. Partition Data (Stateless)
        // Note on "Two-Choice Hashing":
        // While Two-Choice Hashing (selecting the lighter of 2 potential bins) would significantly 
        // reduce max bin load variance, it introduces "Statefulness".
        // The bin assignment for Key K would depend on the load of bins, which depends on other keys.
        // In standard PSI protocols (like RR22), the Decode step must be capable of processing keys 
        // independently or without knowledge of the full set distribution (Sender/Receiver separation).
        // Therefore, we use **Simple Binning** (Stateless Hash) where Bin(K) = H(K) % Bins.
        // We mitigate the resulting variance ("Balls-in-Bins" problem) by using a slightly larger 
        // expansion factor (epsilon ~ 1.35) which is bandwidth-acceptable and ensures stability.
        
        std::vector<std::vector<uint64_t>> bin_keys(NUM_BINS);
        std::vector<std::vector<__m128i>> bin_vals(NUM_BINS);
        
        // Pre-allocate to reduce reallocation overhead (assume ~uniform distribution)
        // 1.5x margin for pre-allocation safety
        size_t est_size = (n / NUM_BINS) * 3 / 2;
        for(int b=0; b<NUM_BINS; ++b) {
            bin_keys[b].reserve(est_size);
            bin_vals[b].reserve(est_size);
        }
        
        const __m128i* V_ptr = (const __m128i*)values;
        for(uint64_t i=0; i<n; ++i) {
            uint64_t b = get_bin_index(keys[i], seed);
            bin_keys[b].push_back(keys[i]);
            bin_vals[b].push_back(_mm_loadu_si128(&V_ptr[i]));
        }
        
        // 3. Parallel Solve
        // Each bin is solved independently. This logic is perfectly parallelizable (embarrassingly parallel).
        // The working set for each bin (~1000 items) stays hot in L1 Cache.
        memset(output, 0, m * 16);
        __m128i* P_vec = (__m128i*)output;

        #pragma omp parallel for schedule(dynamic)
        for(uint64_t b=0; b<NUM_BINS; ++b) {
            if(bin_keys[b].empty()) continue;
            
            uint64_t offset = bin_offsets[b];
            uint64_t valid_m = m_per_bin[b];
            
            if(!solve_bin(bin_keys[b], bin_vals[b], &P_vec[offset], valid_m, seed)) {
                #pragma omp critical
                {
                    fprintf(stderr, "[ERROR] Bin %lu failed OKVS peeling. Items: %lu / M: %lu (Ratio: %.2f). Try increasing expansion factor.\n", 
                        b, bin_keys[b].size(), valid_m, (double)valid_m / bin_keys[b].size());
                }
            }
        }
    }

    void decode_okvs_opt(uint64_t* keys, uint64_t* storage, uint64_t* output, uint64_t n, uint64_t m, uint64_t* seed_ptr) {
        __m128i seed = _mm_loadu_si128((__m128i*)seed_ptr);
        __m128i* P_vec = (__m128i*)storage;
        __m128i* out_vec = (__m128i*)output;
        
        // Replicate Boundary Logic
        std::vector<uint64_t> bin_offsets(NUM_BINS + 1);
        std::vector<uint64_t> m_per_bin(NUM_BINS); 
        uint64_t base_m = m / NUM_BINS;
        uint64_t remainder = m % NUM_BINS;
        uint64_t current_offset = 0;
        for(uint64_t b=0; b<NUM_BINS; ++b) {
            bin_offsets[b] = current_offset;
            m_per_bin[b] = base_m + (b < remainder ? 1 : 0);
            current_offset += m_per_bin[b];
        }

        // Parallel Stateless Decode
        #pragma omp parallel for schedule(static)
        for(uint64_t i=0; i<n; ++i) {
            uint64_t b = get_bin_index(keys[i], seed);
             
            uint64_t m_local = m_per_bin[b];
            uint64_t offset = bin_offsets[b];
             
            Indices idx = get_bin_local_indices(keys[i], m_local, seed);
             
            __m128i val = _mm_xor_si128(
                _mm_xor_si128(_mm_loadu_si128(&P_vec[offset + idx.h1]), _mm_loadu_si128(&P_vec[offset + idx.h2])),
                _mm_loadu_si128(&P_vec[offset + idx.h3])
            );
            _mm_storeu_si128(&out_vec[i], val);
        }
    }
}
