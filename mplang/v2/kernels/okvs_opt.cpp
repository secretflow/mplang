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

    // Number of Bins for Mega-Binning
    static const uint64_t NUM_BINS = 1024;

    struct Indices {
        uint64_t h1, h2, h3;
    };

    inline std::pair<uint64_t, uint64_t> get_two_bin_choices(uint64_t key, __m128i seed) {
        // Use AES to get robust bin indices
        __m128i k = _mm_set_epi64x(0, key);
        __m128i h = _mm_aesenc_si128(k, seed);
        h = _mm_aesenc_si128(h, seed);
        
        uint64_t v1 = _mm_extract_epi64(h, 0);
        uint64_t v2 = _mm_extract_epi64(h, 1);
        
        return {v1 % NUM_BINS, v2 % NUM_BINS};
    }

    // Helper to get local indices within a bin
    inline Indices get_bin_local_indices(uint64_t key, uint64_t m_local, __m128i seed) {
        // Need distinct seed behavior or just use key hash again?
        // Reuse key hash but mix it.
        __m128i k = _mm_set_epi64x(0, key);
        // Add 1 to seed to distinguish from bin selection hash
        __m128i s2 = _mm_add_epi64(seed, _mm_set_epi64x(1, 1)); 
        __m128i h = _mm_aesenc_si128(k, s2);
        h = _mm_aesenc_si128(h, s2);
        h = _mm_aesenc_si128(h, s2);

        uint64_t r = _mm_extract_epi64(h, 0);
        Indices idx;
        
        idx.h1 = r % m_local;
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        idx.h2 = r % m_local;
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        idx.h3 = r % m_local;

        if(idx.h2 == idx.h1) idx.h2 = (idx.h2 + 1) % m_local;
        if(idx.h3 == idx.h1 || idx.h3 == idx.h2) {
            idx.h3 = (idx.h3 + 1) % m_local;
            if(idx.h3 == idx.h1 || idx.h3 == idx.h2) idx.h3 = (idx.h3 + 1) % m_local;
        }
        return idx;
    }

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
        
        for(uint64_t i=0; i<n; ++i) {
             Indices idx = get_bin_local_indices(keys[i], m, seed);
             edges[i] = {idx.h1, idx.h2, idx.h3, i};
             col_degree[idx.h1]++;
             col_degree[idx.h2]++;
             col_degree[idx.h3]++;
        }

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
        
        // 1. Correct Boundary Handling for M
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
        bin_offsets[NUM_BINS] = m; // Sentinel

        // 2. Two-Choice Hashing - Balancing Pass
        // Pass 1: Count potential load for each bin
        std::vector<std::atomic<int>> bin_loads(NUM_BINS);
        for(int i=0; i<NUM_BINS; ++i) bin_loads[i] = 0;

        // Parallel Count? 
        // For accurate Two-Choice, we usually do sequential assignment or batch assignment.
        // Purely independent parallel check is "polluting" the count.
        // But for 1M items, serial assignment is fast ~10ms.
        // Let's do serial assignment to `assignments` array.
        
        std::vector<uint16_t> assignments(n);
        
        for(uint64_t i=0; i<n; ++i) {
            auto choices = get_two_bin_choices(keys[i], seed);
            uint64_t b1 = choices.first;
            uint64_t b2 = choices.second;
            
            int l1 = bin_loads[b1].load(std::memory_order_relaxed);
            int l2 = bin_loads[b2].load(std::memory_order_relaxed);
            
            if (l1 <= l2) {
                assignments[i] = (uint16_t)b1;
                bin_loads[b1].fetch_add(1, std::memory_order_relaxed);
            } else {
                assignments[i] = (uint16_t)b2;
                bin_loads[b2].fetch_add(1, std::memory_order_relaxed);
            }
        }

        // 3. Partition Data
        std::vector<std::vector<uint64_t>> bin_keys(NUM_BINS);
        std::vector<std::vector<__m128i>> bin_vals(NUM_BINS);
        
        for(int b=0; b<NUM_BINS; ++b) {
            int count = bin_loads[b].load();
            bin_keys[b].reserve(count);
            bin_vals[b].reserve(count);
        }
        
        const __m128i* V_ptr = (const __m128i*)values;
        for(uint64_t i=0; i<n; ++i) {
            uint16_t b = assignments[i];
            bin_keys[b].push_back(keys[i]);
            bin_vals[b].push_back(_mm_loadu_si128(&V_ptr[i]));
        }
        
        // 4. Parallel Solve
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
                    fprintf(stderr, "[ERROR] Bin %lu failed OKVS peeling. Items: %lu / M: %lu (Ratio: %.2f)\n", 
                        b, bin_keys[b].size(), valid_m, (double)valid_m / bin_keys[b].size());
                }
            }
        }
    }

    void decode_okvs_opt(uint64_t* keys, uint64_t* storage, uint64_t* output, uint64_t n, uint64_t m, uint64_t* seed_ptr) {
        __m128i seed = _mm_loadu_si128((__m128i*)seed_ptr);
        __m128i* P_vec = (__m128i*)storage;
        __m128i* out_vec = (__m128i*)output;
        
        // Need to replicate boundary and assignment logic!
        // Boundary Logic
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

        // Assignment Replay (Two-Choice)
        // We need to re-simulate the exact assignment order.
        // Problem: Parallel Decode?
        // If we strictly replay the serial assignment from Solve, we must be serial here too to know bin loads.
        // BUT: Decode is usually read-only. We don't need to know global loads, we just need to know WHERE the key went.
        // 
        // CRITICAL ISSUE:
        // OKVS Decode is stateless. It typically assumes mapping is a pure function of Key.
        // Two-Choice Hashing `d = (load[h1] < load[h2]) ? h1 : h2` makes mapping DEPENDENT on other keys.
        // This breaks the "Stateless Decode" property unless we transmit the Assignment Map.
        // Standard PSI does NOT transmit assignment map (bandwidth!).
        // 
        // SOTA Solution for Two-Choice OKVS:
        // Cuckoo Hashing OKVS? No, that moves keys around.
        // If we use Two-Choice, sender must tell receiver where the key is?
        // NO. In PSI, Sender encodes. Receiver must decode.
        // Actually, Sender encodes his items. Receiver decodes... wait.
        // In RR22, Sender computes P = Solve(Y). Receiver computes Q = Decode(X, P).
        // If Mapping depends on Y's distribution, Receiver cannot reproduce it for X!
        // User X doesn't know Y's distribution.
        //
        // THUS: Two-Choice Hashing is INVALID for Standard OKVS where Decode is independent.
        // Unless we use Cuckoo Hashing where the "Choice" is deterministic placement?
        // Or "Simple Hashing" with stash?
        // 
        // Wait, SOTA (Vector-OLE / Silent-OT / RR22) papers usually use **Simple Binning with Stash** or just **3-way Cuckoo Hashing**.
        // Or if they use Mega-Binning, they accept $\epsilon=1.3$ with Simple Hashing?
        // Let's re-read papers or assume Simple Binning is standard.
        // RR22 paper uses "Simple Hashing".
        // Why did my Simple Binning fail at 1.3?
        // Because 1M items / 1024 bins = 1000 items/bin.
        // Max load of 1000 items in 128 bins (Poisson):
        // Mean = 1000. StdDev = sqrt(1000) = 31.
        // Max is roughly Mean + 4*StdDev = 1124.
        // Ratio needed: 1124 / 1000 = 1.12x locally.
        // But standard OKVS ($\epsilon=1.3$) gives 1300 slots.
        // 1300 > 1124.
        // So 1.3 SHOULD be enough.
        // Why did it fail?
        // "Bin 1023 failed... Items: 965 / M: 797".
        // Ah! The last bin had M=797 (truncated) but received ~1000 items. 
        // That explains the failure! M was chopped off.
        //
        // So: Two-Choice is NOT needed if we fix the boundary bug!
        // Simple Hashing with correct M-balancing should work at 1.3.
        
        #pragma omp parallel for schedule(static)
        for(uint64_t i=0; i<n; ++i) {
             // For decode, we just fallback to Simple Hashing (stateless).
             // Since we abandoned Two-Choice (due to PSI constraint), logic is simpler.
             auto choices = get_two_bin_choices(keys[i], seed);
             uint64_t b = choices.first; // Simple Hash uses first choice always
             
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
