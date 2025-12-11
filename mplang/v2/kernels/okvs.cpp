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

extern "C" {

    // AES-NI Hashing Helper
    struct Indices {
        uint64_t h1, h2, h3;
    };

    inline Indices hash_key(uint64_t key, uint64_t m, __m128i seed) {
        __m128i k = _mm_set_epi64x(0, key);
        __m128i h = _mm_aesenc_si128(k, seed);
        h = _mm_aesenc_si128(h, seed);

        uint64_t v1 = _mm_extract_epi64(h, 0);
        uint64_t v2 = _mm_extract_epi64(h, 1);

        Indices idx;
        idx.h1 = v1 % m;
        idx.h2 = v2 % m;
        idx.h3 = (v1 ^ v2) % m;

        // Enforce distinct indices
        if(idx.h2 == idx.h1) {
            idx.h2 = (idx.h2 + 1) % m;
        }
        if(idx.h3 == idx.h1 || idx.h3 == idx.h2) {
            idx.h3 = (idx.h3 + 1) % m;
            if(idx.h3 == idx.h1 || idx.h3 == idx.h2) {
                idx.h3 = (idx.h3 + 1) % m;
            }
        }

        return idx;
    }

    // Solve OKVS System: H * P = V
    void solve_okvs(uint64_t* keys, uint64_t* values, uint64_t* output, uint64_t n, uint64_t m, uint64_t* seed_ptr) {
        // Load dynamic seed
        __m128i seed = _mm_loadu_si128((__m128i*)seed_ptr);

        struct Row {
            uint64_t h1, h2, h3;
        };
        std::vector<Row> rows(n);

        // 1. Parallel Hash Compute
        #pragma omp parallel for schedule(static)
        for(uint64_t i=0; i<n; ++i) {
            Indices idx = hash_key(keys[i], m, seed);
            rows[i] = {idx.h1, idx.h2, idx.h3};
        }

        // 2. Count Degrees (Serial or Atomic)
        // Since M ~ 1.2N, atomic contention is low? Serial is safe and simple.
        std::vector<int> col_degree(m, 0);
        for(uint64_t i=0; i<n; ++i) {
            col_degree[rows[i].h1]++;
            col_degree[rows[i].h2]++;
            col_degree[rows[i].h3]++;
        }

        // 3. Build CSR Structure (Flat Arrays) to replace vector<vector>
        // col_start[j] points to start of column j's rows in flat_rows
        std::vector<int> col_start(m + 1, 0);
        
        // Prefix sum to compute start positions
        // col_start[0] = 0
        // col_start[j+1] = col_start[j] + degree[j]
        for(uint64_t j=0; j<m; ++j) {
            col_start[j+1] = col_start[j] + col_degree[j];
        }

        // Total edges = 3 * N implies flat_rows size
        std::vector<int> flat_rows(n * 3);
        
        // Temporary copy of start indices to use as fill pointers
        std::vector<int> fill_ptr = col_start;

        for(uint64_t i=0; i<n; ++i) {
            const auto& r = rows[i];
            flat_rows[fill_ptr[r.h1]++] = i;
            flat_rows[fill_ptr[r.h2]++] = i;
            flat_rows[fill_ptr[r.h3]++] = i;
        }
        
        // 4. Initialize Peeling
        std::vector<int> peel_stack;
        peel_stack.reserve(m);
        for(uint64_t j=0; j<m; ++j) {
            if(col_degree[j] == 1) peel_stack.push_back(j);
        }

        std::vector<bool> row_removed(n, false);
        std::vector<bool> col_removed(m, false);

        struct Assignment {
            int col;
            int row;
        };
        std::vector<Assignment> assignment_stack;
        assignment_stack.reserve(n);

        int head = 0;

        // 5. Peeling BFS
        while(head < peel_stack.size()) {
            int j = peel_stack[head++];
            if(col_removed[j]) continue;

            // Find owner row: Iterate over edges of col j using flat arrays
            int owner_row = -1;
            int start = col_start[j];
            int end = col_start[j+1];
            
            for(int k=start; k<end; ++k) {
                int r_idx = flat_rows[k];
                if(!row_removed[r_idx]) {
                    owner_row = r_idx;
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

            // Update neighbors
            const auto& r = rows[owner_row];
            uint64_t nbs[3] = {r.h1, r.h2, r.h3};
            for(uint64_t neighbor : nbs) {
                if(neighbor == (uint64_t)j) continue;
                if(col_removed[neighbor]) continue;

                col_degree[neighbor]--;
                if(col_degree[neighbor] == 1) {
                    peel_stack.push_back((int)neighbor);
                }
            }
        }

        if(assignment_stack.size() != n) {
            fprintf(stderr, "[ERROR] OKVS Peeling Failed. N=%lu M=%lu Solved=%lu\n",
                    n, m, assignment_stack.size());
            // Zero output to identify failure clearly
            memset(output, 0, m * 16);
            return;
        }

        // 6. Back Substitution
        // Use 128-bit intrinsics for value XORing
        __m128i* P_vec = (__m128i*)output;
        __m128i* V_vec = (__m128i*)values;
        memset(output, 0, m * 16);

        // Process in reverse constraint order (LIFO)
        for(int i = (int)assignment_stack.size() - 1; i >= 0; --i) {
            const auto& a = assignment_stack[i];
            const auto& r = rows[a.row];

            __m128i val1 = _mm_loadu_si128(&P_vec[r.h1]);
            __m128i val2 = _mm_loadu_si128(&P_vec[r.h2]);
            __m128i val3 = _mm_loadu_si128(&P_vec[r.h3]);
            __m128i target = _mm_loadu_si128(&V_vec[a.row]);

            __m128i current_sum = _mm_xor_si128(_mm_xor_si128(val1, val2), val3);
            __m128i diff = _mm_xor_si128(target, current_sum);

            _mm_storeu_si128(&P_vec[a.col], diff);
        }
    }

    void decode_okvs(uint64_t* keys, uint64_t* storage, uint64_t* output, uint64_t n, uint64_t m, uint64_t* seed_ptr) {
        __m128i seed = _mm_loadu_si128((__m128i*)seed_ptr);
        __m128i* P_vec = (__m128i*)storage;
        __m128i* out_vec = (__m128i*)output;

        #pragma omp parallel for schedule(static)
        for(uint64_t i=0; i<n; ++i) {
            Indices idx = hash_key(keys[i], m, seed);
            __m128i val = _mm_xor_si128(
                _mm_xor_si128(_mm_loadu_si128(&P_vec[idx.h1]), _mm_loadu_si128(&P_vec[idx.h2])),
                _mm_loadu_si128(&P_vec[idx.h3])
            );
            _mm_storeu_si128(&out_vec[i], val);
        }
    }

    // Helper for key expansion
    inline __m128i aes_keygen_assist(__m128i temp1, __m128i temp2) {
        __m128i temp3;
        temp2 = _mm_shuffle_epi32(temp2, 0xff);
        temp3 = _mm_slli_si128(temp1, 0x4);
        temp1 = _mm_xor_si128(temp1, temp3);
        temp3 = _mm_slli_si128(temp3, 0x4);
        temp1 = _mm_xor_si128(temp1, temp3);
        temp3 = _mm_slli_si128(temp3, 0x4);
        temp1 = _mm_xor_si128(temp1, temp3);
        temp1 = _mm_xor_si128(temp1, temp2);
        return temp1;
    }

    void aes_key_expand(__m128i user_key, __m128i* key_schedule) {
        key_schedule[0] = user_key;
        key_schedule[1] = aes_keygen_assist(key_schedule[0], _mm_aeskeygenassist_si128(key_schedule[0], 0x01));
        key_schedule[2] = aes_keygen_assist(key_schedule[1], _mm_aeskeygenassist_si128(key_schedule[1], 0x02));
        key_schedule[3] = aes_keygen_assist(key_schedule[2], _mm_aeskeygenassist_si128(key_schedule[2], 0x04));
        key_schedule[4] = aes_keygen_assist(key_schedule[3], _mm_aeskeygenassist_si128(key_schedule[3], 0x08));
        key_schedule[5] = aes_keygen_assist(key_schedule[4], _mm_aeskeygenassist_si128(key_schedule[4], 0x10));
        key_schedule[6] = aes_keygen_assist(key_schedule[5], _mm_aeskeygenassist_si128(key_schedule[5], 0x20));
        key_schedule[7] = aes_keygen_assist(key_schedule[6], _mm_aeskeygenassist_si128(key_schedule[6], 0x40));
        key_schedule[8] = aes_keygen_assist(key_schedule[7], _mm_aeskeygenassist_si128(key_schedule[7], 0x80));
        key_schedule[9] = aes_keygen_assist(key_schedule[8], _mm_aeskeygenassist_si128(key_schedule[8], 0x1b));
        key_schedule[10] = aes_keygen_assist(key_schedule[9], _mm_aeskeygenassist_si128(key_schedule[9], 0x36));
    }

    // AES-128 Expansion
    void aes_128_expand(uint64_t* seeds, uint64_t* output, uint64_t num_seeds, uint64_t length) {
        __m128i* seeds_vec = (__m128i*)seeds;
        __m128i* out_vec = (__m128i*)output;

        // Fixed Key (Arbitrary constant)
        // Using PI fractional part (Nothing-up-my-sleeve numbers)
        // 0x243F6A8885A308D3 (PI_FRAC_1)
        // 0x13198A2E03707344 (PI_FRAC_2)
        __m128i fixed_key = _mm_set_epi64x(0x243F6A8885A308D3, 0x13198A2E03707344);
        __m128i round_keys[11];
        aes_key_expand(fixed_key, round_keys);

        // For each seed
        #pragma omp parallel for schedule(static)
        for(uint64_t i=0; i<num_seeds; ++i) {
             __m128i s = _mm_loadu_si128(&seeds_vec[i]);

             // Expand to 'length' blocks
             for(uint64_t j=0; j<length; ++j) {
                 // Block = Seed ^ j
                 // Note: j is passed as counter mix
                 __m128i ctr = _mm_set_epi64x(0, j);
                 __m128i block = _mm_xor_si128(s, ctr);

                 // Encrypt Block
                 __m128i state = _mm_xor_si128(block, round_keys[0]);
                 for(int r=1; r<10; ++r) {
                     state = _mm_aesenc_si128(state, round_keys[r]);
                 }
                 state = _mm_aesenclast_si128(state, round_keys[10]);

                 // Store
                 // Output is flat: [seed0_0, seed0_1 ... seed1_0 ...]
                 _mm_storeu_si128(&out_vec[i * length + j], state);
             }
        }
    }
}
