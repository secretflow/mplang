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
    void solve_okvs(uint64_t* keys, uint64_t* values, uint64_t* output, uint64_t n, uint64_t m) {
        // Fixed seed for deterministic hashing
        __m128i seed = _mm_set_epi64x(0xDEADBEEF, 0xCAFEBABE);

        std::vector<std::vector<int>> col_to_rows(m);
        struct Row {
            uint64_t h1, h2, h3;
        };
        std::vector<Row> rows(n);

        // Build Graph
        // Parallelizing this requires care with col_to_rows (concurrent writes).
        // Strategy: Thread-local buffers or atomic locks. 
        // Given m is large (M > N), collisions on col_to_rows are rare but possible.
        // Simple approach: Parallel compute hashes (heavy), Serial build graph (light).

        #pragma omp parallel for schedule(static)
        for(uint64_t i=0; i<n; ++i) {
            Indices idx = hash_key(keys[i], m, seed);
            rows[i] = {idx.h1, idx.h2, idx.h3};
        }

        // Serial Graph Build (Memory bound, hard to parallelize efficiently without locks)
        for(uint64_t i=0; i<n; ++i) {
            const auto& r = rows[i];
            col_to_rows[r.h1].push_back(i);
            col_to_rows[r.h2].push_back(i);
            col_to_rows[r.h3].push_back(i);
        }

        // Compute Initial Column Degrees
        std::vector<int> col_degree(m, 0);
        for(uint64_t j=0; j<m; ++j) {
            col_degree[j] = col_to_rows[j].size();
        }

        // Initialize Peel Stack with degree-1 columns
        std::vector<int> peel_stack;
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

        // Peeling BFS
        while(head < peel_stack.size()) {
            int j = peel_stack[head++];
            if(col_removed[j]) continue;

            // Find owner row (the single active row for this col)
            int owner_row = -1;
            for(int r_idx : col_to_rows[j]) {
                if(!row_removed[r_idx]) {
                    owner_row = r_idx;
                    break;
                }
            }

            if(owner_row == -1) {
                // If col degree was 1 but row was removed by another col,
                // then this col has degree 0. Ignore.
                col_removed[j] = true;
                continue;
            }

            // Peel (j, owner_row)
            // j is determined by owner_row.
            // Neighbors of owner_row are determined LATER? No.
            // j is a leaf.
            // Solving j requires other variables in owner_row to be known.
            // Other variables are NOT degree 1 (or were processed earlier?).
            // If they are not processed, they will be processed later.
            // Wait.
            // If `k` in owner_row is NOT processed, it means `k` will be processed LATER?
            // If `k` is processed later, it means `k` is "above" `j` in dependency?
            // Peeling order: leaves first.
            // So `j` is a leaf. `k` is internal.
            // To solve `j`, we need `k`.
            // So `k` must be solved BEFORE `j`.
            // If `k` is solved later in peeling order (queue),
            // then `j` depends on something processed later.
            // Contradiction?
            // No. Peeling Order removes `j`.
            // `k` remains.
            // `k` is solved later.
            // So `j` needs `k` (solved later).
            // So logic implies: To solve `j`, we must wait for `k`.
            // So we solve `k` first.
            // Since `k` is peeled LATER,
            // we must solving in REVERSE of Peeling Order.
            // Queue: `j`, ..., `k`.
            // Solve: `k`, ..., `j`.
            // So LIFO (Stack) is correct.

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
            fprintf(stderr, "OKVS Core detected. Failed to peel all rows. N=%lu M=%lu Solved=%lu\n",
                    n, m, assignment_stack.size());
            // If we fail, just return (leaving output 0s or partial).
            return;
        }

        // Solve (Back Substitution: Reverse Order)
        __m128i* P_vec = (__m128i*)output;
        __m128i* V_vec = (__m128i*)values;
        memset(output, 0, m * 16);

        for(int i = (int)assignment_stack.size() - 1; i >= 0; --i) {
            const auto& a = assignment_stack[i];
            const auto& r = rows[a.row];

            __m128i val1 = _mm_loadu_si128(&P_vec[r.h1]);
            __m128i val2 = _mm_loadu_si128(&P_vec[r.h2]);
            __m128i val3 = _mm_loadu_si128(&P_vec[r.h3]);
            __m128i target = _mm_loadu_si128(&V_vec[a.row]);

            // Current sum of all cols involved
            __m128i current_sum = _mm_xor_si128(_mm_xor_si128(val1, val2), val3);

            // diff needed to make sum equal to target
            // P[a.col] is currently 0.
            __m128i diff = _mm_xor_si128(target, current_sum);

            _mm_storeu_si128(&P_vec[a.col], diff);
        }
    }

    void decode_okvs(uint64_t* keys, uint64_t* storage, uint64_t* output, uint64_t n, uint64_t m) {
        __m128i seed = _mm_set_epi64x(0xDEADBEEF, 0xCAFEBABE);
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
        __m128i fixed_key = _mm_set_epi64x(0x1234567890ABCDEF, 0xFEDCBA0987654321);
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
