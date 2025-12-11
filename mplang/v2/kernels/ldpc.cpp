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
#include <cstring>
#include <vector>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

/**
 * @brief LDPC Encoding: Compute Syndrome s = H * x
 * 
 * H is a sparse M x N binary matrix (CSR format).
 * x is a dense N-vector of 128-bit blocks (N * 16 bytes).
 * s is a dense M-vector of 128-bit blocks (M * 16 bytes).
 * 
 * Logic: For each row i of H, s[i] = XOR(x[j]) for all j where H[i, j] = 1.
 * 
 * @param message_ptr  Pointer to message x (N * 2 uint64_t)
 * @param indices_ptr  Pointer to CSR indices (uint64_t)
 * @param indptr_ptr   Pointer to CSR indptr (M+1 uint64_t)
 * @param output_ptr   Pointer to output s (M * 2 uint64_t)
 * @param m            Number of rows in H (syndrome length)
 * @param n            Number of cols in H (message length)
 */
void ldpc_encode(const uint64_t* message_ptr, 
                 const uint64_t* indices_ptr, 
                 const uint64_t* indptr_ptr, 
                 uint64_t* output_ptr, 
                 uint64_t m, 
                 uint64_t n) {
    
    // Check alignment
    // We assume message_ptr and output_ptr are 16-byte aligned for SSE/AVX?
    // JAX/Numpy arrays are usually aligned.
    
    // Cast to __m128i for efficiency
    // But we need to handle potential unaligned access if numpy doesn't align.
    // _mm_loadu_si128 handles unaligned.

    const __m128i* x_vec = (const __m128i*)message_ptr;
    __m128i* s_vec = (__m128i*)output_ptr;

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < m; ++i) {
        // Row i
        __m128i sum = _mm_setzero_si128();
        
        uint64_t start = indptr_ptr[i];
        uint64_t end = indptr_ptr[i+1];
        
        for (uint64_t k = start; k < end; ++k) {
            uint64_t col_idx = indices_ptr[k];
            // XOR accumulation
            // Use loadu for safety
            __m128i val = _mm_loadu_si128(&x_vec[col_idx]);
            sum = _mm_xor_si128(sum, val);
        }
        
        _mm_storeu_si128(&s_vec[i], sum);
    }
}

}
