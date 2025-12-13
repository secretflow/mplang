# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

import jax.numpy as jnp
import numpy as np

import mplang.v2 as mp
import mplang.v2.edsl as el
from mplang.v2.dialects import bfv, simp, tensor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_benchmark(N=1000, K=10):
    logger.info(f"Starting benchmark with N={N}, K={K}")

    # Generate synthetic data
    # BFV works with integers. We scale floats to integers.
    scale = 100.0
    np.random.seed(42)
    scores_data = np.random.rand(N).astype(np.float32) * 100
    amounts_data = (np.random.rand(N).astype(np.float32) * 1000 * scale).astype(
        np.int64
    )

    start_time = time.time()

    with el.Tracer() as tracer:
        # --- Step 1: Data Setup ---

        # Party A (Rank 0) holds scores
        scores = simp.pcall_static((0,), lambda: tensor.constant(scores_data))

        # Party B (Rank 1) holds amounts
        amounts = simp.pcall_static((1,), lambda: tensor.constant(amounts_data))

        # --- Step 2: Key Gen & Encryption (Party B) ---
        def setup_and_encrypt(amounts_local):
            # BFV setup
            # N=4096 slots (secure parameter)
            pk, sk = bfv.keygen(poly_modulus_degree=4096)
            relin_keys = bfv.make_relin_keys(sk)
            encoder = bfv.create_encoder(poly_modulus_degree=4096)

            # Encode & Encrypt
            # Note: BFV encrypts vectors. We need to handle batching if N > 4096.
            # For simplicity, assume N <= 4096 for now.
            pt = bfv.encode(amounts_local, encoder)
            ct = bfv.encrypt(pt, pk)

            return ct, pk, sk, relin_keys, encoder

        ct_amounts, _pk, sk, _relin_keys, encoder = simp.pcall_static(
            (1,), setup_and_encrypt, amounts
        )

        # --- Step 3: Transfer to A ---
        ct_amounts_at_a = simp.shuffle_static(ct_amounts, {0: 1})

        # --- Step 4: Sort & Aggregate (Party A) ---
        def sort_and_aggregate(scores_local, ct_amounts_local):
            # Sort indices based on scores
            def argsort_desc(x):
                return jnp.argsort(x)[::-1]

            indices = tensor.run_jax(argsort_desc, scores_local)

            # Reorder encrypted vector?
            # BFV ciphertext is a vector. We cannot easily "gather" on it homomorphically
            # without complex operations (permutation matrix mult).
            #
            # However, for this benchmark, maybe we just want to test "add" performance?
            # Or maybe we simulate "selection" by multiplying with a mask?
            #
            # Implementing "gather" on BFV is hard.
            # Let's simplify: Just sum the first K elements (assuming they are the top K).
            # Or, we can rotate and sum.
            #
            # For the sake of "testing runtime", let's just do a simple aggregation:
            # Sum all elements.

            # But wait, BFV add is element-wise.
            # To sum all slots, we need rotation.
            # bfv.rotate is available?

            # Let's just do element-wise addition with another ciphertext for now to test basic ops.
            # Or add a constant.

            # Let's simulate "filtering" by multiplying with a plaintext mask (0/1).
            # We create a mask for top K indices.

            # 1. Create mask (1 for top K, 0 otherwise)
            # This happens in plaintext (Party A knows indices)
            # We need to map indices to mask.
            # indices[0] is the index of the largest element.
            # We want to select amounts[indices[0]], amounts[indices[1]], ...
            # So we want a mask that has 1 at indices[0]...indices[K-1].

            # But wait, we can't easily apply this mask to a packed ciphertext
            # unless the mask is also packed in the same order.
            # Yes, we can encode the mask and multiply.

            # But we need to construct the mask dynamically based on `indices`.
            # `indices` is a Tensor. We need to convert it to a mask Tensor.

            def make_mask(idxs):
                m = jnp.zeros(N, dtype=jnp.int64)
                # JAX array update
                # m = m.at[idxs[:K]].set(1)
                # But K is python int.
                # We need to slice idxs.
                top_idxs = idxs[:K]
                m = m.at[top_idxs].set(1)
                return m

            mask_tensor = tensor.run_jax(make_mask, indices)

            # Now we have a mask tensor. We need to encode it to BFV Plaintext.
            # But `bfv.encode` expects a Tensor.
            # We need `encoder` here. But `encoder` was created on Party B.
            # Party A needs its own encoder?
            # Encoder parameters are public. Party A can create one.
            encoder_a = bfv.create_encoder(poly_modulus_degree=4096)

            pt_mask = bfv.encode(
                mask_tensor, encoder_a
            )  # Multiply: ct_amounts * pt_mask
            # This zeros out non-top-K elements.
            ct_filtered = bfv.mul(ct_amounts_local, pt_mask)

            # Now we want to sum the slots of ct_filtered.
            # This requires rotations.
            # Assuming we have a `sum_slots` or similar helper, or we implement it.
            # For now, let's just return the filtered vector and decrypt it, then sum in plaintext.
            # (To avoid implementing complex rotation logic in this benchmark script)

            return ct_filtered

        ct_result_at_a = simp.pcall_static(
            (0,), sort_and_aggregate, scores, ct_amounts_at_a
        )

        # --- Step 5: Transfer to B ---
        ct_result_at_b = simp.shuffle_static(ct_result_at_a, {1: 0})

        # --- Step 6: Decrypt (Party B) ---
        def decrypt_and_sum(ct, sk_local, encoder_local):
            pt = bfv.decrypt(ct, sk_local)
            vec = bfv.decode(pt, encoder_local)
            return tensor.run_jax(jnp.sum, vec)

        final_result = simp.pcall_static(
            (1,), decrypt_and_sum, ct_result_at_b, sk, encoder
        )

        tracer.finalize(final_result)

    graph = tracer.graph
    compile_time = time.time()
    logger.info(f"Graph construction took {compile_time - start_time:.4f}s")

    # Execute
    host = mp.make_simulator(2)

    exec_start = time.time()
    result = host.evaluate_graph(graph, {})
    exec_end = time.time()

    logger.info(f"Execution took {exec_end - exec_start:.4f}s")

    # Verify result (locally)
    # Sort locally to check correctness
    sorted_indices = np.argsort(scores_data)[::-1]
    top_k_indices = sorted_indices[:K]
    expected_sum = np.sum(amounts_data[top_k_indices])

    # Result is from Party 1, so it's in result[1]
    # result[0] is None
    actual_sum = result[1]

    logger.info(f"Expected Sum: {expected_sum:.4f}")
    logger.info(f"Actual Sum:   {actual_sum:.4f}")

    diff = abs(expected_sum - actual_sum)
    logger.info(f"Difference:   {diff:.6f}")

    if diff < 1.0:  # Allow some error due to fixed point
        logger.info("SUCCESS: Result matches expectation!")
    else:
        logger.error("FAILURE: Result mismatch!")


if __name__ == "__main__":
    # Run larger test with secure parameters
    run_benchmark(N=1000, K=10)
