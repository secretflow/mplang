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
from collections.abc import Callable
from typing import TypeVar

import jax.numpy as jnp
import numpy as np

import mplang.v2 as mp
import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects import phe, simp, tensor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


T = TypeVar("T")


def tree_reduce(fn: Callable[[T, T], T], items: list[T]) -> T:
    """Tree-structured reduction with O(log n) depth for parallel execution.

    Unlike functools.reduce which creates a linear dependency chain O(n),
    tree_reduce creates a balanced tree structure O(log n) that enables
    parallel execution when the backend supports it.

    Example:
        functools.reduce: ((((1+2)+3)+4)+5)  # depth=4, sequential
        tree_reduce:      (((1+2)+(3+4))+5)  # depth=3, level 0 parallelizable
    """
    if len(items) == 0:
        raise ValueError("Cannot reduce empty list")
    items = list(items)  # Make a copy to avoid mutating input
    while len(items) > 1:
        next_level = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                next_level.append(fn(items[i], items[i + 1]))
            else:
                next_level.append(items[i])
        items = next_level
    return items[0]


def run_benchmark():
    logger.info("Starting PHE Sort & Aggregate Benchmark")

    # Larger dataset for benchmarking
    N = 1000
    K = 10

    with el.Tracer() as tracer:
        # --- Step 1: Data Setup (Distributed) ---

        # Party A (Rank 0) creates scores
        def create_scores():
            np.random.seed(42)
            return tensor.constant(np.random.rand(N).astype(np.float32) * 100)

        scores = simp.pcall_static((0,), create_scores)

        # Party B (Rank 1) creates amounts
        def create_amounts():
            np.random.seed(43)
            return tensor.constant(np.random.rand(N).astype(np.float32) * 1000)

        amounts = simp.pcall_static((1,), create_amounts)

        # --- Step 2: Key Generation & Encryption (Party B) ---

        def setup_and_encrypt(amounts_local):
            # Generate keys
            pk, sk = phe.keygen(key_size=2048)
            # Create encoder
            encoder = phe.create_encoder(dtype=elt.f32, fxp_bits=16)
            # Encrypt amounts
            enc_amounts = phe.encrypt_auto(amounts_local, encoder, pk)
            return enc_amounts, pk, sk, encoder

        # Execute on Party B (Rank 1)
        enc_amounts, _pk, sk, encoder = simp.pcall_static(
            (1,), setup_and_encrypt, amounts
        )

        # --- Step 3: Transfer Encrypted Data to A ---

        # Send encrypted amounts from B (1) to A (0)
        enc_amounts_at_a = simp.shuffle_static(enc_amounts, {0: 1})

        # --- Step 4: Sorting (Party A) ---
        def sort_and_aggregate(scores_local, enc_amounts_local, k_val):
            # 1. Compute sort indices
            def argsort_desc(x):
                return jnp.argsort(x)[::-1]

            indices = tensor.run_jax(argsort_desc, scores_local)

            # 2. Reorder encrypted amounts
            sorted_enc = tensor.gather(enc_amounts_local, indices)

            # 3. Aggregate top K
            top_k = tensor.slice_tensor(sorted_enc, (0,), (k_val,))

            # Sum top K using tree reduction for O(log n) depth
            items = [tensor.slice_tensor(top_k, (i,), (i + 1,)) for i in range(k_val)]
            total = tree_reduce(phe.add, items)

            return total

        # Execute on Party A (Rank 0)
        total_enc_at_a = simp.pcall_static(
            (0,), sort_and_aggregate, scores, enc_amounts_at_a, K
        )

        # --- Step 5: Transfer Result to B for Decryption ---

        # Send aggregated result from A (0) to B (1)
        total_enc_at_b = simp.shuffle_static(total_enc_at_a, {1: 0})

        # --- Step 6: Decryption (Party B) ---

        def decrypt_result(total_enc, encoder_local, sk_local):
            return phe.decrypt_auto(total_enc, encoder_local, sk_local)

        final_result = simp.pcall_static(
            (1,), decrypt_result, total_enc_at_b, encoder, sk
        )

        # Finalize the trace
        graph = tracer.finalize(final_result)

    # Execute
    sim = mp.make_simulator(2)
    try:
        logger.info("Executing graph...")
        result = sim.evaluate_graph(graph, {})
        logger.info(f"Result: {result}")

        # Verification
        # Compute expected value
        np.random.seed(42)
        scores_data = np.random.rand(N).astype(np.float32) * 100
        np.random.seed(43)
        amounts_data = np.random.rand(N).astype(np.float32) * 1000

        sorted_indices = np.argsort(scores_data)[::-1]
        top_k_amounts = amounts_data[sorted_indices[:K]]
        expected = float(np.sum(top_k_amounts))

        # Result is on Party 1
        actual_val = result[1]
        # It might be an array or scalar
        if hasattr(actual_val, "item"):
            actual = float(actual_val.item())
        elif isinstance(actual_val, (list, np.ndarray)):
            actual = float(actual_val[0])
        else:
            actual = float(actual_val)

        logger.info(f"Expected: {expected}")
        logger.info(f"Actual:   {actual}")

        if abs(actual - expected) < 1e-2:
            logger.info("SUCCESS: Result matches expectation!")
        else:
            logger.info("FAILURE: Result mismatch!")

    finally:
        sim.shutdown()


if __name__ == "__main__":
    run_benchmark()
