import jax.numpy as jnp
import numpy as np

import mplang2.dialects.phe as phe
import mplang2.dialects.simp as simp
import mplang2.dialects.tensor as tensor
import mplang2.edsl as el
import mplang2.edsl.typing as elt
from mplang2.edsl.printer import GraphPrinter


def create_example():
    # 1. Define the computation in a trace context
    with el.Tracer() as tracer:
        # --- Step 1: Data Setup (Distributed) ---

        # Party A (Rank 0) creates scores
        def create_scores():
            return tensor.constant(
                np.array([10.0, 50.0, 20.0, 40.0, 30.0], dtype=np.float32)
            )

        scores = simp.pcall_static((0,), create_scores)
        # Or we can use the following:
        # scores = simp.pcall_static(
        #     (0,),
        #     tensor.constant,
        #     np.array([10.0, 50.0, 20.0, 40.0, 30.0], dtype=np.float32),
        # )

        # Party B (Rank 1) creates amounts
        def create_amounts():
            return tensor.constant(
                np.array([100.0, 500.0, 200.0, 400.0, 300.0], dtype=np.float32)
            )

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
        # Note: pcall returns a tuple if the function returns a tuple
        enc_amounts, _pk, sk, encoder = simp.pcall_static(
            (1,), setup_and_encrypt, amounts
        )

        # --- Step 3: Transfer Encrypted Data to A ---

        # Send encrypted amounts from B (1) to A (0)
        enc_amounts_at_a = simp.shuffle_static(enc_amounts, {0: 1})

        # --- Step 4: Sorting (Party A) ---
        def sort_and_aggregate(scores_local, enc_amounts_local):
            # 1. Compute sort indices
            def argsort_desc(x):
                return jnp.argsort(x)[::-1]

            indices = tensor.run_jax(argsort_desc, scores_local)

            # 2. Reorder encrypted amounts
            sorted_enc = tensor.gather(enc_amounts_local, indices)

            # 3. Aggregate top K
            K = 3
            top_k = tensor.slice_tensor(sorted_enc, (0,), (K,))

            # Sum top K
            total = tensor.slice_tensor(top_k, (0,), (1,))
            for i in range(1, K):
                next_elem = tensor.slice_tensor(top_k, (i,), (i + 1,))
                total = phe.add(total, next_elem)

            return total

        # Execute on Party A (Rank 0)
        total_enc_at_a = simp.pcall_static(
            (0,), sort_and_aggregate, scores, enc_amounts_at_a
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

        # Finalize the trace to get the execution graph
        graph = tracer.finalize(final_result)

        return graph


def main():
    print("Building PHE Sort & Aggregate Example (SIMP Dialect)...")
    graph = create_example()

    print("\nExecution Graph:")
    print("-" * 50)
    printer = GraphPrinter()
    print(printer.format(graph))
    print("-" * 50)


if __name__ == "__main__":
    main()
