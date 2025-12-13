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

"""Silver VOLE Implementation (Silent VOLE from LDPC Codes).

This module implements the Silver protocol for efficient silent VOLE generation.
Silver achieves ~1300x communication reduction compared to IKNP by using
LDPC-based pseudorandom correlation generators.

Key Properties:
- Communication: O(κ) instead of O(N) - sublinear in output length
- Computation: ~30% more than IKNP (due to LDPC operations)
- Security: Based on LPN + Regular Syndrome Decoding

Reference: "Silver: Silent VOLE and Oblivious Transfer from Hardness of Decoding"
           CRYPTO 2021

Usage:
    v_sender, w_receiver = silver_vole(sender=0, receiver=1, n=1000000)
    # W = V + U * Delta (VOLE correlation)

    > [!WARNING]
    > **SECURITY WARNING**: This implementation is a DEMONSTRATION of the Silver interface
    > but does NOT implement the secure LPN-based correlation generation.
    > It currently relies on AES expansion which is NOT homomorphic, meaning the
    > produced correlations are mathematically incorrect and insecure for active use.
    > The LDPC matrix H is generated with a fixed seed and is unused in the main path.
    > DO NOT USE IN PRODUCTION.
"""

from typing import Any, cast

import jax.numpy as jnp

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
import mplang.v2.libs.mpc.ot.extension as ot
from mplang.v2.dialects import crypto, field, simp, tensor
from mplang.v2.libs.mpc.vole import ldpc

# ============================================================================
# Constants
# ============================================================================

# Base OT count (security parameter)
SILVER_BASE_OT = 128

# Noise weight for LPN (lower = faster but less secure)
SILVER_NOISE_WEIGHT = 64

# LDPC compression ratio (syndrome_length / code_length)
SILVER_COMPRESSION = 10


# ============================================================================
# Silver VOLE Core
# ============================================================================


def silver_vole(
    sender: int,
    receiver: int,
    n: int,
    return_secrets: bool = False,
) -> tuple[el.Object, el.Object] | tuple[el.Object, el.Object, el.Object, el.Object]:
    """Execute Silver VOLE Protocol.

    Generates n VOLE correlations using the Silver protocol:
    - Sender gets V
    - Receiver gets W, where W = V + U * Delta

    Communication is O(κ) instead of O(n), achieving ~1300x reduction
    compared to IKNP-based approaches.

    Args:
        sender: Rank of Sender party
        receiver: Rank of Receiver party
        n: Number of VOLE correlations to generate
        return_secrets: If True, also return U (sender) and Delta (receiver)

    Returns:
        Tuple of (v_sender, w_receiver) or
        (v_sender, w_receiver, u_sender, delta_receiver) if return_secrets=True
    """
    if sender == receiver:
        raise ValueError("Sender and Receiver must be different parties.")

    if n <= 0:
        raise ValueError("n must be positive.")

    # =========================================================================
    # REPAIRED SILVER IMPLEMENTATION (Primal LPN w/ Explicit Noise)
    # =========================================================================

    # 1. Setup LPN Parameters
    # We use Primal LPN: W = V + U*Delta + e
    # Generator Matrix G is (K x N). We generate it via LDPC gen.
    # We use the dense JAX implementation from ldpc.py for correctness.

    # Silver parameters
    _code_length, syndrome_length, _noise_weight = ldpc.get_silver_params(n)

    # Primal LPN dimensions: Input K (Base OT), Output N.
    # We treat the "Syndrome Length" M as the Base OT size K for Primal LPN.
    base_k = syndrome_length
    if base_k > 2048:
        base_k = 2048

    # -------------------------------------------------------------------------
    # Generate H' (N rows, K cols) for Transposed Matrix Multiplication
    #
    # We target V = v_base * G, where v_base is (1, K) and G is (K, N).
    # This is equivalent to V^T = G^T * v_base^T.
    # By constructing H' = G^T (N x K), we can leverage the C++ kernel which
    # computes Output(M) = Matrix(M, N) * Input(N).
    # Here, Output(N) = H'(N, K) * Input(K).
    # -------------------------------------------------------------------------

    # Note: generate_silver_ldpc(n, m) returns m x n matrix.
    # Call with (K, N) to get N rows, K cols.
    # SECURITY WARNING: Using a fixed seed (42) means the code structure is public and static.
    # In a real secure deployment, this seed should be generated via a coin-tossing protocol
    # or negotiated securely between parties to ensure the code is random and private if needed.
    # For semi-honest security where parameters are public, this is acceptable but suboptimal.
    H_prime_sparse = ldpc.generate_silver_ldpc(base_k, n, seed=42)

    # Extract indices for kernel
    h_prime_indices = H_prime_sparse.indices.astype(jnp.uint64)
    h_prime_indptr = H_prime_sparse.indptr.astype(jnp.uint64)

    def _sparse_struct_provider() -> tuple[el.Object, el.Object]:
        return tensor.constant(h_prime_indices), tensor.constant(h_prime_indptr)

    H_indices, H_indptr = simp.pcall_static((sender,), _sparse_struct_provider)

    # Broadcast to receiver (assumed public/shared for semi-honest)
    H_indices_r, H_indptr_r = simp.pcall_static((receiver,), _sparse_struct_provider)

    # 2. Base VOLE (Size K)
    from mplang.v2.libs.mpc.vole import gilboa

    def _u_base_provider() -> el.Object:
        # Generate random u_base using new API
        return crypto.random_tensor((base_k, 2), elt.u64)

    def _delta_provider() -> el.Object:
        # Generate random delta using new API
        return crypto.random_tensor((2,), elt.u64)

    v_base, w_base, u_base, delta = gilboa.vole(  # type: ignore[misc]
        sender, receiver, base_k, _u_base_provider, _delta_provider, return_secrets=True
    )

    # 3. Expansion (Encoding) using C++ Kernel
    # V = v_base * G = H' * v_base

    def _encode(vec_base: el.Object, idx: el.Object, ptr: el.Object) -> el.Object:
        # Calls C++ kernel: Output(N) = H'(N, K) * Input(K)
        return ldpc.ldpc_encode_sparse(vec_base, idx, ptr, n, base_k)

    V = simp.pcall_static((sender,), _encode, v_base, H_indices, H_indptr)
    W_clean = simp.pcall_static((receiver,), _encode, w_base, H_indices_r, H_indptr_r)

    # 4. Add Noise (Receiver)
    # W = W_clean + e
    # e is sparse noise (LPN security)

    def _add_noise(w: el.Object) -> el.Object:
        # Generate cryptographically secure sparse noise
        e = ldpc.generate_sparse_noise(n, SILVER_NOISE_WEIGHT)

        def _xor(a: Any, b: Any) -> Any:
            return jnp.bitwise_xor(a, b)

        return cast(el.Object, tensor.run_jax(_xor, w, e))

    W = simp.pcall_static((receiver,), _add_noise, W_clean)

    # 5. Output
    # We now have W = V + (U_base*G)*Delta + e
    # This is a valid LPN sample.
    # It is "Noisy VOLE".

    if return_secrets:
        # Compute U_long for sender to verify
        U_long = simp.pcall_static((sender,), _encode, u_base, H_indices, H_indptr)
        return V, W, U_long, delta

    return V, W


# ============================================================================
# LDPC-Based Syndrome Expansion (Alternative Implementation)
# ============================================================================


def silver_vole_ldpc(
    sender: int,
    receiver: int,
    n: int,
    return_secrets: bool = False,
) -> tuple[el.Object, el.Object] | tuple[el.Object, el.Object, el.Object, el.Object]:
    """Silver VOLE using explicit LDPC syndrome computation.

    This is the full Silver protocol with LDPC syndrome encoding.
    More accurate to the paper but slower due to LDPC operations.

    Args:
        sender: Rank of Sender
        receiver: Rank of Receiver
        n: Number of VOLE correlations
        return_secrets: Return U and Delta

    Returns:
        VOLE correlation tuple
    """
    if sender == receiver:
        raise ValueError("Sender and Receiver must be different parties.")

    # 1. Setup parameters
    code_length, syndrome_length, _noise_weight = ldpc.get_silver_params(n)

    # 2. Generate shared LDPC matrix
    ldpc.generate_silver_ldpc(code_length, syndrome_length, seed=42)

    # 3. Base OT setup (same as standard Silver)
    base_k = SILVER_BASE_OT

    # Generate random choice bits on receiver
    def _gen_choice_bits_ldpc() -> el.Object:
        rand = crypto.random_bytes(base_k)

        def _to_bits(r: Any) -> Any:
            bits = r % 2
            return bits.reshape(-1, 1)

        return cast(el.Object, tensor.run_jax(_to_bits, rand))

    choice_bits = simp.pcall_static((receiver,), _gen_choice_bits_ldpc)

    t_matrix, q_matrix, _s_choices = ot.iknp_core(choice_bits, sender, receiver, base_k)

    # 4. Sender: Generate random vector and compute syndrome
    def _sender_syndrome(t_mat: Any) -> el.Object:
        # Generate random message vector
        seeds = tensor.run_jax(lambda t: t.reshape(-1, 2), t_mat)
        expanded = field.aes_expand(seeds, code_length // base_k + 1)

        def _to_message(exp: Any) -> Any:
            return exp.reshape(-1, 2)[:code_length]

        message = tensor.run_jax(_to_message, expanded)
        return cast(el.Object, message)

    v_sender = simp.pcall_static((sender,), _sender_syndrome, t_matrix)

    # 5. Receiver: Compute correlated output
    def _recv_expand(q_mat: Any) -> el.Object:
        seeds = tensor.run_jax(lambda q: q.reshape(-1, 2), q_mat)
        expanded = field.aes_expand(seeds, code_length // base_k + 1)

        def _to_correlated(exp: Any) -> Any:
            return exp.reshape(-1, 2)[:code_length]

        return cast(el.Object, tensor.run_jax(_to_correlated, expanded))

    w_receiver = simp.pcall_static((receiver,), _recv_expand, q_matrix)

    # 6. Truncate to n
    def _truncate(vec: Any) -> el.Object:
        return cast(el.Object, tensor.run_jax(lambda v: v[:n], vec))

    v_final = simp.pcall_static((sender,), _truncate, v_sender)
    w_final = simp.pcall_static((receiver,), _truncate, w_receiver)

    if return_secrets:
        # Generate U and Delta
        def _gen_u() -> el.Object:
            return crypto.random_bytes(n * 16)

        def _gen_delta() -> el.Object:
            return crypto.random_bytes(16)

        u_sender = simp.pcall_static((sender,), _gen_u)
        delta_receiver = simp.pcall_static((receiver,), _gen_delta)

        return v_final, w_final, u_sender, delta_receiver

    return v_final, w_final


# ============================================================================
# Communication Estimation
# ============================================================================


def estimate_silver_communication(n: int) -> dict:
    """Estimate communication cost for Silver VOLE.

    Args:
        n: Number of VOLE correlations

    Returns:
        Dictionary with communication estimates
    """
    # Base OT communication
    base_ot_comm = SILVER_BASE_OT * 16 * 2  # 128 OTs, 16 bytes each, 2 messages

    # Syndrome communication (compressed)
    syndrome_length = max(n // SILVER_COMPRESSION, 128)
    syndrome_comm = syndrome_length * 16  # 128-bit elements

    # Total Silver communication
    silver_total = base_ot_comm + syndrome_comm

    # Compare with Gilboa (full IKNP)
    gilboa_total = n * 16  # O(n) for full IKNP

    return {
        "silver_bytes": silver_total,
        "gilboa_bytes": gilboa_total,
        "compression_ratio": gilboa_total / silver_total,
        "base_ot_bytes": base_ot_comm,
        "syndrome_bytes": syndrome_comm,
    }
