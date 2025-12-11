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

"""Tests for LDPC module."""

import numpy as np
import pytest
import scipy.sparse as sp

from mplang.v2.libs.mpc.vole.ldpc import (
    generate_silver_ldpc,
    get_silver_params,
    verify_ldpc_structure,
)


def ldpc_encode_numpy(message: np.ndarray, H: sp.csr_matrix) -> np.ndarray:
    """Compute syndrome S = H * message using NumPy (test utility).

    Args:
        message: Message vector of shape (n,) or (n, 2)
        H: LDPC parity check matrix of shape (m, n)

    Returns:
        Syndrome vector of shape (m,) or (m, 2)
    """
    m, _n = H.shape

    if message.ndim == 1:
        # Binary case: S = H @ message mod 2
        syndrome = (H @ message.astype(np.int64)) % 2
        return syndrome.astype(np.uint8)
    else:
        # GF(2^128) case: XOR accumulation
        syndrome = np.zeros((m, message.shape[1]), dtype=message.dtype)
        for i in range(m):
            start, end = H.indptr[i], H.indptr[i + 1]
            cols = H.indices[start:end]
            for j in cols:
                syndrome[i] ^= message[j]
        return syndrome


class TestLDPCGeneration:
    """Test LDPC matrix generation."""

    def test_ldpc_shape(self):
        """Verify LDPC matrix has correct shape."""
        n, m = 1000, 100
        H = generate_silver_ldpc(n, m)

        assert H.shape == (m, n)
        assert isinstance(H, sp.csr_matrix)

    def test_ldpc_sparsity(self):
        """Verify LDPC matrix is sparse."""
        n, m = 1000, 100
        H = generate_silver_ldpc(n, m)

        density = H.nnz / (m * n)
        assert density < 0.1  # Less than 10% fill

    def test_ldpc_deterministic(self):
        """Verify same seed produces same matrix."""
        n, m = 500, 50

        H1 = generate_silver_ldpc(n, m, seed=42)
        H2 = generate_silver_ldpc(n, m, seed=42)

        assert (H1 != H2).nnz == 0  # No differences

    def test_ldpc_different_seeds(self):
        """Verify different seeds produce different matrices."""
        n, m = 500, 50

        H1 = generate_silver_ldpc(n, m, seed=42)
        H2 = generate_silver_ldpc(n, m, seed=43)

        assert (H1 != H2).nnz > 0  # Some differences

    def test_verify_structure(self):
        """Verify LDPC structure validation."""
        n, m = 1000, 100
        H = generate_silver_ldpc(n, m)

        assert verify_ldpc_structure(H)


class TestLDPCEncoding:
    """Test LDPC syndrome encoding."""

    def test_encode_binary(self):
        """Test binary syndrome computation."""
        n, m = 100, 10
        H = generate_silver_ldpc(n, m, seed=42)

        message = np.random.randint(0, 2, size=n, dtype=np.uint8)
        syndrome = ldpc_encode_numpy(message, H)

        assert syndrome.shape == (m,)
        assert np.all(syndrome <= 1)  # Binary

    def test_encode_128bit(self):
        """Test 128-bit syndrome computation."""
        n, m = 100, 10
        H = generate_silver_ldpc(n, m, seed=42)

        message = np.random.randint(0, 2**63, size=(n, 2), dtype=np.uint64)
        syndrome = ldpc_encode_numpy(message, H)

        assert syndrome.shape == (m, 2)
        assert syndrome.dtype == np.uint64

    def test_encode_linearity(self):
        """Verify linearity: encode(a XOR b) = encode(a) XOR encode(b)."""
        n, m = 100, 10
        H = generate_silver_ldpc(n, m, seed=42)

        a = np.random.randint(0, 2, size=n, dtype=np.uint8)
        b = np.random.randint(0, 2, size=n, dtype=np.uint8)

        # Encode individually
        sa = ldpc_encode_numpy(a, H)
        sb = ldpc_encode_numpy(b, H)

        # Encode XOR
        s_xor = ldpc_encode_numpy((a ^ b).astype(np.uint8), H)

        # Verify linearity
        np.testing.assert_array_equal(s_xor, (sa ^ sb) % 2)


class TestSilverParams:
    """Test Silver parameter selection."""

    def test_params_scaling(self):
        """Verify parameters scale appropriately."""
        # Small n
        cl1, sl1, _nw1 = get_silver_params(1000)
        assert cl1 == 1000
        assert sl1 >= 128

        # Large n
        cl2, sl2, _nw2 = get_silver_params(1000000)
        assert cl2 == 1000000
        assert sl2 >= 128
        assert sl2 < cl2 / 5  # Good compression


class TestCommunicationEstimates:
    """Test communication cost estimation."""

    def test_silver_vs_gilboa(self):
        """Verify Silver has lower communication than Gilboa."""
        from mplang.v2.libs.mpc.vole.silver import estimate_silver_communication

        n = 1000000
        est = estimate_silver_communication(n)

        assert est["silver_bytes"] < est["gilboa_bytes"]
        assert est["compression_ratio"] > 9  # At least 9x compression
        print(f"Silver: {est['silver_bytes'] / 1024:.2f} KB")
        print(f"Gilboa: {est['gilboa_bytes'] / 1024 / 1024:.2f} MB")
        print(f"Compression: {est['compression_ratio']:.1f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
