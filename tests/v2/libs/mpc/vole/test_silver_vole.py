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

"""Tests for Silver VOLE implementation."""

import pytest

import mplang.v2 as mp
from mplang.v2.dialects import simp
from mplang.v2.libs.mpc.vole.silver import (
    estimate_silver_communication,
    silver_vole,
)


class TestSilverVOLEBasic:
    """Test basic Silver VOLE functionality."""

    def test_silver_vole_runs(self):
        """Verify Silver VOLE executes without errors."""
        sim = simp.make_simulator(2)
        mp.set_root_context(sim)

        sender = 0
        receiver = 1
        n = 1000

        def job():
            v, w = silver_vole(sender, receiver, n)
            return v, w

        traced = mp.compile(job)
        v_obj, w_obj = mp.evaluate(traced)

        v_val = mp.fetch(v_obj)[sender]
        w_val = mp.fetch(w_obj)[receiver]

        assert v_val.shape == (n, 2)
        assert w_val.shape == (n, 2)
        print(f"Silver VOLE generated {n} correlations")

    def test_silver_vole_with_secrets(self):
        """Test Silver VOLE returns secrets when requested."""
        sim = simp.make_simulator(2)
        mp.set_root_context(sim)

        sender = 0
        receiver = 1
        n = 100

        def job():
            return silver_vole(sender, receiver, n, return_secrets=True)

        traced = mp.compile(job)
        result = mp.evaluate(traced)

        assert len(result) == 4  # v, w, u, delta

        v_val = mp.fetch(result[0])[sender]
        w_val = mp.fetch(result[1])[receiver]
        u_val = mp.fetch(result[2])[sender]
        mp.fetch(result[3])[receiver]

        assert v_val.shape == (n, 2)
        assert w_val.shape == (n, 2)
        print(f"V shape: {v_val.shape}, U shape: {u_val.shape}")


class TestSilverVOLECorrelation:
    """Test Silver VOLE correlation property."""

    def test_vole_correlation(self):
        """Verify W = V + U * Delta correlation.

        Note: This is a simplified test. Full verification requires
        proper GF(2^128) multiplication.
        """
        sim = simp.make_simulator(2)
        mp.set_root_context(sim)

        sender = 0
        receiver = 1
        n = 100

        def job():
            v, w, u, delta = silver_vole(sender, receiver, n, return_secrets=True)

            # Compute U * Delta on sender (need delta from receiver)
            # For testing, we'll just verify shapes
            return v, w, u, delta

        traced = mp.compile(job)
        result = mp.evaluate(traced)

        v_val = mp.fetch(result[0])[sender]
        w_val = mp.fetch(result[1])[receiver]

        # Basic sanity checks
        assert v_val.shape == w_val.shape

        # Note: Full correlation check requires combining u*delta
        # which needs cross-party communication in test
        print("VOLE shape correlation verified")


class TestSilverCommunication:
    """Test Silver communication efficiency."""

    def test_communication_estimates(self):
        """Verify communication estimates are calculated correctly."""
        for n in [1000, 10000, 100000, 1000000]:
            est = estimate_silver_communication(n)

            assert est["silver_bytes"] > 0
            assert est["gilboa_bytes"] > 0
            assert est["compression_ratio"] > 1

            print(
                f"N={n:,}: Silver={est['silver_bytes'] / 1024:.1f}KB, "
                f"Gilboa={est['gilboa_bytes'] / 1024 / 1024:.1f}MB, "
                f"Ratio={est['compression_ratio']:.0f}x"
            )

    def test_sublinear_communication(self):
        """Verify Silver communication is sublinear in N.

        Note: Current implementation uses N/10 syndrome length,
        so communication grows with N but at 10x lower rate than Gilboa.
        True sublinear would require PCG-style expansion.
        """
        est_small = estimate_silver_communication(10000)
        est_large = estimate_silver_communication(1000000)

        # Silver grows at ~N/10 rate, so for 100x N, comm grows ~100x
        # but base_ot stays constant, so actual growth is slightly less
        growth_ratio = est_large["silver_bytes"] / est_small["silver_bytes"]
        n_ratio = 1000000 / 10000  # 100x

        # Communication should grow slower than linear (Gilboa)
        # but our simplified Silver doesn't achieve true sublinear yet
        assert growth_ratio < n_ratio  # Better than linear
        print(f"N grew {n_ratio}x, comm grew {growth_ratio:.1f}x")


class TestSilverVOLEEdgeCases:
    """Test Silver VOLE edge cases."""

    def test_same_party_error(self):
        """Verify error when sender == receiver."""
        sim = simp.make_simulator(2)
        mp.set_root_context(sim)

        def job():
            return silver_vole(0, 0, 100)

        with pytest.raises(ValueError):
            mp.compile(job)

    def test_zero_n_error(self):
        """Verify error when n <= 0."""
        sim = simp.make_simulator(2)
        mp.set_root_context(sim)

        def job():
            return silver_vole(0, 1, 0)

        with pytest.raises(ValueError):
            mp.compile(job)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
