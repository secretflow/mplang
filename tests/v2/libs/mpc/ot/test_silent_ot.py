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

import numpy as np

import mplang.v2 as mp
from mplang.v2.dialects import simp
from mplang.v2.libs.mpc.ot import silent as silent_ot


class TestSilentOT:
    def test_silent_vole_correlation(self):
        """Verify Silent Random VOLE correlation: W = V + U * Delta."""

        sim = simp.make_simulator(2)
        mp.set_root_context(sim)

        N = 10000
        sender = 0
        receiver = 1

        def job():
            # Run Silent VOLE
            # v, w, u, delta, vb, wb, ub
            # v: (N, 2), w: (N, 2), u: (N, 2), delta: (2,)
            # vb: (base_N, 2), wb: (base_N, 2), ub: (base_N, 2)
            res = silent_ot.silent_vole_random_u(sender, receiver, N, base_k=128)
            return res

        traced = mp.compile(job)
        v_obj, w_obj, u_obj, delta_obj = mp.evaluate(traced)

        # Fetch results
        # These are distributed objects.
        # v, u are on Sender (P0). w, delta are on Receiver (P1).

        v = mp.fetch(v_obj)
        v_val = v[sender]
        w = mp.fetch(w_obj)
        w_val = w[receiver]
        u = mp.fetch(u_obj)
        u_val = u[sender]
        delta = mp.fetch(delta_obj)
        delta_val = delta[receiver]

        from mplang.v2.backends.field_impl import _gf128_mul_impl

        # 2. Verify Expanded VOLE
        # W = V + U * Delta
        # This algebra is in GF(2^128). We need to simulate GF(128) mul/add.

        # To verify on host, we need gf128 mul.
        # Let's use `mp.compile` to run verification verification on one party?
        # Or just drag backend implementation here.

        # V + U * Delta
        # U * Delta
        # U is (N, 2), Delta is (2,)
        # Broadcast Delta
        delta_broad = np.tile(delta_val, (N, 1))

        term = _gf128_mul_impl(u_val, delta_broad)
        rhs = v_val ^ term

        # Check equality
        # W ?= V + U*D
        np.testing.assert_array_equal(w_val, rhs, err_msg="Expanded VOLE Broken")

    def test_silent_vole_randomness(self):
        """Verify that outputs are distinguishable from zero (basic randomness check)."""
        sim = simp.make_simulator(2)
        mp.set_root_context(sim)
        N = 1000

        def job():
            return silent_ot.silent_vole_random_u(0, 1, N, base_k=128)

        res_objs = mp.evaluate(mp.compile(job))
        res = mp.fetch(res_objs)

        # Sender: v, u
        v_val = res[0][0]
        u_val = res[2][0]

        # Check not all zeros
        assert np.any(v_val != 0)
        assert np.any(u_val != 0)

        # Check u is not same as v
        assert not np.array_equal(u_val, v_val)
