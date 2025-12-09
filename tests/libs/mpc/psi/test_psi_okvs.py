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

import unittest

import jax.numpy as jnp
import numpy as np

import mplang.v2.dialects.field as field
import mplang.v2.dialects.tensor as tensor
import mplang.v2.edsl.typing as elt
from mplang.v2.edsl import Interpreter


class TestPsiOkvs(unittest.TestCase):
    def setUp(self):
        self.interp = Interpreter()

    def _to_obj(self, np_arr, dtype=None):
        """Helper to wrap numpy array using tensor.constant."""
        # Note: tensor.constant infers dtype and shape from data
        # We ignore explicit dtype/shape args as they are redundant for valid inputs
        return tensor.constant(np_arr)

    def test_okvs_linearity(self):
        """Verify Linear Property of OKVS Decode: D(P ^ W) = D(P) ^ D(W)."""
        N = 100  # Input items
        M = 128  # Storage size (approx 1.2 * N)

        # 1. Generate Inputs (Keys)
        keys = np.random.randint(0, 1000000, size=(N,), dtype=np.uint64)

        # 2. Generate Two Storage Vectors (P and W)
        # P could be from Encoding, W from VOLE.
        # Just use random W for this test.
        p_val = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=(M, 2), dtype=np.uint64)
        w_val = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=(M, 2), dtype=np.uint64)

        # 3. Compute Combined Storage Q = P ^ W
        q_val = p_val ^ w_val

        # 5. Decode Separately
        # Need context
        with self.interp:
            # 4. Wrap (Inside context)
            keys_obj = self._to_obj(keys)
            p_obj = self._to_obj(p_val)
            w_obj = self._to_obj(w_val)
            q_obj = self._to_obj(q_val)

            d_p = field.decode_okvs(keys_obj, p_obj)
            d_w = field.decode_okvs(keys_obj, w_obj)
            d_q = field.decode_okvs(keys_obj, q_obj)

            # 6. Verify D(Q) == D(P) ^ D(W)
            # We can use field.add (XOR) or unwrap and check in numpy

            res_p = d_p.runtime_obj.unwrap()
            res_w = d_w.runtime_obj.unwrap()
            res_q = d_q.runtime_obj.unwrap()

            computed_xor = res_p ^ res_w

            np.testing.assert_array_equal(
                res_q,
                computed_xor,
                err_msg="OKVS Linearity Check Failed: D(P^W) != D(P)^D(W)",
            )
            print("OKVS Linearity Verified.")

    def test_vole_psi_simulation(self):
        """Simulate RR22-lite PSI flow using OKVS and VOLE components."""
        N = 100  # Intersection size
        M = 140  # OKVS/VOLE size (Increased for success probability)

        # 1. Setup VOLE Secrets (Simulated for speed)
        # Sender has U, V. Recv has Delta, W = V + U*Delta
        u_sender = np.random.randint(
            0, 0xFFFFFFFFFFFFFFFF, size=(M, 2), dtype=np.uint64
        )
        v_sender = np.random.randint(
            0, 0xFFFFFFFFFFFFFFFF, size=(M, 2), dtype=np.uint64
        )

        delta_val = np.array([3, 0], dtype=np.uint64)  # Delta = 3

        # Receiver computes W = V + U*Delta
        # Use field.mul for correctness

        # Wrap for calculation
        # Wrap for calculation
        # u_obj = self._to_obj(u_sender)
        # delta_obj = self._to_obj(delta_val, dtype=elt.Tensor[elt.u64, (2,)])
        # v_obj = self._to_obj(v_sender)

        with self.interp:
            u_obj = self._to_obj(u_sender)
            delta_obj = self._to_obj(delta_val, dtype=elt.Tensor[elt.u64, (2,)])
            v_obj = self._to_obj(v_sender)

            # W = V + U * Delta (Broadcasting delta)
            # W = V + U * Delta (Broadcasting delta)
            # Need to broadcast delta? field.mul handles (M,2) * (2,) ??
            # Usually needs explicit broadcast or check field.mul impl.
            # checks: `field._gf128_mul_impl` assumes (N, 2) * (N, 2) or broadcast?
            # It uses `gf128_mul_batch`.
            # If shapes differ, we might need manual broadcast.

            # Manual Expand Delta
            def _expand_delta(d, ref):
                n = ref.shape[0]
                return jnp.tile(d, (n, 1))

            delta_expanded = tensor.run_jax(_expand_delta, delta_obj, u_obj)
            prod = field.mul(u_obj, delta_expanded)
            w_obj = field.add(v_obj, prod)

            # Unwrap W for Receiver's use
            w_recv = w_obj.runtime_obj.unwrap()

        # 2. Protocol Start
        # Inputs (Full intersection for now)
        common_items = np.random.randint(0, 1000000, size=(N,), dtype=np.uint64)

        # Receiver:
        # a. Map inputs to "Random Value" via Hashing (H(y))
        # H(y) here simulated by simple random.
        h_y = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=(N, 2), dtype=np.uint64)

        # b. Encode: P = Solve(Y, H(y))
        # c. Mask: Q = P ^ W (Sender receives Q)
        # Wait, masking P directly works if P is "dense"?
        # Yes, P is Storage (size M). W is Storage-sized vector (size M).
        # We need P to encode H(y).

        # Wrap
        # keys_obj = self._to_obj(common_items)
        # vals_obj = self._to_obj(h_y)
        # w_recv_obj = self._to_obj(w_recv)

        with self.interp:
            keys_obj = self._to_obj(common_items)
            vals_obj = self._to_obj(h_y)
            w_recv_obj = self._to_obj(w_recv)

            # Receiver Ops
            # Receiver Ops
            p_storage = field.solve_okvs(keys_obj, vals_obj, m=M)
            q_storage = field.add(p_storage, w_recv_obj)  # Mask P with W

            # Sender Ops (receives Q_storage)
            # Sender computes S = Decode(Q, x)
            # S = Decode(P^W, x) = Decode(P,x) ^ Decode(W,x)
            # If x=y: S = H(x) ^ (Decode(V,x) ^ Decode(U,x)*Delta)

            s_decoded = field.decode_okvs(keys_obj, q_storage)

            # Sender unmasks V:
            # T = S ^ Decode(V,x)
            v_sender_obj = self._to_obj(v_sender)
            v_decoded = field.decode_okvs(keys_obj, v_sender_obj)

            t_val = field.add(s_decoded, v_decoded)

            # Sender checks against H(x):
            # Final = T ^ H(x)
            # Should equal Decode(U, x) * Delta

            h_x_obj = vals_obj  # Same inputs for simulation
            final_sender = field.add(t_val, h_x_obj)

            # Reference Check: Decode(U, x) * Delta
            u_sender_obj = self._to_obj(u_sender)
            u_decoded = field.decode_okvs(keys_obj, u_sender_obj)

            ref_prod = field.mul(u_decoded, delta_expanded)

            # Compare
            final_val = final_sender.runtime_obj.unwrap()
            ref_val = ref_prod.runtime_obj.unwrap()

            np.testing.assert_array_equal(
                final_val,
                ref_val,
                err_msg="PSI Logic Check Failed: Sender cannot derive U*Delta!",
            )
            print("PSI Logic Verified: Sender successfully derived U*Delta share.")

    def test_psi_okvs_integration(self):
        """Test full End-to-End PSI OKVS Protocol via psi_okvs.py using SimpSimulator."""
        import mplang.v2.dialects.simp as simp
        import mplang.v2.libs.mpc.psi.okvs as psi_okvs
        from mplang.v2.backends.simp_simulator import SimpSimulator

        N = 100
        # Use IDENTICAL items for both parties to verify T == U* * Delta relation
        # (This relation only holds for intersection items)
        # Ensure UNIQUE items because OKVS Peeling solver fails if duplicate keys exist
        # (identical keys map to same positions, preventing degree-1 reduction).
        rng = np.random.default_rng()
        shared_items = rng.choice(1000000, size=N, replace=False).astype(np.uint64)

        sender_items = shared_items
        receiver_items = shared_items

        # SimpSimulator
        sim = SimpSimulator(world_size=2)

        # Define Data Providers

        # Since pcall_static captures closure, and SimpSimulator executes in threads,
        # accessing `sender_items` from outer scope works.

        # Build Graph
        # We use strict explicit placement.

        def run_protocol():
            # 1. Place Inputs
            s_items_Handle = simp.pcall_static((0,), lambda: sender_items)
            r_items_Handle = simp.pcall_static((1,), lambda: receiver_items)

            # 2. Run Protocol
            # Returns (T, U*, Delta)
            t_handle, u_star_handle, d_handle = psi_okvs.psi_intersect(
                0, 1, N, s_items_Handle, r_items_Handle
            )

            # 3. Reveal for Verification
            # Bring everything to Party 0 (Host/Verifier in this context)
            # shuffle T from 0->0 (no-op but makes view consistent)
            t_host = simp.shuffle_static(t_handle, {0: 0})
            u_star_host = simp.shuffle_static(u_star_handle, {0: 0})
            # shuffle Delta from 1->0
            d_host = simp.shuffle_static(d_handle, {0: 1})

            return t_host, u_star_host, d_host

        # Execute
        with sim:
            t_obj, u_star_obj, d_obj = run_protocol()

        # Verify Results
        # SimpSimulator Run returns InterpObject for the LAST expression in with block?
        # No, 'with sim' context just traces.
        # We need to run it.
        # SimpSimulator acts as Interpreter. When we do 'sim.evaluate_graph' or similar.
        # But `with sim:` executes eager ops if they are eager primitives.
        # `psi_intersect` uses `tensor.run_jax` which is eager in Interpreter?
        # NO. `tensor.run_jax` is Traced into Graph in SimpSimulator.

        # Wait, `SimpSimulator` inherits from `Interpreter`.
        # `Interpreter` defaults to Eager Execution for `bind`?
        # `Interpreter` default `run_jax` executes immediately?
        # NO. `SimpSimulator` overrides `_evaluate_graph`?
        # `SimpSimulator` acts as an Orchestrator.

        # Let's look at `test_simp_simulator.py`.
        # `with interp:` -> `res = ...` -> `res` is `InterpObject`.
        # Accessing `res.runtime_obj` forces execution if not done?
        # `SimpSimulator` is Eager-by-default Interpreter?
        # YES. `Interpreter` base class implements Eager execution unless specific tracing/graph mode used.
        # `pcall_static` implementation in `simp_impl` causes execution?

        # Check `mplang/v2/backends/simp_impl.py`.
        # `_pcall_static_impl` calls `interpreter.submit(...)`.

        # So yes, `t_obj` will hold the result (Future/HostVar).

        t_val = t_obj.runtime_obj.values[0].unwrap()  # Rank 0 value
        u_star_val = u_star_obj.runtime_obj.values[0].unwrap()  # Rank 0 value
        d_val = d_obj.runtime_obj.values[0].unwrap()  # Rank 0 value (received from 1)

        # Verify T = U* * Delta (Logic Check)
        # We need to compute U* * Delta + V*? No, T = U_star * Delta.
        # Wait, logic is T = S + V + H(x).
        # And S + V = U_star * Delta.
        # So T - H(x) = U_star * Delta ?
        # No, from `verify_psi_okvs_logic.py`: "Verify T == U_decoded * Delta".
        # Let's trust that logic.

        # Math verification on Host
        # Need field multiplication.
        # We can reuse field implementation or simple numpy logic if available.
        # Or Just use `verify_psi_okvs_logic` style check.
        # But we don't have field.mul available on Host easily without Interpreter context.

        # Let's assume passed if no crash for now?
        # Or assert correctness.
        # We can use `mplang.v2.backends.field_impl._gf128_mul_impl` directly!

        from mplang.v2.backends.field_impl import _gf128_mul_impl

        # Expand Delta to (N, 2)
        d_exp = np.tile(d_val, (N, 1))

        # Calc product
        expected_t = _gf128_mul_impl(u_star_val, d_exp)

        np.testing.assert_array_equal(
            t_val, expected_t, err_msg="PSI OKVS Logic Verification Failed"
        )
        print("Integration Test Passed: T = U* * Delta")


if __name__ == "__main__":
    unittest.main()
