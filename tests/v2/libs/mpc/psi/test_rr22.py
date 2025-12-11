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

from typing import Any

import jax.numpy as jnp
import numpy as np

import mplang.v2 as mp
import mplang.v2.dialects.field as field
import mplang.v2.dialects.simp as simp
import mplang.v2.dialects.tensor as tensor
import mplang.v2.edsl.typing as elt
from mplang.v2.edsl import Interpreter
from mplang.v2.libs.mpc.psi import rr22 as psi_okvs


def _to_obj(np_arr: Any, dtype: Any = None) -> Any:
    """Helper to wrap numpy array using tensor.constant."""
    return tensor.constant(np_arr)


def test_okvs_linearity() -> None:
    """Verify Linear Property of OKVS Decode: D(P ^ W) = D(P) ^ D(W)."""
    interp = Interpreter()
    N = 100  # Input items
    M = 128  # Storage size (approx 1.2 * N)

    # 1. Generate Inputs (Keys)
    keys = np.random.randint(0, 1000000, size=(N,), dtype=np.uint64)

    # 2. Generate Two Storage Vectors (P and W)
    p_val = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=(M, 2), dtype=np.uint64)
    w_val = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=(M, 2), dtype=np.uint64)

    # 3. Compute Combined Storage Q = P ^ W
    q_val = p_val ^ w_val

    # 5. Decode Separately
    with interp:
        # 4. Wrap (Inside context)
        keys_obj = _to_obj(keys)
        p_obj = _to_obj(p_val)
        w_obj = _to_obj(w_val)
        q_obj = _to_obj(q_val)
        seed_obj = _to_obj(np.array([0, 0], dtype=np.uint64))

        d_p = field.decode_okvs(keys_obj, p_obj, seed_obj)
        d_w = field.decode_okvs(keys_obj, w_obj, seed_obj)
        d_q = field.decode_okvs(keys_obj, q_obj, seed_obj)

        # 6. Verify D(Q) == D(P) ^ D(W)
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


def test_vole_psi_simulation() -> None:
    """Simulate RR22-lite PSI flow using OKVS and VOLE components."""
    interp = Interpreter()
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

    with interp:
        u_obj = _to_obj(u_sender)
        delta_obj = _to_obj(delta_val, dtype=elt.Tensor[elt.u64, (2,)])
        v_obj = _to_obj(v_sender)

        # Manual Expand Delta
        def _expand_delta(d: Any, ref: Any) -> Any:
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

    with interp:
        keys_obj = _to_obj(common_items)
        vals_obj = _to_obj(h_y)
        w_recv_obj = _to_obj(w_recv)

        # Receiver Ops
        seed_obj = _to_obj(np.array([0x123, 0x456], dtype=np.uint64))
        p_storage = field.solve_okvs(keys_obj, vals_obj, M, seed_obj)
        q_storage = field.add(p_storage, w_recv_obj)  # Mask P with W

        # Sender Ops (receives Q_storage)
        # Sender computes S = Decode(Q, x)
        # S = Decode(P^W, x) = Decode(P,x) ^ Decode(W,x)
        s_decoded = field.decode_okvs(keys_obj, q_storage, seed_obj)

        # Sender unmasks V:
        # T = S ^ Decode(V,x)
        v_sender_obj = _to_obj(v_sender)
        v_decoded = field.decode_okvs(keys_obj, v_sender_obj, seed_obj)

        t_val = field.add(s_decoded, v_decoded)

        # Sender checks against H(x):
        # Final = T ^ H(x)
        # Should equal Decode(U, x) * Delta

        h_x_obj = vals_obj  # Same inputs for simulation
        final_sender = field.add(t_val, h_x_obj)

        # Reference Check: Decode(U, x) * Delta
        u_sender_obj = _to_obj(u_sender)
        u_decoded = field.decode_okvs(keys_obj, u_sender_obj, seed_obj)

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


def test_rr22_integration() -> None:
    """Test full End-to-End PSI OKVS Protocol via rr22.py using SimpSimulator."""
    N = 100
    # Use IDENTICAL items to verify T == U* * Delta
    rng = np.random.default_rng()
    shared_items = rng.choice(1000000, size=N, replace=False).astype(np.uint64)

    sender_items = shared_items
    receiver_items = shared_items

    SENDER = 0
    RECEIVER = 1

    sim = mp.Simulator.simple(2)

    def job() -> Any:
        # 1. Place Inputs
        s_items_Handle = simp.constant((SENDER,), sender_items)
        r_items_Handle = simp.constant((RECEIVER,), receiver_items)

    # 2. Run Protocol
        # Returns intersection_mask (on Sender)
        mask_handle = psi_okvs.psi_intersect(
            SENDER, RECEIVER, N, s_items_Handle, r_items_Handle
        )
        return mask_handle

    # Execute
    traced = mp.compile(sim, job)
    mask_obj = mp.evaluate(sim, traced)

    # Verify Results
    # Mask is on Sender. Should be all 1s (because items are identical).
    mask_val = mp.fetch(sim, mask_obj)[SENDER]

    # Verify All Matched
    # mask_val is (N,) uint8
    expected_mask = np.ones((N,), dtype=np.uint8)

    np.testing.assert_array_equal(
        mask_val,
        expected_mask,
        err_msg="PSI Integration Failed: Not all items matched!",
    )
    print("Integration Test Passed: Sender received correct Intersection Mask.")
